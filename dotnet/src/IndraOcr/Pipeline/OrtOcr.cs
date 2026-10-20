using IndraOcr.Imaging;
using IndraOcr.Lm;
using IndraOcr.Models;
using IndraOcr.Numerics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using MetaType = IndraOcr.Pipeline.Meta;
using VocabType = IndraOcr.Vocab.Vocab;

namespace IndraOcr.Pipeline;

/// <summary>
/// The detection and recognition half of pipeline.py on ONNX Runtime.  Reading
/// order and markdown (S3) are in <see cref="Assembly"/>; the PDF text layer is
/// the Converter's job (S4).
/// </summary>
public sealed class OrtOcr : IDisposable
{
    public const float MinConf = 0.5f;
    public const double RouteSure = 0.95;
    public const int AdaptiveMinH = 9;
    public const int AdaptiveTargetH = 12;
    public const int AdaptiveMaxSide = 2048;
    public const float PictureMin = 0.5f;
    public const float PictureMinArea = 0.004f;
    public const int LmBeam = 8;
    public const double LmCandFloor = 3e-3;

    private readonly List<IDisposable> _owned = new();
    private readonly Dictionary<InferenceSession, LmConfig> _lms = new();

    private readonly record struct LmConfig(NgramLm Lm, double Alpha, double Beta, double Gate);

    public OrtOcr(string modelDir, int threads = 4, bool adaptive = true, string[]? providers = null)
    {
        var so = OrtUtil.Options(threads);
        _owned.Add(so);
        ModelMeta = MetaType.Load(Path.Combine(modelDir, "meta.json"));
        Det = Load(Path.Combine(modelDir, "det.onnx"), so);
        string rec = File.Exists(Path.Combine(modelDir, "rec.int8.onnx"))
            ? Path.Combine(modelDir, "rec.int8.onnx")
            : Path.Combine(modelDir, "rec.onnx");
        Rec = Load(rec, so);
        Mask = Optional(Path.Combine(modelDir, "mask.onnx"), so);
        Order = Optional(Path.Combine(modelDir, "order.onnx"), so);
        Orient = Optional(Path.Combine(modelDir, "orient.onnx"), so);

        Vocab = VocabType.FromJson(ModelMeta.VocabJson);
        foreach (var (group, json) in ModelMeta.VocabGroups)
        {
            string file = Path.Combine(modelDir, $"rec_{group}.onnx");
            if (File.Exists(file))
            {
                Groups[group] = (Load(file, so), VocabType.FromJson(json));
            }
        }
        if (Groups.Count > 0 && File.Exists(Path.Combine(modelDir, "script.onnx")))
        {
            Script = Load(Path.Combine(modelDir, "script.onnx"), so);
        }
        // Language models: lm.npz for rec.onnx, lm_<group>.npz for a group's
        // recognizer.  Each needs its recognizer's top-k outputs.
        var recognizers = new List<(string Name, InferenceSession Session)> { ("base", Rec) };
        foreach (var (group, entry) in Groups)
        {
            recognizers.Add((group, entry.Session));
        }
        foreach (var (name, session) in recognizers)
        {
            string file = Path.Combine(modelDir, name == "base" ? "lm.npz" : $"lm_{name}.npz");
            bool hasTopK = session.OutputMetadata.ContainsKey("blank")
                           && session.OutputMetadata.ContainsKey("topv")
                           && session.OutputMetadata.ContainsKey("topi");
            if (File.Exists(file) && hasTopK)
            {
                var (alpha, beta, gate) = ModelMeta.LmParamsFor(name);
                _lms[session] = new LmConfig(NgramLm.Load(file), alpha, beta, gate);
            }
        }
        LongSide = ModelMeta.DetLongSide;
        Height = ModelMeta.RecHeight;
        Window = ModelMeta.RecWindow;
        Adaptive = adaptive;
    }

    public MetaType ModelMeta { get; }
    public InferenceSession Det { get; }
    public InferenceSession Rec { get; }
    public InferenceSession? Mask { get; }
    public InferenceSession? Order { get; }
    public InferenceSession? Orient { get; }
    public InferenceSession? Script { get; }
    public VocabType Vocab { get; }
    public Dictionary<string, (InferenceSession Session, VocabType Vocab)> Groups { get; } =
        new(StringComparer.Ordinal);
    public int LongSide { get; }
    public int Height { get; }
    public int Window { get; }
    public bool Adaptive { get; }

    private InferenceSession Load(string path, SessionOptions so)
    {
        var s = new InferenceSession(path, so);
        _owned.Add(s);
        return s;
    }

    private InferenceSession? Optional(string path, SessionOptions so)
        => File.Exists(path) ? Load(path, so) : null;

    /// <summary>Rows of a flat [N, 5] box array as spans.</summary>
    public static float[] BoxRow(float[] boxes, int index)
    {
        var row = new float[5];
        Array.Copy(boxes, index * 5, row, 0, Math.Min(5, boxes.Length - index * 5));
        return row;
    }

    // ------------------------------------------------------------------ detect

    private (float[] Boxes, float[]? Picture, int H, int W, double Scale) DetectAt(Mat gray, int longSide)
    {
        var (x, ph, pw, scale) = DetPost.PageToInput(gray, longSide);
        var input = OrtUtil.FloatTensor(x, 1, 1, ph, pw);
        using var outputs = Det.Run(new[] { NamedOnnxValue.CreateFromTensor("x", input) });
        var tensor = outputs.First().AsTensor<float>();
        var dims = tensor.Dimensions.ToArray(); // [1, C, h, w]
        int c = dims[1], h = dims[2], w = dims[3];
        var data = tensor.ToArray();
        var boxes = DetPost.Decode(data, c, h, w);
        float[]? picture = null;
        if (c > 3)
        {
            int plane = h * w;
            picture = data[(3 * plane)..];
        }
        return (boxes, picture, h, w, scale);
    }

    /// <summary>Text-line boxes [N, 5] in page pixels, with the adaptive second pass.</summary>
    public float[] Detect(Mat gray)
    {
        var (boxes, _, h, w, scale) = DetectAt(gray, LongSide);
        int n = boxes.Length / 5;
        if (Adaptive && n >= 8)
        {
            var heights = new float[n];
            for (int i = 0; i < n; i++)
            {
                heights[i] = boxes[i * 5 + 3] - boxes[i * 5 + 1];
            }
            double medH = Np.Median(heights);
            if (medH < AdaptiveMinH)
            {
                int ls = (int)Math.Min(AdaptiveMaxSide, LongSide * (double)AdaptiveTargetH / Math.Max(medH, 1.0)) / 32 * 32;
                if (ls > LongSide)
                {
                    (boxes, _, _, _, scale) = DetectAt(gray, ls);
                }
            }
        }
        n = boxes.Length / 5;
        if (n > 0)
        {
            for (int i = 0; i < n; i++)
            {
                boxes[i * 5 + 0] = (float)Np.Clip(boxes[i * 5 + 0] / scale, 0, gray.Cols);
                boxes[i * 5 + 1] = (float)Np.Clip(boxes[i * 5 + 1] / scale, 0, gray.Rows);
                boxes[i * 5 + 2] = (float)Np.Clip(boxes[i * 5 + 2] / scale, 0, gray.Cols);
                boxes[i * 5 + 3] = (float)Np.Clip(boxes[i * 5 + 3] / scale, 0, gray.Rows);
            }
        }
        return boxes;
    }

    /// <summary>Text-line boxes and picture boxes [P, 4] (none without the picture channel).</summary>
    public (float[] Boxes, float[] Pictures) DetectAll(Mat gray)
    {
        var (boxes, picture, h, w, scale) = DetectAt(gray, LongSide);
        if (Adaptive && boxes.Length / 5 >= 8)
        {
            var heights = new float[boxes.Length / 5];
            for (int i = 0; i < heights.Length; i++)
            {
                heights[i] = boxes[i * 5 + 3] - boxes[i * 5 + 1];
            }
            double medH = Np.Median(heights);
            if (medH < AdaptiveMinH)
            {
                int ls = (int)Math.Min(AdaptiveMaxSide, LongSide * (double)AdaptiveTargetH / Math.Max(medH, 1.0)) / 32 * 32;
                if (ls > LongSide)
                {
                    (boxes, picture, h, w, scale) = DetectAt(gray, ls);
                }
            }
        }
        int n = boxes.Length / 5;
        for (int i = 0; i < n; i++)
        {
            boxes[i * 5 + 0] = (float)Np.Clip(boxes[i * 5 + 0] / scale, 0, gray.Cols);
            boxes[i * 5 + 1] = (float)Np.Clip(boxes[i * 5 + 1] / scale, 0, gray.Rows);
            boxes[i * 5 + 2] = (float)Np.Clip(boxes[i * 5 + 2] / scale, 0, gray.Cols);
            boxes[i * 5 + 3] = (float)Np.Clip(boxes[i * 5 + 3] / scale, 0, gray.Rows);
        }
        var pictures = picture is null
            ? Array.Empty<float>()
            : PictureBoxes(picture, h, w, scale, gray.Rows, gray.Cols);
        return (boxes, pictures);
    }

    /// <summary>Connected regions of det.onnx's picture channel, in page pixels.</summary>
    public static float[] PictureBoxes(
        float[] logit, int h, int w, double scale, int pageH, int pageW)
    {
        var mask = new byte[h * w];
        for (int i = 0; i < mask.Length; i++)
        {
            mask[i] = logit[i] > 0 ? (byte)1 : (byte)0; // log(P/(1-P)) at PICTURE_MIN
        }
        using var mat = ImageOps.FromBytes(mask, h, w);
        var (count, _, stats) = ImageOps.Components(mat, eightConnected: true);
        float f = (float)(2.0 / scale);
        float minArea = PictureMinArea * pageH * pageW;
        var outv = new List<float>();
        for (int i = 1; i < count; i++)
        {
            int x = stats[i * 5 + 0], y = stats[i * 5 + 1];
            int bw = stats[i * 5 + 2], bh = stats[i * 5 + 3];
            float x0 = x * f, y0 = y * f;
            float x1 = Math.Min((x + bw) * f, pageW);
            float y1 = Math.Min((y + bh) * f, pageH);
            if ((x1 - x0) * (y1 - y0) >= minArea)
            {
                outv.Add(x0);
                outv.Add(y0);
                outv.Add(x1);
                outv.Add(y1);
            }
        }
        return outv.ToArray();
    }

    // -------------------------------------------------------------- recognize

    /// <summary>
    /// (text, confidence, tokens) per line.  A line under <see cref="MinConf"/>
    /// comes back with no text.  Tokens are recognizer ids for base lines and
    /// characters for group recognizers.
    /// </summary>
    public RecLine[] Recognize(Mat gray, float[] boxes, int batch = 64)
    {
        int n = boxes.Length / 5;
        var boxList = new List<float[]>(n);
        for (int i = 0; i < n; i++)
        {
            boxList.Add(BoxRow(boxes, i));
        }
        if (n == 0)
        {
            return Array.Empty<RecLine>();
        }

        bool[]? dark = Window > 1 ? LineMask.Darkness(gray, boxList) : null;
        var crops = LineMask.CropLines(gray, boxList, Height, Mask, 1, dark);
        var tall = Window > 1
            ? LineMask.CropLines(gray, boxList, Height, null, Window, dark)
            : crops;

        var outv = new RecLine[n];
        for (int i = 0; i < n; i++)
        {
            outv[i] = RecLine.Empty;
        }
        var bestConf = Enumerable.Repeat(-1.0, n).ToArray();

        var groups = new List<(InferenceSession Session, Vocab.Vocab Vocab, List<int> Sel, bool AsChars)>();
        var baseSel = Enumerable.Range(0, n).ToList();
        if (Script is not null && crops.Count > 0)
        {
            var (names, sure) = LineGroups(crops);
            baseSel = Enumerable.Range(0, n).Where(j => names[j] == "base" || !sure[j]).ToList();
            groups.Add((Rec, Vocab, baseSel, false));
            foreach (var (name, (session, vocab)) in Groups)
            {
                groups.Add((session, vocab, Enumerable.Range(0, n).Where(j => names[j] == name).ToList(), true));
            }
        }
        else
        {
            groups.Add((Rec, Vocab, baseSel, false));
        }

        foreach (var (session, vocab, sel, asChars) in groups)
        {
            if (sel.Count == 0)
            {
                continue;
            }
            var widthList = sel.Select(j => crops[j].Width).ToList();
            foreach (var part in RecPreproc.WidthBatches(widthList, batch))
            {
                var idx = part.Select(p => sel[p]).ToList();
                var src = ReferenceEquals(session, Rec) ? tall : crops;
                var (padded, widths, height, paddedWidth) = RecPreproc.PadBatch(idx.Select(j => src[j]).ToList());
                var input = new DenseTensor<float>(padded, new[] { idx.Count, 1, height, paddedWidth });
                var widthTensor = new DenseTensor<long>(widths, new[] { idx.Count });
                bool useLm = _lms.TryGetValue(session, out var lmConfig);
                var inputValues = new[]
                {
                    NamedOnnxValue.CreateFromTensor("x", input),
                    NamedOnnxValue.CreateFromTensor("widths", widthTensor),
                };
                using var results = useLm
                    ? session.Run(inputValues, new[] { "ids", "conf", "lengths", "blank", "topv", "topi" })
                    : session.Run(inputValues, new[] { "ids", "conf", "lengths" });
                var idsTensor = results[0].AsTensor<long>();
                var confTensor = results[1].AsTensor<float>();
                var lenTensor = results[2].AsTensor<long>();
                var ids = idsTensor.ToArray();
                var pms = confTensor.ToArray();
                var lens = lenTensor.ToArray();
                var idDims = idsTensor.Dimensions.ToArray();
                int steps = idDims[1];
                double stride = paddedWidth / (double)Math.Max(steps, 1);
                float[]? blankArr = null;
                float[]? topvArr = null;
                long[]? topiArr = null;
                int topk = 0;
                if (useLm)
                {
                    blankArr = results[3].AsTensor<float>().ToArray();
                    var topvTensor = results[4].AsTensor<float>();
                    topvArr = topvTensor.ToArray();
                    topiArr = results[5].AsTensor<long>().ToArray();
                    topk = topvTensor.Dimensions[^1];
                }

                for (int k = 0; k < idx.Count; k++)
                {
                    int j = idx[k];
                    int count = (int)lens[k];
                    var seq = new long[count];
                    var pm = new float[count];
                    Array.Copy(ids, k * steps, seq, 0, count);
                    Array.Copy(pms, k * steps, pm, 0, count);

                    var keep = new bool[count];
                    for (int t = 0; t < count; t++)
                    {
                        keep[t] = seq[t] != 0 && (t == 0 || seq[t] != seq[t - 1]);
                    }
                    var keepIdx = Np.NonZero(keep);

                    var box = boxList[j];
                    float x0 = box[0], y0 = box[1], x1 = box[2], y1 = box[3];
                    float hgt = Math.Max(1.0f, y1 - y0);
                    float ix0 = (float)Math.Floor(x0 - RecPreproc.PadX * hgt);
                    float cw = (float)Math.Ceiling(x1 + RecPreproc.PadX * hgt) - ix0;
                    float f = cw / crops[j].Width;

                    double conf = 0;
                    if (keepIdx.Length > 0)
                    {
                        double sum = 0;
                        foreach (int t in keepIdx)
                        {
                            sum += pm[t];
                        }
                        conf = sum / keepIdx.Length;
                    }

                    // A line the recognizer is unsure of (under the LM's gate) is
                    // re-read by the beam search; the greedy confidence stays.
                    int[] keptIds;
                    int[] positions;
                    if (useLm && MinConf <= conf && conf < lmConfig.Gate && count > 0)
                    {
                        var blankLine = new float[count];
                        Array.Copy(blankArr!, k * steps, blankLine, 0, count);
                        var topvLine = new float[count * topk];
                        Array.Copy(topvArr!, k * steps * topk, topvLine, 0, topvLine.Length);
                        var topiLine = new long[count * topk];
                        Array.Copy(topiArr!, k * steps * topk, topiLine, 0, topiLine.Length);
                        var (beamIds, beamSteps) = BeamSearch.Search(
                            blankLine, topvLine, topiLine, count, topk,
                            lmConfig.Lm, lmConfig.Alpha, lmConfig.Beta, LmBeam,
                            candFloor: LmCandFloor);
                        keptIds = beamIds;
                        positions = beamSteps;
                    }
                    else
                    {
                        keptIds = new int[keepIdx.Length];
                        for (int q = 0; q < keepIdx.Length; q++)
                        {
                            keptIds[q] = (int)seq[keepIdx[q]];
                        }
                        positions = keepIdx;
                    }

                    var tokens = new Token[positions.Length];
                    for (int q = 0; q < positions.Length; q++)
                    {
                        int t = positions[q];
                        float x = (float)(ix0 + (t + 0.5) * stride * f);
                        tokens[q] = asChars
                            ? Token.FromChar(vocab.Chars[keptIds[q] - 1], x)
                            : Token.FromId(keptIds[q], x);
                    }

                    if (conf <= bestConf[j])
                    {
                        continue;
                    }
                    bestConf[j] = conf;
                    outv[j] = conf >= MinConf
                        ? new RecLine(vocab.Decode(keptIds), conf, tokens)
                        : new RecLine("", conf, Array.Empty<Token>());
                }
            }
        }
        return outv;
    }

    /// <summary>Groups for line boxes on the page (script.onnx); all "base" without it.</summary>
    public (string[] Groups, bool[] Sure) BoxGroups(Mat gray, float[] boxes)
    {
        int n = boxes.Length / 5;
        if (Script is null || n == 0)
        {
            return (Enumerable.Repeat("base", n).ToArray(), new bool[n]);
        }
        var boxList = new List<float[]>(n);
        for (int i = 0; i < n; i++)
        {
            boxList.Add(BoxRow(boxes, i));
        }
        bool[]? dark = Window > 1 ? LineMask.Darkness(gray, boxList) : null;
        var crops = LineMask.CropLines(gray, boxList, Height, Mask, 1, dark);
        return LineGroups(crops);
    }

    /// <summary>
    /// Per line crop, the group whose recognizer reads it, and whether the
    /// classifier is sure (probability over <see cref="RouteSure"/>).
    /// </summary>
    public (string[] Groups, bool[] Sure) LineGroups(IReadOnlyList<LineCrop> crops)
    {
        var groups = Enumerable.Repeat("base", crops.Count).ToArray();
        var sure = Enumerable.Repeat(true, crops.Count).ToArray();
        if (Script is null)
        {
            return (groups, sure);
        }
        foreach (var part in RecPreproc.WidthBatches(crops.Select(c => c.Width).ToList()))
        {
            var (batch, widths, height, paddedWidth) = RecPreproc.PadBatch(part.Select(j => crops[j]).ToList());
            var input = new DenseTensor<float>(batch, new[] { part.Count, 1, height, paddedWidth });
            var widthTensor = new DenseTensor<long>(widths, new[] { part.Count });
            using var results = Script.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor("x", input),
                NamedOnnxValue.CreateFromTensor("widths", widthTensor),
            });
            var pTensor = results.First().AsTensor<float>();
            var p = pTensor.ToArray();
            var dims = pTensor.Dimensions.ToArray();
            for (int k = 0; k < part.Count; k++)
            {
                int j = part[k];
                string g;
                double pk;
                if (dims.Length == 1)
                {
                    double v = p[k];
                    g = v > 0.5 ? "cjk" : "base";
                    pk = Math.Max(v, 1.0 - v);
                }
                else
                {
                    int best = 0;
                    for (int c = 1; c < dims[1]; c++)
                    {
                        if (p[k * dims[1] + c] > p[k * dims[1] + best])
                        {
                            best = c;
                        }
                    }
                    g = ModelMeta.ScriptGroups[best];
                    pk = p[k * dims[1] + best];
                }
                groups[j] = Groups.ContainsKey(g) ? g : "base";
                sure[j] = pk > RouteSure;
            }
        }
        return (groups, sure);
    }

    public void Dispose()
    {
        foreach (var d in _owned)
        {
            d.Dispose();
        }
        _owned.Clear();
    }
}
