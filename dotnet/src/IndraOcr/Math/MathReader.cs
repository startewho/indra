using System.Text.Json;
using IndraOcr.Imaging;
using IndraOcr.Numerics;
using IndraOcr.Pipeline;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace IndraOcr.MathOcr;

/// <summary>
/// math.py: a classifier finds lines that hold math, and MathReader reads them
/// with an attention decoder as text with inline LaTeX ("we have $x _ { i }
/// \\leq 1$").  Models are optional; the pipeline skips math without them.
/// </summary>
public sealed class MathReader : IDisposable
{
    public const float MinProbability = 0.5f;
    public const double MinConf = 0.9;
    public const double EarlyStopMargin = 0.1;
    public const int Bos = 1;
    public const int Eos = 2;
    public const int NSpecial = 4;

    private static readonly string[] Files =
    {
        "mathcls.onnx", "math_enc.onnx", "math_dec_init.onnx", "math_dec_step.onnx", "math_meta.json",
    };

    private readonly InferenceSession _cls;
    private readonly InferenceSession _enc;
    private readonly InferenceSession _init;
    private readonly InferenceSession _step;
    private readonly HashSet<string> _initInputs;
    private readonly HashSet<string> _stepInputs;
    private readonly string[] _itos;

    private MathReader(InferenceSession cls, InferenceSession enc, InferenceSession init, InferenceSession step,
        string[] itos, int height, int maxTokens, int layers, int heads, int headDim)
    {
        _cls = cls;
        _enc = enc;
        _init = init;
        _step = step;
        _itos = itos;
        Height = height;
        MaxTokens = maxTokens;
        Layers = layers;
        Heads = heads;
        HeadDim = headDim;
        _initInputs = init.InputMetadata.Keys.ToHashSet(StringComparer.Ordinal);
        _stepInputs = step.InputMetadata.Keys.ToHashSet(StringComparer.Ordinal);
    }

    public int Height { get; }
    public int MaxTokens { get; }
    public int Layers { get; }
    public int Heads { get; }
    public int HeadDim { get; }

    public static MathReader? Load(string onnxDir, SessionOptions options)
    {
        if (!Files.All(f => File.Exists(Path.Combine(onnxDir, f))))
        {
            return null;
        }
        var sessions = Files.Take(4)
            .Select(f => new InferenceSession(Path.Combine(onnxDir, f), options))
            .ToArray();
        try
        {
            using var doc = JsonDocument.Parse(
                File.ReadAllText(Path.Combine(onnxDir, "math_meta.json"), System.Text.Encoding.UTF8));
            var cfg = doc.RootElement.GetProperty("config");
            int height = cfg.GetProperty("height").GetInt32();
            int maxTokens = cfg.GetProperty("max_tokens").GetInt32();
            int layers = cfg.GetProperty("dec_layers").GetInt32();
            int heads = cfg.GetProperty("dec_heads").GetInt32();
            int channels = cfg.GetProperty("channels").GetInt32();
            var tokens = doc.RootElement.GetProperty("tokens").EnumerateArray()
                .Select(e => e.GetString()!).ToArray();
            var itos = new[] { "", "", "", "" }.Concat(tokens).ToArray();
            return new MathReader(sessions[0], sessions[1], sessions[2], sessions[3],
                itos, height, maxTokens, layers, heads, channels / heads);
        }
        catch
        {
            foreach (var s in sessions)
            {
                s.Dispose();
            }
            throw;
        }
    }

    /// <summary>Per line crop (the recognizer's 32 px crop): does it hold math?</summary>
    public bool[] Flags(IReadOnlyList<LineCrop> crops)
    {
        var outv = new bool[crops.Count];
        if (crops.Count == 0)
        {
            return outv;
        }
        var widths = crops.Select(c => c.Width).ToList();
        foreach (var part in RecPreproc.WidthBatches(widths))
        {
            var (batch, w, height, paddedWidth) = RecPreproc.PadBatch(part.Select(j => crops[j]).ToList());
            var input = new DenseTensor<float>(batch, new[] { part.Count, 1, height, paddedWidth });
            var widthTensor = new DenseTensor<long>(w, new[] { part.Count });
            using var results = _cls.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor("x", input),
                NamedOnnxValue.CreateFromTensor("widths", widthTensor),
            });
            var p = results[0].AsTensor<float>().ToArray();
            for (int k = 0; k < part.Count; k++)
            {
                outv[part[k]] = p[k] > MinProbability;
            }
        }
        return outv;
    }

    /// <summary>(text with inline LaTeX, mean token probability) for each box.</summary>
    public (string Text, double Conf)[] Read(Mat gray, float[] boxes)
    {
        int n = boxes.Length / 5;
        var outv = new (string, double)[n];
        if (n == 0)
        {
            return outv;
        }
        var boxList = new List<float[]>(n);
        for (int i = 0; i < n; i++)
        {
            boxList.Add(new[]
            {
                boxes[i * 5], boxes[i * 5 + 1], boxes[i * 5 + 2], boxes[i * 5 + 3],
            });
        }
        var crops = LineMask.CropLines(gray, boxList, Height);
        var widths = crops.Select(c => c.Width).ToList();
        // Few, wide batches: a step costs about the same for 8 lines as for 40.
        foreach (var part in RecPreproc.WidthBatches(widths, maxRatio: 4.0, maxArea: 16 * 64 * 640))
        {
            var idx = part;
            var (batch, w, height, paddedWidth) = RecPreproc.PadBatch(idx.Select(j => crops[j]).ToList());
            var input = new DenseTensor<float>(batch, new[] { idx.Count, 1, height, paddedWidth });
            var widthTensor = new DenseTensor<long>(w, new[] { idx.Count });
            using var enc = _enc.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor("x", input),
                NamedOnnxValue.CreateFromTensor("widths", widthTensor),
            });
            var memory = enc[0].AsTensor<float>();
            var mask = enc[1].AsTensor<float>();
            var initInputs = new List<NamedOnnxValue>(_initInputs.Count);
            if (_initInputs.Contains("memory"))
            {
                initInputs.Add(NamedOnnxValue.CreateFromTensor("memory", memory));
            }
            if (_initInputs.Contains("mask"))
            {
                initInputs.Add(NamedOnnxValue.CreateFromTensor("mask", mask));
            }
            using var init = _init.Run(initInputs, new[] { "cross_k", "cross_v" });
            var crossK = init[0].AsTensor<float>();
            var crossV = init[1].AsTensor<float>();

            int b = idx.Count;
            var selfK = new DenseTensor<float>(
                Array.Empty<float>(), new[] { Layers, b, Heads, 0, HeadDim });
            var selfV = selfK;
            var tokens = Enumerable.Repeat((long)Bos, b).ToArray();
            var seqs = Enumerable.Range(0, b).Select(_ => new List<int>()).ToArray();
            var probs = Enumerable.Range(0, b).Select(_ => new List<double>()).ToArray();
            var done = new bool[b];
            for (int t = 0; t < MaxTokens - 1; t++)
            {
                var tokenTensor = new DenseTensor<long>(tokens, new[] { b });
                var posTensor = new DenseTensor<long>(new[] { (long)t }, Array.Empty<int>());
                var stepInputs = new List<NamedOnnxValue>(7);
                void Feed(string name, NamedOnnxValue value)
                {
                    if (_stepInputs.Contains(name))
                    {
                        stepInputs.Add(value);
                    }
                }
                Feed("token", NamedOnnxValue.CreateFromTensor("token", tokenTensor));
                Feed("pos", NamedOnnxValue.CreateFromTensor("pos", posTensor));
                Feed("self_k", NamedOnnxValue.CreateFromTensor("self_k", selfK));
                Feed("self_v", NamedOnnxValue.CreateFromTensor("self_v", selfV));
                Feed("cross_k", NamedOnnxValue.CreateFromTensor("cross_k", crossK));
                Feed("cross_v", NamedOnnxValue.CreateFromTensor("cross_v", crossV));
                Feed("mask", NamedOnnxValue.CreateFromTensor("mask", mask));
                using var step = _step.Run(stepInputs, new[] { "logits", "new_k", "new_v" });
                var logits = step[0].AsTensor<float>();
                // The result tensors are views over native memory owned by `step`:
                // copy the cache before it is disposed.
                var nk = step[1].AsTensor<float>();
                selfK = new DenseTensor<float>(nk.ToArray(), nk.Dimensions.ToArray());
                var nv = step[2].AsTensor<float>();
                selfV = new DenseTensor<float>(nv.ToArray(), nv.Dimensions.ToArray());

                var logitData = logits.ToArray();
                var dims = logits.Dimensions.ToArray();
                int classes = dims[^1];
                for (int row = 0; row < b; row++)
                {
                    int best = 0;
                    float max = float.NegativeInfinity;
                    for (int c = 0; c < classes; c++)
                    {
                        float v = logitData[row * classes + c];
                        if (v > max)
                        {
                            max = v;
                            best = c;
                        }
                    }
                    double sum = 0;
                    for (int c = 0; c < classes; c++)
                    {
                        sum += Math.Exp(logitData[row * classes + c] - max);
                    }
                    double top = 1.0 / sum; // softmax probability of the argmax token
                    tokens[row] = best;
                    if (done[row])
                    {
                        continue;
                    }
                    probs[row].Add(top);
                    if (best == Eos)
                    {
                        done[row] = true;
                    }
                    else if (probs[row].Count >= 8 && probs[row].Average() < MinConf - EarlyStopMargin)
                    {
                        done[row] = true; // rejected: keeps a low mean probability
                    }
                    else if (best >= NSpecial)
                    {
                        seqs[row].Add(best);
                    }
                }
                if (done.All(v => v))
                {
                    break;
                }
            }
            for (int k = 0; k < b; k++)
            {
                string text = Detokenize(seqs[k].Select(i => _itos[i]).ToList());
                outv[idx[k]] = (text, probs[k].Count > 0 ? probs[k].Average() : 0.0);
            }
        }
        return outv;
    }

    /// <summary>Tokens -> text; inside math (between "$" tokens) joined by spaces.</summary>
    public static string Detokenize(IReadOnlyList<string> tokens)
    {
        var outv = new List<string>();
        var math = new List<string>();
        bool inMath = false;
        foreach (string t in tokens)
        {
            if (t == "$")
            {
                if (inMath)
                {
                    outv.Add("$" + string.Join(" ", math) + "$");
                    math.Clear();
                }
                inMath = !inMath;
            }
            else if (inMath)
            {
                math.Add(t);
            }
            else
            {
                outv.Add(t);
            }
        }
        if (inMath && math.Count > 0)
        {
            outv.Add("$" + string.Join(" ", math) + "$");
        }
        return string.Join(" ", string.Concat(outv).Split(
            (char[]?)null, StringSplitOptions.RemoveEmptyEntries));
    }

    public void Dispose()
    {
        _cls.Dispose();
        _enc.Dispose();
        _init.Dispose();
        _step.Dispose();
    }
}
