using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace IndraOcr.Paddle;

/// <summary>Locates the preinstalled PaddleOCR ONNX models (no runtime download).</summary>
public static class PaddleModelFiles
{
    public static readonly string[] Required = { "ch_PP-OCRv4_det_infer.onnx", "ch_PP-OCRv4_rec_infer.onnx" };

    public static string DefaultDir()
        => Path.Combine(AppContext.BaseDirectory, "Data", "paddleocr");

    public static string Resolve(string? folder = null)
    {
        string dir = string.IsNullOrWhiteSpace(folder) ? DefaultDir() : Path.GetFullPath(folder);
        var missing = Required.Where(f => !File.Exists(Path.Combine(dir, f))).ToList();
        if (missing.Count > 0)
        {
            throw new IOException(
                $"the PaddleOCR model folder {dir} lacks {string.Join(", ", missing)}; " +
                "the desktop package ships these files under Data\\paddleocr");
        }
        return dir;
    }
}

/// <summary>
/// PaddleOCR PP-OCRv4 detection + recognition via ONNX Runtime.  This is the
/// high-level <see cref="Pipeline.IOcrEngine"/>: no PDF text-layer, math or
/// Indra layout heads.  Models are shipped with the app.
/// </summary>
public sealed class PaddleOcrEngine : Pipeline.IOcrEngine
{
    private readonly InferenceSession _det;
    private readonly InferenceSession _rec;
    private readonly string[] _chars;
    private readonly int _threads;

    public PaddleOcrEngine(string modelDir, int threads = 4)
    {
        string dir = PaddleModelFiles.Resolve(modelDir);
        _threads = Math.Max(1, threads);
        var options = new SessionOptions
        {
            IntraOpNumThreads = Math.Max(1, threads),
            InterOpNumThreads = 1,
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
        };
        _det = new InferenceSession(Path.Combine(dir, "ch_PP-OCRv4_det_infer.onnx"), options);
        _rec = new InferenceSession(Path.Combine(dir, "ch_PP-OCRv4_rec_infer.onnx"), options);
        string character = _rec.ModelMetadata.CustomMetadataMap.TryGetValue("character", out var value)
            ? value
            : "";
        _chars = new[] { "" }.Concat(character.Split('\n').Select(s => s.TrimEnd('\r'))).Append(" ").ToArray();
    }

    public Pipeline.OcrEngineKind Kind => Pipeline.OcrEngineKind.PaddleOcr;
    public string DisplayName => "PaddleOCR (PP-OCRv4)";
    public Pipeline.OcrCapabilities Capabilities =>
        Pipeline.OcrCapabilities.Images |
        Pipeline.OcrCapabilities.PdfRenderedPages;

    public Pipeline.Page Read(Mat gray, bool turn = true)
    {
        var total = Stopwatch.StartNew();
        var (boxes, detMs) = Detect(gray);
        var lines = new List<Pipeline.Line>(boxes.Count);
        var rec = Stopwatch.StartNew();
        foreach (var box in boxes)
        {
            string text = Recognize(gray, box, out double conf);
            if (text.Length == 0)
            {
                continue;
            }
            lines.Add(new Pipeline.Line
            {
                Box = new[] { (float)box.X0, (float)box.Y0, (float)box.X1, (float)box.Y1 },
                Text = text,
                Conf = conf,
            });
        }
        double recMs = rec.Elapsed.TotalMilliseconds;
        var page = new Pipeline.Page
        {
            Source = "paddleocr",
            Note = "order=paddle; engine=paddleocr",
            Markdown = string.Join("\n", lines.Select(l => l.Text)),
        };
        page.Lines.AddRange(lines);
        page.Timings["det_ms"] = detMs;
        page.Timings["rec_ms"] = recMs;
        page.Timings["total_ms"] = total.Elapsed.TotalMilliseconds;
        return page;
    }

    private readonly record struct Box(double X0, double Y0, double X1, double Y1);

    private (List<Box> Boxes, double Ms) Detect(Mat gray)
    {
        const int limit = 736;
        int h = gray.Rows, w = gray.Cols;
        double scale = Math.Min(h, w) < limit
            ? limit / (double)Math.Min(h, w)
            : 1.0;
        int rw = Math.Max(32, (int)Math.Round(w * scale / 32) * 32);
        int rh = Math.Max(32, (int)Math.Round(h * scale / 32) * 32);

        using var resized = new Mat();
        Cv2.Resize(gray, resized, new Size(rw, rh), interpolation: InterpolationFlags.Linear);
        var input = new DenseTensor<float>(new[] { 1, 3, rh, rw });
        float[] mean = { 0.485f, 0.456f, 0.406f };
        float[] std = { 0.229f, 0.224f, 0.225f };
        var indexer = resized.GetGenericIndexer<byte>();
        for (int y = 0; y < rh; y++)
        {
            for (int x = 0; x < rw; x++)
            {
                float v = indexer[y, x] / 255f;
                for (int c = 0; c < 3; c++)
                {
                    input[0, c, y, x] = (v - mean[c]) / std[c];
                }
            }
        }

        var sw = Stopwatch.StartNew();
        using var results = _det.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor(_det.InputNames[0], input),
        });
        var map = results.First().AsTensor<float>();
        int mh = map.Dimensions[2], mw = map.Dimensions[3];
        using var mask = new Mat(mh, mw, MatType.CV_8UC1);
        for (int y = 0; y < mh; y++)
        {
            for (int x = 0; x < mw; x++)
            {
                mask.Set(y, x, map[0, 0, y, x] >= 0.3f ? (byte)255 : (byte)0);
            }
        }
        using var kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(2, 2));
        Cv2.Dilate(mask, mask, kernel);
        Cv2.FindContours(mask, out Point[][] contours, out _, RetrievalModes.List,
            ContourApproximationModes.ApproxSimple);

        var boxes = new List<Box>();
        foreach (var contour in contours)
        {
            var rect = Cv2.BoundingRect(contour);
            if (rect.Width < 4 || rect.Height < 4)
            {
                continue;
            }
            double score = 0;
            int samples = 0;
            for (int y = rect.Top; y < rect.Bottom; y += 2)
            {
                for (int x = rect.Left; x < rect.Right; x += 2)
                {
                    score += map[0, 0, y, x];
                    samples++;
                }
            }
            if (samples == 0 || score / samples < 0.5)
            {
                continue;
            }
            // A small expansion replaces pyclipper's unclip for the first
            // version; recognition tolerates a few pixels of padding.
            int padX = Math.Max(1, (int)Math.Round(rect.Height * 0.12));
            int padY = Math.Max(1, (int)Math.Round(rect.Height * 0.04));
            double x0 = Math.Max(0, (rect.Left - padX) / (double)rw * w);
            double y0 = Math.Max(0, (rect.Top - padY) / (double)rh * h);
            double x1 = Math.Min(w, (rect.Right + padX) / (double)rw * w);
            double y1 = Math.Min(h, (rect.Bottom + padY) / (double)rh * h);
            if (x1 - x0 >= 3 && y1 - y0 >= 3)
            {
                boxes.Add(new Box(x0, y0, x1, y1));
            }
        }
        boxes.Sort((a, b) => Math.Abs(a.Y0 - b.Y0) < 10
            ? a.X0.CompareTo(b.X0)
            : a.Y0.CompareTo(b.Y0));
        return (boxes, sw.Elapsed.TotalMilliseconds);
    }

    private string Recognize(Mat gray, Box box, out double confidence)
    {
        int x0 = Math.Max(0, (int)Math.Floor(box.X0));
        int y0 = Math.Max(0, (int)Math.Floor(box.Y0));
        int x1 = Math.Min(gray.Cols, (int)Math.Ceiling(box.X1));
        int y1 = Math.Min(gray.Rows, (int)Math.Ceiling(box.Y1));
        if (x1 <= x0 || y1 <= y0)
        {
            confidence = 0;
            return "";
        }
        using var crop = new Mat(gray, new Rect(x0, y0, x1 - x0, y1 - y0)).Clone();
        int h = 48;
        int w = Math.Max(1, (int)Math.Round(crop.Cols * (h / (double)crop.Rows)));
        w = Math.Min(w, 1200);
        using var resized = new Mat();
        Cv2.Resize(crop, resized, new Size(w, h), interpolation: InterpolationFlags.Linear);
        var input = new DenseTensor<float>(new[] { 1, 3, h, w });
        var src = resized.GetGenericIndexer<byte>();
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                float v = (src[y, x] / 255f - 0.5f) / 0.5f;
                input[0, 0, y, x] = v;
                input[0, 1, y, x] = v;
                input[0, 2, y, x] = v;
            }
        }
        using var results = _rec.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor(_rec.InputNames[0], input),
        });
        var logits = results.First().AsTensor<float>();
        int steps = logits.Dimensions[1];
        int classes = logits.Dimensions[2];
        var text = new System.Text.StringBuilder();
        var probs = new List<double>();
        int previous = 0;
        for (int t = 0; t < steps; t++)
        {
            int best = 0;
            float bestValue = logits[0, t, 0];
            for (int c = 1; c < classes; c++)
            {
                float value = logits[0, t, c];
                if (value > bestValue)
                {
                    bestValue = value;
                    best = c;
                }
            }
            if (best != 0 && best != previous)
            {
                text.Append(best < _chars.Length ? _chars[best] : "");
                probs.Add(bestValue);
            }
            previous = best;
        }
        confidence = probs.Count == 0 ? 0 : probs.Average();
        return text.ToString();
    }

    public void Dispose()
    {
        _det.Dispose();
        _rec.Dispose();
    }
}
