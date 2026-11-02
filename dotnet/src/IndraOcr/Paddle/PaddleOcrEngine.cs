using System.Diagnostics;
using IndraOcr.Layout;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace IndraOcr.Paddle;

/// <summary>Locates the preinstalled PaddleOCR ONNX models (no runtime download).</summary>
public static class PaddleModelFiles
{
    public static readonly string[] Required = { "pp_ocrv6_det.onnx", "pp_ocrv6_rec.onnx", "ppocrv6_dict.txt", "pp_doclayout_s.onnx" };

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
    private readonly InferenceSession? _layout;
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
        _det = new InferenceSession(Path.Combine(dir, "pp_ocrv6_det.onnx"), options);
        _rec = new InferenceSession(Path.Combine(dir, "pp_ocrv6_rec.onnx"), options);
        string layoutPath = Path.Combine(dir, "pp_doclayout_s.onnx");
        if (File.Exists(layoutPath))
        {
            _layout = new InferenceSession(layoutPath, options);
        }
        string character = File.ReadAllText(Path.Combine(dir, "ppocrv6_dict.txt"));
        _chars = new[] { "" }.Concat(character.Replace("\r", "").Split('\n')).Append(" ").ToArray();
    }

    public Pipeline.OcrEngineKind Kind => Pipeline.OcrEngineKind.PaddleOcr;
    public string DisplayName => "PaddleOCR (PP-OCRv6)";
    public Pipeline.OcrCapabilities Capabilities =>
        Pipeline.OcrCapabilities.Images |
        Pipeline.OcrCapabilities.PdfRenderedPages |
        Pipeline.OcrCapabilities.LayoutBlocks;

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
        var (layoutBoxes, layoutMs) = DetectLayout(gray);
        var blocks = new List<Block>();
        foreach (var lb in layoutBoxes)
        {
            var inside = lines.Where(line =>
            {
                double cx = (line.Box[0] + line.Box[2]) / 2;
                double cy = (line.Box[1] + line.Box[3]) / 2;
                return cx >= lb.X0 && cx <= lb.X1 && cy >= lb.Y0 && cy <= lb.Y1;
            }).Select(line => line.Text).Where(text => text.Length > 0).ToList();
            blocks.Add(new Block
            {
                Label = lb.Label,
                Bbox = new[] { (float)lb.X0, (float)lb.Y0, (float)lb.X1, (float)lb.Y1 },
                Text = string.Join("\n", inside),
            });
        }
        string markdown = blocks.Count > 0
            ? string.Join("\n\n", blocks.Select(b => b.Text).Where(t => t.Length > 0))
            : string.Join("\n", lines.Select(l => l.Text));
        var page = new Pipeline.Page
        {
            Source = "paddleocr",
            Note = "order=paddle; engine=paddleocr; model=PP-OCRv6_small; layout=pp-doclayout-s",
            Markdown = markdown,
        };
        page.Lines.AddRange(lines);
        page.Blocks.AddRange(blocks);
        page.Timings["det_ms"] = detMs;
        page.Timings["rec_ms"] = recMs;
        page.Timings["layout_ms"] = layoutMs;
        page.Timings["total_ms"] = total.Elapsed.TotalMilliseconds;
        return page;
    }

    private readonly record struct LayoutBox(double X0, double Y0, double X1, double Y1, string Label, double Score);

    private (List<LayoutBox> Boxes, double Ms) DetectLayout(Mat gray)
    {
        if (_layout is null)
        {
            return (new List<LayoutBox>(), 0);
        }
        var sw = Stopwatch.StartNew();
        const int side = 480;
        using var resized = new Mat();
        Cv2.Resize(gray, resized, new Size(side, side), interpolation: InterpolationFlags.Linear);
        var image = new DenseTensor<float>(new[] { 1, 3, side, side });
        float[] mean = { 0.485f, 0.456f, 0.406f };
        float[] std = { 0.229f, 0.224f, 0.225f };
        for (int y = 0; y < side; y++)
        {
            for (int x = 0; x < side; x++)
            {
                float v = resized.At<byte>(y, x) / 255f;
                for (int c = 0; c < 3; c++)
                {
                    image[0, c, y, x] = (v - mean[c]) / std[c];
                }
            }
        }
        var scale = new DenseTensor<float>(new[]
        {
            side / (float)Math.Max(1, gray.Rows),
            side / (float)Math.Max(1, gray.Cols),
        }, new[] { 1, 2 });
        using var results = _layout.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor(_layout.InputNames[0], image),
            NamedOnnxValue.CreateFromTensor(_layout.InputNames[1], scale),
        });
        var dets = results[0].AsTensor<float>();
        int num = Math.Min(dets.Dimensions[0], results[1].AsTensor<int>()[0]);
        var boxes = new List<LayoutBox>();
        for (int i = 0; i < num; i++)
        {
            double x0 = dets[i, 2], y0 = dets[i, 3], x1 = dets[i, 4], y1 = dets[i, 5];
            if (x1 <= x0 || y1 <= y0)
            {
                continue;
            }
            double score = dets[i, 1];
            if (score < 0.5)
            {
                continue;
            }
            boxes.Add(new LayoutBox(
                Math.Max(0, x0), Math.Max(0, y0),
                Math.Min(gray.Cols, x1), Math.Min(gray.Rows, y1),
                LayoutLabel((int)dets[i, 0]), score));
        }
        return (boxes, sw.Elapsed.TotalMilliseconds);
    }

    private static string LayoutLabel(int id) => id switch
    {
        0 or 11 => "Section-Header",
        1 or 17 or 18 or 20 or 21 => "Picture",
        2 or 4 or 5 or 10 or 13 or 15 or 22 => "Text",
        3 or 19 => "Text",
        6 or 9 => "Caption",
        7 => "Text",
        8 => "Table",
        12 => "Footnote",
        14 => "Text",
        16 => "Text",
        _ => "Text",
    };

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
