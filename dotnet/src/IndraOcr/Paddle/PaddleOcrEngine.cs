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
        var blockLines = new List<(Block Block, List<int> Lines)>();
        var used = new bool[lines.Count];
        foreach (var lb in layoutBoxes.OrderBy(b => (b.X1 - b.X0) * (b.Y1 - b.Y0)))
        {
            var indices = new List<int>();
            for (int i = 0; i < lines.Count; i++)
            {
                if (used[i])
                {
                    continue;
                }
                var line = lines[i];
                double cx = (line.Box[0] + line.Box[2]) / 2;
                double cy = (line.Box[1] + line.Box[3]) / 2;
                if (cx >= lb.X0 && cx <= lb.X1 && cy >= lb.Y0 && cy <= lb.Y1)
                {
                    used[i] = true;
                    indices.Add(i);
                }
            }
            indices.Sort((a, b) => CompareLines(lines[a], lines[b]));
            blockLines.Add((new Block
            {
                Label = lb.Label,
                Bbox = new[] { (float)lb.X0, (float)lb.Y0, (float)lb.X1, (float)lb.Y1 },
                Text = string.Join("\n", indices.Select(i => lines[i].Text)),
            }, indices));
        }
        var orderedBlocks = OrderBlocks(blockLines.Select(x => x.Block).ToList(), gray.Cols, gray.Rows);
        var orderedLines = new List<Pipeline.Line>(lines.Count);
        var outputBlocks = new List<Block>(orderedBlocks.Count);
        foreach (var block in orderedBlocks)
        {
            var pair = blockLines.First(x => ReferenceEquals(x.Block, block));
            outputBlocks.Add(block);
            foreach (int i in pair.Lines)
            {
                if (!orderedLines.Contains(lines[i]))
                {
                    orderedLines.Add(lines[i]);
                }
            }
        }
        for (int i = 0; i < lines.Count; i++)
        {
            if (!used[i])
            {
                orderedLines.Add(lines[i]);
            }
        }
        string markdown = outputBlocks.Count > 0
            ? string.Join("\n\n", outputBlocks.Select(b => b.Text).Where(t => t.Length > 0))
            : string.Join("\n", orderedLines.Select(l => l.Text));
        var page = new Pipeline.Page
        {
            Source = "paddleocr",
            Note = "order=paddle; engine=paddleocr; model=PP-OCRv6_small; layout=pp-doclayout-s",
            Markdown = markdown,
        };
        page.Lines.AddRange(orderedLines);
        page.Blocks.AddRange(outputBlocks);
        page.Timings["det_ms"] = detMs;
        page.Timings["rec_ms"] = recMs;
        page.Timings["layout_ms"] = layoutMs;
        page.Timings["total_ms"] = total.Elapsed.TotalMilliseconds;
        return page;
    }

    private static int CompareLines(Pipeline.Line a, Pipeline.Line b)
        => Math.Abs(a.Box[1] - b.Box[1]) < 8
            ? a.Box[0].CompareTo(b.Box[0])
            : a.Box[1].CompareTo(b.Box[1]);

    /// <summary>XY-cut over layout blocks; recurses on the widest whitespace gap.</summary>
    private static List<Block> OrderBlocks(List<Block> blocks, int pageWidth, int pageHeight)
    {
        if (blocks.Count <= 1)
        {
            return blocks;
        }
        if (TryCut(blocks, horizontal: true, out var top, out var bottom))
        {
            return OrderBlocks(top, pageWidth, pageHeight)
                .Concat(OrderBlocks(bottom, pageWidth, pageHeight)).ToList();
        }
        if (TryCut(blocks, horizontal: false, out var left, out var right))
        {
            return OrderBlocks(left, pageWidth, pageHeight)
                .Concat(OrderBlocks(right, pageWidth, pageHeight)).ToList();
        }
        return blocks.OrderBy(b => b.Bbox[1]).ThenBy(b => b.Bbox[0]).ToList();
    }

    private static bool TryCut(
        IReadOnlyList<Block> blocks, bool horizontal, out List<Block> first, out List<Block> second)
    {
        first = new List<Block>();
        second = new List<Block>();
        var intervals = blocks
            .Select(b => horizontal
                ? (Start: b.Bbox[1], End: b.Bbox[3])
                : (Start: b.Bbox[0], End: b.Bbox[2]))
            .OrderBy(x => x.Start)
            .ToList();
        double end = intervals[0].End;
        double bestGap = 8;
        double cut = double.NaN;
        for (int i = 1; i < intervals.Count; i++)
        {
            var interval = intervals[i];
            if (interval.Start - end > bestGap)
            {
                bestGap = interval.Start - end;
                cut = (end + interval.Start) / 2;
            }
            end = Math.Max(end, interval.End);
        }
        if (double.IsNaN(cut))
        {
            return false;
        }
        foreach (var block in blocks)
        {
            double center = horizontal
                ? (block.Bbox[1] + block.Bbox[3]) / 2
                : (block.Bbox[0] + block.Bbox[2]) / 2;
            (center < cut ? first : second).Add(block);
        }
        return first.Count > 0 && second.Count > 0;
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

    private readonly record struct Box(
        double X0, double Y0, double X1, double Y1, Point2f[]? Quad = null);

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
            var rect = Cv2.MinAreaRect(contour);
            if (rect.Size.Width < 4 || rect.Size.Height < 4)
            {
                continue;
            }
            var bounds = Cv2.BoundingRect(contour);
            double score = 0;
            int samples = 0;
            for (int y = bounds.Top; y < bounds.Bottom; y += 2)
            {
                for (int x = bounds.Left; x < bounds.Right; x += 2)
                {
                    score += map[0, 0, y, x];
                    samples++;
                }
            }
            if (samples == 0 || score / samples < 0.35)
            {
                continue;
            }
            var quad = OrderQuad(Cv2.BoxPoints(rect));
            var center = new Point2f(
                (float)quad.Average(p => p.X), (float)quad.Average(p => p.Y));
            var expanded = quad.Select(p => new Point2f(
                (float)((p.X - center.X) * 1.12 + center.X),
                (float)((p.Y - center.Y) * 1.12 + center.Y))).ToArray();
            for (int i = 0; i < expanded.Length; i++)
            {
                expanded[i].X = (float)Math.Clamp(expanded[i].X / rw * w, 0, w);
                expanded[i].Y = (float)Math.Clamp(expanded[i].Y / rh * h, 0, h);
            }
            double x0 = expanded.Min(p => p.X), x1 = expanded.Max(p => p.X);
            double y0 = expanded.Min(p => p.Y), y1 = expanded.Max(p => p.Y);
            if (x1 - x0 >= 3 && y1 - y0 >= 3)
            {
                boxes.Add(new Box(x0, y0, x1, y1, expanded));
            }
        }
        boxes.Sort((a, b) => Math.Abs(a.Y0 - b.Y0) < 10
            ? a.X0.CompareTo(b.X0)
            : a.Y0.CompareTo(b.Y0));
        return (boxes, sw.Elapsed.TotalMilliseconds);
    }

    private static Point2f[] OrderQuad(Point2f[] points)
    {
        var byX = points.OrderBy(p => p.X).ToArray();
        var left = byX.Take(2).OrderBy(p => p.Y).ToArray();
        var right = byX.Skip(2).OrderBy(p => p.Y).ToArray();
        return new[] { left[0], right[0], right[1], left[1] };
    }

    private static Mat Crop(Mat gray, Box box)
    {
        if (box.Quad is not { Length: 4 } quad)
        {
            int x0 = Math.Max(0, (int)Math.Floor(box.X0));
            int y0 = Math.Max(0, (int)Math.Floor(box.Y0));
            int x1 = Math.Min(gray.Cols, (int)Math.Ceiling(box.X1));
            int y1 = Math.Min(gray.Rows, (int)Math.Ceiling(box.Y1));
            if (x1 <= x0 || y1 <= y0)
            {
                return new Mat();
            }
            return new Mat(gray, new Rect(x0, y0, x1 - x0, y1 - y0)).Clone();
        }
        double width = Math.Max(Distance(quad[0], quad[1]), Distance(quad[3], quad[2]));
        double height = Math.Max(Distance(quad[0], quad[3]), Distance(quad[1], quad[2]));
        int w = Math.Max(1, (int)Math.Round(width));
        int h = Math.Max(1, (int)Math.Round(height));
        if (w < 3 || h < 3)
        {
            return new Mat();
        }
        var dst = new[]
        {
            new Point2f(0, 0), new Point2f(w, 0),
            new Point2f(w, h), new Point2f(0, h),
        };
        using var transform = Cv2.GetPerspectiveTransform(quad, dst);
        var crop = new Mat();
        Cv2.WarpPerspective(gray, crop, transform, new Size(w, h),
            InterpolationFlags.Cubic, BorderTypes.Replicate);
        return crop;
    }

    private static double Distance(Point2f a, Point2f b)
        => Math.Sqrt((a.X - b.X) * (a.X - b.X) + (a.Y - b.Y) * (a.Y - b.Y));

    private string Recognize(Mat gray, Box box, out double confidence)
    {
        using var crop = Crop(gray, box);
        if (crop.Empty())
        {
            confidence = 0;
            return "";
        }
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
