using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using IndraOcr.Document;
using IndraOcr.Imaging;
using IndraOcr.Models;
using IndraOcr.Pdf;
using IndraOcr.Pipeline;
using OpenCvSharp;

namespace IndraOcr.Viewer;

/// <summary>
/// viewer.py's Engine: holds uploaded documents and reads one page per request
/// with the options the page sends, so the browser shows each page as it is done.
/// </summary>
public sealed class ViewerEngine : IDisposable
{
    public const int MaxSide = 1600;
    private static readonly string[] TimingKeys =
    {
        "text_ms", "det_ms", "rec_ms", "ocr_ms", "math_ms", "order_ms", "total_ms",
    };

    private readonly string? _modelPath;
    private readonly int _threads;
    private readonly JsonObject _defaults;
    private readonly Dictionary<(bool Math, bool Furniture), OrtOcr> _ocrs = new();
    private readonly Dictionary<string, byte[]> _docs = new(StringComparer.Ordinal);
    private readonly List<string> _order = new();
    private readonly string _tmp;
    private readonly object _lock = new();

    public ViewerEngine(string? modelPath, int threads, JsonObject defaults)
    {
        _modelPath = modelPath;
        _threads = threads;
        _defaults = defaults;
        _tmp = Path.Combine(Path.GetTempPath(), "indra_ocr_viewer_" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(_tmp);
        // Warm the default recognizer now, like the Python engine: the first
        // call's allocations are not part of any page's timings.
        bool math = defaults["math"]?.GetValue<bool>() ?? false;
        bool furniture = defaults["furniture"]?.GetValue<bool>() ?? false;
        _ = Ocr(math, furniture);
    }

    private OrtOcr Ocr(bool math, bool furniture)
    {
        var key = (math, furniture);
        if (!_ocrs.TryGetValue(key, out var ocr))
        {
            string models = ModelFiles.Resolve(_modelPath, math: math);
            ocr = new OrtOcr(models, threads: _threads, math: math, furniture: furniture);
            using (var white = new Mat(200, 200, MatType.CV_8UC1, new Scalar(255)))
            {
                _ = ocr.Read(white);
            }
            _ocrs[key] = ocr;
        }
        return ocr;
    }

    public JsonObject Info() => new()
    {
        ["tesseract"] = false,
        ["version"] = IndraOcrInfo.Version,
        ["defaults"] = _defaults.DeepClone(),
    };

    private static bool IsPdf(byte[] data)
        => data.Length >= 5 && data[0] == (byte)'%' && data[1] == (byte)'P' && data[2] == (byte)'D'
           && data[3] == (byte)'F' && data[4] == (byte)'-';

    public JsonObject Upload(byte[] data)
    {
        lock (_lock)
        {
            string id = Guid.NewGuid().ToString("N")[..12];
            int pages;
            if (IsPdf(data))
            {
                string path = Path.Combine(_tmp, id + ".pdf");
                File.WriteAllBytes(path, data);
                try
                {
                    using var doc = PdfDocument.Open(path);
                    pages = doc.PageCount;
                }
                catch
                {
                    File.Delete(path);
                    throw;
                }
            }
            else
            {
                using var img = Cv2.ImDecode(data, ImreadModes.Grayscale);
                if (img.Empty())
                {
                    throw new InvalidOperationException("not a PDF or a readable image");
                }
                pages = 1;
            }
            _docs[id] = data;
            _order.Add(id);
            while (_order.Count > 5) // keep the last few uploads only
            {
                string old = _order[0];
                _order.RemoveAt(0);
                _docs.Remove(old);
                try
                {
                    File.Delete(Path.Combine(_tmp, old + ".pdf"));
                }
                catch (IOException)
                {
                    // already gone
                }
            }
            return new JsonObject { ["id"] = id, ["pages"] = pages };
        }
    }

    private static bool Bool(JsonElement opts, string name, bool fallback)
        => opts.ValueKind == JsonValueKind.Object
           && opts.TryGetProperty(name, out var v)
           && v.ValueKind is JsonValueKind.True or JsonValueKind.False
            ? v.GetBoolean()
            : fallback;

    private static double Number(JsonElement opts, string name, double fallback)
        => opts.ValueKind == JsonValueKind.Object
           && opts.TryGetProperty(name, out var v)
           && v.ValueKind == JsonValueKind.Number
            ? v.GetDouble()
            : fallback;

    private static string[] Box(double[] box, double scale, int count = 4)
        => box.Take(count).Select(v => Math.Round(v * scale, 1).ToString("0.0#", CultureInfo.InvariantCulture))
            .Select(double.Parse).Select(v => v.ToString("R", CultureInfo.InvariantCulture)).ToArray();

    private static JsonArray BoxArray(double[] box, double scale, int count = 4)
    {
        var arr = new JsonArray();
        foreach (double v in box.Take(count))
        {
            arr.Add(Math.Round(v * scale, 1));
        }
        return arr;
    }

    /// <summary>One page with the options the browser sent (viewer.Engine.page).</summary>
    public JsonObject Page(string id, int n, JsonElement opts)
    {
        lock (_lock)
        {
            if (!_docs.TryGetValue(id, out var data))
            {
                throw new InvalidOperationException("unknown document (upload it again)");
            }
            string mode = opts.ValueKind == JsonValueKind.Object
                          && opts.TryGetProperty("mode", out var m)
                          && m.ValueKind == JsonValueKind.String
                ? m.GetString() ?? "auto"
                : "auto";
            if (mode is not ("auto" or "ocr" or "text"))
            {
                throw new ArgumentException("mode must be one of ('auto', 'ocr', 'text')");
            }
            bool math = Bool(opts, "math", false);
            bool furniture = Bool(opts, "furniture", false);
            double dpiValue = Number(opts, "dpi", 0);
            double dpi = dpiValue > 0 ? dpiValue : 200.0;
            bool imageOcr = Bool(opts, "image_ocr", true);

            var ocr = Ocr(math, furniture);
            var conv = new Converter(ocr, dpi, mode, imageOcr);
            Page page;
            Mat gray;
            double renderMs = 0;
            if (IsPdf(data))
            {
                string path = Path.Combine(_tmp, id + ".pdf");
                using var doc = PdfDocument.Open(path);
                using var pdfPage = doc.LoadPage(n);
                double scale = conv.PageScale(pdfPage);
                var sw = Stopwatch.StartNew();
                byte[] bytes = pdfPage.RenderGray(scale);
                renderMs = sw.Elapsed.TotalMilliseconds;
                int width = (int)Math.Ceiling(pdfPage.Width * scale);
                int height = (int)Math.Ceiling(pdfPage.Height * scale);
                gray = ImageOps.FromBytes(bytes, height, width);
                page = conv.PdfPage(path, doc, n);
            }
            else
            {
                gray = Cv2.ImDecode(data, ImreadModes.Grayscale);
                if (gray.Empty())
                {
                    throw new InvalidOperationException("not a PDF or a readable image");
                }
                page = ocr.Read(gray);
            }

            try
            {
                if (page.Rotation != 0) // boxes are in the upright page: show it
                {
                    var upright = ImageOps.Rot90(gray, -(page.Rotation / 90));
                    gray.Dispose();
                    gray = upright;
                }
                int h = gray.Rows, w = gray.Cols;
                double s = Math.Min(1.0, MaxSide / (double)Math.Max(h, w));
                Mat small = s < 1
                    ? ImageOps.Resize(gray, (int)Math.Round(w * s), (int)Math.Round(h * s), s)
                    : gray.Clone();
                try
                {
                    Cv2.ImEncode(".jpg", small, out byte[] jpg,
                        new[] { (int)ImwriteFlags.JpegQuality, 82 });
                    var lines = new JsonArray();
                    foreach (var line in page.Lines)
                    {
                        lines.Add(new JsonObject
                        {
                            ["box"] = BoxArray(line.Box.Select(v => (double)v).ToArray(), s),
                            ["text"] = line.Text,
                            ["conf"] = line.Conf,
                        });
                    }
                    var blocks = new JsonArray();
                    foreach (var block in page.Blocks)
                    {
                        blocks.Add(new JsonObject
                        {
                            ["label"] = block.Label,
                            ["box"] = BoxArray(block.Bbox.Select(v => (double)v).ToArray(), s),
                            ["text"] = block.Text,
                        });
                    }
                    var timings = new JsonObject { ["render_ms"] = Math.Round(renderMs, 1) };
                    foreach (string key in TimingKeys)
                    {
                        timings[key] = 0.0;
                    }
                    foreach (var (key, value) in page.Timings)
                    {
                        timings[key] = Math.Round(value, 1);
                    }

                    var jsonLines = new JsonArray();
                    foreach (var line in page.Lines)
                    {
                        jsonLines.Add(new JsonObject
                        {
                            ["box"] = BoxArray(line.Box.Select(v => (double)v).ToArray(), 1.0),
                            ["text"] = line.Text,
                            ["conf"] = line.Conf,
                        });
                    }
                    var jsonBlocks = new JsonArray();
                    foreach (var block in page.Blocks)
                    {
                        jsonBlocks.Add(new JsonObject
                        {
                            ["label"] = block.Label,
                            ["bbox"] = BoxArray(block.Bbox.Select(v => (double)v).ToArray(), 1.0),
                            ["text"] = block.Text,
                        });
                    }
                    var jsonTimings = new JsonObject();
                    foreach (var (key, value) in page.Timings)
                    {
                        jsonTimings[key] = value;
                    }

                    var rec = new JsonObject
                    {
                        ["page"] = n + 1,
                        ["w"] = small.Cols,
                        ["h"] = small.Rows,
                        ["img"] = "data:image/jpeg;base64," + Convert.ToBase64String(jpg),
                        ["lines"] = lines,
                        ["blocks"] = blocks,
                        ["markdown"] = page.Markdown ?? page.Text,
                        ["source"] = page.Source,
                        ["note"] = page.Note,
                        ["rotation"] = page.Rotation,
                        ["skew"] = page.Skew,
                        ["vertical"] = page.Vertical,
                        ["json"] = new JsonObject
                        {
                            ["page"] = n,
                            ["source"] = page.Source,
                            ["rotation"] = page.Rotation,
                            ["skew"] = page.Skew,
                            ["vertical"] = page.Vertical,
                            ["note"] = page.Note,
                            ["lines"] = jsonLines,
                            ["markdown"] = page.Markdown,
                            ["blocks"] = jsonBlocks,
                            ["timings"] = jsonTimings,
                        },
                        ["timings"] = timings,
                    };
                    double total = page.Timings.GetValueOrDefault("total_ms");
                    Console.WriteLine(
                        $"page {n + 1}: {page.Source}, {page.Lines.Count} lines, {page.Blocks.Count} blocks, " +
                        $"{total.ToString("0", CultureInfo.InvariantCulture)} ms " +
                        $"({mode}{(math ? ", math" : "")})");
                    return rec;
                }
                finally
                {
                    small.Dispose();
                }
            }
            finally
            {
                gray.Dispose();
            }
        }
    }

    public void Dispose()
    {
        foreach (var ocr in _ocrs.Values)
        {
            ocr.Dispose();
        }
        _ocrs.Clear();
        try
        {
            Directory.Delete(_tmp, recursive: true);
        }
        catch (IOException)
        {
            // best effort
        }
    }
}
