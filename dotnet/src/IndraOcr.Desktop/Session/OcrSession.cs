using System.Diagnostics;
using System.Text.Json.Nodes;
using System.Text.Json;
using Avalonia.Media.Imaging;
using IndraOcr.Document;
using IndraOcr.Imaging;
using IndraOcr.Models;
using IndraOcr.Pdf;
using IndraOcr.Pipeline;
using OpenCvSharp;
using ConverterType = IndraOcr.Document.Converter;

namespace IndraOcr.Desktop.Session;

/// <summary>
/// The desktop equivalent of viewer.py's Engine: one open document, one page read
/// per call, with the options the sidebar currently shows.
/// </summary>
public sealed class OcrSession : IDisposable
{
    public const int MaxSide = 1600;
    private static readonly string[] TimingKeys =
    {
        "text_ms", "det_ms", "rec_ms", "ocr_ms", "math_ms", "order_ms", "total_ms",
    };

    private readonly Dictionary<(bool Math, bool Furniture), OrtOcr> _ocrs = new();
    private PdfDocument? _pdf;
    private string? _modelDir;

    public string? FilePath { get; private set; }
    public string? FileName => FilePath is null ? null : System.IO.Path.GetFileName(FilePath);
    public int PageCount { get; private set; }
    public bool IsPdf { get; private set; }

    /// <summary>Folder the user chose; null means the environment/cache (auto download).</summary>
    public string? ModelDir => _modelDir;

    public static string DefaultModelDir()
        => Environment.GetEnvironmentVariable("INDRA_OCR_MODEL_DIR")
           ?? ModelFiles.CacheDir(IndraOcrInfo.ModelRevision);

    public static bool ModelsPresent(string? folder, bool math)
    {
        string dir = string.IsNullOrWhiteSpace(folder) ? DefaultModelDir() : folder;
        bool rec = File.Exists(System.IO.Path.Combine(dir, "rec.onnx"))
                   || File.Exists(System.IO.Path.Combine(dir, "rec.int8.onnx"));
        bool core = File.Exists(System.IO.Path.Combine(dir, "det.onnx"))
                    && File.Exists(System.IO.Path.Combine(dir, "meta.json"))
                    && rec;
        bool mathOk = !math || ModelFiles.MathFiles.All(f => File.Exists(System.IO.Path.Combine(dir, f)));
        return core && mathOk;
    }

    public void SetModelDir(string? folder)
    {
        string? normalized = string.IsNullOrWhiteSpace(folder) ? null : folder;
        if (normalized == _modelDir)
        {
            return;
        }
        foreach (var ocr in _ocrs.Values)
        {
            ocr.Dispose();
        }
        _ocrs.Clear();
        _modelDir = normalized;
    }

    public void Open(string path)
    {
        Close();
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"no such file: {path}", path);
        }
        byte[] head = new byte[1024];
        using (var f = File.OpenRead(path))
        {
            int n = f.Read(head, 0, head.Length);
            if (n < 5 || head[0] != (byte)'%' || head[1] != (byte)'P' || head[2] != (byte)'D'
                || head[3] != (byte)'F' || head[4] != (byte)'-')
            {
                IsPdf = false;
            }
            else
            {
                IsPdf = true;
            }
        }
        if (IsPdf)
        {
            _pdf = PdfDocument.Open(path);
            PageCount = _pdf.PageCount;
        }
        else
        {
            using var img = Cv2.ImDecode(File.ReadAllBytes(path), ImreadModes.Grayscale);
            if (img.Empty())
            {
                throw new InvalidOperationException("not a PDF or a readable image");
            }
            PageCount = 1;
        }
        FilePath = path;
    }

    /// <summary>Download (or verify) the models and return the folder that holds them.</summary>
    public string ResolveModels(bool math) => ModelFiles.Resolve(_modelDir, math: math);

    private void EnsureModels(ReadOptions opts)
    {
        var key = (opts.Math, opts.Furniture);
        if (_ocrs.ContainsKey(key))
        {
            return;
        }
        string models = ResolveModels(opts.Math);
        var ocr = new OrtOcr(
            models,
            threads: Math.Max(1, Environment.ProcessorCount),
            math: opts.Math,
            furniture: opts.Furniture);
        using (var white = new Mat(200, 200, MatType.CV_8UC1, new Scalar(255)))
        {
            _ = ocr.Read(white); // warm-up: first-call allocations are not timed
        }
        _ocrs[key] = ocr;
    }

    public PageResult ReadPage(int index, ReadOptions opts)
    {
        if (_pdf is null && !IsPdf)
        {
            if (FilePath is null)
            {
                throw new InvalidOperationException("no document is open");
            }
        }
        else if (_pdf is null)
        {
            throw new InvalidOperationException("no PDF is open");
        }
        if (index < 0 || index >= PageCount)
        {
            throw new ArgumentOutOfRangeException(nameof(index), $"page {index} does not exist");
        }

        EnsureModels(opts);
        var ocr = _ocrs[(opts.Math, opts.Furniture)];
        var conv = new ConverterType(ocr, opts.Dpi, opts.Mode, opts.ImageOcr);

        Page page;
        Mat gray;
        double renderMs = 0;
        if (IsPdf)
        {
            using var pdfPage = _pdf!.LoadPage(index);
            double scale = conv.PageScale(pdfPage);
            var rs = Stopwatch.StartNew();
            byte[] bytes = pdfPage.RenderGray(scale);
            renderMs = rs.Elapsed.TotalMilliseconds;
            int width = (int)Math.Ceiling(pdfPage.Width * scale);
            int height = (int)Math.Ceiling(pdfPage.Height * scale);
            gray = ImageOps.FromBytes(bytes, height, width);
            page = conv.PdfPage(FilePath!, _pdf, index);
        }
        else
        {
            gray = ImageOps.LoadGray(FilePath!);
            page = ocr.Read(gray);
        }

        try
        {
            if (page.Rotation != 0)
            {
                var upright = ImageOps.Rot90(gray, -(page.Rotation / 90));
                gray.Dispose();
                gray = upright;
            }
            int h = gray.Rows, w = gray.Cols;
            double s = Math.Min(1.0, MaxSide / (double)Math.Max(h, w));
            using var small = s < 1
                ? ImageOps.Resize(gray, (int)Math.Round(w * s), (int)Math.Round(h * s), s)
                : gray.Clone();
            Cv2.ImEncode(".jpg", small, out byte[] jpg, new[] { (int)ImwriteFlags.JpegQuality, 82 });

            var lines = page.Lines.Select(line => new LineView(
                line.Box[0] * s, line.Box[1] * s, line.Box[2] * s, line.Box[3] * s,
                line.Text, line.Conf)).ToList();
            var blocks = page.Blocks.Select(block => new BlockView(
                block.Label,
                block.Bbox[0] * s, block.Bbox[1] * s, block.Bbox[2] * s, block.Bbox[3] * s,
                block.Text)).ToList();

            var timings = new Dictionary<string, double> { ["render_ms"] = renderMs };
            foreach (string key in TimingKeys)
            {
                timings[key] = 0;
            }
            foreach (var (key, value) in page.Timings)
            {
                timings[key] = value;
            }

            return new PageResult
            {
                Index = index,
                ImageJpeg = jpg,
                Lines = lines,
                Blocks = blocks,
                Markdown = page.Markdown ?? page.Text,
                Json = BuildJson(index, page),
                Source = page.Source,
                Note = page.Note,
                Rotation = page.Rotation,
                Skew = page.Skew,
                Vertical = page.Vertical,
                Timings = timings,
                TotalMs = page.Timings.GetValueOrDefault("total_ms"),
            };
        }
        finally
        {
            gray.Dispose();
        }
    }

    private static string BuildJson(int index, Page page)
    {
        var lines = new JsonArray();
        foreach (var line in page.Lines)
        {
            lines.Add((JsonNode)new JsonObject
            {
                ["box"] = new JsonArray(line.Box.Select(v => (JsonNode?)(double)v).ToArray()),
                ["text"] = line.Text,
                ["conf"] = line.Conf,
            });
        }
        var blocks = new JsonArray();
        foreach (var block in page.Blocks)
        {
            blocks.Add((JsonNode)new JsonObject
            {
                ["label"] = block.Label,
                ["bbox"] = new JsonArray(block.Bbox.Select(v => (JsonNode?)(double)v).ToArray()),
                ["text"] = block.Text,
            });
        }
        var timings = new JsonObject();
        foreach (var (key, value) in page.Timings)
        {
            timings[key] = value;
        }
        var root = new JsonObject
        {
            ["page"] = index,
            ["source"] = page.Source,
            ["rotation"] = page.Rotation,
            ["skew"] = page.Skew,
            ["vertical"] = page.Vertical,
            ["note"] = page.Note,
            ["lines"] = lines,
            ["markdown"] = page.Markdown,
            ["blocks"] = blocks,
            ["timings"] = timings,
        };
        return root.ToJsonString(new JsonSerializerOptions
        {
            Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
            WriteIndented = true,
        });
    }

    private void Close()
    {
        _pdf?.Dispose();
        _pdf = null;
        FilePath = null;
        PageCount = 0;
        IsPdf = false;
    }

    public void Dispose()
    {
        Close();
        foreach (var ocr in _ocrs.Values)
        {
            ocr.Dispose();
        }
        _ocrs.Clear();
    }
}
