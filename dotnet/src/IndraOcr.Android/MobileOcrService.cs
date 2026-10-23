using IndraOcr.Imaging;
using IndraOcr.Models;
using IndraOcr.Pipeline;
using OpenCvSharp;

namespace IndraOcr.AndroidApp;

public sealed record MobileLine(double X0, double Y0, double X1, double Y1, string Text, double Conf);

public sealed record MobileBlock(
    string Label, double X0, double Y0, double X1, double Y1, string Text);

public sealed class MobilePageResult
{
    public required byte[] PreviewJpeg { get; init; }
    public required IReadOnlyList<MobileLine> Lines { get; init; }
    public required IReadOnlyList<MobileBlock> Blocks { get; init; }
    public required string Markdown { get; init; }
    public required string Source { get; init; }
    public required int Rotation { get; init; }
    public required double Skew { get; init; }
    public required double TotalMs { get; init; }
}

/// <summary>Image-only OCR for the Android app; PDFs and math stay desktop-only.</summary>
public sealed class MobileOcrService : IDisposable
{
    private const int MaxSide = 1600;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private OrtOcr? _ocr;

    public MobileOcrService(string modelDir)
    {
        ModelDir = modelDir;
    }

    public string ModelDir { get; }

    public string ResolveModels() => ModelFiles.Resolve(ModelDir, math: false, allowDownload: true);

    public async Task<MobilePageResult> RecognizeAsync(string path, CancellationToken token)
    {
        await _gate.WaitAsync(token);
        try
        {
            return await Task.Run(() => Recognize(path, token), token);
        }
        finally
        {
            _gate.Release();
        }
    }

    private MobilePageResult Recognize(string path, CancellationToken token)
    {
        string models = ResolveModels();
        token.ThrowIfCancellationRequested();
        _ocr ??= new OrtOcr(
            models,
            threads: Math.Max(1, Environment.ProcessorCount),
            math: false,
            furniture: false);

        using var loaded = ImageOps.LoadGray(path);
        token.ThrowIfCancellationRequested();
        var page = _ocr.Read(loaded);
        token.ThrowIfCancellationRequested();

        Mat? rotated = null;
        var upright = loaded;
        if (page.Rotation != 0)
        {
            rotated = ImageOps.Rot90(loaded, -(page.Rotation / 90));
            upright = rotated;
        }
        try
        {
            int h = upright.Rows, w = upright.Cols;
            double scale = Math.Min(1.0, MaxSide / (double)Math.Max(h, w));
            using var small = scale < 1
                ? ImageOps.Resize(upright, (int)Math.Round(w * scale), (int)Math.Round(h * scale), scale)
                : upright.Clone();
            Cv2.ImEncode(".jpg", small, out byte[] jpeg,
                new[] { (int)ImwriteFlags.JpegQuality, 84 });

            var lines = page.Lines.Select(line => new MobileLine(
                line.Box[0] * scale, line.Box[1] * scale,
                line.Box[2] * scale, line.Box[3] * scale,
                line.Text, line.Conf)).ToList();
            var blocks = page.Blocks.Select(block => new MobileBlock(
                block.Label,
                block.Bbox[0] * scale, block.Bbox[1] * scale,
                block.Bbox[2] * scale, block.Bbox[3] * scale,
                block.Text)).ToList();

            return new MobilePageResult
            {
                PreviewJpeg = jpeg,
                Lines = lines,
                Blocks = blocks,
                Markdown = page.Markdown ?? page.Text,
                Source = page.Source,
                Rotation = page.Rotation,
                Skew = page.Skew,
                TotalMs = page.Timings.GetValueOrDefault("total_ms"),
            };
        }
        finally
        {
            rotated?.Dispose();
        }
    }

    public void Dispose()
    {
        _ocr?.Dispose();
        _ocr = null;
        _gate.Dispose();
    }
}
