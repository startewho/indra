using IndraOcr.Imaging;
using IndraOcr.Paddle;

namespace IndraOcr.Tests;

public class PaddleEngineTests
{
    private static string Models()
        => Path.Combine(TestPaths.Root, "dotnet", "src", "IndraOcr.Desktop", "Data", "paddleocr");

    [SkippableFact]
    public void PaddleReadsAnImageWithoutDownloading()
    {
        Skip.If(!File.Exists(Path.Combine(Models(), "ch_PP-OCRv4_det_infer.onnx")),
            "bundled PaddleOCR models not found");
        using var engine = new PaddleOcrEngine(Models(), threads: 2);
        using var gray = ImageOps.LoadGray(TestPaths.Full("examples/samples/scan.png"));
        var page = engine.Read(gray);
        Assert.Equal("paddleocr", page.Source);
        Assert.True(page.Lines.Count > 10);
        Assert.Contains(page.Lines, line => line.Text.Length > 0);
        Assert.Contains("det_ms", page.Timings.Keys);
        Assert.Contains("rec_ms", page.Timings.Keys);
    }
}
