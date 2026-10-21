using IndraOcr.Desktop.Session;

namespace IndraOcr.Tests;

/// <summary>The desktop app's session layer: open a document and read one page.</summary>
public class DesktopSessionTests
{
    private static string Scan() => TestPaths.Full("examples/samples/scan.png");

    private static string Pdf(string name)
        => Path.Combine(TestPaths.Root, "dotnet", "tests", "parity", "pdfs", name);

    private static ReadOptions Auto() => new("auto", false, false, true, 200, 20);

    [SkippableFact]
    public void AnImageReadsOnePageWithBoxesMarkdownAndJson()
    {
        Skip.If(TestPaths.Models is null, "models not found");
        using var session = new OcrSession();
        session.SetModelDir(TestPaths.Models);
        session.Open(Scan());
        Assert.False(session.IsPdf);
        Assert.Equal(1, session.PageCount);

        var page = session.ReadPage(0, Auto());
        Assert.Equal("ocr", page.Source);
        Assert.Equal(36, page.Lines.Count);
        Assert.Equal(9, page.Blocks.Count);
        Assert.True(page.ImageJpeg.Length > 1000);
        Assert.Equal(0xFF, page.ImageJpeg[0]);
        Assert.Equal(0xD8, page.ImageJpeg[1]);
        Assert.False(string.IsNullOrEmpty(page.Markdown));
        Assert.Contains("\"page\": 0", page.Json);
        Assert.Contains("\"source\": \"ocr\"", page.Json);
        Assert.True(page.TotalMs > 0);
        Assert.True(page.Timings["det_ms"] > 0);
        Assert.All(page.Lines, line => Assert.True(line.X1 > line.X0));
    }

    [SkippableFact]
    public void APdfReadsBothPagesAndKeepsTheRotation()
    {
        Skip.If(TestPaths.Models is null, "models not found");
        using var session = new OcrSession();
        session.SetModelDir(TestPaths.Models);
        session.Open(Pdf("digital.pdf"));
        Assert.True(session.IsPdf);
        Assert.Equal(2, session.PageCount);

        var first = session.ReadPage(0, Auto());
        Assert.Equal("text", first.Source);
        Assert.Equal(0, first.Rotation);
        Assert.Equal(5, first.Lines.Count);

        var second = session.ReadPage(1, Auto());
        Assert.Equal("text", second.Source);
        Assert.Equal(270, second.Rotation);
        Assert.Equal(2, second.Lines.Count);
    }

    [Fact]
    public void AnUnreadableFileIsRejected()
    {
        using var session = new OcrSession();
        string path = Path.Combine(TestPaths.Root, "dotnet", "tests", "parity", "inputs", "notes.txt");
        Skip.IfNot(File.Exists(path), "no notes.txt fixture");
        var error = Assert.Throws<InvalidOperationException>(() => session.Open(path));
        Assert.Contains("not a PDF or a readable image", error.Message);
    }

    [SkippableFact]
    public void ModelPresenceChecksTheFolder()
    {
        Skip.If(TestPaths.Models is null, "models not found");
        Assert.True(OcrSession.ModelsPresent(TestPaths.Models, math: false));
        string empty = Path.Combine(Path.GetTempPath(), "indra-models-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(empty);
        try
        {
            Assert.False(OcrSession.ModelsPresent(empty, math: false));
        }
        finally
        {
            Directory.Delete(empty, recursive: true);
        }
    }
}
