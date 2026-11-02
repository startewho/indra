using IndraOcr.Desktop.Session;

namespace IndraOcr.Tests;

/// <summary>Exporting every recognized page as Markdown or plain text.</summary>
public class ExportContentTests
{
    private static PageResult Page(int index, string markdown, string text) => new()
    {
        Index = index,
        ImageJpeg = Array.Empty<byte>(),
        Lines = Array.Empty<LineView>(),
        Blocks = Array.Empty<BlockView>(),
        Markdown = markdown,
        Text = text,
        Json = "",
        Source = "ocr",
        Note = "",
        Rotation = 0,
        Skew = 0,
        Vertical = false,
        Timings = new Dictionary<string, double>(),
        TotalMs = 0,
    };

    [Fact]
    public void ASinglePageIsExportedWithoutAHeader()
    {
        var pages = new[] { Page(0, "# Title\n\nBody", "Title\nBody") };

        Assert.Equal("# Title\n\nBody\n", ExportContent.Build(pages, plainText: false));
        Assert.Equal("Title\nBody\n", ExportContent.Build(pages, plainText: true));
    }

    [Fact]
    public void MarkdownAddsPageHeadingsAndSeparators()
    {
        var pages = new[]
        {
            Page(1, "Second", "Second"),
            Page(0, "First", "First"),
        };

        string result = ExportContent.Build(pages, plainText: false);

        Assert.Equal("## Page 1\n\nFirst\n\n---\n\n## Page 2\n\nSecond\n", result);
    }

    [Fact]
    public void TextUsesPlainLinesAndPageHeadings()
    {
        var pages = new[]
        {
            Page(0, "# One", "first line\nsecond line"),
            Page(1, "**Two**", "third line"),
        };

        string result = ExportContent.Build(pages, plainText: true);

        Assert.Contains("===== Page 1 =====", result);
        Assert.Contains("first line\nsecond line", result);
        Assert.Contains("===== Page 2 =====", result);
        Assert.Contains("third line", result);
        Assert.DoesNotContain("# One", result);
        Assert.DoesNotContain("**Two**", result);
    }

    [Fact]
    public void NoPagesExportsAnEmptyString()
    {
        Assert.Equal("", ExportContent.Build(Array.Empty<PageResult>(), plainText: false));
        Assert.Equal("", ExportContent.Build(Array.Empty<PageResult>(), plainText: true));
    }
}
