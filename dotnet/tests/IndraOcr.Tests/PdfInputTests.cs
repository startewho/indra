using IndraOcr.Document;
using IndraOcr.Pipeline;

namespace IndraOcr.Tests;

/// <summary>S4 input handling: passwords, TIFF frames, page limits and bad files.</summary>
public class PdfInputTests
{
    private static string Pdf(string name) => Path.Combine(TestPaths.Root, "dotnet", "tests", "parity", "pdfs", name);

    private static string File_(string name) => Path.Combine(TestPaths.Root, "dotnet", "tests", "parity", "inputs", name);

    [SkippableFact]
    public void APasswordProtectedPdfAsksForThePasswordAndOpensWithIt()
    {
        Skip.If(TestPaths.Models is null, "models not found");
        using var ocr = new OrtOcr(TestPaths.Models!, threads: 1);
        var conv = new Converter(ocr, mode: "text");

        var error = Assert.Throws<InputError>(() => conv.Convert(Pdf("locked.pdf")).ToList());
        Assert.Contains("password-protected PDF", error.Message);
        Assert.Contains("--password", error.Message);

        var pages = conv.Convert(Pdf("locked.pdf"), password: "pw").ToList();
        Assert.Equal(2, pages.Count); // the two digital.pdf pages
        Assert.Equal(0, pages[0].Index);
        Assert.Equal("text", pages[0].Page.Source);
    }

    [SkippableFact]
    public void ImageOcrCanBeDisabled()
    {
        Skip.If(TestPaths.Models is null, "models not found");
        using var ocr = new OrtOcr(TestPaths.Models!, threads: 1);
        var withImages = new Converter(ocr, mode: "auto").Convert(Pdf("hybrid.pdf")).First().Page;
        Assert.Equal("hybrid", withImages.Source);
        Assert.True(withImages.Lines.Count > 10);

        var without = new Converter(ocr, mode: "auto", imageOcr: false)
            .Convert(Pdf("hybrid.pdf")).First().Page;
        Assert.Equal("text", without.Source);
        Assert.Single(without.Lines);
        Assert.Equal("Digital caption line", without.Lines[0].Text);
    }

    [SkippableFact]
    public void EveryTiffFrameIsAPageAndMaxPagesLimitsThem()
    {
        Skip.If(TestPaths.Models is null, "models not found");
        string path = File_("frames.tiff");
        Skip.IfNot(File.Exists(path), "no TIFF fixture (run the make_inputs step)");
        using var ocr = new OrtOcr(TestPaths.Models!, threads: 1);
        var conv = new Converter(ocr, mode: "ocr");

        var pages = conv.Convert(path).ToList();
        Assert.Equal(2, pages.Count);
        Assert.Equal(new[] { 0, 1 }, pages.Select(p => p.Index).ToArray());
        Assert.All(pages, p => Assert.Equal("ocr", p.Page.Source));
        Assert.All(pages, p => Assert.NotEmpty(p.Page.Lines));

        var one = conv.Convert(path, maxPages: 1).ToList();
        Assert.Single(one);
    }

    [Fact]
    public void AnUnreadableFileIsAnInputError()
    {
        using var ocr = StubOcr();
        var conv = new Converter(ocr);
        var error = Assert.Throws<InputError>(() => conv.Convert(File_("notes.txt")).ToList());
        Assert.Contains("not a PDF or a readable image", error.Message);
    }

    [Fact]
    public void AMissingFileIsAnInputError()
    {
        using var ocr = StubOcr();
        var conv = new Converter(ocr);
        var error = Assert.Throws<InputError>(() => conv.Convert(File_("nope.pdf")).ToList());
        Assert.Contains("no such file", error.Message);
    }

    [Fact]
    public void ANegativePageLimitIsAUsageError()
    {
        using var ocr = StubOcr();
        var conv = new Converter(ocr);
        Assert.Throws<ArgumentException>(() => conv.Convert(Pdf("digital.pdf"), maxPages: -1).ToList());
    }

    [Fact]
    public void PdfIsDetectedFromTheBytes()
    {
        Assert.True(Converter.IsPdf(Pdf("digital.pdf")));
        Assert.False(Converter.IsPdf(File_("notes.txt")));
        Assert.False(Converter.IsPdf(File_("frames.tiff")));
    }

    /// <summary>A converter that never reaches the OCR path (input errors only).</summary>
    private static OrtOcr StubOcr()
    {
        Skip.If(TestPaths.Models is null, "models not found");
        return new OrtOcr(TestPaths.Models!, threads: 1);
    }
}
