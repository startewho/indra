using IndraOcr.Document;
using IndraOcr.Pipeline;

namespace IndraOcr.Tests;

/// <summary>S4 parity: PDF pages through all three Converter modes (golden-pdf/&lt;mode&gt;).</summary>
public class PdfPageParityTests
{
    [SkippableTheory]
    [InlineData("auto")]
    [InlineData("text")]
    [InlineData("ocr")]
    public void EveryGoldenPdfPageMatchesPython(string mode)
    {
        Skip.If(TestPaths.Models is null, "models not found (set INDRA_OCR_MODEL_DIR)");
        string root = Path.Combine(TestPaths.Root, "dotnet", "tests", "parity", "golden-pdf", mode);
        var files = Directory.Exists(root)
            ? Directory.GetFiles(root, "page-*.json", SearchOption.AllDirectories)
            : Array.Empty<string>();
        Skip.If(files.Length == 0, $"no golden PDF pages for mode {mode} (run tools/dump_golden.py)");

        using var ocr = new OrtOcr(TestPaths.Models!, threads: 1);
        foreach (string path in files.OrderBy(p => p, StringComparer.Ordinal))
        {
            var golden = PageComparison.Load(path);
            string name = $"{mode}:{Path.GetRelativePath(root, path)}";
            var conv = new Converter(ocr, dpi: 200.0, mode: mode);
            var pages = conv.Convert(TestPaths.Full(golden.File)).ToList();
            Assert.True(golden.Page < pages.Count,
                $"{name}: golden page {golden.Page} but only {pages.Count} converted");
            PageComparison.AssertMatches(golden, pages[golden.Page].Page, name);
        }
    }
}
