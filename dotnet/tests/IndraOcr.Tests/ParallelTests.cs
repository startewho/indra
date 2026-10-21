using System.Text.Json;
using IndraOcr.Parallel;
using IndraOcr.Pipeline;

namespace IndraOcr.Tests;

/// <summary>S6 parity: convert_many order, error isolation and page parallelism.</summary>
public class ParallelTests
{
    private static string Pdf(string name) => Path.Combine(TestPaths.Root, "dotnet", "tests", "parity", "pdfs", name);

    [SkippableFact]
    public void ConvertManyKeepsInputOrderAndIsolatesErrors()
    {
        Skip.If(TestPaths.Models is null, "models not found");
        string scan = TestPaths.Full("examples/samples/scan.png");
        string missing = Path.Combine(TestPaths.Root, "no-such.pdf");
        var results = PageRunner.ConvertMany(
                new[] { scan, missing, Pdf("digital.pdf") },
                cores: 2, modelDir: TestPaths.Models, mode: "auto")
            .ToList();

        Assert.Equal(3, results.Count);
        Assert.Equal(scan, results[0].Path);
        Assert.Null(results[0].Error);
        Assert.Single(results[0].Pages);
        Assert.Equal("ocr", results[0].Pages[0].Page.Source);

        Assert.Equal(missing, results[1].Path);
        Assert.NotNull(results[1].Error);
        Assert.Contains("no such file", results[1].Error!.Message);

        Assert.Equal(Pdf("digital.pdf"), results[2].Path);
        Assert.Null(results[2].Error);
        Assert.Equal(2, results[2].Pages.Count);
        Assert.Equal(new[] { 0, 1 }, results[2].Pages.Select(p => p.Index).ToArray());
        Assert.All(results[2].Pages, p => Assert.Equal("text", p.Page.Source));
    }

    [SkippableFact]
    public void ParallelPdfPagesMatchTheReferenceGoldens()
    {
        Skip.If(TestPaths.Models is null, "models not found");
        string pdf = Pdf("multipage.pdf");
        var result = PageRunner.ConvertMany(
                new[] { pdf }, cores: 4, modelDir: TestPaths.Models, mode: "auto")
            .Single();
        Assert.Null(result.Error);
        Assert.Equal(4, result.Pages.Count);

        string root = Path.Combine(TestPaths.Root, "dotnet", "tests", "parity", "golden-pdf", "auto");
        var goldens = Directory.GetFiles(root, "page-*.json", SearchOption.AllDirectories)
            .Where(p => Path.GetFileName(Path.GetDirectoryName(p)!).StartsWith("multipage-", StringComparison.Ordinal))
            .OrderBy(p => p, StringComparer.Ordinal)
            .ToList();
        Assert.Equal(4, goldens.Count);
        for (int i = 0; i < goldens.Count; i++)
        {
            var golden = PageComparison.Load(goldens[i]);
            var page = result.Pages[i].Page;
            Assert.Equal(i, result.Pages[i].Index);
            PageComparison.AssertMatches(golden, page, $"multipage page {i} (cores=4)");
        }
    }

    [SkippableFact]
    public void CoresDoNotChangeTheOutput()
    {
        Skip.If(TestPaths.Models is null, "models not found");
        string pdf = Pdf("multipage.pdf");
        var serial = PageRunner.ConvertMany(
                new[] { pdf }, cores: 1, modelDir: TestPaths.Models, mode: "auto")
            .Single();
        var parallel = PageRunner.ConvertMany(
                new[] { pdf }, cores: 4, modelDir: TestPaths.Models, mode: "auto")
            .Single();
        Assert.Equal(serial.Pages.Count, parallel.Pages.Count);
        for (int i = 0; i < serial.Pages.Count; i++)
        {
            var a = serial.Pages[i].Page;
            var b = parallel.Pages[i].Page;
            Assert.Equal(a.Text, b.Text);
            Assert.Equal(a.Markdown, b.Markdown);
            Assert.Equal(a.Lines.Count, b.Lines.Count);
            Assert.Equal(a.Rotation, b.Rotation);
            Assert.Equal(a.Note, b.Note);
        }
    }

    [SkippableFact]
    public void AFileThatCannotBeOpenedIsReportedOnce()
    {
        Skip.If(TestPaths.Models is null, "models not found");
        var results = PageRunner.ConvertMany(
                new[] { Pdf("locked.pdf") }, cores: 2, modelDir: TestPaths.Models)
            .ToList();
        Assert.Single(results);
        Assert.NotNull(results[0].Error);
        Assert.Contains("password-protected PDF", results[0].Error!.Message);
    }
}
