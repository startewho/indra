using System.Text.Json;
using CliType = IndraOcr.Cli.Cli;

namespace IndraOcr.Tests;

/// <summary>S6 parity: CLI options, formats, output files and exit codes.</summary>
public class CliTests
{
    private static (int Code, string Out, string Err) Run(params string[] args)
    {
        var oldOut = Console.Out;
        var oldErr = Console.Error;
        var captureOut = new StringWriter();
        var captureErr = new StringWriter();
        Console.SetOut(captureOut);
        Console.SetError(captureErr);
        try
        {
            int code = new CliType().Run(args);
            return (code, captureOut.ToString(), captureErr.ToString());
        }
        finally
        {
            Console.SetOut(oldOut);
            Console.SetError(oldErr);
        }
    }

    private static string Scan() => TestPaths.Full("examples/samples/scan.png");

    [Fact]
    public void VersionAndHelp()
    {
        var version = Run("--version");
        Assert.Equal(0, version.Code);
        Assert.Equal($"indra-ocr {IndraOcrInfo.Version}\n", version.Out.Replace("\r\n", "\n"));

        var help = Run("--help");
        Assert.Equal(0, help.Code);
        Assert.Contains("usage: indra-ocr", help.Out);
        Assert.Contains("--format", help.Out);
    }

    [Fact]
    public void BadArgumentsExitWithTwo()
    {
        var format = Run("--format", "nope", Scan());
        Assert.Equal(2, format.Code);
        Assert.Contains("invalid choice", format.Err);

        var pages = Run(Scan(), "--max-pages", "-1");
        Assert.Equal(2, pages.Code);
        Assert.Contains("must be 0 (all pages) or more", pages.Err);

        var missing = Run("--out");
        Assert.Equal(2, missing.Code);
        Assert.Contains("expected one argument", missing.Err);

        var unknown = Run("--nope", Scan());
        Assert.Equal(2, unknown.Code);
        Assert.Contains("unrecognized arguments", unknown.Err);
    }

    [SkippableFact]
    public void MissingInputFailsWithOne()
    {
        Skip.If(TestPaths.Models is null, "models not found");
        string path = Path.Combine(TestPaths.Root, "no-such.pdf");
        var run = Run(path, "--cores", "1", "--model-dir", TestPaths.Models!);
        Assert.Equal(1, run.Code);
        Assert.Contains("no such file", run.Err);
        Assert.Contains("1 of 1 inputs failed", run.Err);
    }

    [SkippableFact]
    public void MarkdownOutputMatchesTheGoldens()
    {
        Skip.If(TestPaths.Models is null, "models not found");
        var run = Run(Scan(), "--cores", "1", "--model-dir", TestPaths.Models!);
        Assert.Equal(0, run.Code);
        Assert.Equal("", run.Err);

        string golden = Directory.GetFiles(TestPaths.Golden, "page-*.json", SearchOption.AllDirectories)
            .First(p => Path.GetFileName(Path.GetDirectoryName(p)!).StartsWith("scan-", StringComparison.Ordinal));
        var page = PageComparison.Load(golden);
        Assert.Equal((page.Markdown ?? "") + "\n", run.Out.Replace("\r\n", "\n"));
    }

    [SkippableFact]
    public void JsonOutputHasTheReferenceFields()
    {
        Skip.If(TestPaths.Models is null, "models not found");
        var run = Run(Scan(), "--format", "json", "--cores", "1", "--model-dir", TestPaths.Models!);
        Assert.Equal(0, run.Code);
        var lines = run.Out.Split('\n', StringSplitOptions.RemoveEmptyEntries);
        Assert.Single(lines);
        using var doc = JsonDocument.Parse(lines[0]);
        var root = doc.RootElement;
        Assert.Equal(Scan(), root.GetProperty("file").GetString());
        Assert.Equal(0, root.GetProperty("page").GetInt32());
        Assert.Equal("ocr", root.GetProperty("source").GetString());
        Assert.Equal(36, root.GetProperty("lines").GetArrayLength());
        Assert.False(string.IsNullOrEmpty(root.GetProperty("markdown").GetString()));
        Assert.Equal(9, root.GetProperty("blocks").GetArrayLength());
        Assert.True(root.TryGetProperty("timings", out _));
    }

    [SkippableFact]
    public void OutWritesOneFilePerInputAndKeepsDuplicateNames()
    {
        Skip.If(TestPaths.Models is null, "models not found");
        string root = Path.Combine(Path.GetTempPath(), "indra-cli-" + Guid.NewGuid().ToString("N"));
        try
        {
            Directory.CreateDirectory(Path.Combine(root, "a"));
            Directory.CreateDirectory(Path.Combine(root, "b"));
            string first = Path.Combine(root, "a", "scan.png");
            string second = Path.Combine(root, "b", "scan.png");
            File.Copy(Scan(), first);
            File.Copy(Scan(), second);
            string outDir = Path.Combine(root, "out");

            var run = Run(first, second, "--out", outDir, "--cores", "1", "--model-dir", TestPaths.Models!);
            Assert.Equal(0, run.Code);
            Assert.True(File.Exists(Path.Combine(outDir, "scan.md")));
            Assert.True(File.Exists(Path.Combine(outDir, "scan_2.md")));
            Assert.True(new FileInfo(Path.Combine(outDir, "scan.md")).Length > 0);
            Assert.True(new FileInfo(Path.Combine(outDir, "scan_2.md")).Length > 0);
        }
        finally
        {
            if (Directory.Exists(root))
            {
                Directory.Delete(root, recursive: true);
            }
        }
    }

    [SkippableFact]
    public void TimingsGoToStderr()
    {
        Skip.If(TestPaths.Models is null, "models not found");
        var run = Run(Scan(), "--cores", "1", "--model-dir", TestPaths.Models!, "--timings");
        Assert.Equal(0, run.Code);
        Assert.Contains("page 1: ocr,", run.Err);
        Assert.Contains(" ms ", run.Err);
    }
}
