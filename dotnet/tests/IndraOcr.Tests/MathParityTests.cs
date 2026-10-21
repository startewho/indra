using System.Text.Json;
using IndraOcr.Imaging;
using IndraOcr.Pipeline;
using ConverterType = IndraOcr.Document.Converter;

namespace IndraOcr.Tests;

/// <summary>
/// S5 parity: math line classification, the attention decoder's readings, and
/// whole pages with math enabled (tools/dump_math_fixtures.py, dump_golden --math).
/// </summary>
public class MathParityTests
{
    private sealed record GoldenLine(float[] Box, bool Flag, string? Text, double? Conf);

    private sealed record GoldenPage(string Input, GoldenLine[] Lines);

    private sealed record Fixture(GoldenPage[] Pages);

    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNameCaseInsensitive = true,
    };

    private static bool HasMathModels()
        => TestPaths.Models is not null
           && File.Exists(Path.Combine(TestPaths.Models, "mathcls.onnx"))
           && File.Exists(Path.Combine(TestPaths.Models, "math_meta.json"));

    [SkippableFact]
    public void MathReaderMatchesPython()
    {
        Skip.If(!HasMathModels(), "math models not found (set INDRA_OCR_MODEL_DIR)");
        string path = Path.Combine(TestPaths.Root, "dotnet", "tests", "parity", "math-fixtures.json");
        Skip.IfNot(File.Exists(path), "no math fixture (run tools/dump_math_fixtures.py)");
        var fixture = JsonSerializer.Deserialize<Fixture>(File.ReadAllText(path), JsonOpts)!;

        using var ocr = new OrtOcr(TestPaths.Models!, threads: 1, math: true);
        Assert.NotNull(ocr.MathRec);
        foreach (var page in fixture.Pages)
        {
            using var gray = ImageOps.LoadGray(TestPaths.Full(page.Input));
            var boxes = ocr.DetectAll(gray).Boxes;
            Assert.Equal(page.Lines.Length, boxes.Length / 5);
            var boxList = Enumerable.Range(0, boxes.Length / 5)
                .Select(i => OrtOcr.BoxRow(boxes, i))
                .ToList();
            var dark = LineMask.Darkness(gray, boxList);
            var crops = LineMask.CropLines(gray, boxList, ocr.Height, ocr.Mask, 1, dark);
            var flags = ocr.MathRec!.Flags(crops);

            var math = new List<int>();
            for (int i = 0; i < flags.Length; i++)
            {
                Assert.True(page.Lines[i].Flag == flags[i],
                    $"{page.Input} line {i} math flag: expected {page.Lines[i].Flag}, got {flags[i]}");
                if (flags[i])
                {
                    math.Add(i);
                }
            }

            var sub = new float[math.Count * 5];
            for (int k = 0; k < math.Count; k++)
            {
                Array.Copy(boxes, math[k] * 5, sub, k * 5, 4);
            }
            var readings = ocr.MathRec.Read(gray, sub);
            for (int k = 0; k < math.Count; k++)
            {
                var golden = page.Lines[math[k]];
                Assert.True(golden.Text == readings[k].Text,
                    $"{page.Input} line {math[k]} math text: expected '{golden.Text}', got '{readings[k].Text}'");
                Assert.True(Math.Abs((golden.Conf ?? 0) - readings[k].Conf) <= 1e-5,
                    $"{page.Input} line {math[k]} math conf: expected {golden.Conf}, got {readings[k].Conf}");
            }
        }
    }

    [SkippableFact]
    public void TextLayerMathLinesAreReplacedLikePython()
    {
        Skip.If(!HasMathModels(), "math models not found (set INDRA_OCR_MODEL_DIR)");
        string path = Path.Combine(TestPaths.Root, "dotnet", "tests", "parity", "math-fixtures.json");
        Skip.IfNot(File.Exists(path), "no math fixture (run tools/dump_math_fixtures.py)");
        var fixture = JsonSerializer.Deserialize<Fixture>(File.ReadAllText(path), JsonOpts)!;
        var page = fixture.Pages.First(p => p.Input.Contains("equations", StringComparison.Ordinal));

        using var ocr = new OrtOcr(TestPaths.Models!, threads: 1, math: true);
        using var gray = ImageOps.LoadGray(TestPaths.Full(page.Input));
        var boxes = ocr.DetectAll(gray).Boxes;
        int n = boxes.Length / 5;
        var rec = new Pipeline.RecLine?[n];
        for (int i = 0; i < n; i++)
        {
            float x = (boxes[i * 5] + boxes[i * 5 + 2]) / 2;
            rec[i] = new Pipeline.RecLine("x", 1.0,
                new[] { Pipeline.Token.FromChar("x", x, 0) });
        }
        var conv = new ConverterType(ocr, mode: "text");
        conv.ReadMath(gray, boxes, rec);

        for (int i = 0; i < n; i++)
        {
            var golden = page.Lines[i];
            bool replaced = golden.Flag
                            && (golden.Text?.Contains('$') ?? false)
                            && (golden.Conf ?? 0) >= MathOcr.MathReader.MinConf;
            if (replaced)
            {
                Assert.True(golden.Text == rec[i]!.Text,
                    $"line {i} text-layer math: expected '{golden.Text}', got '{rec[i]!.Text}'");
                Assert.True(rec[i]!.Tokens.Length == 1 && rec[i]!.Tokens[0].Seq == 0,
                    $"line {i}: one token with the layer's first content index");
            }
            else
            {
                Assert.True(rec[i]!.Text == "x", $"line {i} should keep the layer's text");
            }
        }
    }

    [SkippableFact]
    public void MathPagesMatchPython()
    {
        Skip.If(!HasMathModels(), "math models not found (set INDRA_OCR_MODEL_DIR)");
        string root = Path.Combine(TestPaths.Root, "dotnet", "tests", "parity", "golden-math");
        var files = Directory.Exists(root)
            ? Directory.GetFiles(root, "page-*.json", SearchOption.AllDirectories)
            : Array.Empty<string>();
        Skip.If(files.Length == 0, "no math golden pages (run tools/dump_golden.py --math)");

        using var ocr = new OrtOcr(TestPaths.Models!, threads: 1, math: true);
        foreach (string path in files.OrderBy(p => p, StringComparer.Ordinal))
        {
            var golden = PageComparison.Load(path);
            string name = Path.GetRelativePath(root, path);
            using var gray = ImageOps.LoadGray(TestPaths.Full(golden.File));
            var page = ocr.Read(gray);
            PageComparison.AssertMatches(golden, page, name);
        }
    }
}
