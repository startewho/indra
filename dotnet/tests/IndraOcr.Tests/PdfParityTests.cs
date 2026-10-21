using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using IndraOcr.Imaging;
using IndraOcr.Pdf;
using IndraOcr.TextLayer;
using OpenCvSharp;

namespace IndraOcr.Tests;

/// <summary>
/// S4 parity: PDFium rendering (pixel hashes) and the text-layer reader against
/// tools/dump_pdf_fixtures.py.
/// </summary>
public class PdfParityTests
{
    private sealed record RenderInfo(double Scale, int[] Shape, string Sha256);

    private sealed record LayerLine(double[] Bbox, string Text, int Quarter, int? Seq, JsonElement[] Chars);

    private sealed record StatsInfo(
        int Chars, int Rotated, int Bad, int[] Quarters, int OffPage, int[] Steps,
        int Turn, int Frame,
        [property: JsonPropertyName("visual_rtl")] double? VisualRtl,
        [property: JsonPropertyName("no_ink")] double? NoInk);

    private sealed record LayerInfo(
        double[] Size, bool Trusted, string Reason, StatsInfo Stats,
        LayerLine[] Lines, LayerLine[] Rotated);

    private sealed record PageInfo(
        int Index, double[] Size, int Rotation, RenderInfo[] Renders,
        Dictionary<string, LayerInfo> Layers,
        [property: JsonPropertyName("page_turn")] int PageTurn);

    private sealed record Fixture(Dictionary<string, PageInfo[]> Pdfs);

    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNameCaseInsensitive = true,
    };

    private static string Hash(byte[] data) => Convert.ToHexString(SHA256.HashData(data)).ToLowerInvariant();

    [SkippableFact]
    public void PdfiumRenderingAndTextLayersMatchPython()
    {
        string path = Path.Combine(TestPaths.Root, "dotnet", "tests", "parity", "pdf-fixtures.json");
        Skip.IfNot(File.Exists(path), "no PDF fixtures (run tools/dump_pdf_fixtures.py)");
        var fixture = JsonSerializer.Deserialize<Fixture>(File.ReadAllText(path), JsonOpts)!;

        foreach (var (name, pages) in fixture.Pdfs)
        {
            string pdfPath = Path.Combine(TestPaths.Root, "dotnet", "tests", "parity", "pdfs", name);
            using var doc = PdfDocument.Open(pdfPath);
            foreach (var golden in pages)
            {
                using var page = doc.LoadPage(golden.Index);

                foreach (var render in golden.Renders)
                {
                    var bytes = page.RenderGray(render.Scale);
                    Assert.Equal(render.Shape[0] * render.Shape[1], bytes.Length);
                    Assert.True(render.Sha256 == Hash(bytes),
                        $"{name} page {golden.Index} render at {render.Scale}: pixels differ");
                }

                foreach (var (key, layer) in golden.Layers)
                {
                    int turn = key == "vertical" ? 0 : int.Parse(key);
                    int frame = key == "vertical" ? 3 : turn;
                    var raw = page.RenderGray(1.0);
                    using var mat = ImageOps.FromBytes(raw, (int)page.Height, (int)page.Width);
                    int rotation = key == "vertical" ? 1 : ((-turn % 4) + 4) % 4;
                    using var turned = ImageOps.Rot90(mat, rotation);
                    var gray = ImageOps.ToBytes(turned);
                    var got = TextLayerReader.Read(
                        page, gray, turned.Cols, turned.Rows, 1.0, turn, frame);

                    Assert.True(layer.Trusted == got.Trusted,
                        $"{name} p{golden.Index} layer {key} trusted: expected {layer.Trusted}, got {got.Trusted}");
                    Assert.True(layer.Reason == got.Reason,
                        $"{name} p{golden.Index} layer {key} reason: expected '{layer.Reason}', got '{got.Reason}'");
                    Assert.Equal(layer.Size, new[] { got.Size.Width, got.Size.Height });

                    var s = layer.Stats;
                    var g = got.Stats;
                    Assert.Equal(s.Chars, g.Chars);
                    Assert.Equal(s.Rotated, g.Rotated);
                    Assert.Equal(s.Bad, g.Bad);
                    Assert.Equal(s.Quarters ?? new[] { 0, 0, 0, 0 }, g.Quarters);
                    Assert.Equal(s.OffPage, g.OffPage);
                    Assert.Equal(s.Steps ?? new[] { 0, 0 }, g.Steps);
                    Assert.Equal(s.Turn, g.Turn);
                    Assert.Equal(s.Frame, g.Frame);
                    Assert.True(NullableClose(s.VisualRtl, g.VisualRtl), $"{name} layer {key} visual_rtl");
                    Assert.True(NullableClose(s.NoInk, g.NoInk), $"{name} layer {key} no_ink");

                    AssertLines(name, golden.Index, key, "lines", layer.Lines, got.Lines);
                    AssertLines(name, golden.Index, key, "rotated", layer.Rotated, got.Rotated);
                }
            }
        }
    }

    private static bool NullableClose(double? a, double? b)
        => a is null ? b is null : b is not null && Math.Abs(a.Value - b.Value) <= 1e-9;

    private static void AssertLines(
        string name, int page, string key, string what, LayerLine[] expected, List<TextLine> actual)
    {
        Assert.True(expected.Length == actual.Count,
            $"{name} p{page} layer {key} {what}: expected {expected.Length} runs, got {actual.Count}");
        for (int i = 0; i < actual.Count; i++)
        {
            var e = expected[i];
            var a = actual[i];
            Assert.True(e.Text == a.Text,
                $"{name} p{page} layer {key} {what} {i} text: expected '{e.Text}', got '{a.Text}'");
            Assert.True(e.Quarter == a.Quarter, $"{name} p{page} {what} {i} quarter");
            Assert.True((e.Seq ?? -1) == a.Seq, $"{name} p{page} {what} {i} seq");
            for (int k = 0; k < 4; k++)
            {
                Assert.True(Math.Abs(e.Bbox[k] - a.Bbox[k]) <= 1e-6,
                    $"{name} p{page} {what} {i} bbox[{k}]: {e.Bbox[k]} vs {a.Bbox[k]}");
            }
            Assert.True(e.Chars.Length == a.Chars.Count,
                $"{name} p{page} {what} {i} char count: {e.Chars.Length} vs {a.Chars.Count}");
            for (int c = 0; c < a.Chars.Count; c++)
            {
                string ec = e.Chars[c][0].GetString()!;
                Assert.True(ec == a.Chars[c].Ch,
                    $"{name} p{page} {what} {i} char {c}: '{ec}' vs '{a.Chars[c].Ch}'");
                var box = e.Chars[c][1].EnumerateArray().Select(v => v.GetDouble()).ToArray();
                for (int k = 0; k < 4; k++)
                {
                    Assert.True(Math.Abs(box[k] - a.Chars[c].Box[k]) <= 1e-6,
                        $"{name} p{page} {what} {i} char {c} box[{k}]");
                }
            }
        }
    }
}
