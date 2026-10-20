using System.Text.Json;
using IndraOcr.Imaging;
using IndraOcr.Numerics;
using IndraOcr.Pipeline;

namespace IndraOcr.Tests;

/// <summary>
/// S2 parity: detection + recognition from the page image, compared with the
/// per-line output tools/dump_artifacts.py recorded from the Python reference.
/// </summary>
public class RecognitionParityTests
{
    private sealed record GoldenLine(
        float[] Box, string Group, bool Sure, string Text, double Conf, int[] Ids,
        bool Dark, CropInfo Crop, CropInfo Tall);

    private sealed record CropInfo(int[] Shape, string Sha256);

    private static string Hash(LineCrop crop)
    {
        var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(crop.Data.AsSpan());
        return Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(bytes)).ToLowerInvariant();
    }

    private static readonly JsonSerializerOptions JsonOpts = new() { PropertyNameCaseInsensitive = true };

    [SkippableFact]
    public void RecognitionMatchesPythonOnEveryArtifactPage()
    {
        Skip.If(TestPaths.Models is null, "models not found (set INDRA_OCR_MODEL_DIR)");
        var pages = TestPaths.ArtifactPages();
        Skip.If(pages.Count == 0, "no artifacts (run tools/dump_artifacts.py)");

        using var ocr = new OrtOcr(TestPaths.Models!, threads: 1);
        foreach (var page in pages)
        {
            string dir = Path.Combine(TestPaths.Artifacts, page.Dir.Replace('/', Path.DirectorySeparatorChar));
            using var gray = ImageOps.LoadGray(TestPaths.Full(page.Input));

            var (boxes, pictures) = ocr.DetectAll(gray);
            var golden = JsonSerializer.Deserialize<List<GoldenLine>>(
                File.ReadAllText(Path.Combine(dir, "rec.json")), JsonOpts)!;
            var goldenBoxes = Npy.Read(Path.Combine(dir, "boxes.npy")).AsFloat32();
            Assert.Equal(goldenBoxes.Length, boxes.Length);
            for (int i = 0; i < boxes.Length; i++)
            {
                Assert.True(Math.Abs(goldenBoxes[i] - boxes[i]) <= 0.5,
                    $"{page.Dir} box[{i}]: expected {goldenBoxes[i]}, got {boxes[i]}");
            }
            _ = pictures;

            var rec = ocr.Recognize(gray, boxes);

            var (groups, sure) = ocr.BoxGroups(gray, boxes);
            for (int i = 0; i < groups.Length; i++)
            {
                Assert.True(golden[i].Group == groups[i],
                    $"{page.Dir} line {i} group: expected {golden[i].Group}, got {groups[i]}");
                Assert.True(golden[i].Sure == sure[i],
                    $"{page.Dir} line {i} sure: expected {golden[i].Sure}, got {sure[i]}");
            }

            var boxList = Enumerable.Range(0, boxes.Length / 5)
                .Select(i => OrtOcr.BoxRow(boxes, i)).ToList();
            var dark = ocr.Window > 1 ? LineMask.Darkness(gray, boxList) : null;
            var crops = LineMask.CropLines(gray, boxList, ocr.Height, ocr.Mask, 1, dark);
            var tall = ocr.Window > 1
                ? LineMask.CropLines(gray, boxList, ocr.Height, null, ocr.Window, dark)
                : crops;

            Assert.Equal(golden.Count, rec.Length);
            for (int i = 0; i < rec.Length; i++)
            {
                var g = golden[i];
                Assert.True(g.Dark == dark![i],
                    $"{page.Dir} line {i} dark: expected {g.Dark}, got {dark[i]}");
                Assert.True(g.Crop.Shape.SequenceEqual(new[] { crops[i].Height, crops[i].Width }),
                    $"{page.Dir} line {i} crop shape: expected [{string.Join(',', g.Crop.Shape)}], " +
                    $"got [{crops[i].Height},{crops[i].Width}]");
                Assert.True(g.Crop.Sha256 == Hash(crops[i]),
                    $"{page.Dir} line {i} crop pixels differ");
                Assert.True(g.Tall.Shape.SequenceEqual(new[] { tall[i].Height, tall[i].Width }),
                    $"{page.Dir} line {i} tall shape: expected [{string.Join(',', g.Tall.Shape)}], " +
                    $"got [{tall[i].Height},{tall[i].Width}]");
                Assert.True(g.Tall.Sha256 == Hash(tall[i]),
                    $"{page.Dir} line {i} tall crop pixels differ");
                Assert.True(g.Text == rec[i].Text,
                    $"{page.Dir} line {i} text: expected '{g.Text}' (group {g.Group}, conf {g.Conf}, " +
                    $"ids [{string.Join(',', g.Ids)}]), got '{rec[i].Text}' (conf {rec[i].Conf}, " +
                    $"ids [{string.Join(',', rec[i].Tokens.Select(t => t.Id))}])");
                Assert.True(Math.Abs(g.Conf - rec[i].Conf) <= 1e-4,
                    $"{page.Dir} line {i} conf: expected {g.Conf}, got {rec[i].Conf}");
                int[] ids = rec[i].Tokens.Where(t => t.Char is null).Select(t => t.Id).ToArray();
                Assert.True(g.Ids.SequenceEqual(ids),
                    $"{page.Dir} line {i} ids: expected [{string.Join(',', g.Ids)}], " +
                    $"got [{string.Join(',', ids)}]");
            }
        }
    }
}
