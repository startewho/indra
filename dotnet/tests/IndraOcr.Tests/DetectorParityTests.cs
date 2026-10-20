using System.Text.Json;
using IndraOcr.Imaging;
using IndraOcr.Numerics;
using IndraOcr.Pipeline;
using Microsoft.ML.OnnxRuntime;

namespace IndraOcr.Tests;

/// <summary>
/// S2 parity: the detector, from the page image to the boxes, against the
/// artifacts tools/dump_artifacts.py wrote with the Python reference.
/// </summary>
public class DetectorParityTests
{
    [SkippableFact]
    public void DetectorMatchesPythonOnEveryArtifactPage()
    {
        Skip.If(TestPaths.Models is null, "models not found (set INDRA_OCR_MODEL_DIR)");
        var pages = TestPaths.ArtifactPages();
        Skip.If(pages.Count == 0, "no artifacts (run tools/dump_artifacts.py)");

        using var session = new InferenceSession(
            Path.Combine(TestPaths.Models!, "det.onnx"), OrtUtil.Options(1));
        using var ocr = new OrtOcr(TestPaths.Models!, threads: 1);

        foreach (var page in pages)
        {
            string dir = Path.Combine(TestPaths.Artifacts, page.Dir.Replace('/', Path.DirectorySeparatorChar));
            using var meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "meta.json")));
            int longSide = meta.RootElement.GetProperty("det_long_side").GetInt32();

            using var gray = ImageOps.LoadGray(TestPaths.Full(page.Input));
            var goldenX = Npy.Read(Path.Combine(dir, "det_x.npy"));
            var (x, h, w, scale) = DetPost.PageToInput(gray, longSide);

            Assert.Equal(goldenX.Shape[0], h);
            Assert.Equal(goldenX.Shape[1], w);
            AssertClose(goldenX.AsFloat32(), x, 1e-6, $"{page.Dir} detector input");

            var input = OrtUtil.FloatTensor(x, 1, 1, h, w);
            using var outputs = session.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor("x", input),
            });
            var outTensor = outputs.First().AsTensor<float>();
            var (detOut, dims4) = (outTensor.ToArray(), outTensor.Dimensions.ToArray());
            var goldenOut = Npy.Read(Path.Combine(dir, "det_out.npy"));
            Assert.Equal(4, dims4.Length);
            Assert.Equal(1, dims4[0]);
            var dims = new[] { dims4[1], dims4[2], dims4[3] };
            Assert.Equal(goldenOut.Shape, dims);
            AssertClose(goldenOut.AsFloat32(), detOut, 1e-6, $"{page.Dir} detector output");

            var boxes = ocr.DetectAll(gray).Boxes;
            var golden = Npy.Read(Path.Combine(dir, "boxes.npy"));
            float[] expected = golden.AsFloat32();
            Assert.Equal(expected.Length, boxes.Length);

            for (int i = 0; i < boxes.Length; i++)
            {
                double delta = Math.Abs(expected[i] - boxes[i]);
                double tol = i % 5 == 4 ? 1e-4 : 0.5;
                Assert.True(delta <= tol,
                    $"{page.Dir} box value {i}: expected {expected[i]}, got {boxes[i]} (delta {delta})");
            }
        }
    }

    internal static void AssertClose(float[] expected, float[] actual, double tol, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        double worst = 0;
        int worstAt = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = Math.Abs((double)expected[i] - actual[i]);
            if (d > worst)
            {
                worst = d;
                worstAt = i;
            }
        }
        if (worst > tol)
        {
            Assert.Fail(
                $"{what}: max |delta| {worst:g6} at {worstAt} " +
                $"(expected {expected[worstAt]}, got {actual[worstAt]}, tol {tol:g3})");
        }
    }
}
