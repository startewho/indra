using IndraOcr.Numerics;

namespace IndraOcr.Tests;

public class NpyTests
{
    [SkippableFact]
    public void ReadsTheReferenceDetectorArtifacts()
    {
        var pages = TestPaths.ArtifactPages();
        Skip.If(pages.Count == 0, "no artifacts (run tools/dump_artifacts.py)");
        string dir = Path.Combine(TestPaths.Artifacts, pages[0].Dir.Replace('/', Path.DirectorySeparatorChar));

        var x = Npy.Read(Path.Combine(dir, "det_x.npy"));
        Assert.Equal(2, x.Shape.Length);
        Assert.Equal("<f4", x.Descr);
        Assert.Equal(x.Shape[0] * x.Shape[1], x.Length);

        var outv = Npy.Read(Path.Combine(dir, "det_out.npy"));
        Assert.Equal(3, outv.Shape.Length);
        Assert.Equal(3, outv.Shape[0]);
        Assert.Equal(x.Shape[0] / 2, outv.Shape[1]);
        Assert.Equal(x.Shape[1] / 2, outv.Shape[2]);

        var boxes = Npy.Read(Path.Combine(dir, "boxes.npy"));
        Assert.Equal(2, boxes.Shape.Length);
        Assert.Equal(5, boxes.Shape[1]);
    }

    [SkippableFact]
    public void ReadsTheNgramLmArchive()
    {
        Skip.If(TestPaths.Models is null, "models not found");
        string path = Path.Combine(TestPaths.Models!, "lm.npz");
        Skip.IfNot(File.Exists(path), "lm.npz not in the model folder");

        var arrays = Npz.ReadAll(path);
        Assert.Contains("order", arrays.Keys);
        Assert.Contains("vocab_size", arrays.Keys);
        int order = (int)arrays["order"].AsInt64()[0];
        Assert.InRange(order, 2, 6);
        for (int n = 0; n < order; n++)
        {
            Assert.Contains($"k{n}", arrays.Keys);
            Assert.Contains($"v{n}", arrays.Keys);
            Assert.Contains($"hk{n}", arrays.Keys);
            Assert.Contains($"hb{n}", arrays.Keys);
            Assert.Equal("u8", arrays[$"k{n}"].Descr.Replace("<", ""));
        }
    }
}
