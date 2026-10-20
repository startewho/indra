using IndraOcr.Numerics;

namespace IndraOcr.Tests;

public class NpTests
{
    [Fact]
    public void MedianUsesTheMeanOfTheTwoMiddleValues()
    {
        Assert.Equal(3.0, Np.Median(new float[] { 5, 1, 3 }));
        Assert.Equal(2.5, Np.Median(new float[] { 4, 1, 3, 2 }));
        Assert.Equal(2.5, Np.Median(new float[] { 1, 2, 3, 4 }));
    }

    [Fact]
    public void PercentilesFromHistogramMatchNumpy()
    {
        var ramp = Enumerable.Range(0, 256).Select(v => (byte)v).ToArray();
        var got = Np.PercentilesFromHistogram(ramp, new[] { 0.02, 0.5, 0.98 });
        Assert.Equal(5.1, got[0], 9);
        Assert.Equal(127.5, got[1], 9);
        Assert.Equal(249.9, got[2], 9);

        var mixed = Enumerable.Repeat((byte)0, 100)
            .Concat(Enumerable.Repeat((byte)255, 100))
            .Concat(Enumerable.Repeat((byte)128, 50))
            .ToArray();
        var got2 = Np.PercentilesFromHistogram(mixed, new[] { 0.02, 0.5, 0.98 });
        Assert.Equal(0.0, got2[0], 9);
        Assert.Equal(128.0, got2[1], 9);
        Assert.Equal(255.0, got2[2], 9);
    }

    [Fact]
    public void SearchSortedMatchesNumpysSides()
    {
        var a = new float[] { 1, 2, 2, 3 };
        Assert.Equal(1, Np.SearchSorted(a, 2, right: false));
        Assert.Equal(3, Np.SearchSorted(a, 2, right: true));
        Assert.Equal(4, Np.SearchSorted(a, 9, right: false));
        Assert.Equal(0, Np.SearchSorted(a, 0, right: true));
    }

    [Fact]
    public void LogAddExpMatchesNumpy()
    {
        Assert.Equal(-1.1131289938851001, Np.LogAddExp(-1.5, -2.25), 12);
        Assert.Equal(double.NegativeInfinity, Np.LogAddExp(double.NegativeInfinity, double.NegativeInfinity));
        Assert.Equal(-2.5, Np.LogAddExp(double.NegativeInfinity, -2.5));
    }

    [Fact]
    public void BincountSumsWeights()
    {
        var got = Np.Bincount(new[] { 0, 1, 1, 3 }, new double[] { 0.5, 1.5, 2.5, 4.0 }, 5);
        Assert.Equal(new[] { 0.5, 4.0, 0.0, 4.0, 0.0 }, got);
    }

    [Fact]
    public void CumSumIsSequential()
    {
        Assert.Equal(new double[] { 1, 3, 6 }, Np.CumSum(new double[] { 1, 2, 3 }));
    }

    [Fact]
    public void ArgSortAndArgMaxFollowNumpy()
    {
        var a = new float[] { 3, 1, 2, 1 };
        Assert.Equal(0, Np.ArgMax(a));
        var idx = Np.ArgSort(a);
        Assert.Equal(new[] { 1, 3, 2, 0 }, idx);
        Assert.Equal(new[] { 1, 3, 2, 0 }, Np.ArgSortStable(a));
    }

    [Fact]
    public void ClipMatchesNumpyBounds()
    {
        Assert.Equal(0.0, Np.Clip(-1.0, 0.0, 10.0));
        Assert.Equal(10.0, Np.Clip(11.0, 0.0, 10.0));
        Assert.Equal(5.0, Np.Clip(5.0, 0.0, 10.0));
    }
}
