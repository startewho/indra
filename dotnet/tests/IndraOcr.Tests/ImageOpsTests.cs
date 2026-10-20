using IndraOcr.Imaging;
using OpenCvSharp;

namespace IndraOcr.Tests;

public class ImageOpsTests
{
    [Fact]
    public void Rot90MatchesNumpyCounterClockwise()
    {
        using var a = new Mat(2, 3, MatType.CV_8UC1, new Scalar(0));
        byte[,] values = { { 1, 2, 3 }, { 4, 5, 6 } };
        for (int r = 0; r < 2; r++)
        {
            for (int c = 0; c < 3; c++)
            {
                a.Set(r, c, values[r, c]);
            }
        }

        // np.rot90: out[i, j] = a[j, cols - 1 - i]
        using var ccw = ImageOps.Rot90(a, 1);
        Assert.Equal(3, ccw.Rows);
        Assert.Equal(2, ccw.Cols);
        byte[,] expected = { { 3, 6 }, { 2, 5 }, { 1, 4 } };
        for (int r = 0; r < expected.GetLength(0); r++)
        {
            for (int c = 0; c < expected.GetLength(1); c++)
            {
                Assert.Equal(expected[r, c], ccw.At<byte>(r, c));
            }
        }

        using var cw = ImageOps.Rot90(a, -1);
        Assert.Equal(4, cw.At<byte>(0, 0));
        Assert.Equal(1, cw.At<byte>(0, 1));
        Assert.Equal(6, cw.At<byte>(2, 0));

        using var half = ImageOps.Rot90(a, 2);
        Assert.Equal(6, half.At<byte>(0, 0));
        Assert.Equal(1, half.At<byte>(1, 2));

        using var none = ImageOps.Rot90(a, 4);
        Assert.Equal(1, none.At<byte>(0, 0));
    }

    [Fact]
    public void InkMapIsOneMinusGrayscaleOver255()
    {
        using var m = new Mat(1, 2, MatType.CV_8UC1, new Scalar(0));
        m.Set(0, 0, (byte)0);
        m.Set(0, 1, (byte)255);
        var ink = ImageOps.ToInk(m);
        Assert.Equal(1.0f, ink[0]);
        Assert.Equal(0.0f, ink[1]);
    }
}
