using IndraOcr.Imaging;
using IndraOcr.Numerics;
using OpenCvSharp;

namespace IndraOcr.Pipeline;

/// <summary>Page turning, deskewing and the orientation model's input (pipeline.py helpers).</summary>
public static class Skew
{
    public const double DeskewMax = 45.0;
    public const double DeskewMin = 0.3;
    public const int OrientSide = 768;
    public const int PreviewSide = 1600;

    /// <summary>uint8 ink map (255 = ink) of the page shrunk to side pixels.</summary>
    public static Mat InkThumb(Mat gray, int side)
    {
        int h = gray.Rows, w = gray.Cols;
        double s = side / (double)Math.Max(h, w);
        int nw = Math.Max(1, (int)Math.Round(w * s));
        int nh = Math.Max(1, (int)Math.Round(h * s));
        using var small = new Mat();
        Cv2.Resize(gray, small, new Size(nw, nh), interpolation: InterpolationFlags.Area);
        var inv = new Mat();
        Cv2.BitwiseNot(small, inv);
        return inv;
    }

    /// <summary>How sharply the horizontal ink profile changes row to row, rotated by a degrees.</summary>
    public static double RowSharpness(Mat ink, double a)
    {
        int w = ink.Cols, h = ink.Rows;
        var m = Cv2.GetRotationMatrix2D(new Point2f(w / 2f, h / 2f), a, 1.0);
        using var r = new Mat();
        Cv2.WarpAffine(ink, r, m, new Size(w, h));
        var bytes = ImageOps.ToBytes(r);
        var p = new double[h];
        for (int row = 0; row < h; row++)
        {
            long sum = 0;
            int off = row * w;
            for (int col = 0; col < w; col++)
            {
                sum += bytes[off + col];
            }
            p[row] = sum;
        }
        double total = 0;
        for (int row = 1; row < h; row++)
        {
            double d = p[row] - p[row - 1];
            total += d * d;
        }
        return total;
    }

    /// <summary>
    /// Degrees (counter-clockwise) that straighten the text rows: the angle whose
    /// horizontal ink profile changes most sharply row to row.  Searched over
    /// +-45 in 2 degree steps, then refined in 0.5 and 0.1 degree steps.
    /// </summary>
    public static double EstimateSkew(Mat gray, int side = 800)
    {
        using var fine = InkThumb(gray, side);
        if (Cv2.Mean(fine).Val0 < 0.003 * 255)
        {
            return 0.0; // a blank page has no rows to straighten
        }
        using var coarse = InkThumb(gray, side / 2);
        var grid = new double[46];
        for (int i = 0; i < 46; i++)
        {
            grid[i] = -DeskewMax + 2.0 * i;
        }
        var coarseScores = grid.Select(a => RowSharpness(coarse, a)).ToArray();
        var top = Np.ArgSort(coarseScores.Select(v => (float)v).ToArray());
        var starts = new List<double> { 0.0 };
        foreach (int i in top.TakeLast(2))
        {
            if (!starts.Contains(grid[i]))
            {
                starts.Add(grid[i]);
            }
        }
        var near = new List<double>();
        foreach (double a0 in starts)
        {
            for (int k = 0; k <= 8; k++)
            {
                double a = a0 - 2.0 + 0.5 * k;
                if (!near.Any(v => Math.Abs(v - a) < 1e-9))
                {
                    near.Add(a);
                }
            }
        }
        double a1 = 0, bestScore = double.NegativeInfinity;
        foreach (double a in near)
        {
            double s = RowSharpness(fine, a);
            if (s > bestScore)
            {
                bestScore = s;
                a1 = a;
            }
        }
        double angle = 0;
        double lastBest = double.NegativeInfinity;
        for (int k = 0; k <= 10; k++)
        {
            double a = a1 - 0.5 + 0.1 * k;
            double s = RowSharpness(fine, a);
            if (s > lastBest)
            {
                lastBest = s;
                angle = a;
            }
        }
        return angle;
    }

    /// <summary>The page shrunk once to at most side pixels a side (area averaging).</summary>
    public static Mat Preview(Mat gray, int side = PreviewSide)
    {
        int h = gray.Rows, w = gray.Cols;
        double s = side / (double)Math.Max(h, w);
        if (s >= 1)
        {
            return gray;
        }
        var dst = new Mat();
        Cv2.Resize(gray, dst, new Size(Math.Max(1, (int)Math.Round(w * s)), Math.Max(1, (int)Math.Round(h * s))),
            interpolation: InterpolationFlags.Area);
        return dst;
    }

    /// <summary>The page rotated by degrees counter-clockwise about its centre (same size).</summary>
    public static Mat RotatePage(Mat gray, double degrees)
    {
        int h = gray.Rows, w = gray.Cols;
        var m = Cv2.GetRotationMatrix2D(new Point2f(w / 2f, h / 2f), degrees, 1.0);
        var dst = new Mat();
        Cv2.WarpAffine(gray, dst, m, new Size(w, h),
            InterpolationFlags.Linear, BorderTypes.Replicate);
        return dst;
    }

    /// <summary>A box of the page that RotatePage made, mapped back to the page before it.</summary>
    public static float[] UnrotateBox(ReadOnlySpan<float> box, double degrees, int w, int h)
    {
        var m = Cv2.GetRotationMatrix2D(new Point2f(w / 2f, h / 2f), -degrees, 1.0);
        double[] xs = { box[0], box[2], box[0], box[2] };
        double[] ys = { box[1], box[1], box[3], box[3] };
        double minX = double.MaxValue, minY = double.MaxValue, maxX = double.MinValue, maxY = double.MinValue;
        for (int i = 0; i < 4; i++)
        {
            double x = m.At<double>(0, 0) * xs[i] + m.At<double>(0, 1) * ys[i] + m.At<double>(0, 2);
            double y = m.At<double>(1, 0) * xs[i] + m.At<double>(1, 1) * ys[i] + m.At<double>(1, 2);
            minX = Math.Min(minX, x);
            minY = Math.Min(minY, y);
            maxX = Math.Max(maxX, x);
            maxY = Math.Max(maxY, y);
        }
        return new float[]
        {
            (float)Math.Round(Math.Clamp(minX, 0, w), 1),
            (float)Math.Round(Math.Clamp(minY, 0, h), 1),
            (float)Math.Round(Math.Clamp(maxX, 0, w), 1),
            (float)Math.Round(Math.Clamp(maxY, 0, h), 1),
        };
    }

    /// <summary>[side, side] float32 ink map (1 = ink): long side scaled to side, padded.</summary>
    public static float[] OrientInput(Mat gray, int side = OrientSide)
    {
        int h = gray.Rows, w = gray.Cols;
        double s = side / (double)Math.Max(h, w);
        using var small = new Mat();
        Cv2.Resize(gray, small,
            new Size(Math.Max(1, (int)Math.Round(w * s)), Math.Max(1, (int)Math.Round(h * s))),
            interpolation: InterpolationFlags.Area);
        var bytes = ImageOps.ToBytes(small);
        var outv = new float[side * side];
        int sh = small.Rows, sw = small.Cols;
        for (int r = 0; r < sh; r++)
        {
            for (int c = 0; c < sw; c++)
            {
                outv[r * side + c] = 1.0f - bytes[r * sw + c] / 255.0f;
            }
        }
        return outv;
    }
}
