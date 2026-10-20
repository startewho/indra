using IndraOcr.Imaging;
using IndraOcr.Numerics;
using OpenCvSharp;

namespace IndraOcr.Pipeline;

/// <summary>One recognizer input: an ink map [Height, Width], row major.</summary>
public readonly record struct LineCrop(float[] Data, int Height, int Width);

/// <summary>
/// rec_preproc.py: the line crop rule shared by training and inference.  The
/// text scale and margins must match the recognizer's training exactly.
/// </summary>
public static class RecPreproc
{
    public const int LineHeight = 32;
    public const float PadY = 0.2f;
    public const float PadX = 0.3f;
    public const int WidthMultiple = 8;

    /// <summary>
    /// Light text on a dark ground (a banner, a dark table head): the ground (the
    /// median of the line box) lies nearer the dark end than the text does, so the
    /// bright tail is the longer one.  Such a line is read inverted.
    /// </summary>
    public static bool IsDarkLine(Mat gray, ReadOnlySpan<float> box)
    {
        int h = gray.Rows, w = gray.Cols;
        int x0 = (int)box[0], y0 = (int)box[1], x1 = (int)box[2], y1 = (int)box[3];
        int r0 = Math.Max(0, y0), r1 = Math.Min(h, y1 + 1);
        int c0 = Math.Max(0, x0), c1 = Math.Min(w, x1 + 1);
        if (r1 <= r0 || c1 <= c0)
        {
            return false;
        }
        var pixels = new byte[(r1 - r0) * (c1 - c0)];
        int k = 0;
        for (int r = r0; r < r1; r++)
        {
            for (int c = c0; c < c1; c++)
            {
                pixels[k++] = gray.At<byte>(r, c);
            }
        }
        var q = Np.PercentilesFromHistogram(pixels, new[] { 0.02, 0.5, 0.98 });
        double lo = q[0], med = q[1], hi = q[2];
        return hi - med > 2 * (med - lo) + 8;
    }

    /// <summary>
    /// Crop one line from a uint8 gray page: expand by the pads, crop (white
    /// outside the page), resize to <paramref name="height"/> px high keeping the
    /// aspect ratio.  Returns an ink map (1 = ink).  A tall window keeps the text
    /// scale and adds the page above and below the line.
    /// </summary>
    public static LineCrop CropLine(
        Mat gray,
        ReadOnlySpan<float> box,
        int height = LineHeight,
        float padY = PadY,
        float padX = PadX,
        int maxWidth = 4096,
        int window = 1,
        bool? dark = null)
    {
        double padYEff = (window * (1 + 2.0 * padY) - 1) / 2;
        int hgt = height * window;

        float x0 = box[0], y0 = box[1], x1 = box[2], y1 = box[3];
        float h = Math.Max(1.0f, y1 - y0);
        float X0 = x0 - padX * h, X1 = x1 + padX * h;
        float Y0 = (float)(y0 - padYEff * h), Y1 = (float)(y1 + padYEff * h);
        int H = gray.Rows, W = gray.Cols;
        int ix0 = (int)Math.Floor(X0), iy0 = (int)Math.Floor(Y0);
        int ix1 = (int)Math.Ceiling(X1), iy1 = (int)Math.Ceiling(Y1);
        int cw = Math.Max(1, ix1 - ix0), ch = Math.Max(1, iy1 - iy0);
        int sx0 = Math.Max(0, ix0), sy0 = Math.Max(0, iy0);
        int sx1 = Math.Min(W, ix1), sy1 = Math.Min(H, iy1);
        bool isDark = dark ?? IsDarkLine(gray, box);

        using var crop = new Mat(ch, cw, MatType.CV_8UC1, new Scalar(isDark ? 0 : 255));
        if (sx1 > sx0 && sy1 > sy0)
        {
            using var src = new Mat(gray, new Rect(sx0, sy0, sx1 - sx0, sy1 - sy0));
            using var dst = new Mat(crop, new Rect(sx0 - ix0, sy0 - iy0, sx1 - sx0, sy1 - sy0));
            src.CopyTo(dst);
        }
        int outW = Math.Max(WidthMultiple, Math.Min(maxWidth, (int)Math.Round(cw * (double)hgt / ch)));
        using var resized = ImageOps.Resize(crop, outW, hgt, (double)hgt / ch);
        var bytes = ImageOps.ToBytes(resized);
        var ink = new float[bytes.Length];
        for (int i = 0; i < bytes.Length; i++)
        {
            float v = bytes[i] / 255.0f;
            ink[i] = isDark ? v : 1.0f - v;
        }
        return new LineCrop(ink, hgt, outW);
    }

    /// <summary>Right-pad ink maps to one width: [B, 1, H, W] and the true widths.</summary>
    public static (float[] Data, long[] Widths, int Height, int PaddedWidth) PadBatch(
        IReadOnlyList<LineCrop> lines, int multiple = WidthMultiple)
    {
        if (lines.Count == 0)
        {
            return (Array.Empty<float>(), Array.Empty<long>(), 0, 0);
        }
        int H = lines[0].Height;
        int wm = lines.Max(l => l.Width);
        wm = (int)Math.Ceiling(wm / (double)multiple) * multiple;
        var outv = new float[lines.Count * H * wm];
        var widths = new long[lines.Count];
        for (int i = 0; i < lines.Count; i++)
        {
            var line = lines[i];
            for (int r = 0; r < H; r++)
            {
                Array.Copy(line.Data, r * line.Width, outv, (i * H + r) * wm, line.Width);
            }
            widths[i] = line.Width;
        }
        return (outv, widths, H, wm);
    }

    /// <summary>
    /// Group line indices into batches of similar width, so little compute goes to
    /// padding.  Narrowest first; a batch closes at <paramref name="maxBatch"/>
    /// lines, when the next line is wider than maxRatio times the batch's first
    /// line, or when lines x width would pass maxArea (ONNX Runtime keeps its
    /// largest allocation).
    /// </summary>
    public static List<List<int>> WidthBatches(
        IReadOnlyList<int> widths,
        int maxBatch = 64,
        double maxRatio = 1.3,
        int minWidth = 64,
        int maxArea = 64 * 640)
    {
        var order = Enumerable.Range(0, widths.Count).OrderBy(i => widths[i]).ToList();
        var batches = new List<List<int>>();
        foreach (int i in order)
        {
            int wi = Math.Max(widths[i], minWidth);
            if (batches.Count > 0)
            {
                var last = batches[^1];
                int first = Math.Max(widths[last[0]], minWidth);
                if (last.Count < maxBatch
                    && wi <= maxRatio * first
                    && (last.Count + 1) * wi <= maxArea)
                {
                    last.Add(i);
                    continue;
                }
            }
            batches.Add(new List<int> { i });
        }
        return batches;
    }
}
