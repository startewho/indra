using IndraOcr.Imaging;
using IndraOcr.Numerics;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace IndraOcr.Pipeline;

/// <summary>
/// line_mask.py: CenterMask window before the recognizer crops the line.  Used
/// only when the model folder ships mask.onnx (the default models do not).
/// </summary>
public static class LineMask
{
    public const int WindowH = 96;
    public const double WindowAbove = 1.0;
    public const double WindowSide = 0.5;
    public const float Keep = 0.5f;

    /// <summary>The window around a line box (page pixels), clipped to the page.</summary>
    public static (float X0, float Y0, float X1, float Y1) WindowBox(
        ReadOnlySpan<float> box, float pageW, float pageH)
    {
        float x0 = box[0], y0 = box[1], x1 = box[2], y1 = box[3];
        float h = Math.Max(1.0f, y1 - y0);
        return (
            Math.Max(0.0f, (float)(x0 - WindowSide * h)),
            Math.Max(0.0f, (float)(y0 - WindowAbove * h)),
            Math.Min(pageW, (float)(x1 + WindowSide * h)),
            Math.Min(pageH, (float)(y1 + WindowAbove * h)));
    }

    /// <summary>Scale a window to (w, h): linear down to half scale, area below.</summary>
    public static Mat ResizeWindow(Mat a, int width, int height)
    {
        double shrink = height / (double)Math.Max(1, a.Rows);
        var dst = new Mat();
        Cv2.Resize(a, dst, new Size(width, height),
            interpolation: shrink >= 0.5 ? InterpolationFlags.Linear : InterpolationFlags.Area);
        return dst;
    }

    private static (Mat Window, int X, int Y) Window(Mat gray, ReadOnlySpan<float> box)
    {
        var (x0, y0, x1, y1) = WindowBox(box, gray.Cols, gray.Rows);
        int ix0 = (int)Math.Round(x0), iy0 = (int)Math.Round(y0);
        int ix1 = (int)Math.Round(x1), iy1 = (int)Math.Round(y1);
        return (new Mat(gray, new Rect(ix0, iy0, Math.Max(1, ix1 - ix0), Math.Max(1, iy1 - iy0))), ix0, iy0);
    }

    /// <summary>
    /// Centre-line keep masks (bool, window resolution) for grayscale windows.
    /// Probabilities are scaled back bilinearly before the threshold.
    /// </summary>
    public static bool[][] MaskWindows(InferenceSession session, IReadOnlyList<Mat> windows)
    {
        if (windows.Count == 0)
        {
            return Array.Empty<bool[]>();
        }
        var sizes = new (int W, int H)[windows.Count];
        var xs = new float[windows.Count][];
        var widths = new int[windows.Count];
        for (int i = 0; i < windows.Count; i++)
        {
            var w = windows[i];
            sizes[i] = (w.Cols, w.Rows);
            int width = Math.Max(32, (int)Math.Round(w.Cols * (double)WindowH / Math.Max(w.Rows, 1) / 8) * 8);
            using var resized = ResizeWindow(w, width, WindowH);
            var bytes = ImageOps.ToBytes(resized);
            var ink = new float[bytes.Length];
            for (int k = 0; k < bytes.Length; k++)
            {
                ink[k] = 1.0f - bytes[k] / 255.0f;
            }
            xs[i] = ink;
            widths[i] = width;
        }

        var result = new bool[windows.Count][];
        var order = Np.ArgSortStable(widths.Select(v => (float)v).ToArray());
        for (int start = 0; start < order.Length; start += 16)
        {
            var idx = order.Skip(start).Take(16).ToArray();
            int wm = idx.Max(i => widths[i]);
            var batch = new float[idx.Length * WindowH * wm];
            for (int j = 0; j < idx.Length; j++)
            {
                int i = idx[j];
                for (int r = 0; r < WindowH; r++)
                {
                    Array.Copy(xs[i], r * widths[i], batch, (j * WindowH + r) * wm, widths[i]);
                }
            }
            var input = OrtUtil.FloatTensor(batch, idx.Length, 1, WindowH, wm);
            using var outputs = session.Run(new[] { NamedOnnxValue.CreateFromTensor("x", input) });
            var p = outputs.First().AsTensor<float>();
            var data = p.ToArray();
            var dims = p.Dimensions.ToArray();
            int pw = dims[3];
            for (int j = 0; j < idx.Length; j++)
            {
                int i = idx[j];
                var plane = new float[WindowH * pw];
                Array.Copy(data, j * WindowH * pw, plane, 0, plane.Length);
                using var pm = new Mat(WindowH, pw, MatType.CV_32FC1);
                System.Runtime.InteropServices.Marshal.Copy(plane, 0, pm.Data, plane.Length);
                using var up = new Mat();
                Cv2.Resize(pm, up, new Size(sizes[i].W, sizes[i].H), interpolation: InterpolationFlags.Linear);
                var pf = new float[sizes[i].W * sizes[i].H];
                System.Runtime.InteropServices.Marshal.Copy(up.Data, pf, 0, pf.Length);
                var keep = new bool[pf.Length];
                for (int k = 0; k < pf.Length; k++)
                {
                    keep[k] = pf[k] >= Keep;
                }
                result[i] = keep;
            }
        }
        return result;
    }

    /// <summary>For each line box: (window with foreign ink whitened, the box inside it).</summary>
    public static List<(Mat Window, float[] Box)> MaskedWindows(
        Mat gray, IReadOnlyList<float[]> boxes, InferenceSession session)
    {
        var wins = new List<Mat>(boxes.Count);
        var offs = new List<(int X, int Y)>(boxes.Count);
        foreach (var b in boxes)
        {
            var (w, x, y) = Window(gray, b);
            wins.Add(w);
            offs.Add((x, y));
        }
        var keeps = MaskWindows(session, wins);
        var outv = new List<(Mat, float[])>(boxes.Count);
        for (int i = 0; i < boxes.Count; i++)
        {
            var w = wins[i];
            var keep = keeps[i];
            var m = w.Clone();
            if (w.Total() > 0)
            {
                int rows = w.Rows, cols = w.Cols;
                var sub = new List<byte>();
                for (int r = 0; r < rows; r += 4)
                {
                    for (int c = 0; c < cols; c += 4)
                    {
                        sub.Add(w.At<byte>(r, c));
                    }
                }
                var paper = (byte)Np.Percentile(sub.Select(v => (float)v).ToArray(), 0.9);
                for (int r = 0; r < rows; r++)
                {
                    for (int c = 0; c < cols; c++)
                    {
                        if (!keep[r * w.Cols + c])
                        {
                            m.Set(r, c, paper);
                        }
                    }
                }
            }
            var b = boxes[i];
            outv.Add((m, new[]
            {
                b[0] - offs[i].X, b[1] - offs[i].Y, b[2] - offs[i].X, b[3] - offs[i].Y,
            }));
        }
        return outv;
    }

    /// <summary>Recognizer inputs for the line boxes of one page.</summary>
    public static List<LineCrop> CropLines(
        Mat gray,
        IReadOnlyList<float[]> boxes,
        int height,
        InferenceSession? mask = null,
        int window = 1,
        IReadOnlyList<bool>? dark = null)
    {
        var outv = new List<LineCrop>(boxes.Count);
        if (mask is null || window > 1)
        {
            for (int i = 0; i < boxes.Count; i++)
            {
                outv.Add(RecPreproc.CropLine(
                    gray, boxes[i], height, window: window, dark: dark?[i]));
            }
            return outv;
        }
        foreach (var (win, box) in MaskedWindows(gray, boxes, mask))
        {
            using (win)
            {
                outv.Add(RecPreproc.CropLine(win, box, height));
            }
        }
        return outv;
    }

    /// <summary>is_dark_line of each line box (light text on a dark ground).</summary>
    public static bool[] Darkness(Mat gray, IReadOnlyList<float[]> boxes)
    {
        var outv = new bool[boxes.Count];
        for (int i = 0; i < boxes.Count; i++)
        {
            outv[i] = RecPreproc.IsDarkLine(gray, boxes[i]);
        }
        return outv;
    }
}
