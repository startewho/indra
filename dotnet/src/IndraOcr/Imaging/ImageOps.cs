using System.Runtime.InteropServices;
using OpenCvSharp;

namespace IndraOcr.Imaging;

/// <summary>Image loading and the OpenCV operations the pipeline uses.</summary>
public static class ImageOps
{
    /// <summary>Grayscale page of an image file (first frame).</summary>
    public static Mat LoadGray(string path)
    {
        var m = Cv2.ImRead(path, ImreadModes.Grayscale);
        if (m.Empty())
        {
            throw new IOException($"{path}: not a readable image");
        }
        return m;
    }

    /// <summary>Every frame of a multi-page TIFF (or the single frame of other files).</summary>
    public static List<Mat> LoadGrayMulti(string path)
    {
        var frames = new List<Mat>();
        if (Cv2.ImReadMulti(path, out Mat[]? mats, ImreadModes.Grayscale) && mats is { Length: > 0 })
        {
            frames.AddRange(mats);
            return frames;
        }
        var one = Cv2.ImRead(path, ImreadModes.Grayscale);
        if (!one.Empty())
        {
            frames.Add(one);
        }
        return frames;
    }

    /// <summary>cv2.resize; area for shrinking, linear for growing.</summary>
    public static Mat Resize(Mat src, int width, int height, double scale)
    {
        var dst = new Mat();
        var interp = scale < 1 ? InterpolationFlags.Area : InterpolationFlags.Linear;
        Cv2.Resize(src, dst, new Size(width, height), interpolation: interp);
        return dst;
    }

    /// <summary>np.rot90(page, k): counter-clockwise for positive k.</summary>
    public static Mat Rot90(Mat src, int k)
    {
        int turns = ((k % 4) + 4) % 4;
        if (turns == 0)
        {
            return src.Clone();
        }
        var dst = new Mat();
        Cv2.Rotate(src, dst, turns switch
        {
            1 => RotateFlags.Rotate90Counterclockwise,
            2 => RotateFlags.Rotate180,
            _ => RotateFlags.Rotate90Clockwise,
        });
        return dst;
    }

    /// <summary>uint8 grayscale Mat -> float32 ink map (0 = paper, 1 = ink), row major.</summary>
    public static float[] ToInk(Mat gray)
    {
        var bytes = ToBytes(gray);
        var ink = new float[bytes.Length];
        for (int i = 0; i < bytes.Length; i++)
        {
            ink[i] = 1.0f - bytes[i] / 255.0f;
        }
        return ink;
    }

    public static byte[] ToBytes(Mat m)
    {
        if (m.Type() != MatType.CV_8UC1)
        {
            throw new ArgumentException($"expected CV_8UC1, got {m.Type()}");
        }
        var data = new byte[checked((int)(m.Total() * m.ElemSize()))];
        if (!m.IsContinuous())
        {
            m = m.Clone();
        }
        Marshal.Copy(m.Data, data, 0, data.Length);
        return data;
    }

    public static Mat FromBytes(byte[] data, int height, int width)
    {
        var m = new Mat(height, width, MatType.CV_8UC1);
        Marshal.Copy(data, 0, m.Data, data.Length);
        return m;
    }

    public static int[] ToInt32(Mat m)
    {
        if (m.Type() != MatType.CV_32SC1)
        {
            throw new ArgumentException($"expected CV_32SC1, got {m.Type()}");
        }
        var data = new int[checked((int)m.Total())];
        if (!m.IsContinuous())
        {
            m = m.Clone();
        }
        Marshal.Copy(m.Data, data, 0, data.Length);
        return data;
    }

    /// <summary>
    /// cv2.connectedComponentsWithStats; returns the label count, the label map
    /// and the [n, 5] stats (left, top, width, height, area) flattened.
    /// </summary>
    public static (int count, int[] labels, int[] stats) Components(Mat mask, bool eightConnected)
    {
        using var labels = new Mat();
        using var stats = new Mat();
        using var centroids = new Mat();
        int count = Cv2.ConnectedComponentsWithStats(
            mask, labels, stats, centroids,
            eightConnected ? PixelConnectivity.Connectivity8 : PixelConnectivity.Connectivity4);
        return (count, ToInt32(labels), ToInt32(stats));
    }
}
