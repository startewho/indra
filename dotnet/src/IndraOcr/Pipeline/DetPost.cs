using IndraOcr.Imaging;
using OpenCvSharp;

namespace IndraOcr.Pipeline;

/// <summary>det_post.py: detector input and box decoding, shared with training.</summary>
public static class DetPost
{
    public const float CoreFrac = 0.4f;
    public const float CoreXShrink = 0.15f;
    public const int OutStride = 2;
    public const int InputMultiple = 32;

    /// <summary>
    /// Resize a uint8 page so its long side is <paramref name="longSide"/> and pad
    /// to a multiple of 32 with white: float32 ink map [H, W] and the scale.
    /// </summary>
    public static (float[] X, int Height, int Width, double Scale) PageToInput(Mat gray, int longSide)
    {
        int h = gray.Rows, w = gray.Cols;
        double scale = (double)longSide / Math.Max(h, w);
        int nh = Math.Max(1, (int)Math.Round(h * scale));
        int nw = Math.Max(1, (int)Math.Round(w * scale));
        Mat resized = gray;
        bool own = false;
        if (nh != h || nw != w)
        {
            resized = ImageOps.Resize(gray, nw, nh, scale);
            own = true;
        }
        try
        {
            int ph = (int)Math.Ceiling(nh / (double)InputMultiple) * InputMultiple;
            int pw = (int)Math.Ceiling(nw / (double)InputMultiple) * InputMultiple;
            var x = new float[ph * pw];
            var bytes = ImageOps.ToBytes(resized);
            for (int row = 0; row < nh; row++)
            {
                int src = row * nw;
                int dst = row * pw;
                for (int col = 0; col < nw; col++)
                {
                    x[dst + col] = 1.0f - bytes[src + col] / 255.0f;
                }
            }
            return (x, ph, pw, scale);
        }
        finally
        {
            if (own)
            {
                resized.Dispose();
            }
        }
    }

    /// <summary>
    /// Decode one page.  <paramref name="outv"/> is [C, h, w] (core logit, top
    /// distance, bottom distance).  Returns [N, 5] boxes in input pixels:
    /// x0, y0, x1, y1, score.
    /// </summary>
    public static float[] Decode(
        ReadOnlySpan<float> outv,
        int channels,
        int height,
        int width,
        float thresh = 0.5f,
        int minArea = 2,
        float growY = 0f,
        float growX = 0f)
    {
        int plane = height * width;
        var prob = new float[plane];
        for (int i = 0; i < plane; i++)
        {
            prob[i] = 1.0f / (1.0f + MathF.Exp(-outv[i]));
        }
        var mask = new byte[plane];
        for (int i = 0; i < plane; i++)
        {
            mask[i] = prob[i] > thresh ? (byte)1 : (byte)0;
        }

        using var maskMat = new Mat(height, width, MatType.CV_8UC1);
        System.Runtime.InteropServices.Marshal.Copy(mask, 0, maskMat.Data, mask.Length);
        var (count, labels, stats) = ImageOps.Components(maskMat, eightConnected: false);
        if (count <= 1)
        {
            return Array.Empty<float>();
        }

        const int stride = OutStride;
        var cnt = new double[count];
        var y0 = new double[count];
        var y1 = new double[count];
        var score = new double[count];
        var top = outv.Slice(plane, plane);
        var bot = outv.Slice(2 * plane, plane);
        for (int i = 0; i < plane; i++)
        {
            int lab = labels[i];
            if (lab <= 0)
            {
                continue;
            }
            int row = i / width;
            float ys = (row + 0.5f) * stride;
            cnt[lab] += 1.0;
            y0[lab] += ys - top[i];
            y1[lab] += ys + bot[i];
            score[lab] += prob[i];
        }
        for (int i = 1; i < count; i++)
        {
            double c = Math.Max(cnt[i], 1.0);
            y0[i] /= c;
            y1[i] /= c;
            score[i] /= c;
        }

        var boxes = new List<float>();
        for (int i = 1; i < count; i++)
        {
            int area = stats[i * 5 + 4];
            if (area < minArea)
            {
                continue;
            }
            int left = stats[i * 5 + 0];
            int widthPx = stats[i * 5 + 2];
            float lh = Math.Max(1.0f, (float)(y1[i] - y0[i]));
            float ex = Math.Min(CoreXShrink * lh, 0.5f * lh);
            float x0 = left * stride - ex;
            float x1 = (left + widthPx) * stride + ex;
            boxes.Add(x0 - growX * lh);
            boxes.Add((float)y0[i] - growY * lh);
            boxes.Add(x1 + growX * lh);
            boxes.Add((float)y1[i] + growY * lh);
            boxes.Add((float)score[i]);
        }
        return boxes.ToArray();
    }
}
