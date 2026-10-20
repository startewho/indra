namespace IndraOcr.Numerics;

/// <summary>
/// The small slice of numpy the port needs, with numpy's semantics spelled out.
/// Keeping these in one place makes the parity tests meaningful: every helper
/// has a Python counterpart with the same name in the original module.
/// </summary>
public static class Np
{
    public static double Median(ReadOnlySpan<float> a)
    {
        if (a.Length == 0)
        {
            throw new ArgumentException("median of an empty array");
        }
        var copy = a.ToArray();
        Array.Sort(copy);
        int n = copy.Length;
        return n % 2 == 1 ? copy[n / 2] : (copy[n / 2 - 1] + (double)copy[n / 2]) / 2.0;
    }

    public static double Median(ReadOnlySpan<double> a)
    {
        if (a.Length == 0)
        {
            throw new ArgumentException("median of an empty array");
        }
        var copy = a.ToArray();
        Array.Sort(copy);
        int n = copy.Length;
        return n % 2 == 1 ? copy[n / 2] : (copy[n / 2 - 1] + copy[n / 2]) / 2.0;
    }

    /// <summary>np.median(x) &gt; t, without copying when it can be avoided.</summary>
    public static int ArgMax(ReadOnlySpan<float> a)
    {
        if (a.Length == 0)
        {
            return -1;
        }
        int best = 0;
        for (int i = 1; i < a.Length; i++)
        {
            if (a[i] > a[best])
            {
                best = i;
            }
        }
        return best;
    }

    public static int ArgMin(ReadOnlySpan<float> a)
    {
        if (a.Length == 0)
        {
            return -1;
        }
        int best = 0;
        for (int i = 1; i < a.Length; i++)
        {
            if (a[i] < a[best])
            {
                best = i;
            }
        }
        return best;
    }

    /// <summary>
    /// np.argsort(a) ascending.  numpy's default is an unstable introsort;
    /// <see cref="Array.Sort{TKey,TValue}(TKey[],TValue[])"/> is an unstable
    /// introsort too, which matches it on ties often enough that the remaining
    /// differences are caught by the golden tests rather than hidden by a sort.
    /// </summary>
    public static int[] ArgSort(ReadOnlySpan<float> a)
    {
        var keys = a.ToArray();
        var idx = new int[keys.Length];
        for (int i = 0; i < idx.Length; i++)
        {
            idx[i] = i;
        }
        Array.Sort(keys, idx);
        return idx;
    }

    /// <summary>Python's sorted(range(n), key=...) (stable), e.g. width_batches.</summary>
    public static int[] ArgSortStable(ReadOnlySpan<float> a)
    {
        var idx = new int[a.Length];
        for (int i = 0; i < idx.Length; i++)
        {
            idx[i] = i;
        }
        var keys = a.ToArray();
        Array.Sort(idx, (x, y) =>
        {
            int c = keys[x].CompareTo(keys[y]);
            return c != 0 ? c : x.CompareTo(y);
        });
        return idx;
    }

    /// <summary>np.searchsorted(sorted, v, side="left" | "right").</summary>
    public static int SearchSorted(ReadOnlySpan<float> sorted, float v, bool right)
    {
        int lo = 0, hi = sorted.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            bool before = right ? sorted[mid] <= v : sorted[mid] < v;
            if (before)
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid;
            }
        }
        return lo;
    }

    public static int SearchSorted(ReadOnlySpan<double> sorted, double v, bool right)
    {
        int lo = 0, hi = sorted.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            bool before = right ? sorted[mid] <= v : sorted[mid] < v;
            if (before)
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid;
            }
        }
        return lo;
    }

    public static int SearchSorted(ReadOnlySpan<ulong> sorted, ulong v, bool right)
    {
        int lo = 0, hi = sorted.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            bool before = right ? sorted[mid] <= v : sorted[mid] < v;
            if (before)
            {
                lo = mid + 1;
            }
            else
            {
                hi = mid;
            }
        }
        return lo;
    }

    /// <summary>np.bincount(idx, minlength=n).  Indices must be non-negative.</summary>
    public static double[] Bincount(ReadOnlySpan<int> idx, int minLength)
    {
        int n = minLength;
        foreach (int v in idx)
        {
            if (v + 1 > n)
            {
                n = v + 1;
            }
        }
        var outv = new double[n];
        foreach (int v in idx)
        {
            outv[v] += 1.0;
        }
        return outv;
    }

    /// <summary>np.bincount(idx, weights=w, minlength=n).</summary>
    public static double[] Bincount(ReadOnlySpan<int> idx, ReadOnlySpan<double> weights, int minLength)
    {
        if (idx.Length != weights.Length)
        {
            throw new ArgumentException("bincount: idx and weights differ in length");
        }
        int n = minLength;
        foreach (int v in idx)
        {
            if (v + 1 > n)
            {
                n = v + 1;
            }
        }
        var outv = new double[n];
        for (int i = 0; i < idx.Length; i++)
        {
            outv[idx[i]] += weights[i];
        }
        return outv;
    }

    /// <summary>
    /// np.percentile(a, 100*q, method="linear"): position (n-1)*q, interpolated.
    /// numpy sorts (introselect) but the value is order independent.
    /// </summary>
    public static double Percentile(ReadOnlySpan<float> a, double q)
    {
        if (a.Length == 0)
        {
            throw new ArgumentException("percentile of an empty array");
        }
        var copy = a.ToArray();
        Array.Sort(copy);
        double x = q * (copy.Length - 1);
        int i = (int)Math.Floor(x);
        double g = x - i;
        int j = Math.Min(i + 1, copy.Length - 1);
        return copy[i] + g * (copy[j] - (double)copy[i]);
    }

    /// <summary>
    /// rec_preproc._percentiles: np.percentile of a uint8 array from its
    /// histogram, with numpy's lerp ("from the nearer end").
    /// </summary>
    public static double[] PercentilesFromHistogram(ReadOnlySpan<byte> b, double[] qs)
    {
        var counts = new double[256];
        foreach (byte v in b)
        {
            counts[v] += 1.0;
        }
        var cum = CumSum(counts);
        double n = b.Length;
        var result = new double[qs.Length];
        for (int k = 0; k < qs.Length; k++)
        {
            double x = qs[k] * (n - 1);
            int i = (int)Math.Floor(x);
            double g = x - i;
            double lo = SearchSorted(cum, i, right: true);
            double hi = SearchSorted(cum, Math.Min(i + 1, n - 1), right: true);
            double d = hi - lo;
            result[k] = g >= 0.5 ? hi - d * (1 - g) : lo + d * g;
        }
        return result;
    }

    public static double[] CumSum(ReadOnlySpan<double> a)
    {
        var outv = new double[a.Length];
        double acc = 0;
        for (int i = 0; i < a.Length; i++)
        {
            acc += a[i];
            outv[i] = acc;
        }
        return outv;
    }

    /// <summary>np.logaddexp: log(exp(a) + exp(b)), stable.</summary>
    public static double LogAddExp(double a, double b)
    {
        if (double.IsNegativeInfinity(a))
        {
            return b;
        }
        if (double.IsNegativeInfinity(b))
        {
            return a;
        }
        double m = Math.Max(a, b);
        return m + Math.Log(Math.Exp(a - m) + Math.Exp(b - m));
    }

    public static float LogAddExp(float a, float b) => (float)LogAddExp((double)a, (double)b);

    public static double LogSumExp(ReadOnlySpan<double> a)
    {
        double m = double.NegativeInfinity;
        foreach (double v in a)
        {
            if (v > m)
            {
                m = v;
            }
        }
        if (double.IsNegativeInfinity(m))
        {
            return m;
        }
        double s = 0;
        foreach (double v in a)
        {
            s += Math.Exp(v - m);
        }
        return m + Math.Log(s);
    }

    /// <summary>np.diff (first order).</summary>
    public static float[] Diff(ReadOnlySpan<float> a)
    {
        if (a.Length < 2)
        {
            return Array.Empty<float>();
        }
        var outv = new float[a.Length - 1];
        for (int i = 1; i < a.Length; i++)
        {
            outv[i - 1] = a[i] - a[i - 1];
        }
        return outv;
    }

    /// <summary>np.nonzero(a)[0] for a 1-D array; predicate form.</summary>
    public static int[] NonZero(ReadOnlySpan<bool> mask)
    {
        var outv = new List<int>();
        for (int i = 0; i < mask.Length; i++)
        {
            if (mask[i])
            {
                outv.Add(i);
            }
        }
        return outv.ToArray();
    }

    public static float Clip(float v, float lo, float hi) => v < lo ? lo : v > hi ? hi : v;

    public static double Clip(double v, double lo, double hi) => v < lo ? lo : v > hi ? hi : v;

    public static float Max(ReadOnlySpan<float> a)
    {
        float m = float.NegativeInfinity;
        foreach (float v in a)
        {
            if (v > m)
            {
                m = v;
            }
        }
        return m;
    }

    public static double Sum(ReadOnlySpan<float> a)
    {
        double s = 0;
        foreach (float v in a)
        {
            s += v;
        }
        return s;
    }
}
