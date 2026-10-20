using IndraOcr.Numerics;

namespace IndraOcr.Lm;

/// <summary>
/// lm.py: a token n-gram language model stored as a backoff table (npz, no
/// pickle).  Per history length n, sorted keys of (history, token) pairs with
/// their log-probabilities, and sorted history keys with their backoff weights.
/// A key packs tokens <see cref="Bits"/> bits each (0 = line start, t + 1 =
/// token t).
/// </summary>
public sealed class NgramLm
{
    public const int DefaultBits = 11;
    private const int TagShift = 61;
    private const int Bos = -1;

    private readonly ulong[][] _k;
    private readonly double[][] _v;
    private readonly ulong[][] _hk;
    private readonly double[][] _hb;
    private readonly Dictionary<(ulong history, int token), double> _cache = new();
    private readonly double _maxLogp;

    private NgramLm(int order, int vocabSize, int bits,
        ulong[][] k, double[][] v, ulong[][] hk, double[][] hb)
    {
        Order = order;
        VocabSize = vocabSize;
        Bits = bits;
        _k = k;
        _v = v;
        _hk = hk;
        _hb = hb;
        double top = -Math.Log(vocabSize);
        foreach (var vals in v)
        {
            if (vals.Length > 0)
            {
                top = Math.Max(top, vals.Max());
            }
        }
        double backoff = 0;
        foreach (var h in hb)
        {
            if (h.Length > 0)
            {
                backoff += Math.Max(0.0, h.Max());
            }
        }
        _maxLogp = top + backoff;
    }

    public int Order { get; }
    public int VocabSize { get; }
    public int Bits { get; }

    /// <summary>An upper bound of every value <see cref="Logp"/> returns.</summary>
    public double MaxLogp => _maxLogp;

    public static NgramLm Load(string path)
    {
        var z = Npz.ReadAll(path);
        int order = (int)z["order"].AsInt64()[0];
        int vocab = (int)z["vocab_size"].AsInt64()[0];
        int bits = z.TryGetValue("bits", out var bitsArr) ? (int)bitsArr.AsInt64()[0] : DefaultBits;
        if (order * bits > TagShift)
        {
            throw new ArgumentException($"order {order} x {bits} bits do not fit a key");
        }
        var k = new ulong[order][];
        var v = new double[order][];
        var hk = new ulong[order][];
        var hb = new double[order][];
        for (int n = 0; n < order; n++)
        {
            k[n] = z[$"k{n}"].AsUInt64();
            v[n] = z[$"v{n}"].AsFloat64();
            hk[n] = z[$"hk{n}"].AsUInt64();
            hb[n] = z[$"hb{n}"].AsFloat64();
        }
        return new NgramLm(order, vocab, bits, k, v, hk, hb);
    }

    /// <summary>lm._pack: tokens (0 = line start) packed into one key.</summary>
    private ulong Pack(ReadOnlySpan<int> tokens)
    {
        ulong key = 0;
        foreach (int t in tokens)
        {
            key = (key << Bits) | (uint)(t == Bos ? 0 : t + 1);
        }
        return key;
    }

    private static int Find(ulong[] keys, ulong key)
    {
        int i = Array.BinarySearch(keys, key);
        return i >= 0 ? i : -1;
    }

    /// <summary>
    /// log P(token | history), with backoff.  History tokens are recognizer class
    /// ids (the same ids the model emits); an empty history is the line start.
    /// </summary>
    public double Logp(ReadOnlySpan<int> history, int token)
    {
        // lm.logp_keys semantics: the key is the last order - 1 tokens packed,
        // with missing high bits left at zero (a line start).  A short history can
        // therefore match a longer one padded with BOS, which is what the beam
        // search sees; the scalar lm.logp would stop at the real length.
        int top = Order - 1;
        ulong cacheKey = Pack(Tail(history, top));
        if (_cache.TryGetValue((cacheKey, token), out double cached))
        {
            return cached;
        }
        int m = 0; // the longest history the model has seen
        for (int n = 1; n <= top; n++)
        {
            if (Find(_hk[n], Pack(Tail(history, n))) < 0)
            {
                break;
            }
            m = n;
        }
        double acc = 0;
        double result = double.NaN;
        for (int n = m; n >= 0; n--)
        {
            var hist = Tail(history, n);
            int i = Find(_k[n], PackWith(hist, token));
            if (i >= 0)
            {
                result = acc + _v[n][i];
                break;
            }
            int j = Find(_hk[n], Pack(hist));
            if (j < 0)
            {
                break;
            }
            acc += _hb[n][j];
        }
        if (double.IsNaN(result))
        {
            result = acc - Math.Log(VocabSize);
        }
        if (_cache.Count < 1_000_000)
        {
            _cache[(cacheKey, token)] = result;
        }
        return result;
    }

    private static ReadOnlySpan<int> Tail(ReadOnlySpan<int> history, int n)
        => n <= 0 ? ReadOnlySpan<int>.Empty : history[Math.Max(0, history.Length - n)..];

    private ulong PackWith(ReadOnlySpan<int> history, int token)
    {
        ulong key = 0;
        foreach (int t in history)
        {
            key = (key << Bits) | (uint)(t == Bos ? 0 : t + 1);
        }
        return (key << Bits) | (uint)(token + 1);
    }
}
