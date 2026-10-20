using IndraOcr.Numerics;

namespace IndraOcr.Lm;

/// <summary>
/// lm.py: CTC prefix beam search with a token n-gram language model, for lines
/// the recognizer is unsure of.  score = log P_ctc + alpha * log P_lm + beta *
/// tokens; the recognizer's own probabilities still decide.
/// </summary>
public static class BeamSearch
{
    private sealed class Beams
    {
        private readonly List<string> _order = new();
        private readonly Dictionary<string, Entry> _map = new(StringComparer.Ordinal);

        public int Count => _order.Count;

        public void Clear()
        {
            _order.Clear();
            _map.Clear();
        }

        public bool TryGet(string prefix, out Entry entry) => _map.TryGetValue(prefix, out entry);

        public Entry this[string prefix] => _map[prefix];

        /// <summary>Insert, or merge the CTC probabilities of an existing prefix.</summary>
        public void Add(string prefix, double b, double nb, double s, int[] steps)
        {
            if (_map.TryGetValue(prefix, out var old))
            {
                _map[prefix] = old with
                {
                    B = Np.LogAddExp(old.B, b),
                    Nb = Np.LogAddExp(old.Nb, nb),
                };
            }
            else
            {
                _order.Add(prefix);
                _map[prefix] = new Entry(b, nb, s, steps);
            }
        }

        public IEnumerable<KeyValuePair<string, Entry>> Items()
        {
            foreach (string p in _order)
            {
                yield return new KeyValuePair<string, Entry>(p, _map[p]);
            }
        }

        public void Fill(IEnumerable<KeyValuePair<string, Entry>> sorted)
        {
            Clear();
            foreach (var (p, e) in sorted)
            {
                _order.Add(p);
                _map[p] = e;
            }
        }
    }

    private readonly record struct Entry(double B, double Nb, double S, int[] Steps)
    {
        public double Score => Np.LogAddExp(B, Nb) + S;
    }

    /// <summary>
    /// <paramref name="blank"/>: [T] blank log-probabilities; topv / topi: [T, k]
    /// the k best classes per step (blank included) and their log-probabilities.
    /// A class under <paramref name="candFloor"/> probability at a step is not
    /// tried there.  Returns (token ids, the step where each token was first
    /// emitted).
    /// </summary>
    public static (int[] Ids, int[] Steps) Search(
        float[] blank,
        float[] topv,
        long[] topi,
        int steps,
        int k,
        NgramLm lm,
        double alpha = 0.5,
        double beta = 1.5,
        int beam = 8,
        double blankSkip = 0.999,
        double candFloor = 1e-3,
        Action<int, IReadOnlyList<(string Prefix, double Score, int[] Steps)>>? trace = null)
    {
        var beams = new Beams();
        beams.Add("", 0.0, double.NegativeInfinity, 0.0, Array.Empty<int>());
        double logSkip = Math.Log(blankSkip);
        double logFloor = Math.Log(candFloor);

        for (int t = 0; t < steps; t++)
        {
            double b0 = blank[t];
            if (b0 > logSkip)
            {
                var copy = new Beams();
                foreach (var (p, e) in beams.Items())
                {
                    copy.Add(p, Np.LogAddExp(e.B, e.Nb) + b0, double.NegativeInfinity, e.S, e.Steps);
                }
                beams = copy;
                continue;
            }

            var nxt = new Beams();
            var cands = new List<(int C, double Lp)>();
            for (int j = 0; j < k; j++)
            {
                int c = (int)topi[t * k + j];
                double lp = topv[t * k + j];
                if (c != 0 && lp >= logFloor)
                {
                    cands.Add((c, lp));
                }
            }

            // Every beam's own entries first (the blank, and a repeat that collapses).
            foreach (var (prefix, e) in beams.Items())
            {
                nxt.Add(prefix, Np.LogAddExp(e.B, e.Nb) + b0, double.NegativeInfinity, e.S, e.Steps);
                int last = prefix.Length > 0 ? prefix[^1] : -1;
                foreach (var (c, lp) in cands)
                {
                    if (c == last)
                    {
                        nxt.Add(prefix, double.NegativeInfinity, e.Nb + lp, e.S, e.Steps);
                    }
                }
            }

            double floor = double.NegativeInfinity;
            if (beams.Count >= beam)
            {
                floor = double.PositiveInfinity;
                foreach (var (_, e) in nxt.Items())
                {
                    floor = Math.Min(floor, e.Score);
                }
            }
            double reach = beta + alpha * lm.MaxLogp;

            var ext = new List<(string Prefix, int C, double NewFrom, double S, int[] Steps)>();
            foreach (var (prefix, e) in beams.Items())
            {
                double tot = Np.LogAddExp(e.B, e.Nb);
                int last = prefix.Length > 0 ? prefix[^1] : -1;
                foreach (var (c, lp) in cands)
                {
                    // after a blank, the same class is a new token
                    double newFrom = c == last ? e.B + lp : tot + lp;
                    if (double.IsNegativeInfinity(newFrom))
                    {
                        continue;
                    }
                    string q = prefix + (char)c;
                    if (nxt.TryGet(q, out _))
                    {
                        nxt.Add(q, double.NegativeInfinity, newFrom, e.S, e.Steps);
                    }
                    else if (newFrom + e.S + reach >= floor - 1e-9)
                    {
                        ext.Add((prefix, c, newFrom, e.S, e.Steps));
                    }
                }
            }

            foreach (var (prefix, c, newFrom, s, st) in ext)
            {
                var history = History(prefix, lm.Order - 1);
                double lmp = lm.Logp(history, c);
                nxt.Add(prefix + (char)c, double.NegativeInfinity, newFrom,
                    s + alpha * lmp + beta, Append(st, t));
            }

            var sorted = nxt.Items().OrderByDescending(kv => kv.Value.Score).Take(beam).ToList();
            beams = new Beams();
            beams.Fill(sorted);
            trace?.Invoke(t, sorted.Take(3).Select(kv => (kv.Key, kv.Value.Score, kv.Value.Steps)).ToList());
        }

        var best = beams.Items().OrderByDescending(kv => kv.Value.Score).First();
        return (best.Key.Select(ch => (int)ch).ToArray(), best.Value.Steps);
    }

    private static int[] History(string prefix, int max)
    {
        int n = Math.Min(prefix.Length, max);
        var hist = new int[n];
        for (int i = 0; i < n; i++)
        {
            hist[i] = prefix[prefix.Length - n + i];
        }
        return hist;
    }

    private static int[] Append(int[] a, int v)
    {
        var b = new int[a.Length + 1];
        Array.Copy(a, b, a.Length);
        b[^1] = v;
        return b;
    }
}
