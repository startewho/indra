using IndraOcr.Numerics;

namespace IndraOcr.Ordering;

/// <summary>order_post.py: line features, reading-order decoding and clustering.</summary>
public static class OrderPost
{
    public const int NGeom = 10;
    public const int NText = 6;
    public const int NFeats = NGeom + NText;
    public const string SentenceEnd = ".!?:;。！？：；．";

    public static readonly string[] BlockClasses =
    {
        "Text", "Section-Header", "Page-Header", "Page-Footer", "List-Group",
        "Table", "Caption", "Footnote", "Table-Of-Contents", "Form",
    };

    public static readonly HashSet<string> GridLabels = new() { "Table", "Form", "Table-Of-Contents" };
    public static readonly HashSet<int> GridIds =
        new(BlockClasses.Select((c, i) => (c, i)).Where(t => GridLabels.Contains(t.c)).Select(t => t.i));

    public static int BlockClass(string label)
    {
        int i = Array.IndexOf(BlockClasses, label);
        return i >= 0 ? i : 0;
    }

    /// <summary>Length with East Asian wide characters counted twice.</summary>
    public static int DisplayLen(string text)
    {
        int n = 0;
        foreach (int cp in Unicode.CodePoints(text))
        {
            n += Unicode.IsWide(cp) ? 2 : 1;
        }
        return n;
    }

    /// <summary>[N, 4] boxes -> [N, 16] float32 features.</summary>
    public static float[] LineFeatures(
        float[] boxes, int n, float pageW, float pageH, IReadOnlyList<string>? texts)
    {
        var feats = new float[n * NFeats];
        if (n == 0)
        {
            return feats;
        }
        var w = new float[n];
        var h = new float[n];
        for (int i = 0; i < n; i++)
        {
            w[i] = Math.Max(boxes[i * 4 + 2] - boxes[i * 4 + 0], 1e-3f);
            h[i] = Math.Max(boxes[i * 4 + 3] - boxes[i * 4 + 1], 1e-3f);
        }
        double medH = Np.Median(h);
        for (int i = 0; i < n; i++)
        {
            float x0 = boxes[i * 4], y0 = boxes[i * 4 + 1], x1 = boxes[i * 4 + 2], y1 = boxes[i * 4 + 3];
            int o = i * NFeats;
            feats[o + 0] = x0 / pageW;
            feats[o + 1] = y0 / pageH;
            feats[o + 2] = x1 / pageW;
            feats[o + 3] = y1 / pageH;
            feats[o + 4] = (x0 + x1) / (2 * pageW);
            feats[o + 5] = (y0 + y1) / (2 * pageH);
            feats[o + 6] = w[i] / pageW;
            feats[o + 7] = (float)Math.Log(h[i] / medH);
            feats[o + 8] = (float)(Math.Log(w[i] / h[i]) / 4.0);
            feats[o + 9] = (float)(medH / pageH * 20.0);
        }
        if (texts is not null)
        {
            for (int i = 0; i < n; i++)
            {
                string t = texts[i].Trim();
                if (t.Length == 0)
                {
                    continue;
                }
                int o = i * NFeats + NGeom;
                feats[o + 0] = (float)(Math.Log(1.0 + DisplayLen(t)) / 5.0);
                feats[o + 1] = (float)t.Count(char.IsDigit) / t.Length;
                bool marker = t[0] is '•' or '-' or '–' or '▪' or '◦' or '*' or '·' or '→'
                              || (t.Length > 3 && t[..Math.Min(3, t.Length)].TrimEnd('.', ')').All(char.IsDigit));
                feats[o + 2] = marker ? 1f : 0f;
                feats[o + 3] = t.EndsWith('-') ? 1f : 0f;
                feats[o + 4] = SentenceEnd.Contains(t[^1]) ? 1f : 0f;
                feats[o + 5] = char.IsUpper(t[0]) ? 1f : 0f;
            }
        }
        return feats;
    }

    private static int ArgMax(ReadOnlySpan<double> a, int length)
    {
        int best = 0;
        for (int i = 1; i < length; i++)
        {
            if (a[i] > a[best])
            {
                best = i;
            }
        }
        return best;
    }

    /// <summary>
    /// Order from a pairwise logit matrix [N, N] (a[i, j] &gt; 0: i before j).
    /// Row-sum ranking, then insertion local search.
    /// </summary>
    public static int[] DecodeOrder(double[] a, int n, int sweeps = 4)
    {
        if (n <= 1)
        {
            return Enumerable.Range(0, n).ToArray();
        }
        var lp = new double[n * n];
        for (int i = 0; i < n * n; i++)
        {
            // log P(i before j) = log sigmoid(a)
            lp[i] = -Np.LogAddExp(0.0, -a[i]);
        }
        var key = new double[n];
        for (int i = 0; i < n; i++)
        {
            double row = 0, col = 0;
            for (int j = 0; j < n; j++)
            {
                row += lp[i * n + j];
                col += lp[j * n + i];
            }
            key[i] = -row + col;
        }
        var order = Np.ArgSort(key.Select(v => (float)v).ToArray());
        for (int sweep = 0; sweep < sweeps; sweep++)
        {
            bool improved = false;
            foreach (int x in order.ToArray())
            {
                int m = n - 1;
                var rest = new int[m];
                for (int i = 0, k = 0; i < n; i++)
                {
                    if (order[i] != x)
                    {
                        rest[k++] = order[i];
                    }
                }
                var gains = new double[m + 1];
                double prefix = 0;
                for (int k = 0; k <= m; k++)
                {
                    gains[k] = prefix;
                    if (k < m)
                    {
                        prefix += lp[rest[k] * n + x];
                    }
                }
                double suffix = 0;
                for (int k = m - 1; k >= 0; k--)
                {
                    suffix += lp[x * n + rest[k]];
                    gains[k] += suffix;
                }
                gains[m] += 0;
                int kBest = ArgMax(gains, m + 1);
                int cur = Array.IndexOf(order, x);
                if (gains[kBest] > gains[cur] + 1e-6)
                {
                    var next = new int[n];
                    Array.Copy(rest, 0, next, 0, kBest);
                    next[kBest] = x;
                    Array.Copy(rest, kBest, next, kBest + 1, m - kBest);
                    order = next;
                    improved = true;
                }
            }
            if (!improved)
            {
                break;
            }
        }
        return order;
    }

    /// <summary>Rows top to bottom (centres within half a line height share a row), left to right.</summary>
    public static int[] OrderTopDown(float[] boxes, int n)
    {
        if (n == 0)
        {
            return Array.Empty<int>();
        }
        var cy = new double[n];
        var h = new float[n];
        for (int i = 0; i < n; i++)
        {
            cy[i] = (boxes[i * 4 + 1] + boxes[i * 4 + 3]) / 2.0;
            h[i] = boxes[i * 4 + 3] - boxes[i * 4 + 1];
        }
        double med = Np.Median(h);
        double scale = Math.Max(med * 0.5, 1e-3);
        return Enumerable.Range(0, n)
            .OrderBy(i => Math.Round(cy[i] / scale))
            .ThenBy(i => boxes[i * 4])
            .ToArray();
    }

    private static (double Gap, int K, int[] Order) BestCut(float[] boxes, int n, int axis)
    {
        var lo = new float[n];
        var hi = new float[n];
        for (int i = 0; i < n; i++)
        {
            lo[i] = axis == 1 ? boxes[i * 4 + 1] : boxes[i * 4 + 0];
            hi[i] = axis == 1 ? boxes[i * 4 + 3] : boxes[i * 4 + 2];
        }
        var order = Enumerable.Range(0, n).OrderBy(i => lo[i]).ToArray();
        double end = hi[order[0]];
        double bestGap = 0;
        int bestK = -1;
        for (int k = 1; k < n; k++)
        {
            double gap = lo[order[k]] - end;
            if (gap > bestGap)
            {
                bestGap = gap;
                bestK = k;
            }
            end = Math.Max(end, hi[order[k]]);
        }
        return (bestGap, bestK, order);
    }

    /// <summary>Recursive XY-cut: split at the widest whitespace gap on either axis.</summary>
    public static int[] OrderXycut(float[] boxes, int n)
    {
        int[] Rec(int[] ids)
        {
            if (ids.Length <= 1)
            {
                return ids;
            }
            var b = new float[ids.Length * 4];
            for (int i = 0; i < ids.Length; i++)
            {
                Array.Copy(boxes.ToArray(), ids[i] * 4, b, i * 4, 4);
            }
            var (gy, ky, oy) = BestCut(b, ids.Length, 1);
            var (gx, kx, ox) = BestCut(b, ids.Length, 0);
            if (ky < 0 && kx < 0)
            {
                var top = OrderTopDown(b, ids.Length);
                return top.Select(i => ids[i]).ToArray();
            }
            var (k, o) = gy >= gx ? (ky, oy) : (kx, ox);
            var left = Rec(o.Take(k).Select(i => ids[i]).ToArray());
            var right = Rec(o.Skip(k).Select(i => ids[i]).ToArray());
            return left.Concat(right).ToArray();
        }
        return Rec(Enumerable.Range(0, n).ToArray());
    }

    /// <summary>Blocks in reading order by recursive XY-cut (vertical pages).</summary>
    public static List<List<int>> XycutBlocks(float[] boxes, int n, double minGap = 1.0)
    {
        var result = new List<List<int>>();
        if (n == 0)
        {
            return result;
        }
        var h = new float[n];
        for (int i = 0; i < n; i++)
        {
            h[i] = boxes[i * 4 + 3] - boxes[i * 4 + 1];
        }
        double medH = Np.Median(h);

        void Walk(int[] ids)
        {
            if (ids.Length > 1)
            {
                var b = new float[ids.Length * 4];
                for (int i = 0; i < ids.Length; i++)
                {
                    Array.Copy(boxes.ToArray(), ids[i] * 4, b, i * 4, 4);
                }
                var (gy, ky, oy) = BestCut(b, ids.Length, 1);
                var (gx, kx, ox) = BestCut(b, ids.Length, 0);
                var (gap, k, o) = gy >= gx ? (gy, ky, oy) : (gx, kx, ox);
                if (k >= 0 && gap >= minGap * medH)
                {
                    Walk(o.Take(k).Select(i => ids[i]).ToArray());
                    Walk(o.Skip(k).Select(i => ids[i]).ToArray());
                    return;
                }
            }
            var local = new float[ids.Length * 4];
            for (int i = 0; i < ids.Length; i++)
            {
                Array.Copy(boxes.ToArray(), ids[i] * 4, local, i * 4, 4);
            }
            result.Add(OrderXycut(local, ids.Length).Select(k => ids[k]).ToList());
        }
        Walk(Enumerable.Range(0, n).ToArray());
        return result;
    }

    /// <summary>
    /// Average-linkage clustering of pairwise "same group" logits: merge the two
    /// groups with the highest positive mean logit between them.
    /// </summary>
    public static List<List<int>> Cluster(float[] logits, int n)
    {
        if (n == 0)
        {
            return new List<List<int>>();
        }
        var s = new double[n * n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                s[i * n + j] = logits[i * n + j];
            }
            s[i * n + i] = 0;
        }
        var size = Enumerable.Repeat(1.0, n).ToArray();
        var alive = Enumerable.Repeat(true, n).ToArray();
        var members = Enumerable.Range(0, n).Select(i => new List<int> { i }).ToArray();
        var m = (double[])s.Clone();
        for (int i = 0; i < n; i++)
        {
            m[i * n + i] = double.NegativeInfinity;
        }
        var best = new int[n];
        var bestV = new double[n];
        for (int i = 0; i < n; i++)
        {
            best[i] = ArgMax(m.AsSpan(i * n, n), n);
            bestV[i] = m[i * n + best[i]];
        }
        for (int iter = 0; iter < n - 1; iter++)
        {
            int a = ArgMax(bestV, n);
            int b = best[a];
            if (bestV[a] <= 0)
            {
                break;
            }
            if (a > b)
            {
                (a, b) = (b, a);
            }
            for (int j = 0; j < n; j++)
            {
                s[a * n + j] += s[b * n + j];
                s[j * n + a] += s[j * n + b];
            }
            s[a * n + a] = 0;
            size[a] += size[b];
            alive[b] = false;
            members[a].AddRange(members[b]);
            var row = new double[n];
            for (int j = 0; j < n; j++)
            {
                row[j] = s[a * n + j] / (size[a] * size[j]);
                if (!alive[j])
                {
                    row[j] = double.NegativeInfinity;
                }
            }
            row[a] = double.NegativeInfinity;
            for (int j = 0; j < n; j++)
            {
                m[a * n + j] = row[j];
                m[j * n + a] = row[j];
                m[b * n + j] = double.NegativeInfinity;
                m[j * n + b] = double.NegativeInfinity;
            }
            bestV[b] = double.NegativeInfinity;
            best[a] = ArgMax(row, n);
            bestV[a] = row[best[a]];
            for (int j = 0; j < n; j++)
            {
                if (alive[j] && (best[j] == a || best[j] == b) && j != a)
                {
                    best[j] = ArgMax(m.AsSpan(j * n, n), n);
                    bestV[j] = m[j * n + best[j]];
                }
            }
            for (int j = 0; j < n; j++)
            {
                if (alive[j] && row[j] > bestV[j] && j != a)
                {
                    best[j] = a;
                    bestV[j] = row[j];
                }
            }
        }
        return members.Where((_, i) => alive[i]).Select(x => { x.Sort(); return x; }).ToList();
    }

    /// <summary>Reading order with the layout heads: blocks first, then lines inside blocks.</summary>
    public static int[] DecodeBlocksOrder(
        double[] a, int n, double[] same, float[] boxes, float[]? cls, int? nClasses)
    {
        if (n <= 1)
        {
            return Enumerable.Range(0, n).ToArray();
        }
        var sameF = new float[n * n];
        for (int i = 0; i < n * n; i++)
        {
            sameF[i] = (float)same[i];
        }
        var blocks = Cluster(sameF, n);
        var inner = new List<int[]>();
        foreach (var b in blocks)
        {
            var bx = new float[b.Count * 4];
            for (int i = 0; i < b.Count; i++)
            {
                Array.Copy(boxes.ToArray(), b[i] * 4, bx, i * 4, 4);
            }
            bool grid = false;
            if (cls is not null && nClasses is int nc)
            {
                var sums = new double[nc];
                foreach (int i in b)
                {
                    for (int c = 0; c < nc; c++)
                    {
                        sums[c] += cls[i * nc + c];
                    }
                }
                grid = GridIds.Contains(ArgMax(sums, nc));
            }
            var local = grid ? OrderTopDown(bx, b.Count) : OrderXycut(bx, b.Count);
            inner.Add(local.Select(k => b[k]).ToArray());
        }
        int m = blocks.Count;
        var g = new double[m * n];
        for (int k = 0; k < m; k++)
        {
            double w = 1.0 / blocks[k].Count;
            foreach (int i in blocks[k])
            {
                g[k * n + i] = w;
            }
        }
        var ga = new double[m * n];
        for (int r = 0; r < m; r++)
        {
            for (int k = 0; k < n; k++)
            {
                double sum = 0;
                for (int i = 0; i < n; i++)
                {
                    sum += g[r * n + i] * a[i * n + k];
                }
                ga[r * n + k] = sum;
            }
        }
        var gab = new double[m * m];
        for (int r = 0; r < m; r++)
        {
            for (int q = 0; q < m; q++)
            {
                double sum = 0;
                for (int k = 0; k < n; k++)
                {
                    sum += ga[r * n + k] * g[q * n + k];
                }
                gab[r * m + q] = sum;
            }
        }
        var ab = new double[m * m];
        for (int r = 0; r < m; r++)
        {
            for (int q = r + 1; q < m; q++)
            {
                ab[r * m + q] = gab[r * m + q];
                ab[q * m + r] = -gab[r * m + q];
            }
        }
        var order = DecodeOrder(ab, m);
        return order.SelectMany(k => inner[k]).ToArray();
    }
}
