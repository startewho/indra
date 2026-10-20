using System.Text;
using System.Text.RegularExpressions;
using IndraOcr.Numerics;
using IndraOcr.Ordering;

namespace IndraOcr.Layout;

/// <summary>A layout block: label, bbox in page pixels, and its text.</summary>
public sealed class Block
{
    public required string Label { get; init; }
    public required float[] Bbox { get; init; }
    public required string Text { get; init; }
}

/// <summary>layout_post.py: page text from the ordered lines and the layout heads.</summary>
public static class LayoutPost
{
    public static readonly int Header = Array.IndexOf(OrderPost.BlockClasses, "Page-Header");
    public static readonly int Footer = Array.IndexOf(OrderPost.BlockClasses, "Page-Footer");
    public static readonly int Section = Array.IndexOf(OrderPost.BlockClasses, "Section-Header");
    public static readonly int List = Array.IndexOf(OrderPost.BlockClasses, "List-Group");
    public static readonly int Table = Array.IndexOf(OrderPost.BlockClasses, "Table");

    private static readonly Regex WholeMath = new(@"^\$[^$]+\$$", RegexOptions.Compiled);
    private static readonly Regex Item = new(
        @"^\s*([•\-–▪◦*·→]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)])(\s|$)",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);

    /// <summary>Han, kana, CJK punctuation and fullwidth forms: no spaces between them.</summary>
    public static bool IsCjk(char c)
    {
        int o = c;
        return (o >= 0x3000 && o <= 0x30FF)
               || (o >= 0x3400 && o <= 0x9FFF)
               || (o >= 0xF900 && o <= 0xFAFF)
               || (o >= 0xFF00 && o <= 0xFFEF)
               || (o >= 0x20000 && o <= 0x2FA1F);
    }

    /// <summary>
    /// Join the lines of one paragraph: "tumori-" + "genesis" -> "tumorigenesis";
    /// a line break between two CJK characters takes no space.
    /// </summary>
    public static string JoinLines(IReadOnlyList<string> texts)
    {
        var sb = new StringBuilder();
        foreach (string raw in texts)
        {
            string t = raw.Trim();
            if (t.Length == 0)
            {
                continue;
            }
            string cur = sb.ToString();
            if (cur.Length > 1 && cur.EndsWith('-') && char.IsLetter(cur[^2])
                && char.IsLetter(t[0]) && char.IsLower(t[0]))
            {
                sb.Clear();
                sb.Append(cur[..^1]).Append(t);
            }
            else if (cur.Length > 0 && IsCjk(cur[^1]) && IsCjk(t[0]))
            {
                sb.Append(t);
            }
            else
            {
                if (cur.Length > 0)
                {
                    sb.Append(' ');
                }
                sb.Append(t);
            }
        }
        return sb.ToString();
    }

    /// <summary>Runs of the reading order with a positive same-block logit between neighbours.</summary>
    public static List<List<int>> SplitBlocks(IReadOnlyList<int> order, float[] same, int n)
    {
        var blocks = new List<List<int>>();
        foreach (int i in order)
        {
            if (blocks.Count > 0 && same[blocks[^1][^1] * n + i] > 0)
            {
                blocks[^1].Add(i);
            }
            else
            {
                blocks.Add(new List<int> { i });
            }
        }
        return blocks;
    }

    private static List<List<int>> Rows(float[] boxes, int n)
    {
        var cy = new float[n];
        var h = new float[n];
        for (int i = 0; i < n; i++)
        {
            cy[i] = (float)((boxes[i * 4 + 1] + boxes[i * 4 + 3]) / 2.0);
            h[i] = boxes[i * 4 + 3] - boxes[i * 4 + 1];
        }
        double med = Np.Median(h);
        var rows = new List<List<int>>();
        foreach (int i in Np.ArgSort(cy))
        {
            if (rows.Count > 0)
            {
                var last = rows[^1];
                double mean = last.Sum(k => cy[k]) / last.Count;
                if (Math.Abs(cy[i] - mean) < 0.5 * med)
                {
                    last.Add(i);
                    continue;
                }
            }
            rows.Add(new List<int> { i });
        }
        foreach (var r in rows)
        {
            r.Sort((a, b) => boxes[a * 4].CompareTo(boxes[b * 4]));
        }
        return rows;
    }

    private static List<List<int>> MergeWrapped(List<List<int>> rows, float[] boxes, int n)
    {
        if (rows.Count < 3)
        {
            return rows;
        }
        var h = new float[n];
        for (int i = 0; i < n; i++)
        {
            h[i] = boxes[i * 4 + 3] - boxes[i * 4 + 1];
        }
        double medH = Np.Median(h);
        var gaps = new double[rows.Count - 1];
        for (int i = 0; i < gaps.Length; i++)
        {
            float y0Next = rows[i + 1].Min(k => boxes[k * 4 + 1]);
            float y1Cur = rows[i].Max(k => boxes[k * 4 + 3]);
            gaps[i] = y0Next - y1Cur;
        }
        double med = Np.Median(gaps);
        var outv = new List<List<int>> { new(rows[0]) };
        var prev = rows[0];
        for (int i = 0; i < gaps.Length; i++)
        {
            var r = rows[i + 1];
            if (gaps[i] < 0.5 * med && gaps[i] < 0.6 * medH && r.Count < prev.Count)
            {
                outv[^1].AddRange(r);
            }
            else
            {
                outv.Add(new List<int>(r));
            }
            prev = r;
        }
        return outv;
    }

    /// <summary>An HTML table from cell lines (one text line per cell).</summary>
    public static string TableHtml(float[] boxes, IReadOnlyList<string> texts, int n)
    {
        var rows = Rows(boxes, n);
        float minX = float.MaxValue, maxX = float.MinValue;
        for (int i = 0; i < n; i++)
        {
            minX = Math.Min(minX, boxes[i * 4]);
            maxX = Math.Max(maxX, boxes[i * 4 + 2]);
        }
        float spanX = maxX - minX;
        if (rows.Count > 2 && rows[0].Count == 1 && rows.Max(r => r.Count) > 1
            && boxes[rows[0][0] * 4 + 2] - boxes[rows[0][0] * 4] > 0.6 * spanX)
        {
            var keep = Enumerable.Range(0, n).Where(k => k != rows[0][0]).ToArray();
            var keepBoxes = new float[keep.Length * 4];
            for (int i = 0; i < keep.Length; i++)
            {
                Array.Copy(boxes.ToArray(), keep[i] * 4, keepBoxes, i * 4, 4);
            }
            return texts[rows[0][0]] + "\n\n" + TableHtml(keepBoxes, keep.Select(k => texts[k]).ToList(), keep.Length);
        }
        int width = rows.Max(r => r.Count);
        var full = rows.Where(r => r.Count == width).ToList();
        rows = MergeWrapped(rows, boxes, n);
        var bands = new float[width, 2];
        for (int c = 0; c < width; c++)
        {
            bands[c, 0] = (float)Np.Median(full.Select(r => boxes[r[c] * 4]).ToArray());
            bands[c, 1] = (float)Np.Median(full.Select(r => boxes[r[c] * 4 + 2]).ToArray());
        }
        var edges = new float[width + 1];
        edges[0] = float.NegativeInfinity;
        for (int c = 0; c < width - 1; c++)
        {
            edges[c + 1] = (bands[c, 1] + bands[c + 1, 0]) / 2;
        }
        edges[width] = float.PositiveInfinity;

        var sb = new StringBuilder("<table>");
        for (int ri = 0; ri < rows.Count; ri++)
        {
            string tag = ri == 0 && rows.Count > 1 ? "th" : "td";
            var cells = new SortedDictionary<int, (int Span, string Text)>();
            foreach (int k in rows[ri].OrderBy(k => boxes[k * 4 + 1]).ThenBy(k => boxes[k * 4]))
            {
                float x0 = boxes[k * 4], x1 = boxes[k * 4 + 2];
                int first = Np.SearchSorted(edges, x0, right: true) - 1;
                int last = Np.SearchSorted(edges, x1, right: true) - 1;
                var span = new List<int>();
                for (int c = Math.Max(first, 0); c <= Math.Min(last, width - 1); c++)
                {
                    if (Math.Min(x1, bands[c, 1]) - Math.Max(x0, bands[c, 0]) > 0.5 * (bands[c, 1] - bands[c, 0]))
                    {
                        span.Add(c);
                    }
                }
                if (span.Count == 0)
                {
                    span.Add(Math.Min(Math.Max(first, 0), width - 1));
                }
                int c0 = span[0];
                string text = cells.TryGetValue(c0, out var prev) ? prev.Text + " " + texts[k] : texts[k];
                cells[c0] = (prev.Text is null ? span.Count : Math.Max(prev.Span, span.Count), text);
            }
            sb.Append("\n<tr>");
            int col = 0;
            while (col < width)
            {
                if (cells.TryGetValue(col, out var cell))
                {
                    string attr = cell.Span > 1 ? $" colspan=\"{cell.Span}\"" : "";
                    sb.Append($"<{tag}{attr}>{Escape(cell.Text)}</{tag}>");
                    col += cell.Span;
                }
                else
                {
                    sb.Append($"<{tag}></{tag}>");
                    col++;
                }
            }
            sb.Append("</tr>");
        }
        sb.Append("\n</table>");
        return sb.ToString();
    }

    /// <summary>html.escape(text, quote=False): only &amp;, &lt; and &gt;.</summary>
    public static string Escape(string text)
        => text.Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;");

    private static float[][] CellLogits(float[] tab, int n, List<List<int>> cells, int which)
    {
        int m = cells.Count;
        var outv = new float[m][];
        for (int i = 0; i < m; i++)
        {
            outv[i] = new float[m];
        }
        int offset = which * n * n;
        for (int a = 0; a < m; a++)
        {
            for (int b = a + 1; b < m; b++)
            {
                double sum = 0;
                int count = 0;
                foreach (int i in cells[a])
                {
                    foreach (int j in cells[b])
                    {
                        sum += tab[offset + i * n + j];
                        count++;
                    }
                }
                float mean = (float)(sum / count);
                outv[a][b] = mean;
                outv[b][a] = mean;
            }
        }
        return outv;
    }

    /// <summary>
    /// The lines of one cell are stacked (they overlap horizontally).  When the
    /// cell head joins side-by-side lines in more than 10% of its cells, it has
    /// failed and the geometric assembly is used instead.
    /// </summary>
    public static bool HeadsPlausible(float[] boxes, int n, float[] cellLogits)
    {
        var cells = OrderPost.Cluster(cellLogits, n);
        int bad = 0;
        foreach (var c in cells)
        {
            if (c.Count < 2)
            {
                continue;
            }
            for (int i = 0; i < c.Count && bad >= 0; i++)
            {
                for (int j = i + 1; j < c.Count; j++)
                {
                    float overlap = Math.Min(boxes[c[i] * 4 + 2], boxes[c[j] * 4 + 2])
                                    - Math.Max(boxes[c[i] * 4], boxes[c[j] * 4]);
                    if (overlap < 0)
                    {
                        bad++;
                        i = c.Count;
                        break;
                    }
                }
            }
        }
        return bad <= 0.1 * cells.Count;
    }

    /// <summary>An HTML table from the OrderNet table heads.</summary>
    public static string TableHtmlHeads(float[] boxes, IReadOnlyList<string> texts, float[] tab, int n)
    {
        var cells = OrderPost.Cluster(tab[(2 * n * n)..], n);
        int m = cells.Count;
        var cb = new float[m, 4];
        for (int i = 0; i < m; i++)
        {
            cb[i, 0] = cells[i].Min(k => boxes[k * 4]);
            cb[i, 1] = cells[i].Min(k => boxes[k * 4 + 1]);
            cb[i, 2] = cells[i].Max(k => boxes[k * 4 + 2]);
            cb[i, 3] = cells[i].Max(k => boxes[k * 4 + 3]);
        }
        var ctext = new string[m];
        for (int i = 0; i < m; i++)
        {
            ctext[i] = string.Join(" ", cells[i]
                .OrderBy(k => boxes[k * 4 + 1]).ThenBy(k => boxes[k * 4])
                .Select(k => texts[k]));
        }
        var rowLogits = CellLogits(tab, n, cells, 0);
        var colLogits = CellLogits(tab, n, cells, 1);
        var rows = OrderPost.Cluster(Flatten(rowLogits), m)
            .OrderBy(g => g.Min(c => cb[c, 1])).ToList();
        var cols = OrderPost.Cluster(Flatten(colLogits), m)
            .OrderBy(g => g.Min(c => cb[c, 0])).ToList();
        var rowOf = new int[m];
        var colOf = new int[m];
        for (int r = 0; r < rows.Count; r++)
        {
            foreach (int c in rows[r]) rowOf[c] = r;
        }
        for (int c = 0; c < cols.Count; c++)
        {
            foreach (int k in cols[c]) colOf[k] = c;
        }
        var centers = new double[cols.Count];
        for (int c = 0; c < cols.Count; c++)
        {
            centers[c] = Np.Median(cols[c].Select(k => (cb[k, 0] + cb[k, 2]) / 2).ToArray());
        }
        var hArr = new float[n];
        for (int i = 0; i < n; i++) hArr[i] = boxes[i * 4 + 3] - boxes[i * 4 + 1];
        float h = (float)Np.Median(hArr);

        var grid = new Dictionary<(int Row, int Col), (int Span, string Text)>();
        for (int c = 0; c < m; c++)
        {
            int c0 = colOf[c];
            int span = 1;
            while (c0 + span < cols.Count && cb[c, 0] - h <= centers[c0 + span] && centers[c0 + span] <= cb[c, 2] + h)
            {
                span++;
            }
            var key = (rowOf[c], c0);
            if (grid.TryGetValue(key, out var prev))
            {
                grid[key] = (Math.Max(prev.Span, span), prev.Text + " " + ctext[c]);
            }
            else
            {
                grid[key] = (span, ctext[c]);
            }
        }
        var sb = new StringBuilder("<table>");
        for (int r = 0; r < rows.Count; r++)
        {
            string tag = r == 0 && rows.Count > 1 ? "th" : "td";
            sb.Append("\n<tr>");
            int col = 0;
            while (col < cols.Count)
            {
                if (grid.TryGetValue((r, col), out var cell))
                {
                    string attr = cell.Span > 1 ? $" colspan=\"{cell.Span}\"" : "";
                    sb.Append($"<{tag}{attr}>{Escape(cell.Text)}</{tag}>");
                    col += cell.Span;
                }
                else
                {
                    sb.Append($"<{tag}></{tag}>");
                    col++;
                }
            }
            sb.Append("</tr>");
        }
        sb.Append("\n</table>");
        return sb.ToString();
    }

    private static float[] Flatten(float[][] a)
    {
        int n = a.Length;
        var outv = new float[n * n];
        for (int i = 0; i < n; i++)
        {
            Array.Copy(a[i], 0, outv, i * n, n);
        }
        return outv;
    }

    /// <summary>The layout blocks behind <see cref="PageMarkdown"/>, in reading order.</summary>
    public static List<Block> PageBlocks(
        float[] boxes, int n, IReadOnlyList<string> texts, IReadOnlyList<int> order,
        float[] clsLogits, int nClasses, float[] same)
    {
        var outv = new List<Block>();
        foreach (var blk in SplitBlocks(order, same, n))
        {
            var sums = new double[nClasses];
            foreach (int i in blk)
            {
                for (int c = 0; c < nClasses; c++)
                {
                    sums[c] += clsLogits[i * nClasses + c];
                }
            }
            int best = 0;
            for (int c = 1; c < nClasses; c++)
            {
                if (sums[c] > sums[best]) best = c;
            }
            float x0 = blk.Min(i => boxes[i * 4]);
            float y0 = blk.Min(i => boxes[i * 4 + 1]);
            float x1 = blk.Max(i => boxes[i * 4 + 2]);
            float y1 = blk.Max(i => boxes[i * 4 + 3]);
            outv.Add(new Block
            {
                Label = OrderPost.BlockClasses[best],
                Bbox = new[] { Round1(x0), Round1(y0), Round1(x1), Round1(y1) },
                Text = string.Join("\n", blk.Select(i => texts[i])),
            });
        }
        return outv;
    }

    private static float Round1(float v) => (float)Math.Round(v, 1);

    /// <summary>Markdown for one page.</summary>
    public static string PageMarkdown(
        float[] boxes, int n, IReadOnlyList<string> texts, IReadOnlyList<int> order,
        float[] clsLogits, int nClasses, float[] same,
        bool keepFurniture = false,
        Dictionary<int, List<(float[] Box, string Text)>>? cells = null,
        float[]? tab = null,
        float[]? pictures = null, int pictureCount = 0)
    {
        var parts = new List<string>();
        var partBoxes = new List<(float X0, float Y0, float X1, float Y1)>();
        foreach (var blk in SplitBlocks(order, same, n))
        {
            int before = parts.Count;
            BlockPart(parts, blk, boxes, n, texts, clsLogits, nClasses, keepFurniture, cells, tab);
            if (parts.Count > before)
            {
                partBoxes.Add((
                    blk.Min(i => boxes[i * 4]),
                    blk.Min(i => boxes[i * 4 + 1]),
                    blk.Max(i => boxes[i * 4 + 2]),
                    blk.Max(i => boxes[i * 4 + 3])));
            }
        }
        if (pictures is not null && pictureCount > 0)
        {
            var slots = new Dictionary<int, List<string>>();
            for (int k = 0; k < pictureCount; k++)
            {
                float x0 = pictures[k * 4], y0 = pictures[k * 4 + 1];
                float x1 = pictures[k * 4 + 2], y1 = pictures[k * 4 + 3];
                float cy = (y0 + y1) / 2;
                int at = -1;
                for (int j = 0; j < partBoxes.Count; j++)
                {
                    var b = partBoxes[j];
                    if ((b.Y0 + b.Y1) / 2 > cy && Math.Min(x1, b.X1) > Math.Max(x0, b.X0))
                    {
                        at = j;
                        break;
                    }
                }
                if (at < 0) at = parts.Count;
                if (!slots.TryGetValue(at, out var list))
                {
                    slots[at] = list = new List<string>();
                }
                list.Add($"![](picture-{k})");
            }
            var merged = new List<string>();
            for (int j = 0; j <= parts.Count; j++)
            {
                if (slots.TryGetValue(j, out var list))
                {
                    merged.AddRange(list);
                }
                if (j < parts.Count)
                {
                    merged.Add(parts[j]);
                }
            }
            parts = merged;
        }
        return string.Join("\n\n", parts.Where(p => p.Length > 0));
    }

    private static void BlockPart(
        List<string> parts, List<int> blk, float[] boxes, int n,
        IReadOnlyList<string> texts, float[] clsLogits, int nClasses,
        bool keepFurniture,
        Dictionary<int, List<(float[] Box, string Text)>>? cells,
        float[]? tab)
    {
        var sums = new double[nClasses];
        foreach (int i in blk)
        {
            for (int c = 0; c < nClasses; c++)
            {
                sums[c] += clsLogits[i * nClasses + c];
            }
        }
        int cls = 0;
        for (int c = 1; c < nClasses; c++)
        {
            if (sums[c] > sums[cls]) cls = c;
        }
        var textsB = blk.Select(i => texts[i]).ToList();
        if ((cls == Header || cls == Footer) && !keepFurniture)
        {
            return;
        }
        if (textsB.All(t => WholeMath.IsMatch(t.Trim()))) // display equations
        {
            parts.Add(string.Join("\n", textsB.Select(t => "$$" + t.Trim()[1..^1] + "$$")));
            return;
        }
        if (cls == Section)
        {
            parts.Add("## " + JoinLines(textsB));
        }
        else if (cls == Table && blk.Count > 1 && tab is not null
                 && HeadsPlausible(SubBoxes(boxes, blk), blk.Count, SubTab(tab, blk, n, 2)))
        {
            parts.Add(TableHtmlHeads(SubBoxes(boxes, blk), textsB, SubTabAll(tab, blk, n), blk.Count));
        }
        else if (cls == Table && blk.Count > 1)
        {
            var cb = new List<float[]>();
            var ct = new List<string>();
            foreach (int i in blk)
            {
                if (cells is not null && cells.TryGetValue(i, out var list))
                {
                    foreach (var (bx, tx) in list)
                    {
                        cb.Add(bx);
                        ct.Add(tx);
                    }
                }
                else
                {
                    cb.Add(new[] { boxes[i * 4], boxes[i * 4 + 1], boxes[i * 4 + 2], boxes[i * 4 + 3] });
                    ct.Add(texts[i]);
                }
            }
            var flat = new float[cb.Count * 4];
            for (int i = 0; i < cb.Count; i++)
            {
                Array.Copy(cb[i], 0, flat, i * 4, 4);
            }
            parts.Add(TableHtml(flat, ct, cb.Count));
        }
        else if (cls == List)
        {
            var items = new List<List<string>>();
            foreach (string t in textsB)
            {
                if (items.Count == 0 || Item.IsMatch(t))
                {
                    items.Add(new List<string> { t });
                }
                else
                {
                    items[^1].Add(t);
                }
            }
            parts.Add(string.Join("\n", items.Select(JoinLines)));
        }
        else
        {
            parts.Add(JoinLines(textsB));
        }
    }

    private static float[] SubBoxes(float[] boxes, List<int> blk)
    {
        var outv = new float[blk.Count * 4];
        for (int i = 0; i < blk.Count; i++)
        {
            Array.Copy(boxes.ToArray(), blk[i] * 4, outv, i * 4, 4);
        }
        return outv;
    }

    private static float[] SubTab(float[] tab, List<int> blk, int n, int which)
    {
        int m = blk.Count;
        var outv = new float[m * m];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < m; j++)
            {
                outv[i * m + j] = tab[which * n * n + blk[i] * n + blk[j]];
            }
        }
        return outv;
    }

    /// <summary>The [3, m, m] table heads of one block, flattened.</summary>
    private static float[] SubTabAll(float[] tab, List<int> blk, int n)
    {
        int m = blk.Count;
        var outv = new float[3 * m * m];
        for (int which = 0; which < 3; which++)
        {
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    outv[which * m * m + i * m + j] = tab[which * n * n + blk[i] * n + blk[j]];
                }
            }
        }
        return outv;
    }
}
