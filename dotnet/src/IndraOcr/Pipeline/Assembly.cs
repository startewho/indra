using System.Text.RegularExpressions;
using IndraOcr.Numerics;

namespace IndraOcr.Pipeline;

/// <summary>The parts of pipeline.py that turn recognized lines into a page.</summary>
public static class Assembly
{
    public const float SplitGap = 3.0f;
    public const float CellGap = 1.8f;
    public const float GutterGap = 1.6f;
    public const int GutterEdges = 3;

    private static readonly HashSet<char> LeaderChars = new(".·…‥_-–—=");
    private static readonly HashSet<char> Dashes = new("-–—");
    private static readonly HashSet<char> Bullets = new("•◦▪▫●○■□‣⁃∙*");
    private static readonly Regex Enumerator = new(
        @"\A(?:(?:\d{1,3}|[A-Za-z])[.)]|\((?:\d{1,3}|[A-Za-z]|[ivxlcIVXLC]{1,5})\))\z",
        RegexOptions.CultureInvariant);

    public static bool IsSpace(Token t, int spaceId) => t.Char is null ? t.Id == spaceId : t.Char == " ";

    /// <summary>Text of tokens: ids from the main recognizer, chars from a group recognizer.</summary>
    public static string Decode(IReadOnlyList<Token> toks, Func<IReadOnlyList<int>, string> decodeIds, bool pageRtl)
    {
        if (toks.Count > 0 && toks[0].Char is not null)
        {
            var raw = string.Concat(toks.Select(t => t.Char));
            var text = string.Join(" ", raw.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));
            if (toks[0].Seq < 0)
            {
                text = text.Normalize(System.Text.NormalizationForm.FormC); // jamo -> syllables
                if (Rtl.Bidi.HasRtl(text))
                {
                    text = Rtl.Bidi.Flip(text, baseRtl: pageRtl ? true : null);
                }
            }
            return text;
        }
        return decodeIds(toks.Select(t => t.Id).ToList());
    }

    /// <summary>
    /// Cut a line where two neighbouring characters are more than <paramref name="gap"/>
    /// line heights apart (the detector merged two cells or columns).
    /// </summary>
    public static List<(float[] Box, string Text, Token[] Tokens)> SplitLine(
        float[] box, IReadOnlyList<Token> toks, float gap, IReadOnlyList<float>? at, int spaceId,
        Func<IReadOnlyList<int>, string> decodeIds, bool pageRtl)
    {
        float h = Math.Max(1.0f, box[3] - box[1]);
        var segs = new List<List<Token>> { new() };
        float? prev = null;
        var cuts = (at ?? Array.Empty<float>()).OrderBy(v => v).ToArray();
        foreach (var tk in toks)
        {
            float x = tk.X;
            if (!IsSpace(tk, spaceId))
            {
                bool cut = prev is not null
                           && (x - prev.Value > gap * h || cuts.Any(c => prev.Value < c && c < x));
                if (cut)
                {
                    segs.Add(new List<Token>());
                }
                prev = x;
            }
            segs[^1].Add(tk);
        }
        if (segs.Count == 1)
        {
            return new List<(float[], string, Token[])>
            {
                ((float[])box.Clone(), Decode(toks, decodeIds, pageRtl), toks.ToArray()),
            };
        }
        var outv = new List<(float[], string, Token[])>();
        foreach (var sg in segs)
        {
            string text = Decode(sg, decodeIds, pageRtl).Trim();
            var xs = sg.Where(t => !IsSpace(t, spaceId)).Select(t => t.X).ToArray();
            if (text.Length > 0 && xs.Length > 0)
            {
                var bx = new[]
                {
                    Math.Max(box[0], xs.Min() - 0.35f * h),
                    box[1],
                    Math.Min(box[2], xs.Max() + 0.35f * h),
                    box[3],
                };
                outv.Add((bx, text, sg.ToArray()));
            }
        }
        return outv;
    }

    /// <summary>
    /// Per line, the x positions of gaps that sit on a column gutter: a gap over
    /// GUTTER_GAP line heights where at least GUTTER_EDGES other lines end at its
    /// left side or start at its right side.
    /// </summary>
    public static List<List<float>> GutterCuts(
        float[] boxes, int n, IReadOnlyList<RecLine> rec, int spaceId)
    {
        var cuts = new List<List<float>>(n);
        for (int i = 0; i < n; i++)
        {
            float h = Math.Max(1.0f, boxes[i * 4 + 3] - boxes[i * 4 + 1]);
            var xs = rec[i].Tokens.Where(t => !IsSpace(t, spaceId)).Select(t => t.X).ToArray();
            var mine = new List<float>();
            for (int k = 0; k + 1 < xs.Length; k++)
            {
                float a = xs[k], c = xs[k + 1];
                if (c - a <= GutterGap * h)
                {
                    continue;
                }
                float gl = a + 0.35f * h, gr = c - 0.35f * h;
                int near = 0;
                for (int j = 0; j < n; j++)
                {
                    if (j == i)
                    {
                        continue;
                    }
                    float left = boxes[j * 4], right = boxes[j * 4 + 2];
                    if (Math.Abs(right - gl) < 0.5f * h || Math.Abs(left - gr) < 0.5f * h)
                    {
                        near++;
                    }
                }
                if (near >= GutterEdges)
                {
                    mine.Add((a + c) / 2);
                }
            }
            cuts.Add(mine);
        }
        return cuts;
    }

    /// <summary>
    /// Two kinds of lines carry no text of their own: a line of leader characters
    /// only is dropped; a lone list marker joins the line that starts right after
    /// it on the same row.
    /// </summary>
    public static (float[] Boxes, List<RecLine> Rec) TidyLines(
        float[] boxes, int n, List<RecLine> rec, bool rtl, int spaceId)
    {
        var keep = Enumerable.Repeat(true, n).ToArray();
        var h = new float[n];
        var w = new float[n];
        for (int i = 0; i < n; i++)
        {
            h[i] = boxes[i * 4 + 3] - boxes[i * 4 + 1];
            w[i] = boxes[i * 4 + 2] - boxes[i * 4 + 0];
        }
        for (int i = 0; i < n; i++)
        {
            string core = rec[i].Text.Replace(" ", "");
            if (core.Length > 0 && core.All(LeaderChars.Contains))
            {
                bool wide = w[i] >= 3 * h[i];
                if (core.Length >= 3 || wide || !core.Any(Dashes.Contains))
                {
                    keep[i] = false;
                }
            }
        }
        for (int i = 0; i < n; i++)
        {
            string t = rec[i].Text.Trim();
            if (!keep[i] || !((t.Length == 1 && Bullets.Contains(t[0])) || Enumerator.IsMatch(t)))
            {
                continue;
            }
            int best = -1;
            double bestGap = 0;
            for (int j = 0; j < n; j++)
            {
                if (j == i || !keep[j] || rec[j].Text.Trim().Length == 0)
                {
                    continue;
                }
                float ov = Math.Min(boxes[i * 4 + 3], boxes[j * 4 + 3]) - Math.Max(boxes[i * 4 + 1], boxes[j * 4 + 1]);
                if (ov < 0.5 * Math.Min(h[i], h[j]))
                {
                    continue;
                }
                float gap = rtl
                    ? boxes[i * 4 + 0] - boxes[j * 4 + 2]
                    : boxes[j * 4 + 0] - boxes[i * 4 + 2];
                if (-0.3 * h[i] <= gap && gap <= 2.5 * Math.Max(h[i], h[j])
                    && (best < 0 || gap < bestGap))
                {
                    best = j;
                    bestGap = gap;
                }
            }
            if (best < 0)
            {
                continue;
            }
            var (tj, cj, kj) = (rec[best].Text, rec[best].Conf, rec[best].Tokens);
            bool chars = kj.Length > 0 && kj[0].Char is not null;
            object? sp = chars ? " " : (spaceId >= 0 ? spaceId : null);
            var newTokens = kj;
            if (!rtl && rec[i].Tokens.Length > 0 && kj.Length > 0 && sp is not null)
            {
                bool sameKind = (rec[i].Tokens[0].Char is not null) == chars;
                if (sameKind)
                {
                    float x = (boxes[i * 4 + 2] + boxes[best * 4 + 0]) / 2;
                    int seq = rec[i].Tokens[^1].Seq;
                    var joined = new List<Token>(rec[i].Tokens)
                    {
                        sp is string s
                            ? Token.FromChar(s, x, seq)
                            : new Token((int)sp, null, x, seq),
                    };
                    joined.AddRange(kj);
                    newTokens = joined.ToArray();
                }
            }
            rec[best] = new RecLine($"{t} {tj.Trim()}", cj, newTokens);
            boxes[best * 4 + 0] = Math.Min(boxes[i * 4 + 0], boxes[best * 4 + 0]);
            boxes[best * 4 + 1] = Math.Min(boxes[i * 4 + 1], boxes[best * 4 + 1]);
            boxes[best * 4 + 2] = Math.Max(boxes[i * 4 + 2], boxes[best * 4 + 2]);
            boxes[best * 4 + 3] = Math.Max(boxes[i * 4 + 3], boxes[best * 4 + 3]);
            keep[i] = false;
        }
        var idx = Enumerable.Range(0, n).Where(i => keep[i]).ToArray();
        var outBoxes = new float[idx.Length * 4];
        for (int i = 0; i < idx.Length; i++)
        {
            Array.Copy(boxes, idx[i] * 4, outBoxes, i * 4, 4);
        }
        return (outBoxes, idx.Select(i => rec[i]).ToList());
    }

    /// <summary>Boxes mirrored left to right (reading order on an RTL page).</summary>
    public static float[] Geo(float[] boxes, int n, bool rtl, float width)
    {
        if (!rtl)
        {
            return boxes;
        }
        var m = (float[])boxes.Clone();
        for (int i = 0; i < n; i++)
        {
            m[i * 4 + 0] = width - boxes[i * 4 + 2];
            m[i * 4 + 2] = width - boxes[i * 4 + 0];
        }
        return m;
    }
}
