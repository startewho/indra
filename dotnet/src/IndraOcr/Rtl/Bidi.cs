using System.Text;
using IndraOcr.Numerics;

namespace IndraOcr.Rtl;

/// <summary>
/// rtl.py: right-to-left text for a recognizer that reads every line left to
/// right.  A CTC recognizer emits characters in the order it sees them, so a
/// line with Arabic or Hebrew is stored in visual order; <see cref="Flip"/>
/// turns logical order into visual order and back.
///
/// This is a small subset of the Unicode bidi algorithm (UAX #9): no embeddings
/// or isolates, neutrals resolved by runs.  It is its own inverse.
/// </summary>
public static class Bidi
{
    private static readonly Dictionary<char, char> Mirror = new()
    {
        ['('] = ')', [')'] = '(', ['['] = ']', [']'] = '[', ['{'] = '}', ['}'] = '{',
        ['<'] = '>', ['>'] = '<', ['«'] = '»', ['»'] = '«', ['‹'] = '›', ['›'] = '‹',
    };

    private const char Tatweel = '\u0640';
    private static readonly (int Start, int End)[] Presentation = { (0xFB1D, 0xFDFF), (0xFE70, 0xFEFF) };

    public static bool IsRtl(int cp) => Unicode.IsRtl(cp);

    public static bool HasRtl(string text) => Unicode.CodePoints(text).Any(Unicode.IsRtl);

    /// <summary>Whether the line reads right to left: more RTL letters than LTR.</summary>
    public static bool RtlBase(string text)
    {
        int rtl = 0, ltr = 0;
        foreach (int cp in Unicode.CodePoints(text))
        {
            if (Unicode.IsRtl(cp))
            {
                rtl++;
            }
            else if (Unicode.BidiClass(cp) == Unicode.Bidi.L)
            {
                ltr++;
            }
        }
        return rtl > ltr;
    }

    /// <summary>
    /// Presentation forms to base letters (NFKC on those blocks only), no
    /// tatweel, no combining mark without a base character.
    /// </summary>
    public static string Normalize(string text)
    {
        var sb = new StringBuilder(text.Length);
        foreach (int cp in Unicode.CodePoints(text))
        {
            bool presentation = Presentation.Any(r => cp >= r.Start && cp <= r.End);
            if (presentation)
            {
                sb.Append(char.ConvertFromUtf32(cp).Normalize(NormalizationForm.FormKC));
            }
            else if (cp != Tatweel
                     && (sb.Length > 0 || !IsMark(cp)))
            {
                sb.Append(char.ConvertFromUtf32(cp));
            }
        }
        return sb.ToString();
    }

    private static bool IsMark(int cp)
    {
        var cat = System.Globalization.CharUnicodeInfo.GetUnicodeCategory(cp);
        return cat is System.Globalization.UnicodeCategory.NonSpacingMark
            or System.Globalization.UnicodeCategory.EnclosingMark;
    }

    /// <summary>Base character with its combining marks (harakat, niqqud).</summary>
    public static List<string> Clusters(string text)
    {
        var outv = new List<string>();
        foreach (int cp in Unicode.CodePoints(text))
        {
            string c = char.ConvertFromUtf32(cp);
            if (outv.Count > 0 && IsMark(cp))
            {
                outv[^1] += c;
            }
            else
            {
                outv.Add(c);
            }
        }
        return outv;
    }

    private static char Kind(string cluster)
    {
        int cp = char.ConvertToUtf32(cluster, 0);
        if (Unicode.IsRtl(cp))
        {
            return 'R';
        }
        return Unicode.IsLtr(cp) ? 'L' : 'N';
    }

    /// <summary>[start, end) spans from a strong cluster to the last strong one before a stop.</summary>
    private static List<(int Start, int End)> Runs(string kinds, char strong, string stop)
    {
        var spans = new List<(int, int)>();
        int? start = null, last = null;
        for (int i = 0; i <= kinds.Length; i++)
        {
            char k = i < kinds.Length ? kinds[i] : '#';
            if (k == strong)
            {
                start ??= i;
                last = i;
            }
            else if (stop.IndexOf(k) >= 0 || k == '#')
            {
                if (start is not null)
                {
                    spans.Add((start.Value, last!.Value + 1));
                }
                start = last = null;
            }
        }
        return spans;
    }

    private static string MirrorCluster(string cluster)
    {
        if (cluster.Length == 0)
        {
            return cluster;
        }
        return Mirror.TryGetValue(cluster[0], out char m) ? m + cluster[1..] : cluster;
    }

    /// <summary>
    /// Logical order to visual order, and visual order back to logical order.
    /// <paramref name="baseRtl"/>: the paragraph direction, when known; null
    /// decides by <see cref="RtlBase"/>.
    /// </summary>
    public static string Flip(string text, bool mirror = true, bool? baseRtl = null)
    {
        if (!HasRtl(text))
        {
            return text;
        }
        var cls = Clusters(text);
        var kinds = new string(cls.Select(Kind).ToArray());
        bool rtl = baseRtl ?? RtlBase(text);
        if (rtl)
        {
            // Units: LTR runs kept whole, every other cluster alone; reverse.
            var units = new List<string>();
            var spans = Runs(kinds, 'L', "R").Select(s => (s.Start, s.End)).ToList();
            spans.Add((cls.Count, cls.Count));
            int i = 0;
            foreach (var (a, b) in spans)
            {
                for (int j = i; j < a; j++)
                {
                    units.Add(mirror ? MirrorCluster(cls[j]) : cls[j]);
                }
                if (a < b)
                {
                    units.Add(string.Concat(cls.GetRange(a, b - a)));
                }
                i = b;
            }
            units.Reverse();
            return string.Concat(units);
        }
        // LTR base: reverse each RTL run in place.
        var outv = new List<string>(cls);
        foreach (var (a, b) in Runs(kinds, 'R', "L"))
        {
            var seg = cls.GetRange(a, b - a);
            seg.Reverse();
            for (int k = 0; k < seg.Count; k++)
            {
                outv[a + k] = mirror ? MirrorCluster(seg[k]) : seg[k];
            }
        }
        return string.Concat(outv);
    }

    /// <summary>The recognizer's label for a line stored in logical order.</summary>
    public static string VisualLabel(string logical, bool? baseRtl = null)
        => Flip(Normalize(logical), baseRtl: baseRtl);
}
