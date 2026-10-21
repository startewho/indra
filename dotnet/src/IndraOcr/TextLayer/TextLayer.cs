using System.Text;
using IndraOcr.Pdf;

namespace IndraOcr.TextLayer;

/// <summary>One text-layer run: its box, text and characters with boxes.</summary>
public sealed class TextLine
{
    public required double[] Bbox { get; init; }
    public required string Text { get; init; }
    public List<(string Ch, double[] Box)> Chars { get; init; } = new();
    public int Quarter { get; set; }
    public int Seq { get; set; } = -1;
}

/// <summary>Statistics collected while reading a page's text (textlayer.py).</summary>
public sealed class TextStats
{
    public int Chars;
    public int Rotated;
    public int Bad;
    public int[] Quarters = new int[4];
    public int OffPage;
    public int[] Steps = new int[2];
    public int Turn;
    public int Frame;
    public bool Vertical;
    public double? VisualRtl;
    public double? NoInk;
}

/// <summary>Text-layer runs of a page and the verdict on them.</summary>
public sealed class TextLayer
{
    public required List<TextLine> Lines { get; init; }
    public required (double Width, double Height) Size { get; init; }
    public required bool Trusted { get; init; }
    public string Reason { get; init; } = "";
    public required TextStats Stats { get; init; }
    public List<TextLine> Rotated { get; init; } = new();
}

/// <summary>
/// textlayer.py: PDFium's text runs, and the checks that decide whether to
/// trust them.  A digital PDF already holds its text; when the layer is sound,
/// reading it is faster than OCR and has no recognition errors.
/// </summary>
public static class TextLayerReader
{
    public const double MaxInvisibleFrac = 0.05;
    public const double MaxBadCharFrac = 0.005;
    public const double MaxNoInkLineFrac = 0.03;
    public const double MaxVisualRtlFrac = 0.25;
    public const int MinRtlPairs = 20;
    public const string VisualRtlReason = "right-to-left text in visual order";
    public const string BrokenFontMap = "broken font map";
    public const int Dark = 160;

    public static int BadCharCount(string text)
    {
        int n = 0;
        foreach (int cp in Numerics.Unicode.CodePoints(text))
        {
            var cat = System.Globalization.CharUnicodeInfo.GetUnicodeCategory(cp);
            if (cp == 0xFFFD
                || cat is System.Globalization.UnicodeCategory.PrivateUse
                    or System.Globalization.UnicodeCategory.Surrogate
                    or System.Globalization.UnicodeCategory.OtherNotAssigned
                || (cat == System.Globalization.UnicodeCategory.Control && cp != '\t'))
            {
                n++;
            }
        }
        return n;
    }

    /// <summary>Whether the page's upright characters run top to bottom (vertical writing).</summary>
    public static bool VerticalLayout(TextStats stats)
        => stats.Steps[0] >= 30 && stats.Steps[0] > 2 * stats.Steps[1];

    /// <summary>The quarter turn holding most characters (a table set sideways), or 0.</summary>
    public static int PageTurn(TextStats stats)
    {
        int k = 1;
        int best = 1;
        for (int i = 2; i < 4; i++)
        {
            if (stats.Quarters[i] > stats.Quarters[best])
            {
                best = i;
            }
        }
        k = best;
        return stats.Quarters[k] > 0.5 * Math.Max(stats.Chars, 1) ? k : 0;
    }

    /// <summary>The character of pdfium's UTF-16 code unit u (and the next unit).</summary>
    private static (string Ch, bool UsedSecond) CodePoint(uint u, uint u2)
    {
        if (u is >= 0xD800 and <= 0xDBFF && u2 is >= 0xDC00 and <= 0xDFFF)
        {
            return (char.ConvertFromUtf32(0x10000 + (((int)u - 0xD800) << 10) + ((int)u2 - 0xDC00)), true);
        }
        if (u is >= 0xD800 and <= 0xDFFF)
        {
            return ("\uFFFD", false);
        }
        return (char.ConvertFromUtf32((int)u), false);
    }

    /// <summary>[2, 3] map from PDF page space to the rendered frame (points, top-left origin).</summary>
    private static double[,] DeviceAffine(PdfPage page)
    {
        var (W, H) = page.GetSize();
        const double S = 1000; // device units per point: keeps the integer output precise

        (double X, double Y) Dev(double x, double y)
        {
            page.PageToDevice(
                0, 0, (int)Math.Round(W * S), (int)Math.Round(H * S), 0,
                x, y, out int dx, out int dy);
            return (dx / S, dy / S);
        }

        var o = Dev(0, 0);
        var ex = Dev(1, 0);
        var ey = Dev(0, 1);
        return new[,]
        {
            { ex.X - o.X, ey.X - o.X, o.X },
            { ex.Y - o.Y, ey.Y - o.Y, o.Y },
        };
    }

    /// <summary>Quarter turns (counter-clockwise, like np.rot90) of a baseline direction.</summary>
    private static int? Quarter(double dx, double dy)
    {
        double q = Math.Atan2(dy, dx) / (Math.PI / 2); // pdfium's char angle runs clockwise
        int k = (int)Math.Round(q);
        return Math.Abs(q - k) < 0.01 ? ((k % 4) + 4) % 4 : null;
    }

    /// <summary>Rendered-frame points in the frame of the page turned upright.</summary>
    private static (double X, double Y) TurnFrame(double px, double py, double W, double H, int turn)
        => turn switch
        {
            1 => (H - py, px),
            2 => (W - px, H - py),
            3 => (py, W - px),
            _ => (px, py),
        };

    private static double[,] MatMul(double[,] a, double[,] b, int n, int m, int p)
    {
        var outv = new double[n, p];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                double s = 0;
                for (int k = 0; k < m; k++)
                {
                    s += a[i, k] * b[k, j];
                }
                outv[i, j] = s;
            }
        }
        return outv;
    }

    /// <summary>
    /// Text runs of a page that read left to right once the page is turned
    /// upright by <paramref name="turn"/> quarter turns, in content order.
    /// </summary>
    public static (List<TextLine> Runs, TextStats Stats) ExtractRuns(
        PdfPage page, int turn = 0, int? frame = null)
    {
        int fr = frame ?? turn;
        using var tp = page.LoadTextPage();
        int n = tp.CountChars;

        // A character is upright when its baseline, mapped through the page to
        // render transform, runs left to right after `turn`.
        var aff = DeviceAffine(page);
        // F L F with F = diag(1, -1); the two y flips cancel the sign of the angle.
        var l = new double[2, 2]
        {
            { aff[0, 0], -aff[0, 1] },
            { -aff[1, 0], aff[1, 1] },
        };

        var upright = new Dictionary<float, int?>();
        var quarters = new int[4];
        var badCache = new Dictionary<string, int>();
        var chars = new List<string>();
        var raw = new List<(double L, double B, double R, double T)>();
        var breaks = new List<int>();
        var qs = new List<int>();
        int curQ = 0;
        var stats = new TextStats { Quarters = quarters };
        bool low = false;

        for (int i = 0; i < n; i++)
        {
            if (low)
            {
                low = false;
                continue;
            }
            uint u = tp.GetUnicode(i);
            uint u2 = u is >= 0xD800 and <= 0xDBFF && i + 1 < n ? tp.GetUnicode(i + 1) : 0u;
            var (c, usedSecond) = CodePoint(u, u2);
            low = usedSecond;
            if (c is "\r" or "\n")
            {
                breaks.Add(chars.Count);
                continue;
            }
            if (c == "\x02") // pdfium's mark for a hyphen at a line break
            {
                c = "-";
            }
            if (!IsSpace(c))
            {
                stats.Chars++;
                if (!badCache.TryGetValue(c, out int bad))
                {
                    bad = BadCharCount(c);
                    badCache[c] = bad;
                }
                stats.Bad += bad;
                float a = tp.GetCharAngle(i);
                if (!upright.TryGetValue(a, out int? q))
                {
                    double dx = l[0, 0] * Math.Cos(a) + l[0, 1] * Math.Sin(a);
                    double dy = l[1, 0] * Math.Cos(a) + l[1, 1] * Math.Sin(a);
                    q = Quarter(dx, dy);
                    upright[a] = q;
                }
                if (q is not null)
                {
                    quarters[q.Value]++;
                }
                if (q != turn)
                {
                    stats.Rotated++;
                }
                if (q is null)
                {
                    breaks.Add(chars.Count);
                    continue;
                }
                curQ = ((q.Value - turn) % 4 + 4) % 4;
            }
            chars.Add(c);
            qs.Add(curQ);
            var box = tp.GetCharBox(i) ?? throw new PdfiumException("Failed to get charbox.", 1);
            raw.Add((box.Left, box.Bottom, box.Right, box.Top));
        }

        if (chars.Count == 0)
        {
            return (new List<TextLine>(), stats);
        }

        // Page space -> rendered frame, all characters at once.
        var A = DeviceAffine(page);
        int m = chars.Count;
        var px = new double[m];
        var py = new double[m];
        double[] xsOrder = { 0, 1, 0, 1 };
        double[] ysOrder = { 0, 0, 1, 1 };
        var boxes = new double[m, 4];
        for (int i = 0; i < m; i++)
        {
            var (bl, bb, br, bt) = raw[i];
            double x0 = double.MaxValue, y0 = double.MaxValue, x1 = double.MinValue, y1 = double.MinValue;
            for (int k = 0; k < 4; k++)
            {
                double x = xsOrder[k] == 0 ? bl : br;
                double y = ysOrder[k] == 0 ? bb : bt;
                double dx = A[0, 0] * x + A[0, 1] * y + A[0, 2];
                double dy = A[1, 0] * x + A[1, 1] * y + A[1, 2];
                x0 = Math.Min(x0, dx);
                x1 = Math.Max(x1, dx);
                y0 = Math.Min(y0, dy);
                y1 = Math.Max(y1, dy);
            }
            px[i] = A[0, 0] * (bl + br) / 2 + A[0, 1] * (bb + bt) / 2 + A[0, 2];
            py[i] = A[1, 0] * (bl + br) / 2 + A[1, 1] * (bb + bt) / 2 + A[1, 2];
            boxes[i, 0] = x0;
            boxes[i, 1] = y0;
            boxes[i, 2] = x1;
            boxes[i, 3] = y1;
        }

        // Characters outside the rendered frame are not on the page: dropped.
        var (W, H) = page.GetSize();
        var cx = new double[m];
        var cy = new double[m];
        var visible = new bool[m];
        for (int i = 0; i < m; i++)
        {
            cx[i] = (boxes[i, 0] + boxes[i, 2]) / 2;
            cy[i] = (boxes[i, 1] + boxes[i, 3]) / 2;
            visible[i] = cx[i] >= 0 && cx[i] <= W && cy[i] >= 0 && cy[i] <= H;
        }
        stats.OffPage = visible.Count(v => !v);

        // Steps between neighbouring upright characters of a run.
        int pairs = Math.Max(m - 1, 0);
        var ok = Enumerable.Repeat(true, pairs).ToArray();
        foreach (int b in breaks)
        {
            if (b > 0 && b < m)
            {
                ok[b - 1] = false;
            }
        }
        var sp = chars.Select(IsSpace).ToArray();
        for (int i = 0; i < pairs; i++)
        {
            if (qs[i] != 0 || qs[i + 1] != 0 || sp[i] || sp[i + 1])
            {
                ok[i] = false;
            }
        }
        for (int i = 0; i < pairs; i++)
        {
            double h = Math.Max(boxes[i, 3] - boxes[i, 1], 1e-3);
            double dx = cx[i + 1] - cx[i];
            double dy = cy[i + 1] - cy[i];
            if (Math.Abs(dx) > 3 * h || Math.Abs(dy) > 3 * h)
            {
                ok[i] = false;
            }
        }
        int down = 0, right = 0;
        for (int i = 0; i < pairs; i++)
        {
            double h = Math.Max(boxes[i, 3] - boxes[i, 1], 1e-3);
            double dx = cx[i + 1] - cx[i];
            double dy = cy[i + 1] - cy[i];
            if (ok[i] && dy > 0.3 * h && Math.Abs(dx) < 0.3 * h)
            {
                down++;
            }
            if (ok[i] && Math.Abs(dx) > 0.3 * h && Math.Abs(dy) < 0.3 * h)
            {
                right++;
            }
        }
        stats.Steps = new[] { down, right };

        if (fr != 0)
        {
            for (int i = 0; i < m; i++)
            {
                var (tx, ty) = TurnFrame(px[i], py[i], W, H, fr);
                boxes[i, 0] = boxes[i, 0]; // recomputed below from the corners
                px[i] = tx;
                py[i] = ty;
            }
            // Python maps the four corners of every character; rebuild the boxes.
            for (int i = 0; i < m; i++)
            {
                var (bl, bb, br, bt) = raw[i];
                double x0 = double.MaxValue, y0 = double.MaxValue, x1 = double.MinValue, y1 = double.MinValue;
                for (int k = 0; k < 4; k++)
                {
                    double x = xsOrder[k] == 0 ? bl : br;
                    double y = ysOrder[k] == 0 ? bb : bt;
                    double dx = A[0, 0] * x + A[0, 1] * y + A[0, 2];
                    double dy = A[1, 0] * x + A[1, 1] * y + A[1, 2];
                    var (tx, ty) = TurnFrame(dx, dy, W, H, fr);
                    x0 = Math.Min(x0, tx);
                    x1 = Math.Max(x1, tx);
                    y0 = Math.Min(y0, ty);
                    y1 = Math.Max(y1, ty);
                }
                boxes[i, 0] = x0;
                boxes[i, 1] = y0;
                boxes[i, 2] = x1;
                boxes[i, 3] = y1;
            }
        }

        var runs = new List<TextLine>();
        // A run also ends where the quarter turn changes.
        var turns = new List<int>();
        for (int i = 1; i < m; i++)
        {
            if (qs[i] != qs[i - 1])
            {
                turns.Add(i);
            }
        }
        var cuts = breaks.Concat(turns).Append(m).Distinct().OrderBy(v => v).ToArray();
        int start = 0;
        foreach (int end in cuts)
        {
            var seg = new List<(string Ch, double[] Box)>();
            for (int i = start; i < end; i++)
            {
                if (visible[i])
                {
                    seg.Add((chars[i], new[] { boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3] }));
                }
            }
            int q = start < qs.Count ? qs[start] : 0;
            start = end;
            var ink = seg.Where(s => !IsSpace(s.Ch)).ToList();
            if (ink.Count == 0)
            {
                continue;
            }
            var bbox = new[]
            {
                ink.Min(s => s.Box[0]),
                ink.Min(s => s.Box[1]),
                ink.Max(s => s.Box[2]),
                ink.Max(s => s.Box[3]),
            };
            var text = string.Join(" ", string.Concat(seg.Select(s => s.Ch)).Split(
                (char[]?)null, StringSplitOptions.RemoveEmptyEntries));
            if (fr != turn && q == 0 && ink.Count > 1)
            {
                // A vertical page read turned: a horizontal run is a sideways line.
                var c0 = ink.Select(s => ((s.Box[0] + s.Box[2]) / 2, (s.Box[1] + s.Box[3]) / 2)).ToList();
                double dx = Math.Abs(c0[^1].Item1 - c0[0].Item1);
                double dy = Math.Abs(c0[^1].Item2 - c0[0].Item2);
                if (dy > 2 * dx)
                {
                    q = 1;
                }
            }
            runs.Add(new TextLine
            {
                Bbox = bbox,
                Text = text,
                Chars = seg,
                Quarter = q,
                Seq = runs.Count,
            });
        }
        return (runs, stats);
    }

    private static bool IsSpace(string s)
        => s.Length > 0 && char.IsWhiteSpace(s[0]);

    /// <summary>(pairs, share): neighbouring right-to-left letters inside a run.</summary>
    public static (int Pairs, double Share) VisualRtlFrac(IReadOnlyList<TextLine> runs)
    {
        int pairs = 0, right = 0;
        foreach (var r in runs)
        {
            for (int i = 0; i + 1 < r.Chars.Count; i++)
            {
                string a = r.Chars[i].Ch, b = r.Chars[i + 1].Ch;
                if (a.Length > 0 && b.Length > 0 && Rtl.Bidi.IsRtl(char.ConvertToUtf32(a, 0))
                    && Rtl.Bidi.IsRtl(char.ConvertToUtf32(b, 0)))
                {
                    pairs++;
                    var ba = r.Chars[i].Box;
                    var bb = r.Chars[i + 1].Box;
                    if (bb[0] + bb[2] > ba[0] + ba[2])
                    {
                        right++;
                    }
                }
            }
        }
        return (pairs, right / (double)Math.Max(pairs, 1));
    }

    /// <summary>Share of text objects drawn invisibly (an OCR layer).</summary>
    public static double InvisibleTextFrac(PdfPage page)
    {
        int n = 0, inv = 0;
        foreach (IntPtr obj in page.EnumerateObjects(4))
        {
            if (PdfiumNative.FPDFPageObj_GetType(obj) == PdfiumNative.FPDF_PAGEOBJ_TEXT)
            {
                n++;
                if (PdfiumNative.FPDFTextObj_GetTextRenderMode(obj) == PdfiumNative.FPDF_TEXTRENDERMODE_INVISIBLE)
                {
                    inv++;
                }
            }
        }
        return n > 0 ? inv / (double)n : 0.0;
    }

    /// <summary>Share of runs whose box holds no dark pixel in the render.</summary>
    public static double NoInkFrac(byte[] gray, int width, int height, IReadOnlyList<TextLine> runs, double scale)
    {
        int empty = 0;
        foreach (var ln in runs)
        {
            int r0 = Math.Max(0, (int)(ln.Bbox[1] * scale));
            int r1 = Math.Min(height, (int)(ln.Bbox[3] * scale) + 1);
            int c0 = Math.Max(0, (int)(ln.Bbox[0] * scale));
            int c1 = Math.Min(width, (int)(ln.Bbox[2] * scale) + 1);
            bool any = false;
            for (int r = r0; r < r1 && !any; r++)
            {
                int off = r * width;
                for (int c = c0; c < c1; c++)
                {
                    if (gray[off + c] < Dark)
                    {
                        any = true;
                        break;
                    }
                }
            }
            if (!any)
            {
                empty++;
            }
        }
        return empty / (double)Math.Max(runs.Count, 1);
    }

    /// <summary>Text-layer runs of a page and the verdict on them.</summary>
    public static TextLayer Read(
        PdfPage page, byte[] gray, int width, int height, double scale,
        int turn = 0, int? frame = null)
    {
        int fr = frame ?? turn;
        var (W, H) = page.GetSize();
        var size = fr % 2 != 0 ? (H, W) : (W, H);
        double inv = InvisibleTextFrac(page);
        if (inv > MaxInvisibleFrac)
        {
            return new TextLayer
            {
                Lines = new List<TextLine>(),
                Size = size,
                Trusted = false,
                Reason = $"invisible text {Percent(inv)} (OCR layer)",
                Stats = new TextStats(),
            };
        }
        var (runs, stats) = ExtractRuns(page, turn, frame);
        stats.Turn = turn;
        stats.Frame = fr;
        var rotated = runs.Where(r => r.Quarter != 0).ToList();
        var upright = runs.Where(r => r.Quarter == 0).ToList();
        if (upright.Count == 0)
        {
            return new TextLayer
            {
                Lines = new List<TextLine>(),
                Size = size,
                Trusted = false,
                Reason = "no text layer",
                Stats = stats,
                Rotated = rotated,
            };
        }
        if (stats.Bad > MaxBadCharFrac * Math.Max(stats.Chars, 1))
        {
            return new TextLayer
            {
                Lines = upright,
                Size = size,
                Trusted = false,
                Reason = BrokenFontMap,
                Stats = stats,
                Rotated = rotated,
            };
        }
        // Only upright text has a reading direction on the page.
        var (pairs, visual) = fr == turn ? VisualRtlFrac(upright) : (0, 0.0);
        if (pairs >= MinRtlPairs && visual > MaxVisualRtlFrac)
        {
            stats.VisualRtl = Math.Round(visual, 3);
            return new TextLayer
            {
                Lines = upright,
                Size = size,
                Trusted = false,
                Reason = VisualRtlReason,
                Stats = stats,
                Rotated = rotated,
            };
        }
        double empty = NoInkFrac(gray, width, height, upright, scale);
        stats.NoInk = Math.Round(empty, 4);
        if (empty > MaxNoInkLineFrac)
        {
            return new TextLayer
            {
                Lines = upright,
                Size = size,
                Trusted = false,
                Reason = $"{Percent(empty)} of text runs have no ink",
                Stats = stats,
                Rotated = rotated,
            };
        }
        return new TextLayer
        {
            Lines = upright,
            Size = size,
            Trusted = true,
            Reason = "",
            Stats = stats,
            Rotated = rotated,
        };
    }

    /// <summary>Python's f"{v:.0%}".</summary>
    private static string Percent(double v) => Math.Round(v * 100, MidpointRounding.ToEven).ToString(
        "0", System.Globalization.CultureInfo.InvariantCulture) + "%";
}
