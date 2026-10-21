using IndraOcr.Imaging;
using IndraOcr.Numerics;
using IndraOcr.Pdf;
using IndraOcr.TextLayer;
using OpenCvSharp;
using TextLayerType = IndraOcr.TextLayer.TextLayer;

namespace IndraOcr.Document;

/// <summary>A file that cannot be read: not a PDF or an image, corrupt, or locked.</summary>
public sealed class InputError : Exception
{
    public InputError(string message) : base(message)
    {
    }
}

/// <summary>
/// document.py: the PDF text layer where it can be trusted, OCR elsewhere.
/// </summary>
public sealed class Converter
{
    public const double CheckDpi = 72.0;
    public const double MaxPagePixels = 100e6;
    public const double ReadableMin = 0.9;
    public const int AgreeLines = 6;
    public const double AgreeShare = 0.25;
    public const double LatinCodes = 0.9;
    public const string NotThePage = "text layer does not match the page";

    private readonly Pipeline.OrtOcr _ocr;

    public Converter(Pipeline.OrtOcr ocr, double dpi = 200.0, string mode = "auto", bool imageOcr = true)
    {
        if (mode is not ("auto" or "ocr" or "text"))
        {
            throw new ArgumentException($"mode must be auto, ocr or text, not '{mode}'");
        }
        _ocr = ocr;
        Dpi = dpi;
        Mode = mode;
        ImageOcr = imageOcr;
    }

    public Pipeline.OrtOcr Ocr => _ocr;
    public double Dpi { get; }
    public string Mode { get; }
    public bool ImageOcr { get; }

    /// <summary>Pixels per point for this page, lowered for pages over the pixel cap.</summary>
    public double PageScale(PdfPage page)
    {
        var (w, h) = page.GetSize();
        double k = Dpi / 72.0;
        return Math.Min(k, Math.Sqrt(MaxPagePixels / Math.Max(w * h, 1.0)));
    }

    private static Mat Render(PdfPage page, double scale)
    {
        var bytes = page.RenderGray(scale);
        int width = (int)Math.Ceiling(page.Width * scale);
        int height = (int)Math.Ceiling(page.Height * scale);
        if (bytes.Length == 0)
        {
            return new Mat();
        }
        return FromBytes(bytes, height, width);
    }

    private static Mat FromBytes(byte[] data, int height, int width)
        => ImageOps.FromBytes(data, height, width);

    public static double LatinShare(string text)
    {
        int letters = 0, ascii = 0;
        foreach (char c in text)
        {
            if (char.IsLetter(c))
            {
                letters++;
                if (c < 128)
                {
                    ascii++;
                }
            }
        }
        return ascii / (double)Math.Max(letters, 1);
    }

    /// <summary>Whether a text layer matches its page (legacy fonts store other codes).</summary>
    public static bool LayerAgrees(IReadOnlyList<bool> agree)
        => agree.Count == 0 || agree.Count(v => v) >= AgreeShare * agree.Count;

    /// <summary>Whether the recognizer can read the script of these text-layer runs.</summary>
    public bool Readable(IReadOnlyList<TextLine> runs)
    {
        var letters = new List<string>();
        foreach (var r in runs)
        {
            letters.AddRange(Numerics.Unicode.CodePoints(r.Text).Select(char.ConvertFromUtf32).Where(s => char.IsLetter(s[0])));
        }
        if (letters.Count == 0)
        {
            return true;
        }
        var vocabs = new List<IReadOnlyDictionary<string, int>> { _ocr.Vocab.Index };
        vocabs.AddRange(_ocr.Groups.Values.Select(g => g.Vocab.Index));
        int known = 0;
        foreach (string c in letters)
        {
            bool ok = false;
            foreach (var v in vocabs)
            {
                if (v.ContainsKey(c))
                {
                    ok = true;
                    break;
                }
                var nfd = c.Normalize(System.Text.NormalizationForm.FormD);
                if (nfd.Length > 1 && Numerics.Unicode.CodePoints(nfd)
                        .All(cp => v.ContainsKey(char.ConvertFromUtf32(cp))))
                {
                    ok = true;
                    break;
                }
            }
            if (ok)
            {
                known++;
            }
        }
        return known >= ReadableMin * letters.Count;
    }

    /// <summary>Unicode bidi base direction: most strong characters are right-to-left.</summary>
    public static bool RightToLeft(IReadOnlyList<Pipeline.Token> tokens)
    {
        int rtl = 0, ltr = 0;
        foreach (var t in tokens)
        {
            if (t.Char is null || t.Char.Length == 0)
            {
                continue;
            }
            int cp = char.ConvertToUtf32(t.Char, 0);
            var b = Numerics.Unicode.BidiClass(cp);
            if (b is Numerics.Unicode.Bidi.R or Numerics.Unicode.Bidi.AL)
            {
                rtl++;
            }
            else if (b == Numerics.Unicode.Bidi.L)
            {
                ltr++;
            }
        }
        return rtl > ltr;
    }

    private static bool Covered(float[] box, float[] boxes, int count, double frac = 0.5)
    {
        double area = Math.Max((box[2] - box[0]) * (box[3] - box[1]), 1e-6);
        for (int i = 0; i < count; i++)
        {
            double ix = Math.Max(0, Math.Min(box[2], boxes[i * 4 + 2]) - Math.Max(box[0], boxes[i * 4 + 0]));
            double iy = Math.Max(0, Math.Min(box[3], boxes[i * 4 + 3]) - Math.Max(box[1], boxes[i * 4 + 1]));
            if (ix * iy / area > frac)
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>
    /// Tokens (char, x, content index) per detected line.  A run is one text-layer
    /// line piece; each character goes to the smallest detected box that contains
    /// its centre (a space follows the character before it).
    /// </summary>
    public static (List<Pipeline.Token>[] Tokens, int Lost) AssignRuns(
        IReadOnlyList<TextLine> runs, float[] boxes, int boxCount, double k)
    {
        var parts = new Dictionary<int, List<Pipeline.Token>>[boxCount];
        for (int i = 0; i < boxCount; i++)
        {
            parts[i] = new Dictionary<int, List<Pipeline.Token>>();
        }
        int lost = 0;
        if (boxCount == 0)
        {
            return (Array.Empty<List<Pipeline.Token>>(), runs.Sum(r => r.Chars.Count(c => !char.IsWhiteSpace(c.Ch[0]))));
        }
        var area = new double[boxCount];
        for (int i = 0; i < boxCount; i++)
        {
            area[i] = (boxes[i * 5 + 2] - boxes[i * 5 + 0]) * (boxes[i * 5 + 3] - boxes[i * 5 + 1]);
        }
        for (int pos = 0; pos < runs.Count; pos++)
        {
            var run = runs[pos];
            int idx = run.Seq >= 0 ? run.Seq : pos;
            int j = -1;
            foreach (var (c, b) in run.Chars)
            {
                if (!char.IsWhiteSpace(c[0]))
                {
                    double cx = (b[0] + b[2]) / 2 * k;
                    double cy = (b[1] + b[3]) / 2 * k;
                    int best = -1;
                    double bestArea = double.MaxValue;
                    for (int i = 0; i < boxCount; i++)
                    {
                        if (boxes[i * 5] <= cx && cx <= boxes[i * 5 + 2]
                            && boxes[i * 5 + 1] <= cy && cy <= boxes[i * 5 + 3]
                            && area[i] < bestArea)
                        {
                            best = i;
                            bestArea = area[i];
                        }
                    }
                    if (best < 0)
                    {
                        lost++;
                        j = -1;
                        continue;
                    }
                    j = best;
                    if (!parts[j].TryGetValue(idx, out var list))
                    {
                        parts[j][idx] = list = new List<Pipeline.Token>();
                    }
                    list.Add(Pipeline.Token.FromChar(c, (float)cx, idx));
                }
                else if (j >= 0 && parts[j].TryGetValue(idx, out var list) && list.Count > 0)
                {
                    list.Add(Pipeline.Token.FromChar(c, list[^1].X, idx));
                }
            }
        }
        var toks = new List<Pipeline.Token>[boxCount];
        for (int line = 0; line < boxCount; line++)
        {
            var tk = new List<Pipeline.Token>();
            var byRun = parts[line];
            var all = byRun.Values.SelectMany(v => v).ToList();
            bool rtl = RightToLeft(all);
            var order = rtl
                ? byRun.Keys.OrderByDescending(i => byRun[i].Max(t => t.X)).ToList()
                : byRun.Keys.OrderBy(i => byRun[i].Min(t => t.X)).ToList();
            foreach (int idx in order)
            {
                if (tk.Count > 0)
                {
                    tk.Add(Pipeline.Token.FromChar(" ", tk[^1].X, idx)); // parts of different runs
                }
                tk.AddRange(byRun[idx]);
            }
            toks[line] = tk;
        }
        return (toks, lost);
    }

    /// <summary>One page from its PDF text layer.</summary>
    public Pipeline.Page? TextPage(
        PdfPage page, TextLayerType textLayer, Dictionary<string, double> timings, bool check)
    {
        double k = PageScale(page);
        int turn = textLayer.Stats.Turn;
        int frame = textLayer.Stats.Frame;
        bool vertical = textLayer.Stats.Vertical;

        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var gray = Render(page, k);
        timings["render_ms"] = sw.Elapsed.TotalMilliseconds;
        if (frame != 0)
        {
            using var turned = ImageOps.Rot90(gray, -frame);
            return TextPageOn(page, textLayer, timings, check, turned, k, turn, vertical);
        }
        return TextPageOn(page, textLayer, timings, check, gray, k, turn, vertical);
    }

    /// <summary>
    /// Text-layer lines that hold math (the classifier, on their crops) are read
    /// by MathReader instead: the layer has the symbols but not the LaTeX.  The
    /// line keeps its place in the PDF content order (its first character's index).
    /// </summary>
    internal void ReadMath(Mat gray, float[] boxes, Pipeline.RecLine?[] rec)
    {
        int n = boxes.Length / 5;
        var idx = new List<int>();
        for (int i = 0; i < n; i++)
        {
            if (rec[i] is { } r && r.Tokens.Length > 0
                && r.Tokens[0].Char is not null && r.Tokens[0].Seq >= 0)
            {
                idx.Add(i);
            }
        }
        if (idx.Count == 0)
        {
            return;
        }
        var boxList = new List<float[]>(idx.Count);
        foreach (int i in idx)
        {
            boxList.Add(new[] { boxes[i * 5], boxes[i * 5 + 1], boxes[i * 5 + 2], boxes[i * 5 + 3] });
        }
        var crops = Pipeline.LineMask.CropLines(gray, boxList, _ocr.Height);
        var flags = _ocr.MathRec!.Flags(crops);
        var math = new List<int>();
        for (int k = 0; k < idx.Count; k++)
        {
            if (flags[k])
            {
                math.Add(idx[k]);
            }
        }
        if (math.Count == 0)
        {
            return;
        }
        var sub = new float[math.Count * 5];
        for (int k = 0; k < math.Count; k++)
        {
            Array.Copy(boxes, math[k] * 5, sub, k * 5, 4);
        }
        var readings = _ocr.MathRec.Read(gray, sub);
        for (int k = 0; k < math.Count; k++)
        {
            var (text, conf) = readings[k];
            // A reading without math, or not confident, leaves the layer's text.
            if (text.Contains('$') && conf >= MathOcr.MathReader.MinConf)
            {
                int i = math[k];
                int first = rec[i]!.Tokens.Min(t => t.Seq);
                float xc = (boxes[i * 5] + boxes[i * 5 + 2]) / 2;
                rec[i] = new Pipeline.RecLine(text, 1.0,
                    new[] { Pipeline.Token.FromChar(text, xc, first) });
            }
        }
    }

    private Pipeline.Page? TextPageOn(
        PdfPage page, TextLayerType textLayer, Dictionary<string, double> timings, bool check,
        Mat gray, double k, int turn, bool vertical)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var boxes = _ocr.Detect(gray);
        timings["det_ms"] = sw.Elapsed.TotalMilliseconds;
        int n = boxes.Length / 5;
        var (toks, lost) = AssignRuns(textLayer.Lines, boxes, n, k);
        var rec = new Pipeline.RecLine?[n];
        var empty = new List<int>();
        for (int i = 0; i < n; i++)
        {
            string text = toks[i].Count > 0
                ? Pipeline.Assembly.Decode(toks[i], _ocr.Vocab.Decode, pageRtl: false)
                : "";
            if (text.Length > 0)
            {
                rec[i] = new Pipeline.RecLine(text, 1.0, toks[i].ToArray());
            }
            else
            {
                empty.Add(i);
            }
        }

        if (check)
        {
            var full = Enumerable.Range(0, n)
                .Where(i => rec[i] is not null && rec[i]!.Text.Count(char.IsLetter) >= 4)
                .ToList();
            int stride = Math.Max(1, full.Count / AgreeLines);
            var pick = full.Where((_, idx) => idx % stride == 0).Take(AgreeLines).ToList();
            if (pick.Count > 0)
            {
                var pickBoxes = new float[pick.Count * 5];
                for (int i = 0; i < pick.Count; i++)
                {
                    Array.Copy(boxes, pick[i] * 5, pickBoxes, i * 5, 5);
                }
                var (groups, sure) = _ocr.BoxGroups(gray, pickBoxes);
                // A line agrees unless its layer text is Latin while the router is
                // sure its image is another script.
                var agree = new List<bool>();
                for (int i = 0; i < pick.Count; i++)
                {
                    agree.Add(!(LatinShare(rec[pick[i]]!.Text) >= LatinCodes
                                && groups[i] != "base" && sure[i]));
                }
                timings["check_ms"] = sw.Elapsed.TotalMilliseconds;
                if (!LayerAgrees(agree))
                {
                    return null;
                }
            }
        }

        string source = "text";
        // Sideways runs (a rotated table header, a chart's axis label) come from
        // the layer as their own lines; detected lines on them are not OCR'd.
        var side = textLayer.Rotated.Where(r => r.Text.Trim().Length > 0).ToList();
        var sideBoxes = new float[side.Count * 4];
        for (int i = 0; i < side.Count; i++)
        {
            for (int c = 0; c < 4; c++)
            {
                sideBoxes[i * 4 + c] = (float)(side[i].Bbox[c] * k);
            }
        }
        if (side.Count > 0 && empty.Count > 0)
        {
            empty = empty.Where(i => !Covered(
                new[] { boxes[i * 5], boxes[i * 5 + 1], boxes[i * 5 + 2], boxes[i * 5 + 3] },
                sideBoxes, side.Count)).ToList();
        }

        if (empty.Count > 0 && ImageOcr && Readable(textLayer.Lines))
        {
            sw.Restart();
            var sub = new float[empty.Count * 5];
            for (int i = 0; i < empty.Count; i++)
            {
                Array.Copy(boxes, empty[i] * 5, sub, i * 5, 5);
            }
            var results = _ocr.Recognize(gray, sub, vertical: vertical);
            for (int i = 0; i < empty.Count; i++)
            {
                if (results[i].Text.Trim().Length > 0)
                {
                    rec[empty[i]] = results[i];
                    source = "hybrid";
                }
            }
            timings["rec_ms"] = sw.Elapsed.TotalMilliseconds;
        }

        if (_ocr.MathRec is not null)
        {
            sw.Restart();
            ReadMath(gray, boxes, rec);
            timings["math_ms"] = sw.Elapsed.TotalMilliseconds;
        }

        var keep = Enumerable.Range(0, n).Where(i => rec[i] is not null).ToList();
        var keptBoxes = new float[keep.Count * 5];
        var keptRec = new List<Pipeline.RecLine>();
        for (int i = 0; i < keep.Count; i++)
        {
            Array.Copy(boxes, keep[i] * 5, keptBoxes, i * 5, 5);
            keptRec.Add(rec[keep[i]]!);
        }
        if (side.Count > 0)
        {
            var extra = new float[side.Count * 5];
            for (int i = 0; i < side.Count; i++)
            {
                Array.Copy(sideBoxes, i * 4, extra, i * 5, 4);
                extra[i * 5 + 4] = 1.0f;
            }
            var combined = new float[keptBoxes.Length + extra.Length];
            Array.Copy(keptBoxes, combined, keptBoxes.Length);
            Array.Copy(extra, 0, combined, keptBoxes.Length, extra.Length);
            keptBoxes = combined;
            foreach (var r in side)
            {
                float cx = (float)((r.Bbox[0] + r.Bbox[2]) / 2 * k); // one x: never cut
                keptRec.Add(new Pipeline.RecLine(r.Text, 1.0,
                    r.Text.Select(c => Pipeline.Token.FromChar(c.ToString(), cx, r.Seq)).ToArray()));
            }
        }

        var outPage = _ocr.Assemble(gray.Cols, gray.Rows, keptBoxes, keptRec.ToArray(), timings, vertical);
        outPage.Source = source;
        outPage.Rotation = 90 * turn;
        if (vertical)
        {
            int w = gray.Rows;
            foreach (var line in outPage.Lines)
            {
                line.Box = new[] { w - line.Box[3], line.Box[0], w - line.Box[1], line.Box[2] };
            }
            for (int b = 0; b < outPage.Blocks.Count; b++)
            {
                var bb = outPage.Blocks[b].Bbox;
                outPage.Blocks[b] = new IndraOcr.Layout.Block
                {
                    Label = outPage.Blocks[b].Label,
                    Bbox = new[] { w - bb[3], bb[0], w - bb[1], bb[2] },
                    Text = outPage.Blocks[b].Text,
                };
            }
            outPage.Vertical = true;
        }
        if (lost > 0)
        {
            outPage.Note += $"; {lost} chars outside detected lines";
        }
        return outPage;
    }

    /// <summary>One PDF page: the text layer where it is sound, OCR elsewhere.</summary>
    public Pipeline.Page PdfPage(string pdfPath, PdfDocument doc, int index)
    {
        using var page = doc.LoadPage(index);
        var timings = new Dictionary<string, double>();
        string note = "mode=ocr";
        bool turn;
        if (Mode != "ocr")
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            double s = CheckDpi / 72.0;
            using var check = Render(page, s);
            var tl = TextLayerReader.Read(page, ImageOps.ToBytes(check), check.Cols, check.Rows, s);
            int k = TextLayerReader.PageTurn(tl.Stats);
            int rotation = page.GetRotation();
            if (k != 0 && (rotation + 90 * k) % 360 == 0)
            {
                // The page's /Rotate alone turns the text: read it without.
                page.SetRotation(0);
                using (var upright = Render(page, s))
                {
                    tl = TextLayerReader.Read(page, ImageOps.ToBytes(upright), upright.Cols, upright.Rows, s);
                    if (tl.Lines.Count > 0 && (tl.Trusted || Mode == "text"))
                    {
                        tl.Stats.Turn = k;
                        timings["text_ms"] = sw.Elapsed.TotalMilliseconds;
                        var outPage = TextPage(page, tl, timings, check: Mode == "auto");
                        if (outPage is not null)
                        {
                            return outPage;
                        }
                    }
                }
                page.SetRotation(rotation);
                using var turned = ImageOps.Rot90(check, -k);
                tl = TextLayerReader.Read(page, ImageOps.ToBytes(turned), turned.Cols, turned.Rows, s, k);
            }
            else if (k != 0)
            {
                using var turned = ImageOps.Rot90(check, -k);
                tl = TextLayerReader.Read(page, ImageOps.ToBytes(turned), turned.Cols, turned.Rows, s, k);
            }
            else if (TextLayerReader.VerticalLayout(tl.Stats))
            {
                // Vertical writing: upright characters, read in the page turned a
                // quarter turn counter-clockwise (columns become lines).
                using var turned = ImageOps.Rot90(check, 1);
                tl = TextLayerReader.Read(page, ImageOps.ToBytes(turned), turned.Cols, turned.Rows, s, 0, 3);
                tl.Stats.Vertical = true;
            }
            timings["text_ms"] = sw.Elapsed.TotalMilliseconds;

            // A layer the recognizer cannot read beats OCR, unless its codes are junk.
            bool unreadable = tl.Reason != TextLayerReader.BrokenFontMap && !Readable(tl.Lines);
            note = tl.Reason;
            if (tl.Lines.Count > 0 && (tl.Trusted || Mode == "text" || unreadable))
            {
                var outPage = TextPage(
                    page, tl, timings,
                    check: Mode == "auto" && tl.Trusted && !unreadable);
                if (outPage is not null)
                {
                    return outPage;
                }
                note = NotThePage;
            }
            // A layer with a broken font map still places its glyphs: when most
            // run upright, the page is upright (orient.onnx is not asked).
            var st = tl.Stats;
            turn = !(tl.Reason == TextLayerReader.BrokenFontMap && st.Rotated < 0.5 * st.Chars);
        }
        else
        {
            turn = true;
        }

        var sw2 = System.Diagnostics.Stopwatch.StartNew();
        using var gray = Render(page, PageScale(page));
        timings["render_ms"] = sw2.Elapsed.TotalMilliseconds;
        var result = _ocr.Read(gray, turn);
        foreach (var (key, value) in timings)
        {
            result.Timings[key] = value;
        }
        result.Timings["total_ms"] = result.Timings.Where(kv => kv.Key != "total_ms").Sum(kv => kv.Value);
        result.Note = $"{note}; {result.Note}";
        return result;
    }

    /// <summary>PDF or image file: (index, Page) per page, errors as empty pages.</summary>
    public IEnumerable<(int Index, Pipeline.Page Page)> Convert(
        string path, int maxPages = 0, string? password = null)
    {
        if (maxPages < 0)
        {
            throw new ArgumentException($"max_pages must be 0 (all pages) or more, not {maxPages}");
        }
        if (!File.Exists(path))
        {
            throw new InputError($"{path}: no such file");
        }
        if (!IsPdf(path))
        {
            var frames = ImageOps.LoadGrayMulti(path);
            if (frames.Count == 0)
            {
                throw new InputError($"{path}: not a PDF or a readable image");
            }
            int limit = maxPages == 0 ? frames.Count : Math.Min(maxPages, frames.Count);
            for (int i = 0; i < limit; i++)
            {
                using var frame = frames[i];
                yield return (i, SafeRead(() => _ocr.Read(frame)));
            }
            foreach (var f in frames.Skip(limit))
            {
                f.Dispose();
            }
            yield break;
        }

        PdfDocument doc;
        try
        {
            doc = PdfDocument.Open(path, password);
        }
        catch (PdfiumException e)
        {
            throw OpenError(path, e);
        }
        try
        {
            int n = maxPages == 0 ? doc.PageCount : Math.Min(doc.PageCount, maxPages);
            for (int i = 0; i < n; i++)
            {
                int index = i;
                yield return (index, SafeRead(() => PdfPage(path, doc, index)));
            }
        }
        finally
        {
            doc.Dispose();
        }
    }

    internal static Pipeline.Page SafeRead(Func<Pipeline.Page> read)
    {
        try
        {
            return read();
        }
        catch (Exception e)
        {
            return new Pipeline.Page
            {
                Markdown = "",
                Source = "error",
                Note = $"{e.GetType().Name}: {e.Message}",
                Timings = { ["total_ms"] = 0.0 },
            };
        }
    }

    /// <summary>The InputError message for a document PDFium refused to open.</summary>
    internal static InputError OpenError(string path, PdfiumException e)
    {
        bool locked = e.Code == 4 || e.Message.Contains("password", StringComparison.OrdinalIgnoreCase);
        return new InputError(locked
            ? $"{path}: password-protected PDF (pass password=, or --password)"
            : $"{path}: cannot open the PDF ({e.Message})");
    }

    /// <summary>The file's own bytes say PDF (the header may follow a little junk).</summary>
    public static bool IsPdf(string path)
    {
        using var f = File.OpenRead(path);
        var buf = new byte[1024];
        int n = f.Read(buf, 0, buf.Length);
        return System.Text.Encoding.ASCII.GetString(buf, 0, n).Contains("%PDF-", StringComparison.Ordinal);
    }
}
