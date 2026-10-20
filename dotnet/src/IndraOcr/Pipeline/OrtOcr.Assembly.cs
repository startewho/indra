using IndraOcr.Imaging;
using IndraOcr.Layout;
using IndraOcr.Numerics;
using IndraOcr.Ordering;
using IndraOcr.Rtl;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace IndraOcr.Pipeline;

/// <summary>The page stage of pipeline.py: reading order, layout, markdown and the page turn.</summary>
public sealed partial class OrtOcr
{
    public const int VerticalClass = 4;
    public const double OrientMin = 0.95;

    // ----------------------------------------------------------- reading order

    /// <summary>
    /// Reading order, the layout heads when the order model has them, and the
    /// pairwise order logits.
    /// </summary>
    public (int[] Order, LayoutHeads? Layout, float[]? Pair) ReadingOrder(
        float[] boxes, int n, string[] texts, int width, int height)
    {
        if (n < 2 || Order is null)
        {
            return (OrderPost.OrderXycut(boxes, n), null, null);
        }
        var feats = OrderPost.LineFeatures(boxes, n, width, height, texts);
        var mask = Enumerable.Repeat(true, n).ToArray();
        var input = new DenseTensor<float>(feats, new[] { 1, n, OrderPost.NFeats });
        var maskTensor = new DenseTensor<bool>(mask, new[] { 1, n });
        using var outputs = Order.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor("feats", input),
            NamedOnnxValue.CreateFromTensor("mask", maskTensor),
        });
        var pairTensor = outputs[0].AsTensor<float>();
        var pairF = pairTensor.ToArray();
        var pair = new double[pairF.Length];
        for (int i = 0; i < pair.Length; i++)
        {
            pair[i] = pairF[i];
        }
        if (outputs.Count >= 3)
        {
            var clsTensor = outputs[1].AsTensor<float>();
            int nClasses = clsTensor.Dimensions[2];
            var cls = clsTensor.ToArray();
            var sameTensor = outputs[2].AsTensor<float>();
            var sameF = sameTensor.ToArray();
            var same = new double[sameF.Length];
            for (int i = 0; i < same.Length; i++)
            {
                same[i] = sameF[i];
            }
            float[]? tab = null;
            if (outputs.Count > 3 && TableHeads)
            {
                tab = outputs[3].AsTensor<float>().ToArray();
            }
            var order = OrderPost.DecodeBlocksOrder(pair, n, same, boxes, cls, nClasses);
            return (order, new LayoutHeads
            {
                Cls = cls,
                NClasses = nClasses,
                Same = sameF,
                Tab = tab,
            }, pairF);
        }
        return (OrderPost.DecodeOrder(pair, n), null, pairF);
    }

    // --------------------------------------------------------------- assembly

    private static float[] Boxes4(float[] boxes5, int n)
    {
        var outv = new float[n * 4];
        for (int i = 0; i < n; i++)
        {
            Array.Copy(boxes5, i * 5, outv, i * 4, 4);
        }
        return outv;
    }

    private static float[] Row4(float[] boxes, int i)
    {
        var row = new float[4];
        Array.Copy(boxes, i * 4, row, 0, 4);
        return row;
    }

    private static float[] Geo(float[] boxes, int n, bool rtl, float width)
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

    private static float[] GeoOne(float[] box, bool rtl, float width)
    {
        if (!rtl)
        {
            return box;
        }
        return new[] { width - box[2], box[1], width - box[0], box[3] };
    }

    private static float[] RoundBox(float[] box)
        => box.Select(v => (float)Math.Round(v, 1)).ToArray();

    /// <summary>Lines -> split at wide gaps -> reading order -> markdown.</summary>
    public Page Assemble(
        int width, int height, float[] boxes5, RecLine[] recIn,
        Dictionary<string, double> timings, bool vertical = false, float[]? pictures = null)
    {
        int nIn = boxes5.Length / 5;
        // A right-to-left page: lines read right to left, and reading order and
        // markdown work on the page mirrored left to right.
        int letters = 0, rtlLetters = 0;
        foreach (var r in recIn)
        {
            foreach (char c in r.Text)
            {
                if (!char.IsLetter(c))
                {
                    continue;
                }
                letters++;
                if (Bidi.IsRtl(c))
                {
                    rtlLetters++;
                }
            }
        }
        bool pageRtl = rtlLetters > letters - rtlLetters;
        float widthF = width;
        int spaceId = Vocab.Index.TryGetValue(" ", out int sid) ? sid : -1;

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        var boxesIn = Boxes4(boxes5, nIn);
        var cuts = nIn > 0 ? Assembly.GutterCuts(boxesIn, nIn, recIn, spaceId) : new List<List<float>>();
        var newBoxes = new List<float[]>();
        var newRec = new List<RecLine>();
        for (int i = 0; i < nIn; i++)
        {
            foreach (var part in Assembly.SplitLine(
                         Row4(boxesIn, i), recIn[i].Tokens, Assembly.SplitGap, cuts[i],
                         spaceId, Vocab.Decode, pageRtl))
            {
                newBoxes.Add(part.Box);
                newRec.Add(new RecLine(part.Text, recIn[i].Conf, part.Tokens));
            }
        }
        var flat = new float[newBoxes.Count * 4];
        for (int i = 0; i < newBoxes.Count; i++)
        {
            Array.Copy(newBoxes[i], 0, flat, i * 4, 4);
        }
        var (tidyBoxes, tidyRec) = Assembly.TidyLines(flat, newBoxes.Count, newRec, pageRtl, spaceId);
        var boxes = tidyBoxes;
        var rec = tidyRec;
        int n = rec.Count;
        var texts = rec.Select(r => r.Text).ToArray();

        var (order, layout, _) = ReadingOrder(Geo(boxes, n, pageRtl, widthF), n, texts, width, height);

        if (vertical && layout is not null)
        {
            // The order model saw no vertical pages: blocks and order come from
            // XY-cut instead, without grid classes.
            var blocks = OrderPost.XycutBlocks(boxes, n);
            order = blocks.SelectMany(b => b).ToArray();
            var same = new float[n * n];
            Array.Fill(same, -1f);
            foreach (var b in blocks)
            {
                foreach (int i in b)
                {
                    foreach (int j in b)
                    {
                        same[i * n + j] = 1f;
                    }
                }
            }
            var cls = (float[])layout.Cls.Clone();
            for (int i = 0; i < n; i++)
            {
                foreach (int g in OrderPost.GridIds)
                {
                    cls[i * layout.NClasses + g] = -1e4f;
                }
            }
            layout = new LayoutHeads { Cls = cls, NClasses = layout.NClasses, Same = same, Tab = null };
        }
        else if (layout?.Tab is not null)
        {
            // The table heads learned one line per cell, but the detector can join
            // cells: split table lines into cells, then run the order model again.
            int tableCls = Array.IndexOf(OrderPost.BlockClasses, "Table");
            var isTable = new bool[n];
            bool any = false;
            for (int i = 0; i < n; i++)
            {
                int best = 0;
                for (int c = 1; c < layout.NClasses; c++)
                {
                    if (layout.Cls[i * layout.NClasses + c] > layout.Cls[i * layout.NClasses + best])
                    {
                        best = c;
                    }
                }
                isTable[i] = best == tableCls;
                any |= isTable[i];
            }
            if (any)
            {
                var nb = new List<float[]>();
                var nr = new List<RecLine>();
                for (int i = 0; i < n; i++)
                {
                    var parts = isTable[i]
                        ? Assembly.SplitLine(Row4(boxes, i), rec[i].Tokens, Assembly.CellGap, null,
                            spaceId, Vocab.Decode, pageRtl)
                        : new List<(float[] Box, string Text, Token[] Tokens)>
                        {
                            (Row4(boxes, i), rec[i].Text, rec[i].Tokens),
                        };
                    foreach (var (bx, tx, tk) in parts)
                    {
                        nb.Add(bx);
                        nr.Add(new RecLine(tx, rec[i].Conf, tk));
                    }
                }
                if (nb.Count != n)
                {
                    boxes = new float[nb.Count * 4];
                    for (int i = 0; i < nb.Count; i++)
                    {
                        Array.Copy(nb[i], 0, boxes, i * 4, 4);
                    }
                    rec = nr;
                    n = rec.Count;
                    texts = rec.Select(r => r.Text).ToArray();
                    (order, layout, _) = ReadingOrder(Geo(boxes, n, pageRtl, widthF), n, texts, width, height);
                }
            }
        }

        string note = vertical ? "order=xycut" : "order=model";
        timings["order_ms"] = stopwatch.Elapsed.TotalMilliseconds;
        timings["total_ms"] = timings.Where(kv => kv.Key != "total_ms").Sum(kv => kv.Value);

        var lines = order.Select(i => new Line
        {
            Box = RoundBox(Row4(boxes, i)),
            Text = rec[i].Text,
            Conf = Math.Round(rec[i].Conf, 4),
        }).ToList();

        string? markdown = null;
        var blocksOut = new List<Block>();
        if (layout is not null)
        {
            var geoBoxes = Geo(boxes, n, pageRtl, widthF);
            blocksOut = LayoutPost.PageBlocks(
                geoBoxes, n, texts, order, layout.Cls, layout.NClasses, layout.Same);
            for (int b = 0; b < blocksOut.Count; b++)
            {
                var bb = GeoOne(blocksOut[b].Bbox, pageRtl, widthF);
                blocksOut[b] = new Block
                {
                    Label = blocksOut[b].Label,
                    Bbox = RoundBox(bb),
                    Text = blocksOut[b].Text,
                };
            }
            var cells = new Dictionary<int, List<(float[] Box, string Text)>>();
            for (int i = 0; i < n; i++)
            {
                cells[i] = Assembly.SplitLine(
                        Row4(boxes, i), rec[i].Tokens, Assembly.CellGap, null,
                        spaceId, Vocab.Decode, pageRtl)
                    .Select(p => (GeoOne(p.Box, pageRtl, widthF), p.Text))
                    .ToList();
            }
            bool hasPictures = pictures is not null && pictures.Length > 0;
            var geoPictures = hasPictures ? Geo(pictures!, pictures!.Length / 4, pageRtl, widthF) : null;
            markdown = LayoutPost.PageMarkdown(
                geoBoxes, n, texts, order, layout.Cls, layout.NClasses, layout.Same,
                Furniture, cells, layout.Tab,
                geoPictures, hasPictures ? geoPictures!.Length / 4 : 0);
            if (hasPictures)
            {
                for (int p = 0; p < pictures!.Length / 4; p++)
                {
                    float x0 = pictures[p * 4], y0 = pictures[p * 4 + 1];
                    float x1 = pictures[p * 4 + 2], y1 = pictures[p * 4 + 3];
                    float cy = (y0 + y1) / 2;
                    int at = blocksOut.Count;
                    for (int j = 0; j < blocksOut.Count; j++)
                    {
                        var b = blocksOut[j].Bbox;
                        if ((b[1] + b[3]) / 2 > cy && Math.Min(x1, b[2]) > Math.Max(x0, b[0]))
                        {
                            at = j;
                            break;
                        }
                    }
                    blocksOut.Insert(at, new Block
                    {
                        Label = "Picture",
                        Bbox = new[] { x0, y0, x1, y1 }.Select(v => (float)Math.Round(v, 1)).ToArray(),
                        Text = "",
                    });
                }
            }
        }
        else if (n < 2)
        {
            markdown = string.Join("\n", texts);
        }

        var page = new Page
        {
            Markdown = markdown,
            Note = note,
        };
        page.Lines.AddRange(lines);
        page.Blocks.AddRange(blocksOut);
        foreach (var (k, v) in timings)
        {
            page.Timings[k] = v;
        }
        return page;
    }

    // ------------------------------------------------------------- orientation

    public float[]? OrientationProbs(Mat gray)
    {
        if (Orient is null)
        {
            return null;
        }
        var x = Skew.OrientInput(gray);
        // The orientation model's output: class probabilities.
        var input = new DenseTensor<float>(x, new[] { 1, 1, Skew.OrientSide, Skew.OrientSide });
        using var outputs = Orient.Run(new[] { NamedOnnxValue.CreateFromTensor("x", input) });
        return outputs[0].AsTensor<float>().ToArray();
    }

    /// <summary>Quarter turns the page content is turned by (0 when unsure); VERTICAL for columns.</summary>
    public int Orientation(Mat gray)
    {
        var p = OrientationProbs(gray);
        if (p is null)
        {
            return 0;
        }
        int k = Np.ArgMax(p);
        return p[k] > OrientMin ? k : 0;
    }

    private (Mat Page, int K, double Skew, bool Owned) Turn(
        Mat gray, int k, double a, Mat? small)
    {
        int q = k == VerticalClass ? 1 : -k;
        Mat page;
        bool owned;
        if (k == VerticalClass)
        {
            page = ImageOps.Rot90(gray, 1);
            owned = true;
        }
        else if (k != 0)
        {
            page = ImageOps.Rot90(gray, -k);
            owned = true;
        }
        else
        {
            page = gray;
            owned = false;
        }
        if (k is 1 or 3 or VerticalClass)
        {
            Mat measure = page;
            bool ownMeasure = false;
            if (small is not null)
            {
                measure = ImageOps.Rot90(small, q);
                ownMeasure = true;
            }
            a = Skew.EstimateSkew(measure);
            if (ownMeasure)
            {
                measure.Dispose();
            }
        }
        if (Math.Abs(a) >= Skew.DeskewMin)
        {
            var rotated = Skew.RotatePage(page, a);
            if (owned)
            {
                page.Dispose();
            }
            page = rotated;
            owned = true;
        }
        else
        {
            a = 0.0;
        }
        return (page, k, a, owned);
    }

    /// <summary>The page turned upright and straightened: (page, turns undone, skew undone).</summary>
    public (Mat Page, int K, double Skew, bool Owned) Upright(Mat gray, bool turn = true, int? k = null)
    {
        double a = Skew.EstimateSkew(gray);
        if (k is null)
        {
            k = turn
                ? Orientation(Math.Abs(a) >= Skew.DeskewMin ? Skew.RotatePage(gray, a) : gray)
                : 0;
        }
        return Turn(gray, k.Value, a, null);
    }

    /// <summary>OCR of a page image, with the page turned upright and deskewed first.</summary>
    public Page Read(Mat gray, bool turn = true)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        Mat small = Skew.Preview(gray);
        bool ownsSmall = !ReferenceEquals(small, gray);
        double a = Skew.EstimateSkew(small);
        int k = 0;
        if (turn)
        {
            Mat forOrient = small;
            bool ownOrient = false;
            if (Math.Abs(a) >= Skew.DeskewMin)
            {
                forOrient = Skew.RotatePage(small, a);
                ownOrient = true;
            }
            k = Orientation(forOrient);
            if (ownOrient)
            {
                forOrient.Dispose();
            }
        }
        double orientMs = sw.Elapsed.TotalMilliseconds;
        Page page = ReadPage(gray, k, a, small);
        page.Timings["orient_ms"] = page.Timings.GetValueOrDefault("orient_ms") + orientMs;
        if (ownsSmall)
        {
            small.Dispose();
        }
        return page;
    }

    private Page ReadPage(Mat gray, int k, double a, Mat? small)
    {
        var timings = new Dictionary<string, double>();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var (pageMat, kk, skew, owned) = Turn(gray, k, a, small);
        timings["orient_ms"] = sw.Elapsed.TotalMilliseconds;
        try
        {
            sw.Restart();
            var (boxes, pictures) = DetectAll(pageMat);
            timings["det_ms"] = sw.Elapsed.TotalMilliseconds;
            sw.Restart();
            bool vertical = kk == VerticalClass;
            var rec = Recognize(pageMat, boxes, vertical: vertical);
            timings["rec_ms"] = sw.Elapsed.TotalMilliseconds;
            var page = Assemble(
                pageMat.Cols, pageMat.Rows, boxes, rec, timings,
                vertical, vertical ? null : pictures);
            page.Rotation = vertical ? 0 : 90 * kk;
            page.Skew = Math.Round(skew, 2);
            page.Vertical = vertical;
            if (skew != 0)
            {
                int w = pageMat.Cols, h = pageMat.Rows;
                foreach (var line in page.Lines)
                {
                    line.Box = Skew.UnrotateBox(line.Box, skew, w, h);
                }
                for (int b = 0; b < page.Blocks.Count; b++)
                {
                    page.Blocks[b] = new Block
                    {
                        Label = page.Blocks[b].Label,
                        Bbox = Skew.UnrotateBox(page.Blocks[b].Bbox, skew, w, h),
                        Text = page.Blocks[b].Text,
                    };
                }
            }
            if (vertical)
            {
                int w = pageMat.Rows;
                foreach (var line in page.Lines)
                {
                    line.Box = new[] { w - line.Box[3], line.Box[0], w - line.Box[1], line.Box[2] };
                }
                for (int b = 0; b < page.Blocks.Count; b++)
                {
                    var bb = page.Blocks[b].Bbox;
                    page.Blocks[b] = new Block
                    {
                        Label = page.Blocks[b].Label,
                        Bbox = new[] { w - bb[3], bb[0], w - bb[1], bb[2] },
                        Text = page.Blocks[b].Text,
                    };
                }
            }
            return page;
        }
        finally
        {
            if (owned)
            {
                pageMat.Dispose();
            }
        }
    }
}
