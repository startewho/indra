using System.Text.Json;
using IndraOcr.Layout;
using IndraOcr.Numerics;
using IndraOcr.Ordering;
using IndraOcr.Rtl;

namespace IndraOcr.Tests;

/// <summary>Pure-function parity: bidi flipping, reading order, layout text (tools/dump_stage_fixtures.py).</summary>
public class StageParityTests
{
    private sealed record RtlCase(string Text, string Auto, string Rtl, string Ltr);

    private sealed record OrderCase(
        float[] Page, float[][] Boxes, string[] Texts, float[][] Features,
        int[] Xycut, int[][] Blocks, int[] Decode, int[][] Cluster,
        double[][] A, double[][] Same);

    private sealed record JoinCase(string[] Lines, string Result);

    private sealed record TableCase(float[][] Boxes, string[] Texts, string Html);

    private sealed record LayoutCase(JoinCase[] Join, TableCase Table);

    private sealed record Fixture(RtlCase[] Rtl, OrderCase Order, LayoutCase Layout);

    private static Fixture Load()
    {
        string path = Path.Combine(TestPaths.Root, "dotnet", "tests", "parity", "stage", "fixtures.json");
        Skip.IfNot(File.Exists(path), "no stage fixtures (run tools/dump_stage_fixtures.py)");
        return JsonSerializer.Deserialize<Fixture>(
            File.ReadAllText(path),
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true })!;
    }

    [SkippableFact]
    public void BidiFlipMatchesPython()
    {
        var f = Load();
        foreach (var c in f.Rtl)
        {
            Assert.True(c.Auto == Bidi.Flip(c.Text), $"flip auto: {c.Text}");
            Assert.True(c.Rtl == Bidi.Flip(c.Text, baseRtl: true), $"flip rtl: {c.Text}");
            Assert.True(c.Ltr == Bidi.Flip(c.Text, baseRtl: false), $"flip ltr: {c.Text}");
        }
    }

    [SkippableFact]
    public void ReadingOrderMatchesPython()
    {
        var f = Load();
        var o = f.Order;
        int n = o.Boxes.Length;
        var boxes = FlattenF(o.Boxes);
        var feats = OrderPost.LineFeatures(boxes, n, o.Page[0], o.Page[1], o.Texts);
        var expectedFeats = FlattenF(o.Features);
        for (int i = 0; i < feats.Length; i++)
        {
            Assert.True(Math.Abs(feats[i] - expectedFeats[i]) <= 1e-5,
                $"feature {i}: expected {expectedFeats[i]}, got {feats[i]}");
        }

        Assert.True(o.Xycut.SequenceEqual(OrderPost.OrderXycut(boxes, n)), "order_xycut");
        var blocks = OrderPost.XycutBlocks(boxes, n);
        Assert.Equal(o.Blocks.Length, blocks.Count);
        for (int i = 0; i < blocks.Count; i++)
        {
            Assert.True(o.Blocks[i].SequenceEqual(blocks[i]), $"xycut block {i}");
        }

        var a = FlattenD(o.A);
        Assert.True(o.Decode.SequenceEqual(OrderPost.DecodeOrder(a, n)), "decode_order");

        var same = FlattenToDouble(o.Same);
        var cluster = OrderPost.Cluster(same.Select(v => (float)v).ToArray(), n);
        Assert.Equal(o.Cluster.Length, cluster.Count);
        for (int i = 0; i < cluster.Count; i++)
        {
            Assert.True(o.Cluster[i].SequenceEqual(cluster[i]), $"cluster {i}");
        }
    }

    [SkippableFact]
    public void LayoutTextMatchesPython()
    {
        var f = Load();
        foreach (var j in f.Layout.Join)
        {
            Assert.True(j.Result == LayoutPost.JoinLines(j.Lines), $"join_lines: {string.Join(" | ", j.Lines)}");
        }
        var t = f.Layout.Table;
        int n = t.Boxes.Length;
        var html = LayoutPost.TableHtml(FlattenF(t.Boxes), t.Texts, n);
        Assert.True(t.Html == html, $"table_html:\nexpected:\n{t.Html}\n\ngot:\n{html}");
    }

    private static float[] FlattenF(float[][] rows)
    {
        var flat = new float[rows.Length * rows[0].Length];
        for (int i = 0; i < rows.Length; i++)
        {
            Array.Copy(rows[i], 0, flat, i * rows[i].Length, rows[i].Length);
        }
        return flat;
    }

    private static double[] FlattenD(double[][] rows)
    {
        var flat = new double[rows.Length * rows[0].Length];
        for (int i = 0; i < rows.Length; i++)
        {
            Array.Copy(rows[i], 0, flat, i * rows[i].Length, rows[i].Length);
        }
        return flat;
    }

    private static double[] FlattenToDouble(double[][] rows)
    {
        var flat = new double[rows.Length * rows[0].Length];
        for (int i = 0; i < rows.Length; i++)
        {
            for (int j = 0; j < rows[i].Length; j++)
            {
                flat[i * rows[i].Length + j] = rows[i][j];
            }
        }
        return flat;
    }
}
