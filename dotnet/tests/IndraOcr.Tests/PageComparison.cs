using System.Text.Json;
using IndraOcr.Pipeline;

namespace IndraOcr.Tests;

internal sealed record GoldenLine(float[] Box, string Text, double Conf);

internal sealed record GoldenBlock(string Label, float[] Bbox, string Text);

internal sealed record GoldenPage(
    string File, int Page, string Source, int Rotation, double Skew, bool Vertical,
    string Note, GoldenLine[] Lines, GoldenBlock[] Blocks, string? Markdown);

/// <summary>The golden-page comparison shared by the image and PDF parity tests.</summary>
internal static class PageComparison
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNameCaseInsensitive = true,
    };

    public static GoldenPage Load(string path)
        => JsonSerializer.Deserialize<GoldenPage>(File.ReadAllText(path), JsonOpts)!;

    public static void AssertMatches(GoldenPage golden, Page page, string name)
    {
        Assert.True(golden.Source == page.Source, $"{name} source: expected {golden.Source}, got {page.Source}");
        Assert.True(golden.Rotation == page.Rotation, $"{name} rotation: expected {golden.Rotation}, got {page.Rotation}");
        Assert.True(golden.Vertical == page.Vertical, $"{name} vertical: expected {golden.Vertical}, got {page.Vertical}");
        Assert.True(golden.Note == page.Note, $"{name} note: expected '{golden.Note}', got '{page.Note}'");
        Assert.True(Math.Abs(golden.Skew - page.Skew) <= 0.05,
            $"{name} skew: expected {golden.Skew}, got {page.Skew}");

        Assert.True(golden.Lines.Length == page.Lines.Count,
            $"{name} line count: expected {golden.Lines.Length}, got {page.Lines.Count}");
        for (int i = 0; i < page.Lines.Count; i++)
        {
            var g = golden.Lines[i];
            var l = page.Lines[i];
            Assert.True(g.Text == l.Text,
                $"{name} line {i} text: expected '{g.Text}', got '{l.Text}'");
            // Page confidences are rounded to 4 decimals by both implementations;
            // a last-bit difference in the float32 mean can land on either side of
            // a rounding boundary, so allow one more display step.
            Assert.True(Math.Abs(g.Conf - l.Conf) <= 2.5e-4,
                $"{name} line {i} conf: expected {g.Conf}, got {l.Conf}");
            for (int k = 0; k < 4; k++)
            {
                Assert.True(Math.Abs(g.Box[k] - l.Box[k]) <= 0.5,
                    $"{name} line {i} box[{k}]: expected {g.Box[k]}, got {l.Box[k]}");
            }
        }

        Assert.True(golden.Blocks.Length == page.Blocks.Count,
            $"{name} block count: expected {golden.Blocks.Length}, got {page.Blocks.Count}");
        for (int i = 0; i < page.Blocks.Count; i++)
        {
            var g = golden.Blocks[i];
            var b = page.Blocks[i];
            Assert.True(g.Label == b.Label,
                $"{name} block {i} label: expected {g.Label}, got {b.Label}");
            Assert.True(g.Text == b.Text,
                $"{name} block {i} text: expected '{g.Text}', got '{b.Text}'");
            for (int k = 0; k < 4; k++)
            {
                Assert.True(Math.Abs(g.Bbox[k] - b.Bbox[k]) <= 0.5,
                    $"{name} block {i} bbox[{k}]: expected {g.Bbox[k]}, got {b.Bbox[k]}");
            }
        }

        Assert.True((golden.Markdown ?? "") == (page.Markdown ?? ""),
            $"{name} markdown differs:\nexpected:\n{golden.Markdown}\n\ngot:\n{page.Markdown}");
    }
}
