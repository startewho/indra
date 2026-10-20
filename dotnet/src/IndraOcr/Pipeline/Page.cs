using IndraOcr.Layout;

namespace IndraOcr.Pipeline;

/// <summary>One recognized line: box in page pixels, text and confidence.</summary>
public sealed class Line
{
    public required float[] Box { get; set; }
    public required string Text { get; set; }
    public required double Conf { get; set; }
}

/// <summary>One page: lines in reading order, timings, markdown and layout blocks.</summary>
public sealed class Page
{
    public List<Line> Lines { get; init; } = new();
    public Dictionary<string, double> Timings { get; init; } = new();
    public string? Markdown { get; set; }
    public string Source { get; set; } = "ocr";
    public string Note { get; set; } = "";
    public List<Block> Blocks { get; } = new();
    public int Rotation { get; set; }
    public double Skew { get; set; }
    public bool Vertical { get; set; }

    /// <summary>One line per detected line, in reading order.</summary>
    public string Text => string.Join("\n", Lines.Select(l => l.Text));

    /// <summary>Picture regions on the page, in reading order.</summary>
    public List<float[]> Pictures => Blocks.Where(b => b.Label == "Picture").Select(b => b.Bbox).ToList();
}

/// <summary>The layout heads of the order model, if it has them.</summary>
public sealed class LayoutHeads
{
    public required float[] Cls { get; init; }
    public required int NClasses { get; init; }
    public required float[] Same { get; init; }
    public float[]? Tab { get; init; }
}
