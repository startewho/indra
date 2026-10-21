namespace IndraOcr.Desktop.Session;

/// <summary>Pipeline options the sidebar sends when a document is read.</summary>
public sealed record ReadOptions(
    string Mode,
    bool Math,
    bool Furniture,
    bool ImageOcr,
    double Dpi,
    int MaxPages);

/// <summary>One recognized line in page-image pixels (already scaled for display).</summary>
public sealed record LineView(double X0, double Y0, double X1, double Y1, string Text, double Conf);

/// <summary>One layout block in page-image pixels (already scaled for display).</summary>
public sealed record BlockView(
    string Label, double X0, double Y0, double X1, double Y1, string Text);

/// <summary>One page ready for the UI: bitmap, boxes, markdown, JSON and timings.</summary>
public sealed class PageResult
{
    public required int Index { get; init; }
    public int Number => Index + 1;
    public required byte[] ImageJpeg { get; init; }
    public required IReadOnlyList<LineView> Lines { get; init; }
    public required IReadOnlyList<BlockView> Blocks { get; init; }
    public required string Markdown { get; init; }
    public required string Json { get; init; }
    public required string Source { get; init; }
    public required string Note { get; init; }
    public required int Rotation { get; init; }
    public required double Skew { get; init; }
    public required bool Vertical { get; init; }
    public required IReadOnlyDictionary<string, double> Timings { get; init; }
    public required double TotalMs { get; init; }
}
