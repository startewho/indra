using System.Globalization;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using IndraOcr.Desktop.Session;

namespace IndraOcr.Desktop.Controls;

/// <summary>
/// The page image with line boxes, layout blocks, block labels and the reading
/// order drawn over it — the Avalonia version of the viewer's SVG overlay.
/// </summary>
public sealed class PageOverlay : Control
{
    private static readonly IBrush LineStroke = new SolidColorBrush(Color.Parse("#5B6CF0"), 0.55);
    private static readonly IBrush LineFill = new SolidColorBrush(Color.Parse("#5B6CF0"), 0.18);
    private static readonly IBrush OrderStroke = new SolidColorBrush(Color.Parse("#EA580C"), 0.8);
    private static readonly IBrush OrderFill = new SolidColorBrush(Color.Parse("#EA580C"));
    private static readonly IBrush LabelBackground = new SolidColorBrush(Colors.White, 0.92);
    private static readonly Typeface LabelTypeface = new("Consolas");

    private static readonly Dictionary<string, IBrush> BlockBrushes = new()
    {
        ["Text"] = new SolidColorBrush(Color.Parse("#6B7280")),
        ["Section-Header"] = new SolidColorBrush(Color.Parse("#1D4ED8")),
        ["Page-Header"] = new SolidColorBrush(Color.Parse("#9CA3AF")),
        ["Page-Footer"] = new SolidColorBrush(Color.Parse("#9CA3AF")),
        ["List-Group"] = new SolidColorBrush(Color.Parse("#16A34A")),
        ["Table"] = new SolidColorBrush(Color.Parse("#A21CAF")),
        ["Caption"] = new SolidColorBrush(Color.Parse("#0891B2")),
        ["Footnote"] = new SolidColorBrush(Color.Parse("#D97706")),
        ["Table-Of-Contents"] = new SolidColorBrush(Color.Parse("#0F766E")),
        ["Form"] = new SolidColorBrush(Color.Parse("#DC2626")),
        ["Picture"] = new SolidColorBrush(Color.Parse("#64748B")),
    };

    public Bitmap? Image { get; set; }
    public IReadOnlyList<LineView> Lines { get; set; } = Array.Empty<LineView>();
    public IReadOnlyList<BlockView> Blocks { get; set; } = Array.Empty<BlockView>();
    public bool ShowLines { get; set; } = true;
    public bool ShowBlocks { get; set; } = true;
    public bool ShowLabels { get; set; } = true;
    public bool ShowOrder { get; set; }
    public int HighlightedLine { get; set; } = -1;

    public void Refresh() => InvalidateVisual();

    protected override Size MeasureOverride(Size availableSize)
    {
        if (Image is null)
        {
            return new Size(0, 0);
        }
        double iw = Math.Max(1, Image.PixelSize.Width);
        double ih = Math.Max(1, Image.PixelSize.Height);
        double width = double.IsInfinity(availableSize.Width) ? iw : availableSize.Width;
        double height = width * ih / iw;
        if (!double.IsInfinity(availableSize.Height) && height > availableSize.Height)
        {
            height = availableSize.Height;
            width = height * iw / ih;
        }
        return new Size(Math.Max(1, width), Math.Max(1, height));
    }

    public override void Render(DrawingContext context)
    {
        if (Image is null)
        {
            return;
        }
        double iw = Math.Max(1, Image.PixelSize.Width);
        double ih = Math.Max(1, Image.PixelSize.Height);
        double scale = Math.Min(Bounds.Width / iw, Bounds.Height / ih);
        double ox = (Bounds.Width - iw * scale) / 2;
        double oy = (Bounds.Height - ih * scale) / 2;
        double unit = Math.Max(1.0, iw * scale / 640);

        context.DrawImage(Image, new Rect(ox, oy, iw * scale, ih * scale));

        double X(double v) => ox + v * scale;
        double Y(double v) => oy + v * scale;
        double S(double v) => v * scale;

        if (ShowBlocks)
        {
            foreach (var block in Blocks)
            {
                var brush = BlockBrushes.GetValueOrDefault(block.Label, BlockBrushes["Text"]);
                var pen = new Pen(brush, 2);
                context.DrawRectangle(null, pen, new Rect(
                    X(block.X0), Y(block.Y0),
                    Math.Max(1, S(block.X1 - block.X0)), Math.Max(1, S(block.Y1 - block.Y0))));
                if (ShowLabels)
                {
                    var text = Label(BlockText(block.Label), 11 * unit, brush);
                    double lx = Math.Max(ox, X(block.X0) - 2);
                    double ly = Math.Max(oy + text.Height, Y(block.Y0) - 5 * unit);
                    context.DrawRectangle(LabelBackground, null,
                        new Rect(lx - 2, ly - 1, text.Width + 4, text.Height + 2));
                    context.DrawText(text, new Point(lx, ly));
                }
            }
        }

        if (ShowLines)
        {
            for (int i = 0; i < Lines.Count; i++)
            {
                var line = Lines[i];
                bool on = i == HighlightedLine;
                context.DrawRectangle(
                    on ? LineFill : null,
                    new Pen(LineStroke, on ? 2.2 : 1),
                    new Rect(X(line.X0), Y(line.Y0),
                        Math.Max(1, S(line.X1 - line.X0)), Math.Max(1, S(line.Y1 - line.Y0))));
            }
        }

        if (ShowOrder && Lines.Count > 1)
        {
            var centers = Lines.Select(l => new Point(
                X((l.X0 + l.X1) / 2), Y((l.Y0 + l.Y1) / 2))).ToList();
            for (int i = 1; i < centers.Count; i++)
            {
                context.DrawLine(new Pen(OrderStroke, 1.4), centers[i - 1], centers[i]);
            }
            double radius = 7 * unit;
            for (int i = 0; i < centers.Count; i++)
            {
                context.DrawEllipse(OrderFill, null, centers[i], radius, radius);
                var num = Label((i + 1).ToString(CultureInfo.InvariantCulture), 8 * unit, Brushes.White);
                context.DrawText(num, new Point(
                    centers[i].X - num.Width / 2, centers[i].Y - num.Height / 2));
            }
        }
    }

    private static FormattedText Label(string text, double size, IBrush brush)
        => new(text, CultureInfo.InvariantCulture, FlowDirection.LeftToRight,
            LabelTypeface, Math.Max(7, size), brush);

    /// <summary>The layout block name in the UI language.</summary>
    public static string BlockText(string label) => label switch
    {
        "Text" => Loc.T("Text", "正文"),
        "Section-Header" => Loc.T("Section Header", "章节标题"),
        "Page-Header" => Loc.T("Header/Footer", "页眉/页脚"),
        "Page-Footer" => Loc.T("Header/Footer", "页眉/页脚"),
        "List-Group" => Loc.T("List Group", "列表"),
        "Table" => Loc.T("Table", "表格"),
        "Caption" => Loc.T("Caption", "图注"),
        "Footnote" => Loc.T("Footnote", "脚注"),
        "Table-Of-Contents" => Loc.T("Table Of Contents", "目录"),
        "Form" => Loc.T("Form", "表单"),
        "Picture" => Loc.T("Picture", "图片"),
        _ => label,
    };
}
