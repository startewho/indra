using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using Avalonia.Media.Imaging;

namespace IndraOcr.AndroidApp;

/// <summary>Page image with detected line and layout boxes for the mobile viewer.</summary>
public sealed class MobilePageOverlay : Control
{
    private static readonly IBrush LineStroke = new SolidColorBrush(Color.Parse("#5B6CF0"), 0.8);
    private static readonly IBrush LineFill = new SolidColorBrush(Color.Parse("#5B6CF0"), 0.20);
    private static readonly IBrush BlockStroke = new SolidColorBrush(Color.Parse("#16A34A"), 0.8);

    public Bitmap? Image { get; set; }
    public IReadOnlyList<MobileLine> Lines { get; set; } = Array.Empty<MobileLine>();
    public IReadOnlyList<MobileBlock> Blocks { get; set; } = Array.Empty<MobileBlock>();

    public void Refresh() => InvalidateVisual();

    protected override Size MeasureOverride(Size availableSize)
    {
        if (Image is null)
        {
            return new Size(0, 0);
        }
        double iw = Math.Max(1, Image.PixelSize.Width);
        double ih = Math.Max(1, Image.PixelSize.Height);
        // Fill the available width and let the page scroll vertically; do not
        // shrink a tall page just because the viewport is short.
        double width = double.IsInfinity(availableSize.Width) ? iw : availableSize.Width;
        return new Size(Math.Max(1, width), Math.Max(1, width * ih / iw));
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

        foreach (var block in Blocks)
        {
            context.DrawRectangle(null, new Pen(BlockStroke, 1.4),
                new Rect(X(block.X0), Y(block.Y0),
                    Math.Max(1, S(block.X1 - block.X0)), Math.Max(1, S(block.Y1 - block.Y0))));
        }

        foreach (var line in Lines)
        {
            context.DrawRectangle(LineFill, new Pen(LineStroke, 1),
                new Rect(X(line.X0), Y(line.Y0),
                    Math.Max(1, S(line.X1 - line.X0)), Math.Max(1, S(line.Y1 - line.Y0))));
        }
    }
}
