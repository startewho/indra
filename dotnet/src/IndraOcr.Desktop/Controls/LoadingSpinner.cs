using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using Avalonia.Threading;

namespace IndraOcr.Desktop.Controls;

/// <summary>A small ring of fading dots, shown while a page is being read.</summary>
public sealed class LoadingSpinner : Control
{
    private const int Dots = 10;
    private static readonly IBrush DotBrush = new SolidColorBrush(Color.Parse("#1D4ED8"));
    private readonly DispatcherTimer _timer;
    private bool _active;
    private int _step;

    public LoadingSpinner()
    {
        Width = 40;
        Height = 40;
        _timer = new DispatcherTimer(TimeSpan.FromMilliseconds(70), DispatcherPriority.Render, (_, _) =>
        {
            _step = (_step + 1) % Dots;
            InvalidateVisual();
        });
    }

    public bool IsActive
    {
        get => _active;
        set
        {
            _active = value;
            if (value)
            {
                _timer.Start();
            }
            else
            {
                _timer.Stop();
            }
            InvalidateVisual();
        }
    }

    public override void Render(DrawingContext context)
    {
        double size = Math.Max(8, Math.Min(Bounds.Width, Bounds.Height));
        double radius = size / 2 - 4;
        var center = new Point(Bounds.Width / 2, Bounds.Height / 2);
        for (int i = 0; i < Dots; i++)
        {
            double angle = (i * 360.0 / Dots - 90) * Math.PI / 180;
            var p = new Point(center.X + radius * Math.Cos(angle), center.Y + radius * Math.Sin(angle));
            int age = (i - _step + Dots) % Dots;
            double opacity = 0.12 + 0.88 * (Dots - age) / (double)Dots;
            var brush = new SolidColorBrush(Color.Parse("#1D4ED8"), opacity);
            context.DrawEllipse(brush, null, p, size / 18, size / 18);
        }
    }
}
