using Avalonia;
using IndraOcr.Desktop.Controls;
using IndraOcr.Desktop.Session;

namespace IndraOcr.Tests;

public class PageOverlayTests
{
    private static readonly LineView[] Lines =
    {
        new(10, 10, 50, 20, "first", 1),
        new(60, 10, 90, 20, "second", 1),
    };

    [Fact]
    public void HitTestMapsThroughLetterboxing()
    {
        var bounds = new Size(200, 100);
        var image = new PixelSize(100, 100);

        // The image is centred: 50 px of empty space on both sides.
        Assert.Equal(0, PageOverlay.HitTestLine(new Point(50 + 30, 15), bounds, image, Lines));
        Assert.Equal(1, PageOverlay.HitTestLine(new Point(50 + 75, 15), bounds, image, Lines));
        Assert.Equal(-1, PageOverlay.HitTestLine(new Point(50 + 55, 15), bounds, image, Lines));
        Assert.Equal(-1, PageOverlay.HitTestLine(new Point(10, 15), bounds, image, Lines));
    }

    [Fact]
    public void HitTestPrefersTheSmallestOverlappingBox()
    {
        var lines = new[]
        {
            new LineView(10, 10, 50, 20, "block", 1),
            new LineView(20, 12, 30, 18, "inner", 1),
        };
        var bounds = new Size(100, 100);
        var image = new PixelSize(100, 100);

        Assert.Equal(1, PageOverlay.HitTestLine(new Point(25, 15), bounds, image, lines));
        Assert.Equal(0, PageOverlay.HitTestLine(new Point(45, 15), bounds, image, lines));
    }
}
