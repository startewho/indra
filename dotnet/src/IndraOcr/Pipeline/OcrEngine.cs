using OpenCvSharp;

namespace IndraOcr.Pipeline;

public enum OcrEngineKind
{
    Indra,
    PaddleOcr,
}

[Flags]
public enum OcrCapabilities
{
    None = 0,
    Images = 1 << 0,
    PdfRenderedPages = 1 << 1,
    PdfTextLayer = 1 << 2,
    Math = 1 << 3,
    LayoutBlocks = 1 << 4,
    Vertical = 1 << 5,
    ModelDownload = 1 << 6,
}

/// <summary>The page-level OCR contract shared by all engines.</summary>
public interface IOcrEngine : IDisposable
{
    OcrEngineKind Kind { get; }
    string DisplayName { get; }
    OcrCapabilities Capabilities { get; }
    Page Read(Mat gray, bool turn = true);
}

/// <summary>
/// Low-level Indra operations used by PDF text-layer matching and math.  Only
/// the Indra engine implements this; PaddleOCR uses the high-level Read path.
/// </summary>
public interface IIndraOcrDetails
{
    Vocab.Vocab Vocab { get; }
    IReadOnlyDictionary<string, int> BaseVocab { get; }
    IEnumerable<IReadOnlyDictionary<string, int>> GroupVocabs { get; }
    int LineHeight { get; }
    MathOcr.MathReader? MathRec { get; }

    float[] Detect(Mat gray);
    RecLine[] Recognize(Mat gray, float[] boxes, int batch = 64, bool vertical = false);
    (string[] Groups, bool[] Sure) BoxGroups(Mat gray, float[] boxes);
    Page Assemble(int width, int height, float[] boxes, RecLine[] lines,
        Dictionary<string, double> timings, bool vertical = false, float[]? pictures = null);
}
