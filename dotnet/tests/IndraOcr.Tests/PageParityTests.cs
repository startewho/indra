using IndraOcr.Imaging;
using IndraOcr.Pipeline;

namespace IndraOcr.Tests;

/// <summary>
/// S3 parity: the whole page (orientation, deskew, detection, recognition,
/// reading order, layout, markdown) against tools/dump_golden.py.
/// </summary>
public class PageParityTests
{
    [SkippableFact]
    public void EveryGoldenPageMatchesPython()
    {
        Skip.If(TestPaths.Models is null, "models not found (set INDRA_OCR_MODEL_DIR)");
        var files = Directory.Exists(TestPaths.Golden)
            ? Directory.GetFiles(TestPaths.Golden, "page-*.json", SearchOption.AllDirectories)
            : Array.Empty<string>();
        Skip.If(files.Length == 0, "no golden pages (run tools/dump_golden.py)");

        using var ocr = new OrtOcr(TestPaths.Models!, threads: 1);
        foreach (string path in files.OrderBy(p => p, StringComparer.Ordinal))
        {
            var golden = PageComparison.Load(path);
            string name = Path.GetRelativePath(TestPaths.Golden, path);
            using var gray = ImageOps.LoadGray(TestPaths.Full(golden.File));
            var page = ocr.Read(gray);
            PageComparison.AssertMatches(golden, page, name);
        }
    }
}
