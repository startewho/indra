using IndraOcr.Pipeline;

namespace IndraOcr.Desktop;

/// <summary>Which OCR engine the desktop app last used.</summary>
public static class EnginePreference
{
    private static readonly string Folder = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "indra-ocr");

    private static readonly string FilePath = Path.Combine(Folder, "desktop-engine.txt");

    public static OcrEngineKind Load()
    {
        try
        {
            string value = File.ReadAllText(FilePath).Trim();
            return value.Equals("paddleocr", StringComparison.OrdinalIgnoreCase)
                ? OcrEngineKind.PaddleOcr
                : OcrEngineKind.Indra;
        }
        catch (IOException)
        {
            return OcrEngineKind.Indra;
        }
        catch (UnauthorizedAccessException)
        {
            return OcrEngineKind.Indra;
        }
    }

    public static void Save(OcrEngineKind kind)
    {
        try
        {
            Directory.CreateDirectory(Folder);
            File.WriteAllText(FilePath, kind == OcrEngineKind.PaddleOcr ? "paddleocr" : "indra");
        }
        catch (IOException)
        {
            // keep the session choice
        }
        catch (UnauthorizedAccessException)
        {
            // ignore
        }
    }
}
