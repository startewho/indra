using System.Globalization;

namespace IndraOcr.Desktop;

/// <summary>
/// UI language: English or Chinese, defaulting to the system language.  The
/// choice is remembered in %LOCALAPPDATA%\indra-ocr\desktop-language.txt.
/// Strings are passed as (English, Chinese) pairs at the call site, which keeps
/// the desktop app NativeAOT-friendly (no resource reflection).
/// </summary>
public static class Loc
{
    private static readonly string Folder = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "indra-ocr");

    private static readonly string FilePath = Path.Combine(Folder, "desktop-language.txt");

    static Loc()
    {
        Chinese = Load();
    }

    public static bool Chinese { get; private set; }

    public static event Action? Changed;

    public static void Set(bool chinese)
    {
        if (Chinese == chinese)
        {
            return;
        }
        Chinese = chinese;
        Save();
        Changed?.Invoke();
    }

    public static string T(string english, string chinese) => Chinese ? chinese : english;

    private static bool Load()
    {
        try
        {
            string value = File.ReadAllText(FilePath).Trim();
            if (value.Equals("zh", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
            if (value.Equals("en", StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }
        }
        catch (IOException)
        {
            // first run, or no access: use the system language
        }
        catch (UnauthorizedAccessException)
        {
            // ignore
        }
        return CultureInfo.CurrentUICulture.TwoLetterISOLanguageName
            .Equals("zh", StringComparison.OrdinalIgnoreCase);
    }

    private static void Save()
    {
        try
        {
            Directory.CreateDirectory(Folder);
            File.WriteAllText(FilePath, Chinese ? "zh" : "en");
        }
        catch (IOException)
        {
            // keep the session choice even if the file cannot be written
        }
        catch (UnauthorizedAccessException)
        {
            // ignore
        }
    }
}
