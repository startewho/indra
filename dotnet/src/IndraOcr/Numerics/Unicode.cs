using System.Text;

namespace IndraOcr.Numerics;

/// <summary>Unicode properties .NET does not expose, from the generated tables.</summary>
public static class Unicode
{
    public enum Bidi : byte
    {
        Other = 0,
        L = 1,
        R = 2,
        AL = 3,
        EN = 4,
        AN = 5,
    }

    /// <summary>Bidi class of a code point (Python unicodedata.bidirectional).</summary>
    public static Bidi BidiClass(int codePoint)
    {
        var ranges = UnicodeTables.BidiRanges;
        int lo = 0, hi = ranges.Length / 3 - 1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >> 1;
            int start = ranges[mid * 3], end = ranges[mid * 3 + 1];
            if (codePoint < start)
            {
                hi = mid - 1;
            }
            else if (codePoint > end)
            {
                lo = mid + 1;
            }
            else
            {
                return (Bidi)ranges[mid * 3 + 2];
            }
        }
        return Bidi.Other;
    }

    /// <summary>Right-to-left letter (Arabic, Hebrew, Syriac, Thaana, N'Ko, ...).</summary>
    public static bool IsRtl(int codePoint) => BidiClass(codePoint) is Bidi.R or Bidi.AL;

    /// <summary>
    /// Left-to-right unit inside RTL text: Latin letters and European or
    /// Arabic-Indic digits (digit sequences read left to right in both).
    /// </summary>
    public static bool IsLtr(int codePoint) => BidiClass(codePoint) is Bidi.L or Bidi.EN or Bidi.AN;

    /// <summary>East Asian Width W or F: a wide character counts twice.</summary>
    public static bool IsWide(int codePoint)
    {
        var ranges = UnicodeTables.WideRanges;
        int lo = 0, hi = ranges.Length / 2 - 1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >> 1;
            if (codePoint < ranges[mid * 2])
            {
                hi = mid - 1;
            }
            else if (codePoint > ranges[mid * 2 + 1])
            {
                lo = mid + 1;
            }
            else
            {
                return true;
            }
        }
        return false;
    }

    public static bool IsLetter(char c) => char.IsLetter(c);

    public static bool IsDigit(char c) => char.IsDigit(c);

    /// <summary>Python str.isalpha, per UTF-16 code unit (callers use BMP text).</summary>
    public static bool IsAlpha(char c) => char.IsLetter(c);

    /// <summary>NFC/NFD/NFKC, with the surrogate-safe .NET implementation.</summary>
    public static string Normalize(string text, NormalizationForm form) => text.Normalize(form);

    /// <summary>Iterate code points (Python iterates code points, not UTF-16 units).</summary>
    public static IEnumerable<int> CodePoints(string text)
    {
        for (int i = 0; i < text.Length; i++)
        {
            if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
            {
                yield return char.ConvertToUtf32(text[i], text[i + 1]);
                i++;
            }
            else
            {
                yield return text[i];
            }
        }
    }
}
