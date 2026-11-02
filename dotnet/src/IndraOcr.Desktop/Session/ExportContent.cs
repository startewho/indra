using System.Text;

namespace IndraOcr.Desktop.Session;

/// <summary>Builds the document-wide Markdown or plain-text export.</summary>
public static class ExportContent
{
    /// <summary>
    /// Concatenate every page in reading order. A single page is exported as-is;
    /// multi-page files get a page heading/separator so the boundaries stay clear.
    /// </summary>
    public static string Build(IEnumerable<PageResult> pages, bool plainText)
    {
        var ordered = pages.OrderBy(page => page.Index).ToList();
        if (ordered.Count == 0)
        {
            return "";
        }
        bool multi = ordered.Count > 1;
        var text = new StringBuilder();
        for (int i = 0; i < ordered.Count; i++)
        {
            var page = ordered[i];
            if (plainText)
            {
                if (multi)
                {
                    if (i > 0)
                    {
                        text.Append("\n\n");
                    }
                    text.Append("===== Page ").Append(page.Number).Append(" =====\n\n");
                }
                text.Append(page.Text.Trim());
            }
            else
            {
                if (i > 0)
                {
                    text.Append("\n\n---\n\n");
                }
                if (multi)
                {
                    text.Append("## Page ").Append(page.Number).Append("\n\n");
                }
                text.Append(page.Markdown.Trim());
            }
        }
        return text.ToString().TrimEnd() + "\n";
    }
}
