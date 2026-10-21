using System.Net;
using System.Text;
using System.Text.RegularExpressions;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Documents;
using Avalonia.Layout;
using Avalonia.Media;

namespace IndraOcr.Desktop.Controls;

/// <summary>
/// The same small markdown subset the web viewer renders: headings, paragraphs,
/// lists, HTML tables from the OCR layout, and $math$ as monospace inline code.
/// </summary>
public static class MarkdownRenderer
{
    private static readonly IBrush Ink = new SolidColorBrush(Color.Parse("#111827"));
    private static readonly IBrush Ink2 = new SolidColorBrush(Color.Parse("#374151"));
    private static readonly IBrush Ink3 = new SolidColorBrush(Color.Parse("#6B7280"));
    private static readonly IBrush Accent = new SolidColorBrush(Color.Parse("#1D4ED8"));
    private static readonly IBrush Rule = new SolidColorBrush(Color.Parse("#E5E7EB"));
    private static readonly IBrush HeaderBackground = new SolidColorBrush(Color.Parse("#F3F4F6"));
    private static readonly FontFamily Mono = new("Consolas");
    private static readonly Regex Heading = new(@"^(#{1,6})\s+(.*)$");
    private static readonly Regex ListItem = new(@"^\s*(?:[-*•]|(\d+)[.)])\s+(.*)$");
    private static readonly Regex InlinePattern = new(@"\*\*(.+?)\*\*|\$\$(.+?)\$\$|\$(.+?)\$|`([^`]+)`", RegexOptions.Singleline);
    private static readonly Regex TableRow = new(@"<tr[^>]*>(.*?)</tr>", RegexOptions.IgnoreCase | RegexOptions.Singleline);
    private static readonly Regex TableCell = new(@"<t([dh])([^>]*)>(.*?)</t\1>", RegexOptions.IgnoreCase | RegexOptions.Singleline);
    private static readonly Regex ColSpan = new(@"colspan\s*=\s*[""']?(\d+)", RegexOptions.IgnoreCase);
    private static readonly Regex Tag = new(@"<[^>]+>");

    public static Control Build(string? markdown)
    {
        var panel = new StackPanel { Spacing = 6 };
        string text = (markdown ?? "").Replace("\r\n", "\n");
        if (string.IsNullOrWhiteSpace(text))
        {
            panel.Children.Add(new TextBlock
            {
                Text = "(no text)",
                Foreground = Ink3,
                FontStyle = FontStyle.Italic,
            });
            return panel;
        }

        var lines = text.Split('\n');
        var paragraph = new List<string>();

        void FlushParagraph()
        {
            if (paragraph.Count == 0)
            {
                return;
            }
            panel.Children.Add(Paragraph(string.Join(" ", paragraph)));
            paragraph.Clear();
        }

        for (int i = 0; i < lines.Length; i++)
        {
            string line = lines[i].TrimEnd();
            string trimmed = line.TrimStart();
            if (trimmed.StartsWith("<table", StringComparison.OrdinalIgnoreCase))
            {
                FlushParagraph();
                var html = new StringBuilder();
                while (i < lines.Length)
                {
                    html.AppendLine(lines[i]);
                    if (lines[i].Contains("</table>", StringComparison.OrdinalIgnoreCase))
                    {
                        break;
                    }
                    i++;
                }
                panel.Children.Add(Table(html.ToString()));
                continue;
            }
            if (line.Length == 0)
            {
                FlushParagraph();
                continue;
            }
            var heading = Heading.Match(line);
            if (heading.Success)
            {
                FlushParagraph();
                int level = Math.Min(3, heading.Groups[1].Value.Length);
                panel.Children.Add(new TextBlock
                {
                    Text = heading.Groups[2].Value,
                    FontSize = level switch { 1 => 20, 2 => 17, _ => 15 },
                    FontWeight = FontWeight.SemiBold,
                    Foreground = Ink,
                    Margin = new Thickness(0, 8, 0, 2),
                    TextWrapping = TextWrapping.Wrap,
                });
                continue;
            }
            var item = ListItem.Match(line);
            if (item.Success)
            {
                FlushParagraph();
                panel.Children.Add(ListRow(
                    item.Groups[1].Success ? item.Groups[1].Value + "." : "•",
                    item.Groups[2].Value));
                continue;
            }
            paragraph.Add(line);
        }
        FlushParagraph();
        return panel;
    }

    private static Control Paragraph(string text)
    {
        var block = new SelectableTextBlock
        {
            TextWrapping = TextWrapping.Wrap,
            FontSize = 14,
            LineHeight = 22,
            Foreground = Ink2,
        };
        Inline(block, text);
        return block;
    }

    private static Control ListRow(string marker, string text)
    {
        var grid = new Grid
        {
            ColumnDefinitions = new ColumnDefinitions("Auto,*"),
            Margin = new Thickness(0, 1, 0, 1),
        };
        var bullet = new TextBlock
        {
            Text = marker,
            FontFamily = Mono,
            FontSize = 12.5,
            Foreground = Ink3,
            Margin = new Thickness(4, 3, 8, 0),
            VerticalAlignment = VerticalAlignment.Top,
        };
        var body = new SelectableTextBlock
        {
            TextWrapping = TextWrapping.Wrap,
            FontSize = 14,
            LineHeight = 22,
            Foreground = Ink2,
        };
        Inline(body, text);
        Grid.SetColumn(body, 1);
        grid.Children.Add(bullet);
        grid.Children.Add(body);
        return grid;
    }

    private static void Inline(SelectableTextBlock block, string text)
    {
        int pos = 0;
        foreach (Match match in InlinePattern.Matches(text))
        {
            if (match.Index > pos)
            {
                block.Inlines!.Add(new Run { Text = text[pos..match.Index] });
            }
            if (match.Groups[1].Success)
            {
                block.Inlines!.Add(new Run
                {
                    Text = match.Groups[1].Value,
                    FontWeight = FontWeight.SemiBold,
                });
            }
            else if (match.Groups[2].Success || match.Groups[3].Success)
            {
                string math = match.Groups[2].Success ? match.Groups[2].Value : match.Groups[3].Value;
                block.Inlines!.Add(new Run
                {
                    Text = math,
                    FontFamily = Mono,
                    Foreground = Accent,
                });
            }
            else
            {
                block.Inlines!.Add(new Run
                {
                    Text = match.Groups[4].Value,
                    FontFamily = Mono,
                });
            }
            pos = match.Index + match.Length;
        }
        if (pos < text.Length)
        {
            block.Inlines!.Add(new Run { Text = text[pos..] });
        }
    }

    private static Control Table(string html)
    {
        var rows = TableRow.Matches(html).Select(row => TableCell.Matches(row.Groups[1].Value)).ToList();
        if (rows.Count == 0)
        {
            return Paragraph("");
        }
        int columns = 1;
        foreach (var row in rows)
        {
            int count = 0;
            foreach (Match cell in row)
            {
                var span = ColSpan.Match(cell.Groups[2].Value);
                count += span.Success ? int.Parse(span.Groups[1].Value) : 1;
            }
            columns = Math.Max(columns, count);
        }
        var grid = new Grid
        {
            ColumnDefinitions = new ColumnDefinitions(string.Join(",", Enumerable.Repeat("*", columns))),
            Margin = new Thickness(0, 6, 0, 6),
        };
        for (int r = 0; r < rows.Count; r++)
        {
            grid.RowDefinitions.Add(new RowDefinition(GridLength.Auto));
            int column = 0;
            for (int c = 0; c < rows[r].Count && column < columns; c++)
            {
                var cell = rows[r][c];
                bool header = cell.Groups[1].Value.Equals("h", StringComparison.OrdinalIgnoreCase);
                var span = ColSpan.Match(cell.Groups[2].Value);
                int colspan = span.Success
                    ? Math.Min(int.Parse(span.Groups[1].Value), columns - column)
                    : 1;
                string value = WebUtility.HtmlDecode(Tag.Replace(cell.Groups[3].Value, "")).Trim();
                var text = new SelectableTextBlock
                {
                    Text = value,
                    TextWrapping = TextWrapping.Wrap,
                    FontSize = 12.5,
                    FontWeight = header ? FontWeight.SemiBold : FontWeight.Normal,
                    Foreground = header ? Ink : Ink2,
                };
                var border = new Border
                {
                    BorderBrush = Rule,
                    BorderThickness = new Thickness(0.5),
                    Padding = new Thickness(7, 4),
                    Background = header ? HeaderBackground : null,
                    Child = text,
                };
                Grid.SetRow(border, r);
                Grid.SetColumn(border, column);
                Grid.SetColumnSpan(border, Math.Max(1, colspan));
                grid.Children.Add(border);
                column += Math.Max(1, colspan);
            }
        }
        return grid;
    }
}
