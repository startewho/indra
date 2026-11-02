using Avalonia.Controls;
using Avalonia.Interactivity;

namespace IndraOcr.Desktop;

/// <summary>Asks whether all pages should be exported as Markdown or plain text.</summary>
public partial class ExportFormatWindow : Window
{
    public ExportFormatWindow()
    {
        InitializeComponent();
        Title = Loc.T("Export format", "导出格式");
        TitleText.Text = Loc.T("Export all pages", "导出所有页面");
        IntroText.Text = Loc.T(
            "Choose the file format for the recognized pages.",
            "选择识别结果的导出格式。");
        MarkdownButton.Content = "Markdown (.md)";
        TextButton.Content = Loc.T("Plain text (.txt)", "纯文本 (.txt)");
        CancelButton.Content = Loc.T("Cancel", "取消");
    }

    private void OnMarkdown(object? sender, RoutedEventArgs e) => Close("md");

    private void OnText(object? sender, RoutedEventArgs e) => Close("txt");

    private void OnCancel(object? sender, RoutedEventArgs e) => Close(null);
}
