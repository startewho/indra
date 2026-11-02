using System.Diagnostics;
using Avalonia.Controls;
using Avalonia.Interactivity;
using IndraOcr.Desktop.Session;
using IndraOcr.Paddle;
using IndraOcr.Pipeline;

namespace IndraOcr.Desktop;

public partial class AboutWindow : Window
{
    private readonly OcrEngineKind _currentKind;
    private readonly string? _modelDir;
    private string _indraDir = "";
    private string _paddleDir = "";

    public AboutWindow() : this(OcrEngineKind.Indra, null)
    {
    }

    public AboutWindow(OcrEngineKind currentKind, string? modelDir)
    {
        InitializeComponent();
        _currentKind = currentKind;
        _modelDir = modelDir;
        ApplyLanguage();
    }

    private void ApplyLanguage()
    {
        Title = Loc.T("About", "关于");
        TitleText.Text = "Indra OCR";
        VersionText.Text = "v" + IndraOcrInfo.Version;
        IntroText.Text = Loc.T(
            "On-device OCR with two selectable engines: Indra (ONNX) and PaddleOCR PP-OCRv6. " +
            "Models are bundled with the application; no runtime download is required.",
            "本地 OCR，可切换两种引擎：Indra（ONNX）与 PaddleOCR PP-OCRv6。" +
            "模型随应用预置，运行时无需下载。");

        EngineCaption.Text = Loc.T("OCR ENGINES", "OCR 引擎");
        EngineText.Text = "Indra (ONNX) + PaddleOCR PP-OCRv6 small";
        IndraText.Text = Loc.T(
            "Indra: det + rec ONNX models, layout and reading-order heads, optional math-to-LaTeX.",
            "Indra：det + rec ONNX 模型，版面与阅读顺序，可选数学公式转 LaTeX。");
        PaddleText.Text = Loc.T(
            "PaddleOCR: PP-OCRv6 small det + rec, PP-DocLayout-S layout, XY-cut reading order.",
            "PaddleOCR：PP-OCRv6 small 检测 + 识别，PP-DocLayout-S 版面，XY-cut 阅读顺序。");
        RuntimeText.Text = Loc.T(
            "Runtime: ONNX Runtime CPU",
            "运行时：ONNX Runtime CPU");
        LicenseText.Text = Loc.T(
            "Models: OpenRAIL-M (Indra) · Apache-2.0 (PaddleOCR)",
            "模型许可：OpenRAIL-M（Indra）· Apache-2.0（PaddleOCR）");

        ModelCaption.Text = Loc.T("MODEL FILES (BUNDLED)", "模型文件（随应用预置）");
        IndraModelCaption.Text = Loc.T("Indra (ONNX)", "Indra（ONNX）");
        PaddleModelCaption.Text = Loc.T("PaddleOCR PP-OCRv6", "PaddleOCR PP-OCRv6");

        _indraDir = DirectoryFor(OcrEngineKind.Indra);
        _paddleDir = DirectoryFor(OcrEngineKind.PaddleOcr);
        IndraModelPathText.Text = _indraDir;
        PaddleModelPathText.Text = _paddleDir;
        IndraModelFilesText.Text = DescribeFolder(
            _indraDir, OcrSession.ModelsPresent(_indraDir, false, OcrEngineKind.Indra));
        PaddleModelFilesText.Text = DescribeFolder(
            _paddleDir, OcrSession.ModelsPresent(_paddleDir, false, OcrEngineKind.PaddleOcr));

        LinksCaption.Text = Loc.T("LINKS", "链接");
        CloseButton.Content = Loc.T("Close", "关闭");
    }

    private string DirectoryFor(OcrEngineKind kind)
    {
        if (kind == _currentKind && !string.IsNullOrWhiteSpace(_modelDir))
        {
            return Path.GetFullPath(_modelDir);
        }
        return OcrSession.DefaultModelDir(kind);
    }

    private static string DescribeFolder(string folder, bool ready)
    {
        if (!Directory.Exists(folder))
        {
            return Loc.T("Folder not found.", "目录不存在。");
        }
        var files = Directory.GetFiles(folder, "*", SearchOption.AllDirectories);
        long bytes = files.Sum(f => new FileInfo(f).Length);
        double mb = bytes / 1024.0 / 1024.0;
        string state = ready
            ? Loc.T("ready", "就绪")
            : Loc.T("required files missing", "缺少必需文件");
        return Loc.T(
            $"{state} · {files.Length} files · {mb:0.0} MB",
            $"{state} · {files.Length} 个文件 · {mb:0.0} MB");
    }

    private void OnOpenLink(object? sender, RoutedEventArgs e)
    {
        if (sender is Button { Tag: string url })
        {
            Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });
        }
    }

    private void OnClose(object? sender, RoutedEventArgs e) => Close();
}
