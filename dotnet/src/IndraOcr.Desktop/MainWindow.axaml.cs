using System.Diagnostics;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Documents;
using Avalonia.Controls.Primitives;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Layout;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using Avalonia.Platform.Storage;
using IndraOcr.Desktop.Controls;
using IndraOcr.Desktop.Session;
using IndraOcr.Models;

namespace IndraOcr.Desktop;

public partial class MainWindow : Window
{
    private static readonly IBrush Ink = new SolidColorBrush(Color.Parse("#111827"));
    private static readonly IBrush Ink2 = new SolidColorBrush(Color.Parse("#374151"));
    private static readonly IBrush Ink3 = new SolidColorBrush(Color.Parse("#6B7280"));
    private static readonly IBrush Rule = new SolidColorBrush(Color.Parse("#E5E7EB"));
    private static readonly IBrush Surface = new SolidColorBrush(Color.Parse("#F7F8FA"));
    private static readonly IBrush Surface2 = new SolidColorBrush(Color.Parse("#EEF0F4"));
    private static readonly IBrush Accent = new SolidColorBrush(Color.Parse("#1D4ED8"));
    private static readonly IBrush AccentSoft = new SolidColorBrush(Color.Parse("#EEF2FF"));
    private static readonly IBrush Warning = new SolidColorBrush(Color.Parse("#B45309"));
    private static readonly IBrush Ok = new SolidColorBrush(Color.Parse("#15803D"));
    private static readonly IBrush Error = new SolidColorBrush(Color.Parse("#B91C1C"));
    private static readonly FontFamily Mono = new("Consolas");

    private static readonly (string Key, string Label, string Color)[] Stages =
    {
        ("text_ms", "Text layer", "#7C3AED"),
        ("det_ms", "Detect", "#1D4ED8"),
        ("rec_ms", "Recognize", "#16A34A"),
        ("ocr_ms", "OCR in images", "#0891B2"),
        ("math_ms", "Math", "#A21CAF"),
        ("order_ms", "Order", "#D97706"),
    };

    private static readonly Dictionary<string, string> BlockColors = new()
    {
        ["Text"] = "#6B7280",
        ["Section-Header"] = "#1D4ED8",
        ["Page-Header"] = "#9CA3AF",
        ["Page-Footer"] = "#9CA3AF",
        ["List-Group"] = "#16A34A",
        ["Table"] = "#A21CAF",
        ["Caption"] = "#0891B2",
        ["Footnote"] = "#D97706",
        ["Table-Of-Contents"] = "#0F766E",
        ["Form"] = "#DC2626",
    };

    private static string StageLabel(string key) => key switch
    {
        "text_ms" => Loc.T("Text layer", "文字层"),
        "det_ms" => Loc.T("Detect", "检测"),
        "rec_ms" => Loc.T("Recognize", "识别"),
        "ocr_ms" => Loc.T("OCR in images", "图片内 OCR"),
        "math_ms" => Loc.T("Math", "数学"),
        "order_ms" => Loc.T("Order", "阅读序"),
        _ => key,
    };

    private readonly List<PageOverlay> _overlays = new();
    private readonly List<Bitmap> _images = new();
    private readonly List<PageResult> _pages = new();
    private bool _languageReady;
    private readonly List<IReadOnlyDictionary<string, double>> _timings = new();
    private OcrSession? _session;
    private CancellationTokenSource? _cts;
    private Task? _readTask;
    private int _runId;

    public MainWindow() : this(null)
    {
    }

    public MainWindow(string? initialPath = null)
    {
        InitializeComponent();
        VersionText.Text = "v" + IndraOcrInfo.Version;
        ModelPathBox.Text = Environment.GetEnvironmentVariable("INDRA_OCR_MODEL_DIR") ?? "";
        LanguageBox.SelectedIndex = Loc.Chinese ? 1 : 0;
        ApplyLanguage();
        _languageReady = true;

        DragDrop.SetAllowDrop(DropZone, true);
        DropZone.AddHandler(DragDrop.DragOverEvent, OnDragOver);
        DropZone.AddHandler(DragDrop.DropEvent, OnDrop);
        Closing += (_, _) =>
        {
            _cts?.Cancel();
            _session?.Dispose();
            foreach (var image in _images)
            {
                image.Dispose();
            }
            _images.Clear();
        };
        Opened += (_, _) =>
        {
            if (Screens.Primary is { } screen)
            {
                double scale = Math.Max(1.0, screen.Scaling);
                Width = Math.Min(Width, Math.Max(MinWidth, screen.WorkingArea.Width / scale - 40));
                Height = Math.Min(Height, Math.Max(MinHeight, screen.WorkingArea.Height / scale - 40));
            }
        };
        if (initialPath is not null)
        {
            Opened += async (_, _) => await OpenPathAsync(initialPath);
        }
    }

    // ------------------------------------------------------------------ sidebar

    private void OnLanguageChanged(object? sender, SelectionChangedEventArgs e)
    {
        if (!_languageReady)
        {
            return;
        }
        Loc.Set(LanguageBox.SelectedIndex == 1);
        ApplyLanguage();
        RebuildCards();
    }

    private void ApplyLanguage()
    {
        DropTitleText.Text = Loc.T("Drop a PDF or an image", "拖入 PDF 或图片");
        DropSubtitleText.Text = Loc.T("or click to choose a file", "或点击选择文件");
        SourceCaption.Text = Loc.T("SOURCE", "来源");
        PipelineCaption.Text = Loc.T("PIPELINE", "管线");
        ModelCaption.Text = Loc.T("MODEL", "模型");
        ShowCaption.Text = Loc.T("SHOW", "显示");
        RunCaption.Text = Loc.T("RUN", "运行");

        ModeAuto.Content = Loc.T("Auto", "自动");
        ModeOcr.Content = "OCR";
        ModeText.Content = Loc.T("Text layer", "文字层");

        MathText.Text = Loc.T("Math as LaTeX", "数学公式输出 LaTeX");
        FurnitureText.Text = Loc.T("Keep headers and footers", "保留页眉和页脚");
        ImageOcrText.Text = Loc.T("OCR text inside images", "OCR 图片内的文字");
        TesseractText.Text = Loc.T("Compare with Tesseract", "与 Tesseract 对比");
        DpiCaption.Text = "DPI";
        MaxPagesCaption.Text = Loc.T("MAX PAGES", "最大页数");
        RerunButton.Content = Loc.T("Read again", "重新读取");
        BrowseButton.Content = Loc.T("Choose folder...", "选择目录…");
        DownloadButton.Content = Loc.T("Download", "下载模型");
        ModelPathBox.PlaceholderText = Loc.T(
            "Empty: download to the cache", "留空则自动下载到缓存");

        LinesText.Text = Loc.T("Lines", "行");
        BlocksText.Text = Loc.T("Layout blocks", "版面块");
        LabelsText.Text = Loc.T("Block labels", "块标签");
        OrderText.Text = Loc.T("Reading order", "阅读序");

        EmptyTitle.Text = Loc.T("Read a document", "读取文档");
        EmptyText.Text = Loc.T(
            "Drop a PDF or an image on the left. Pages appear as they finish, with lines, layout blocks and reading order drawn over each page.",
            "把 PDF 或图片拖到左侧。页面会逐页出现，并在页面上画出行框、版面块和阅读序。");

        CalloutText.Inlines!.Clear();
        CalloutText.Inlines.Add(new Run(Loc.T(
            "Need handwriting, math and tables at the highest accuracy? ",
            "需要更高精度的手写、公式和表格？")));
        CalloutText.Inlines.Add(new Run("Chandra on Datalab") { Foreground = Accent });
        CalloutText.Inlines.Add(new Run(Loc.T(
            " reads them, with $5 in free credits to start.",
            " 可以识别，并提供 $5 免费额度。")));

        LegendPanel.Children.Clear();
        BuildLegend();
        UpdateModeHint();
        UpdateModelStatus();
        UpdateStats();
        if (BusyOverlay.IsVisible)
        {
            BusyText.Text = Loc.T("Working ...", "处理中…");
        }
        else
        {
            StatusText.Text = "";
            StatusText.Foreground = Ink3;
        }
    }

    private void SetBusy(bool busy, string? text = null)
    {
        BusyOverlay.IsVisible = busy;
        BusySpinner.IsActive = busy;
        Sidebar.IsEnabled = !busy;
        if (busy)
        {
            BusyText.Text = text ?? Loc.T("Working ...", "处理中…");
        }
    }

    private void RebuildCards()
    {
        if (_pages.Count == 0)
        {
            return;
        }
        PagesPanel.Children.Clear();
        foreach (var image in _images)
        {
            image.Dispose();
        }
        _images.Clear();
        _overlays.Clear();
        EmptyState.IsVisible = false;
        foreach (var page in _pages)
        {
            AddPageCard(page, track: false);
        }
        UpdateStats();
    }

    private void BuildLegend()
    {
        foreach (var (label, color) in BlockColors.Where(kv => kv.Key != "Page-Footer"))
        {
            var item = new StackPanel
            {
                Orientation = Orientation.Horizontal,
                Spacing = 5,
                Margin = new Thickness(0, 2, 10, 2),
                Children =
                {
                    new Border
                    {
                        Width = 9,
                        Height = 9,
                        CornerRadius = new CornerRadius(2),
                        Background = new SolidColorBrush(Color.Parse(color)),
                        VerticalAlignment = VerticalAlignment.Center,
                    },
                    new TextBlock
                    {
                        Text = PageOverlay.BlockText(label),
                        FontSize = 11,
                        Foreground = Ink3,
                    },
                },
            };
            LegendPanel.Children.Add(item);
        }
    }

    private string Mode()
    {
        if (ModeOcr.IsChecked == true) return "ocr";
        if (ModeText.IsChecked == true) return "text";
        return "auto";
    }

    private void UpdateModeHint()
    {
        ModeHint.Text = Mode() switch
        {
            "ocr" => Loc.T("Ignores the text layer and OCRs every page.",
                "忽略文字层，所有页面都走 OCR。"),
            "text" => Loc.T("Uses the text layer even when the checks fail.",
                "即使检查不通过也使用文字层。"),
            _ => Loc.T("Reads the PDF text layer where it passes the checks, and OCRs the rest.",
                "文字层通过检查时直接读取，其余部分走 OCR。"),
        };
    }

    private void OnModeChanged(object? sender, RoutedEventArgs e)
    {
        var clicked = (ToggleButton)sender!;
        foreach (var button in new[] { ModeAuto, ModeOcr, ModeText })
        {
            button.IsChecked = button == clicked;
        }
        UpdateModeHint();
    }

    private ReadOptions Options()
    {
        double dpi = DpiBox.SelectedIndex switch { 0 => 150, 2 => 300, _ => 200 };
        int maxPages = (int)Math.Max(1, MaxPagesBox.Value ?? 20);
        return new ReadOptions(
            Mode(),
            MathSwitch.IsChecked == true,
            FurnitureSwitch.IsChecked == true,
            ImageOcrSwitch.IsChecked != false,
            dpi,
            maxPages);
    }

    private void UpdateModelStatus()
    {
        string folder = string.IsNullOrWhiteSpace(ModelPathBox.Text)
            ? OcrSession.DefaultModelDir()
            : ModelPathBox.Text!.Trim();
        bool ready = OcrSession.ModelsPresent(folder, MathSwitch.IsChecked == true);
        ModelStatusText.Text = ready
            ? Loc.T($"Models ready: {folder}", $"模型已就绪：{folder}")
            : Loc.T($"Models not downloaded: {folder}", $"模型未下载：{folder}");
        ModelStatusText.Foreground = ready ? Ok : Warning;
    }

    private void OnMathChanged(object? sender, RoutedEventArgs e) => UpdateModelStatus();

    private void OnDisplayChanged(object? sender, RoutedEventArgs e)
    {
        foreach (var overlay in _overlays)
        {
            overlay.ShowLines = ShowLinesSwitch.IsChecked != false;
            overlay.ShowBlocks = ShowBlocksSwitch.IsChecked != false;
            overlay.ShowLabels = ShowLabelsSwitch.IsChecked != false;
            overlay.ShowOrder = ShowOrderSwitch.IsChecked == true;
            overlay.Refresh();
        }
    }

    // -------------------------------------------------------------------- file

    private void OnChooseFile(object? sender, PointerPressedEventArgs e) => _ = ChooseFileAsync();

    private async Task ChooseFileAsync()
    {
        var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Open a PDF or an image",
            AllowMultiple = false,
            FileTypeFilter = new[]
            {
                new FilePickerFileType("PDF or image")
                {
                    Patterns = new[]
                    {
                        "*.pdf", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp", "*.webp",
                    },
                },
            },
        });
        if (files.Count > 0)
        {
            await OpenPathAsync(files[0].Path.LocalPath);
        }
    }

    private void OnDragOver(object? sender, DragEventArgs e)
    {
        e.DragEffects = DragDropEffects.Copy;
        e.Handled = true;
    }

    private void OnDrop(object? sender, DragEventArgs e)
    {
        var file = e.DataTransfer.TryGetFiles()?.FirstOrDefault();
        if (file is not null)
        {
            _ = OpenPathAsync(file.Path.LocalPath);
        }
    }

    private async Task OpenPathAsync(string path)
    {
        _cts?.Cancel();
        _runId++;
        SetBusy(true, Loc.T("Opening ...", "正在打开…"));
        if (_readTask is not null)
        {
            try
            {
                await _readTask;
            }
            catch
            {
                // the cancelled read already reported what it needed to
            }
        }
        try
        {
            _session?.Dispose();
            _session = new OcrSession();
            _session.SetModelDir(ModelPathBox.Text);
            await Task.Run(() => _session.Open(path));
            FileNameText.Text = $"{_session.FileName} · {new FileInfo(path).Length / 1024} KB";
            RerunButton.IsEnabled = true;
            StartRead();
        }
        catch (Exception e)
        {
            SetBusy(false);
            StatusText.Foreground = Error;
            StatusText.Text = Loc.T("Error: ", "错误：") + e.Message;
        }
    }

    private void OnRerun(object? sender, RoutedEventArgs e) => StartRead();

    private void StartRead()
    {
        if (_session is null)
        {
            return;
        }
        _cts?.Cancel();
        _cts?.Dispose();
        _cts = new CancellationTokenSource();
        _readTask = ReadLoopAsync(_cts.Token);
    }

    private async Task ReadLoopAsync(CancellationToken token)
    {
        if (_session is null)
        {
            return;
        }
        int runId = ++_runId;
        var options = Options();
        int total = Math.Min(_session.PageCount, options.MaxPages);
        PagesPanel.Children.Clear();
        foreach (var image in _images)
        {
            image.Dispose();
        }
        _images.Clear();
        _overlays.Clear();
        _pages.Clear();
        EmptyState.IsVisible = false;
        _timings.Clear();
        StatsText.Text = "";
        StatusText.Foreground = Ink3;
        SetBusy(true, Loc.T("Loading models ...", "正在加载模型…"));
        var watch = Stopwatch.StartNew();
        try
        {
            StatusText.Text = options.Math
                ? Loc.T("Loading models (math models download on first use) ...",
                    "正在加载模型（数学模型首次使用会下载）…")
                : Loc.T("Loading models (first run downloads about 76 MB) ...",
                    "正在加载模型（首次运行约下载 76 MB）…");
            await Task.Run(() => _session.ResolveModels(options.Math), token);

            for (int i = 0; i < total; i++)
            {
                token.ThrowIfCancellationRequested();
                if (runId != _runId)
                {
                    return;
                }
                string reading = Loc.T(
                    $"Reading page {i + 1} of {total}"
                    + (_session.PageCount > total ? $" (first {total} of {_session.PageCount})" : "")
                    + " ...",
                    $"正在识别第 {i + 1} 页，共 {total} 页"
                    + (_session.PageCount > total ? $"（前 {total} 页，共 {_session.PageCount} 页）" : "")
                    + "…");
                StatusText.Text = reading;
                BusyText.Text = reading;
                int index = i;
                var page = await Task.Run(() => _session.ReadPage(index, options), token);
                if (token.IsCancellationRequested || runId != _runId)
                {
                    return;
                }
                AddPageCard(page);
                _timings.Add(page.Timings);
                UpdateStats();
            }
            double elapsed = watch.Elapsed.TotalMilliseconds;
            StatusText.Text = Loc.T(
                $"{total} page{(total == 1 ? "" : "s")} in {elapsed:0} ms",
                $"{total} 页，用时 {elapsed:0} ms");
        }
        catch (OperationCanceledException)
        {
            // a newer read replaced this one
        }
        catch (Exception e)
        {
            StatusText.Foreground = Error;
            StatusText.Text = Loc.T("Error: ", "错误：") + e.Message;
        }
        finally
        {
            if (runId == _runId)
            {
                SetBusy(false);
            }
        }
    }

    private void UpdateStats()
    {
        if (_timings.Count == 0)
        {
            StatsText.Text = "";
            return;
        }
        double total = _timings.Sum(t => t.GetValueOrDefault("total_ms"));
        double perPage = total / _timings.Count;
        double perSecond = total > 0 ? 1000.0 * _timings.Count / total : 0;
        var parts = Stages
            .Select(s => (s.Key, Ms: _timings.Sum(t => t.GetValueOrDefault(s.Key)) / _timings.Count))
            .Where(s => s.Ms > 0)
            .Select(s => $"{StageLabel(s.Key)} {s.Ms:0} ms");
        StatsText.Text = Loc.T(
            $"{perPage:0} ms per page · {perSecond:0.0} pages/s\n",
            $"{perPage:0} ms/页 · {perSecond:0.0} 页/秒\n")
                         + string.Join(" · ", parts);
    }

    // -------------------------------------------------------------------- cards

    private void AddPageCard(PageResult page, bool track = true)
    {
        if (track)
        {
            _pages.Add(page);
        }
        using var stream = new MemoryStream(page.ImageJpeg);
        var bitmap = new Bitmap(stream);
        _images.Add(bitmap);
        var overlay = new PageOverlay
        {
            Image = bitmap,
            Lines = page.Lines,
            Blocks = page.Blocks,
            ShowLines = ShowLinesSwitch.IsChecked != false,
            ShowBlocks = ShowBlocksSwitch.IsChecked != false,
            ShowLabels = ShowLabelsSwitch.IsChecked != false,
            ShowOrder = ShowOrderSwitch.IsChecked == true,
        };
        _overlays.Add(overlay);

        var card = new Border
        {
            Background = Brushes.White,
            BorderBrush = Rule,
            BorderThickness = new Thickness(1),
            CornerRadius = new CornerRadius(12),
            ClipToBounds = true,
        };
        var content = new StackPanel();
        content.Children.Add(Header(page));
        if (!string.IsNullOrEmpty(page.Note))
        {
            content.Children.Add(new Border
            {
                BorderBrush = Rule,
                BorderThickness = new Thickness(0, 0, 0, 1),
                Padding = new Thickness(16, 6),
                Child = new TextBlock
                {
                    Text = page.Note,
                    FontFamily = Mono,
                    FontSize = 12,
                    Foreground = Ink3,
                    TextWrapping = TextWrapping.Wrap,
                },
            });
        }
        content.Children.Add(Body(page, overlay));
        card.Child = content;
        PagesPanel.Children.Add(card);
    }

    private Control Header(PageResult page)
    {
        var grid = new Grid
        {
            ColumnDefinitions = new ColumnDefinitions("Auto,Auto,Auto,*,Auto,Auto"),
            Margin = new Thickness(16, 12),
        };
        var title = new TextBlock
        {
            Text = Loc.T($"Page {page.Number}", $"第 {page.Number} 页"),
            FontSize = 15,
            FontWeight = FontWeight.SemiBold,
            Foreground = Ink,
            VerticalAlignment = VerticalAlignment.Center,
            Margin = new Thickness(0, 0, 10, 0),
        };
        var chip = SourceChip(page.Source);
        chip.VerticalAlignment = VerticalAlignment.Center;
        chip.Margin = new Thickness(0, 0, 10, 0);
        string extra = (page.Rotation != 0
                            ? Loc.T($" · turned {page.Rotation}°", $" · 旋转 {page.Rotation}°")
                            : "")
                       + (page.Skew != 0
                            ? Loc.T($" · straightened {page.Skew:0.0}°", $" · 校正 {page.Skew:0.0}°")
                            : "");
        var meta = new TextBlock
        {
            Text = Loc.T(
                $"{page.Lines.Count} lines · {page.Blocks.Count} blocks{extra}",
                $"{page.Lines.Count} 行 · {page.Blocks.Count} 块{extra}"),
            FontFamily = Mono,
            FontSize = 12,
            Foreground = Ink3,
            VerticalAlignment = VerticalAlignment.Center,
            TextWrapping = TextWrapping.Wrap,
        };
        var bar = TimingBar(page.Timings);
        bar.VerticalAlignment = VerticalAlignment.Center;
        bar.Margin = new Thickness(0, 0, 10, 0);
        var total = new TextBlock
        {
            Text = $"{page.TotalMs:0} ms",
            FontFamily = Mono,
            FontSize = 12,
            Foreground = Ink3,
            VerticalAlignment = VerticalAlignment.Center,
        };
        Grid.SetColumn(title, 0);
        Grid.SetColumn(chip, 1);
        Grid.SetColumn(meta, 2);
        Grid.SetColumn(bar, 4);
        Grid.SetColumn(total, 5);
        grid.Children.Add(title);
        grid.Children.Add(chip);
        grid.Children.Add(meta);
        grid.Children.Add(bar);
        grid.Children.Add(total);
        return new Border
        {
            BorderBrush = Rule,
            BorderThickness = new Thickness(0, 0, 0, 1),
            Child = grid,
        };
    }

    private static Border SourceChip(string source)
    {
        (string Label, string Background, string Foreground) style = source switch
        {
            "text" => (Loc.T("Text layer", "文字层"), "#E8F5E9", "#2E7D32"),
            "hybrid" => (Loc.T("Text layer + OCR", "文字层 + OCR"), "#FFF4E5", "#B45309"),
            "error" => (Loc.T("Error", "错误"), "#FDEcea", "#B91C1C"),
            _ => ("OCR", "#E8ECFF", "#1D4ED8"),
        };
        return new Border
        {
            Background = new SolidColorBrush(Color.Parse(style.Background)),
            CornerRadius = new CornerRadius(5),
            Padding = new Thickness(7, 3),
            Child = new TextBlock
            {
                Text = style.Label.ToUpperInvariant(),
                FontFamily = Mono,
                FontSize = 10.5,
                Foreground = new SolidColorBrush(Color.Parse(style.Foreground)),
            },
        };
    }

    private static Control TimingBar(IReadOnlyDictionary<string, double> timings)
    {
        const double width = 180;
        double total = timings.GetValueOrDefault("total_ms");
        var panel = new StackPanel { Orientation = Orientation.Horizontal };
        var box = new Border
        {
            Width = width,
            Height = 6,
            CornerRadius = new CornerRadius(3),
            ClipToBounds = true,
            Background = Surface2,
            Child = panel,
        };
        if (total <= 0)
        {
            return box;
        }
        foreach (var (key, label, color) in Stages)
        {
            double ms = timings.GetValueOrDefault(key);
            if (ms <= 0)
            {
                continue;
            }
            var segment = new Border
            {
                Width = Math.Max(1, width * ms / total),
                Height = 6,
                Background = new SolidColorBrush(Color.Parse(color)),
            };
            ToolTip.SetTip(segment, $"{StageLabel(key)} {ms:0} ms");
            panel.Children.Add(segment);
        }
        return box;
    }

    private Control Body(PageResult page, PageOverlay overlay)
    {
        var grid = new Grid
        {
            ColumnDefinitions = new ColumnDefinitions("1.05*,*"),
            MinHeight = 420,
        };
        var overlayHost = new Border
        {
            Background = Brushes.Transparent,
            Child = overlay,
        };
        overlayHost.PointerMoved += (_, e) => overlay.HoverAt(e.GetPosition(overlay));
        overlayHost.PointerExited += (_, _) => overlay.ClearHover();
        var left = new Border
        {
            Background = Surface,
            BorderBrush = Rule,
            BorderThickness = new Thickness(0, 0, 1, 0),
            Padding = new Thickness(16),
            Child = new ScrollViewer
            {
                Content = overlayHost,
                HorizontalScrollBarVisibility = ScrollBarVisibility.Disabled,
                VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
            },
        };
        var right = new Border
        {
            Child = Tabs(page, overlay),
        };
        Grid.SetColumn(left, 0);
        Grid.SetColumn(right, 1);
        grid.Children.Add(left);
        grid.Children.Add(right);
        return grid;
    }

    private static Control Tabs(PageResult page, PageOverlay overlay)
    {
        var tabs = new TabControl
        {
            Padding = new Thickness(0),
            HorizontalAlignment = HorizontalAlignment.Stretch,
            VerticalAlignment = VerticalAlignment.Stretch,
        };
        var preview = new ScrollViewer
        {
            Content = MarkdownRenderer.Build(page.Markdown),
            MaxHeight = 620,
            Padding = new Thickness(16, 14),
            HorizontalScrollBarVisibility = ScrollBarVisibility.Disabled,
        };
        var markdown = PaneText(page.Markdown, wrap: true);
        var json = PaneText(page.Json, wrap: false);
        var lines = new ScrollViewer
        {
            Content = LineRows(page, overlay),
            MaxHeight = 620,
            Padding = new Thickness(12, 10),
            HorizontalScrollBarVisibility = ScrollBarVisibility.Disabled,
        };
        tabs.Items.Add(Tab(Loc.T("Preview", "预览"), preview));
        tabs.Items.Add(Tab("Markdown", markdown));
        tabs.Items.Add(Tab(Loc.T("Lines", "行"), lines));
        tabs.Items.Add(Tab("JSON", json));
        tabs.SelectedIndex = 0;
        return tabs;

        static TabItem Tab(string header, object content) => new()
        {
            Header = header,
            Content = content,
            Width = 96,
            HorizontalContentAlignment = HorizontalAlignment.Center,
            Padding = new Thickness(8, 8),
        };
    }

    private static Control PaneText(string text, bool wrap) => new TextBox
    {
        Text = text,
        IsReadOnly = true,
        AcceptsReturn = true,
        TextWrapping = wrap ? TextWrapping.Wrap : TextWrapping.NoWrap,
        FontFamily = Mono,
        FontSize = 12.5,
        BorderThickness = new Thickness(0),
        Background = Brushes.Transparent,
        MaxHeight = 620,
        Padding = new Thickness(14, 10),
        VerticalContentAlignment = VerticalAlignment.Top,
    };

    private static Control LineRows(PageResult page, PageOverlay overlay)
    {
        var list = new StackPanel { Spacing = 1 };
        var rows = new List<(Border Row, TextBlock Index)>();

        void Select(int selected)
        {
            for (int j = 0; j < rows.Count; j++)
            {
                bool on = j == selected;
                rows[j].Row.Background = on ? AccentSoft : Brushes.Transparent;
                rows[j].Row.BorderBrush = on ? Accent : Brushes.Transparent;
                rows[j].Index.Foreground = on
                    ? Accent
                    : new SolidColorBrush(Color.Parse("#9CA3AF"));
                rows[j].Index.FontWeight = on ? FontWeight.SemiBold : FontWeight.Normal;
            }
        }

        for (int i = 0; i < page.Lines.Count; i++)
        {
            var line = page.Lines[i];
            var grid = new Grid
            {
                ColumnDefinitions = new ColumnDefinitions("28,*,44"),
            };
            var index = new TextBlock
            {
                Text = (i + 1).ToString(),
                FontFamily = Mono,
                FontSize = 11,
                Foreground = new SolidColorBrush(Color.Parse("#9CA3AF")),
                VerticalAlignment = VerticalAlignment.Top,
                Margin = new Thickness(0, 3, 6, 0),
                TextAlignment = TextAlignment.Right,
            };
            var text = new SelectableTextBlock
            {
                Text = string.IsNullOrEmpty(line.Text)
                    ? Loc.T("(no text)", "（无文字）")
                    : line.Text,
                TextWrapping = TextWrapping.Wrap,
                FontSize = 13,
                Foreground = string.IsNullOrEmpty(line.Text)
                    ? new SolidColorBrush(Color.Parse("#9CA3AF"))
                    : Ink2,
                FontStyle = string.IsNullOrEmpty(line.Text) ? FontStyle.Italic : FontStyle.Normal,
            };
            var conf = new TextBlock
            {
                Text = line.Conf.ToString("0.00"),
                FontFamily = Mono,
                FontSize = 11,
                Foreground = line.Conf < 0.8
                    ? new SolidColorBrush(Color.Parse("#B91C1C"))
                    : Ink3,
                Margin = new Thickness(6, 2, 0, 0),
                VerticalAlignment = VerticalAlignment.Top,
            };
            Grid.SetColumn(text, 1);
            Grid.SetColumn(conf, 2);
            grid.Children.Add(index);
            grid.Children.Add(text);
            grid.Children.Add(conf);

            var row = new Border
            {
                Padding = new Thickness(4, 3),
                CornerRadius = new CornerRadius(5),
                Background = Brushes.Transparent,
                BorderBrush = Brushes.Transparent,
                BorderThickness = new Thickness(2, 0, 0, 0),
                Child = grid,
            };
            rows.Add((row, index));
            int lineIndex = i;
            row.PointerEntered += (_, _) =>
            {
                Select(lineIndex);
                overlay.HighlightedLine = lineIndex;
                overlay.Refresh();
            };
            row.PointerExited += (_, _) =>
            {
                if (overlay.HighlightedLine != lineIndex)
                {
                    return;
                }
                Select(-1);
                overlay.HighlightedLine = -1;
                overlay.Refresh();
            };
            list.Children.Add(row);
        }

        // Hovering a box on the page selects the matching row here.
        overlay.LineHoverChanged += Select;
        return list;
    }

    // -------------------------------------------------------------------- model

    private async void OnBrowseModel(object? sender, RoutedEventArgs e)
    {
        var folders = await StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
        {
            Title = "Model folder",
            AllowMultiple = false,
        });
        if (folders.Count == 0)
        {
            return;
        }
        ModelPathBox.Text = folders[0].Path.LocalPath;
        _session?.SetModelDir(ModelPathBox.Text);
        UpdateModelStatus();
    }

    private async void OnDownloadModels(object? sender, RoutedEventArgs e)
    {
        DownloadButton.IsEnabled = false;
        SetBusy(true, Loc.T("Downloading models ...", "正在下载模型…"));
        StatusText.Foreground = Ink3;
        StatusText.Text = Loc.T("Downloading models ...", "正在下载模型…");
        try
        {
            string? folder = string.IsNullOrWhiteSpace(ModelPathBox.Text)
                ? null
                : ModelPathBox.Text!.Trim();
            bool math = MathSwitch.IsChecked == true;
            string resolved = await Task.Run(() => ModelFiles.Resolve(folder, math: math));
            ModelPathBox.Text = resolved;
            _session?.SetModelDir(resolved);
            UpdateModelStatus();
            StatusText.Foreground = Ok;
            StatusText.Text = Loc.T($"Models ready: {resolved}", $"模型已就绪：{resolved}");
        }
        catch (Exception ex)
        {
            ModelStatusText.Foreground = Error;
            ModelStatusText.Text = Loc.T("Download failed: ", "下载失败：") + ex.Message;
            StatusText.Foreground = Error;
            StatusText.Text = Loc.T("Download failed: ", "下载失败：") + ex.Message;
        }
        finally
        {
            DownloadButton.IsEnabled = true;
            SetBusy(false);
        }
    }

    // -------------------------------------------------------------------- links

    private static void OpenUrl(string url)
        => Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });

    private void OnOpenGithub(object? sender, RoutedEventArgs e)
        => OpenUrl("https://github.com/datalab-to/indra");

    private void OnOpenModels(object? sender, RoutedEventArgs e)
        => OpenUrl("https://huggingface.co/datalab-to/indra-ocr");

    private void OnOpenChandra(object? sender, RoutedEventArgs e)
        => OpenUrl("https://www.datalab.to/?utm_source=indra-viewer");
}
