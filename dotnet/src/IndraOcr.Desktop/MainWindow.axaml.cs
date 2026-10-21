using System.Diagnostics;
using Avalonia;
using Avalonia.Controls;
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

    private readonly List<PageOverlay> _overlays = new();
    private readonly List<Bitmap> _images = new();
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
        BuildLegend();
        UpdateModeHint();
        UpdateModelStatus();

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
                        Text = label == "Page-Header" ? "Header/Footer" : label.Replace('-', ' '),
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
            "ocr" => "Ignores the text layer and OCRs every page.",
            "text" => "Uses the text layer even when the checks fail.",
            _ => "Reads the PDF text layer where it passes the checks, and OCRs the rest.",
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
            ? $"Models ready: {folder}"
            : $"Models not downloaded: {folder}";
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
            StatusText.Foreground = Error;
            StatusText.Text = "Error: " + e.Message;
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
        EmptyState.IsVisible = false;
        _overlays.Clear();
        _timings.Clear();
        StatsText.Text = "";
        StatusText.Foreground = Ink3;
        BusyBar.IsVisible = true;
        var watch = Stopwatch.StartNew();
        try
        {
            StatusText.Text = options.Math
                ? "Loading models (math models download on first use) ..."
                : "Loading models (first run downloads about 76 MB) ...";
            await Task.Run(() => _session.ResolveModels(options.Math), token);

            for (int i = 0; i < total; i++)
            {
                token.ThrowIfCancellationRequested();
                if (runId != _runId)
                {
                    return;
                }
                StatusText.Text = $"Reading page {i + 1} of {total}"
                                  + (_session.PageCount > total ? $" (first {total} of {_session.PageCount})" : "")
                                  + " ...";
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
            StatusText.Text = $"{total} page{(total == 1 ? "" : "s")} in {watch.Elapsed.TotalMilliseconds:0} ms";
        }
        catch (OperationCanceledException)
        {
            // a newer read replaced this one
        }
        catch (Exception e)
        {
            StatusText.Foreground = Error;
            StatusText.Text = "Error: " + e.Message;
        }
        finally
        {
            if (runId == _runId)
            {
                BusyBar.IsVisible = false;
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
            .Select(s => (s.Label, Ms: _timings.Sum(t => t.GetValueOrDefault(s.Key)) / _timings.Count))
            .Where(s => s.Ms > 0)
            .Select(s => $"{s.Label} {s.Ms:0} ms");
        StatsText.Text = $"{perPage:0} ms per page · {perSecond:0.0} pages/s\n"
                         + string.Join(" · ", parts);
    }

    // -------------------------------------------------------------------- cards

    private void AddPageCard(PageResult page)
    {
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
            Text = $"Page {page.Number}",
            FontSize = 15,
            FontWeight = FontWeight.SemiBold,
            Foreground = Ink,
            VerticalAlignment = VerticalAlignment.Center,
            Margin = new Thickness(0, 0, 10, 0),
        };
        var chip = SourceChip(page.Source);
        chip.VerticalAlignment = VerticalAlignment.Center;
        chip.Margin = new Thickness(0, 0, 10, 0);
        string extra = (page.Rotation != 0 ? $" · turned {page.Rotation}°" : "")
                       + (page.Skew != 0 ? $" · straightened {page.Skew:0.0}°" : "");
        var meta = new TextBlock
        {
            Text = $"{page.Lines.Count} lines · {page.Blocks.Count} blocks{extra}",
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
            "text" => ("Text layer", "#E8F5E9", "#2E7D32"),
            "hybrid" => ("Text layer + OCR", "#FFF4E5", "#B45309"),
            "error" => ("Error", "#FDEcea", "#B91C1C"),
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
            ToolTip.SetTip(segment, $"{label} {ms:0} ms");
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
        var left = new Border
        {
            Background = Surface,
            BorderBrush = Rule,
            BorderThickness = new Thickness(0, 0, 1, 0),
            Padding = new Thickness(16),
            Child = new ScrollViewer
            {
                Content = overlay,
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
        tabs.Items.Add(new TabItem { Header = "Preview", Content = preview });
        tabs.Items.Add(new TabItem { Header = "Markdown", Content = markdown });
        tabs.Items.Add(new TabItem { Header = "Lines", Content = lines });
        tabs.Items.Add(new TabItem { Header = "JSON", Content = json });
        return tabs;
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
                Text = string.IsNullOrEmpty(line.Text) ? "(no text)" : line.Text,
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
                Padding = new Thickness(6, 3),
                CornerRadius = new CornerRadius(5),
                Background = Brushes.Transparent,
                Child = grid,
            };
            int lineIndex = i;
            row.PointerEntered += (_, _) =>
            {
                overlay.HighlightedLine = lineIndex;
                overlay.Refresh();
                row.Background = AccentSoft;
            };
            row.PointerExited += (_, _) =>
            {
                overlay.HighlightedLine = -1;
                overlay.Refresh();
                row.Background = Brushes.Transparent;
            };
            list.Children.Add(row);
        }
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
        BusyBar.IsVisible = true;
        StatusText.Foreground = Ink3;
        StatusText.Text = "Downloading models ...";
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
            StatusText.Text = $"Models ready: {resolved}";
        }
        catch (Exception ex)
        {
            ModelStatusText.Foreground = Error;
            ModelStatusText.Text = "Download failed: " + ex.Message;
            StatusText.Foreground = Error;
            StatusText.Text = "Download failed: " + ex.Message;
        }
        finally
        {
            DownloadButton.IsEnabled = true;
            BusyBar.IsVisible = false;
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
