using Avalonia.Controls;
using Avalonia.Media.Imaging;
using Avalonia.Platform.Storage;
using Avalonia.Threading;

namespace IndraOcr.AndroidApp;

public partial class MobileMainView : UserControl
{
    private MobileOcrService? _service;
    private Bitmap? _preview;
    private string? _imagePath;
    private bool _started;

    public MobileMainView()
    {
        InitializeComponent();
        Loaded += OnLoaded;
        Unloaded += (_, _) => _preview?.Dispose();
    }

    private static string ModelDir
    {
        get
        {
            string root = global::Android.App.Application.Context.FilesDir?.AbsolutePath
                          ?? Path.GetTempPath();
            return Path.Combine(root, "Data");
        }
    }

    private async void OnLoaded(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        if (_started)
        {
            return;
        }
        _started = true;
        await EnsureModelsAsync();
    }

    private async Task EnsureModelsAsync()
    {
        SetBusy(true, "正在检查 / 下载模型（首次约 73 MB）…");
        try
        {
            _service ??= new MobileOcrService(ModelDir);
            await Task.Run(() => _service.ResolveModels());
            SetStatus("模型已就绪");
        }
        catch (Exception ex)
        {
            SetStatus("模型下载失败：" + ex.Message);
        }
        finally
        {
            SetBusy(false);
        }
    }

    private async void OnPickImage(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        var top = TopLevel.GetTopLevel(this);
        if (top?.StorageProvider is null)
        {
            SetStatus("当前平台不支持文件选择");
            return;
        }
        var files = await top.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "选择图片",
            AllowMultiple = false,
            FileTypeFilter = new[]
            {
                new FilePickerFileType("Images")
                {
                    Patterns = new[] { "*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.tif", "*.tiff" },
                    MimeTypes = new[] { "image/*" },
                },
            },
        });
        if (files.Count == 0)
        {
            return;
        }
        try
        {
            var file = files[0];
            string ext = Path.GetExtension(file.Name);
            if (string.IsNullOrEmpty(ext))
            {
                ext = ".img";
            }
            string cache = global::Android.App.Application.Context.CacheDir?.AbsolutePath
                           ?? Path.GetTempPath();
            string path = Path.Combine(cache, "indra-input" + ext);
            await using (var src = await file.OpenReadAsync())
            await using (var dst = File.Create(path))
            {
                await src.CopyToAsync(dst);
            }
            _imagePath = path;
            SetPreview(path);
            Overlay.Lines = Array.Empty<MobileLine>();
            Overlay.Blocks = Array.Empty<MobileBlock>();
            Overlay.Refresh();
            EmptyHint.IsVisible = false;
            RecognizeButton.IsEnabled = true;
            MarkdownBox.Text = "";
            InfoText.Text = file.Name;
            SetStatus("图片已选择，可以开始识别");
        }
        catch (Exception ex)
        {
            SetStatus("打开图片失败：" + ex.Message);
        }
    }

    private async void OnRecognize(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        if (_imagePath is null)
        {
            return;
        }
        SetBusy(true, "正在识别…");
        try
        {
            if (_service is null)
            {
                _service = new MobileOcrService(ModelDir);
            }
            var result = await _service.RecognizeAsync(_imagePath, CancellationToken.None);
            SetPreview(result.PreviewJpeg);
            Overlay.Lines = result.Lines;
            Overlay.Blocks = result.Blocks;
            Overlay.Refresh();
            MarkdownBox.Text = result.Markdown;
            InfoText.Text = $"{result.Lines.Count} 行 · {result.Blocks.Count} 块 · {result.TotalMs:0} ms";
            SetStatus("识别完成");
        }
        catch (Exception ex)
        {
            SetStatus("识别失败：" + ex.Message);
        }
        finally
        {
            SetBusy(false);
        }
    }

    private void SetPreview(string path)
    {
        using var stream = File.OpenRead(path);
        SetPreview(new Bitmap(stream));
    }

    private void SetPreview(byte[] jpeg)
    {
        using var stream = new MemoryStream(jpeg);
        SetPreview(new Bitmap(stream));
    }

    private void SetPreview(Bitmap bitmap)
    {
        var old = _preview;
        _preview = bitmap;
        Overlay.Image = bitmap;
        Overlay.Refresh();
        old?.Dispose();
    }

    private void SetBusy(bool busy, string? status = null)
    {
        BusyBar.IsVisible = busy;
        PickButton.IsEnabled = !busy;
        RecognizeButton.IsEnabled = !busy && _imagePath is not null;
        if (status is not null)
        {
            SetStatus(status);
        }
    }

    private void SetStatus(string text)
    {
        Dispatcher.UIThread.Post(() => StatusText.Text = text);
    }
}
