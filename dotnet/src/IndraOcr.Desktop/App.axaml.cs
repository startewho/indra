using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;

namespace IndraOcr.Desktop;

public partial class App : Application
{
    public override void Initialize()
    {
        AvaloniaXamlLoader.Load(this);
    }

    public override void OnFrameworkInitializationCompleted()
    {
        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            string? initial = Environment.GetCommandLineArgs().Skip(1).FirstOrDefault(File.Exists);
            desktop.MainWindow = new MainWindow(initial);
        }

        base.OnFrameworkInitializationCompleted();
    }
}