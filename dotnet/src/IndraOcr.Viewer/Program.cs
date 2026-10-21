using System.Text;
using IndraOcr;
using IndraOcr.Models;
using IndraOcr.Viewer;

ViewerOptions options;
try
{
    options = ViewerOptions.Parse(args);
}
catch (UsageException e)
{
    Console.Error.WriteLine($"indra-ocr-viewer: error: {e.Message}");
    return 2;
}
if (options.Help)
{
    PrintHelp();
    return 0;
}
if (options.Version)
{
    Console.WriteLine($"indra-ocr {IndraOcrInfo.Version}");
    return 0;
}
try
{
    Console.OutputEncoding = new UTF8Encoding(false);
}
catch (IOException)
{
    // no console
}
try
{
    using var engine = new ViewerEngine(options.ModelDir, options.Threads, options.Defaults());
    var app = ViewerApp.Build(engine, options);
    Console.WriteLine($"Indra OCR viewer on http://localhost:{options.Port}  ({options.Threads} threads)");
    if (options.Tesseract)
    {
        Console.Error.WriteLine("indra-ocr-viewer: Tesseract comparison is not available in the C# viewer");
    }
    app.Run();
    return 0;
}
catch (ModelException e)
{
    Console.Error.WriteLine($"indra-ocr-viewer: {e.Message}");
    return 1;
}

static void PrintHelp()
{
    Console.WriteLine("usage: indra-ocr-viewer [-h] [--model-dir MODEL_DIR] [--mode {auto,ocr,text}]");
    Console.WriteLine("                         [--no-image-ocr] [--version] [--port PORT] [--threads THREADS]");
    Console.WriteLine("                         [--dpi DPI] [--max-pages MAX_PAGES] [--keep-furniture]");
    Console.WriteLine("                         [--tesseract] [--tesseract-lang TESSERACT_LANG] [--math]");
    Console.WriteLine();
    Console.WriteLine("Local viewer: drop in a PDF or an image, see what Indra OCR reads.");
    Console.WriteLine("Open http://localhost:8765 after starting it.");
}
