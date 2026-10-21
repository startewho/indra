using IndraOcr.Document;
using IndraOcr.Models;
using IndraOcr.Pdf;
using IndraOcr.Pipeline;

namespace IndraOcr.Parallel;

/// <summary>One input's conversion: its pages in order, or the InputError that stopped it.</summary>
public sealed class ConvertedFile
{
    public required string Path { get; init; }
    public InputError? Error { get; init; }
    public List<(int Index, Page Page)> Pages { get; init; } = new();
}

/// <summary>
/// parallel.py: pages on all cores.  ONNX Runtime sessions are shared (thread
/// safe); PDFium is not, so each worker thread keeps its own document handle and
/// PDFium calls are serialized inside the wrapper.  Pages of one file run in
/// parallel; files are read one at a time and yielded in input order.
/// </summary>
public static class PageRunner
{
    public const int ThreadsPerWorker = 1;
    public const int MinPoolPages = 4;

    public static int UsableCores()
    {
        int n = Environment.ProcessorCount;
        return n > 0 ? n : 1;
    }

    public static IEnumerable<ConvertedFile> ConvertMany(
        IEnumerable<string> paths,
        int cores = 0,
        int maxPages = 0,
        string? password = null,
        string? modelDir = null,
        double dpi = 200.0,
        string mode = "auto",
        bool imageOcr = true,
        bool furniture = false,
        bool math = false)
    {
        if (maxPages < 0)
        {
            throw new ArgumentException($"max_pages must be 0 (all pages) or more, not {maxPages}");
        }
        // Resolve (and on a first run download) the models once, not per worker.
        string models = ModelFiles.Resolve(modelDir, math: math);
        int nCores = cores > 0 ? cores : UsableCores();

        foreach (string path in paths)
        {
            yield return ConvertOne(path, models, nCores, maxPages, password, dpi, mode, imageOcr, furniture, math);
        }
    }

    private static ConvertedFile ConvertOne(
        string path, string models, int nCores, int maxPages, string? password,
        double dpi, string mode, bool imageOcr, bool furniture, bool math)
    {
        if (!File.Exists(path))
        {
            return new ConvertedFile
            {
                Path = path,
                Error = new InputError($"{path}: no such file"),
            };
        }

        if (!Converter.IsPdf(path))
        {
            // An image file is one unit (every frame is a page, read in order).
            try
            {
                using var ocr = new OrtOcr(models, threads: nCores, furniture: furniture, math: math);
                var conv = new Converter(ocr, dpi, mode, imageOcr);
                return new ConvertedFile { Path = path, Pages = conv.Convert(path, maxPages, password).ToList() };
            }
            catch (InputError e)
            {
                return new ConvertedFile { Path = path, Error = e };
            }
        }

        int count;
        try
        {
            using var doc = PdfDocument.Open(path, password);
            count = doc.PageCount;
        }
        catch (PdfiumException e)
        {
            return new ConvertedFile { Path = path, Error = Converter.OpenError(path, e) };
        }
        if (maxPages > 0)
        {
            count = Math.Min(count, maxPages);
        }
        if (count == 0)
        {
            return new ConvertedFile { Path = path };
        }

        if (nCores < 2 || count < MinPoolPages)
        {
            // Few pages: one process with all cores on each page.
            try
            {
                using var ocr = new OrtOcr(models, threads: nCores, furniture: furniture, math: math);
                var conv = new Converter(ocr, dpi, mode, imageOcr);
                return new ConvertedFile { Path = path, Pages = conv.Convert(path, maxPages, password).ToList() };
            }
            catch (InputError e)
            {
                return new ConvertedFile { Path = path, Error = e };
            }
        }

        var pages = new Page[count];
        using (var ocr = new OrtOcr(models, threads: ThreadsPerWorker, furniture: furniture, math: math))
        {
            var conv = new Converter(ocr, dpi, mode, imageOcr);
            using var docs = new ThreadLocal<DocCache>(() => new DocCache(), trackAllValues: true);
            System.Threading.Tasks.Parallel.For(
                0, count, new System.Threading.Tasks.ParallelOptions { MaxDegreeOfParallelism = nCores }, i =>
                {
                    var doc = docs.Value!.Get(path, password);
                    pages[i] = Converter.SafeRead(() => conv.PdfPage(path, doc, i));
                });
            foreach (var cache in docs.Values)
            {
                cache.Dispose();
            }
        }
        var result = new List<(int, Page)>(count);
        for (int i = 0; i < count; i++)
        {
            result.Add((i, pages[i]));
        }
        return new ConvertedFile { Path = path, Pages = result };
    }

    /// <summary>One PDF document per worker thread (PDFium is not thread-safe).</summary>
    private sealed class DocCache : IDisposable
    {
        private PdfDocument? _doc;
        private string? _path;
        private string? _password;

        public PdfDocument Get(string path, string? password)
        {
            if (_doc is null || _path != path || _password != password)
            {
                _doc?.Dispose();
                _doc = PdfDocument.Open(path, password);
                _path = path;
                _password = password;
            }
            return _doc;
        }

        public void Dispose()
        {
            _doc?.Dispose();
            _doc = null;
        }
    }
}
