using System.Net.Http.Headers;
using System.Text.Json;

namespace IndraOcr.Models;

/// <summary>The models are not available: download failed or the folder lacks files.</summary>
public sealed class ModelException : IOException
{
    public ModelException(string message, Exception? inner = null) : base(message, inner)
    {
    }
}

/// <summary>
/// models.py: a local folder, or a download from the Hugging Face repository.
/// Downloads use the plain resolve URLs (no Xet), and honour HF_ENDPOINT so a
/// mirror works; the cache is per user, per revision.
/// </summary>
public static class ModelFiles
{
    public const string Repo = IndraOcrInfo.ModelRepo;
    public const string Revision = IndraOcrInfo.ModelRevision;

    public static readonly string[] Required = { "det.onnx", "meta.json" };
    public static readonly string[] MathFiles =
    {
        "mathcls.onnx", "math_enc.onnx", "math_dec_init.onnx", "math_dec_step.onnx", "math_meta.json",
    };

    /// <summary>
    /// Folder with det.onnx, rec.onnx (or rec.int8.onnx) and meta.json.  Order of
    /// lookup: <paramref name="path"/>, INDRA_OCR_MODEL_DIR, then the cache (with a
    /// download on first use).  A local folder without order.onnx is accepted.
    /// </summary>
    public static string Resolve(string? path = null, bool math = false, string? revision = null)
    {
        string? explicitPath = path ?? Environment.GetEnvironmentVariable("INDRA_OCR_MODEL_DIR");
        if (!string.IsNullOrWhiteSpace(explicitPath))
        {
            string folder = Path.GetFullPath(explicitPath);
            var missing = Missing(folder, math);
            if (missing.Count > 0)
            {
                throw new ModelException($"the model folder {folder} lacks {string.Join(", ", missing)}");
            }
            return folder;
        }

        string rev = revision ?? Revision;
        string cache = CacheDir(rev);
        if (Missing(cache, math).Count == 0)
        {
            return cache;
        }
        try
        {
            Download(cache, rev, math);
        }
        catch (Exception e) when (e is not ModelException)
        {
            throw new ModelException(
                $"could not download the models from Hugging Face ({Repo}, {rev}): " +
                $"{e.GetType().Name}: {FirstLine(e.Message)}. Check the network connection, or " +
                "point at a local model folder with --model-dir or INDRA_OCR_MODEL_DIR.", e);
        }
        var still = Missing(cache, math);
        if (still.Count > 0)
        {
            throw new ModelException($"the downloaded model folder {cache} lacks {string.Join(", ", still)}");
        }
        return cache;
    }

    public static string CacheDir(string revision)
    {
        string root = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        if (string.IsNullOrEmpty(root))
        {
            root = Path.GetTempPath();
        }
        return Path.Combine(root, "indra-ocr", "models", Repo.Replace("/", "--"), revision);
    }

    private static List<string> Missing(string folder, bool math)
    {
        var missing = new List<string>();
        foreach (string f in Required)
        {
            if (!File.Exists(Path.Combine(folder, f)))
            {
                missing.Add(f);
            }
        }
        if (!File.Exists(Path.Combine(folder, "rec.onnx")) && !File.Exists(Path.Combine(folder, "rec.int8.onnx")))
        {
            missing.Add("rec.onnx");
        }
        if (math)
        {
            missing.AddRange(MathFiles.Where(f => !File.Exists(Path.Combine(folder, f))));
        }
        return missing;
    }

    private static string Endpoint() =>
        (Environment.GetEnvironmentVariable("HF_ENDPOINT") ?? "https://huggingface.co").TrimEnd('/');

    private static void Download(string folder, string revision, bool math)
    {
        Directory.CreateDirectory(folder);
        using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(15) };
        http.DefaultRequestHeaders.UserAgent.Add(new ProductInfoHeaderValue("indra-ocr-csharp", IndraOcrInfo.Version));
        string? token = Environment.GetEnvironmentVariable("HF_TOKEN");
        if (!string.IsNullOrEmpty(token))
        {
            http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", token);
        }

        foreach (string file in ListFiles(http, revision, math))
        {
            string target = Path.Combine(folder, file);
            Directory.CreateDirectory(Path.GetDirectoryName(target)!);
            if (File.Exists(target))
            {
                continue;
            }
            string url = $"{Endpoint()}/{Repo}/resolve/{revision}/{file}?download=true";
            using var resp = http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead).GetAwaiter().GetResult();
            if (!resp.IsSuccessStatusCode)
            {
                throw new HttpRequestException($"GET {url}: {(int)resp.StatusCode} {resp.ReasonPhrase}");
            }
            string tmp = target + ".incomplete";
            using (var src = resp.Content.ReadAsStream())
            using (var dst = File.Create(tmp))
            {
                src.CopyTo(dst);
            }
            File.Move(tmp, target, overwrite: true);
        }
    }

    /// <summary>
    /// Files to fetch: the repository tree when the API answers, otherwise the
    /// known file list (a 404 on an optional file is not an error).
    /// </summary>
    private static IEnumerable<string> ListFiles(HttpClient http, string revision, bool math)
    {
        var wanted = new List<string>();
        try
        {
            string url = $"{Endpoint()}/api/models/{Repo}/tree/{revision}?recursive=true";
            using var resp = http.GetAsync(url).GetAwaiter().GetResult();
            resp.EnsureSuccessStatusCode();
            using var doc = JsonDocument.Parse(resp.Content.ReadAsStream());
            foreach (var item in doc.RootElement.EnumerateArray())
            {
                if (item.TryGetProperty("type", out var type) && type.GetString() != "file")
                {
                    continue;
                }
                string? path = item.GetProperty("path").GetString();
                if (path is null || !Keep(path, math))
                {
                    continue;
                }
                wanted.Add(path);
            }
        }
        catch (Exception) when (wanted.Count == 0)
        {
            wanted.AddRange(KnownFiles.Where(f => Keep(f, math)));
        }
        // The required files first: a failure leaves a clear message.
        return wanted.OrderBy(f => Array.IndexOf(Required, f) is var i && i >= 0 ? i : 99);
    }

    private static readonly string[] KnownFiles =
    {
        "det.onnx", "rec.onnx", "order.onnx", "meta.json", "orient.onnx", "script.onnx",
        "rec_cjk.onnx", "rec_rtl.onnx", "rec_world.onnx",
        "lm.npz", "lm_cjk.npz", "lm_rtl.npz", "lm_world.npz", "mask.onnx",
    };

    private static bool Keep(string path, bool math)
    {
        bool isMath = MathFiles.Contains(path);
        if (isMath && !math)
        {
            return false;
        }
        string name = Path.GetFileName(path);
        return name.EndsWith(".onnx", StringComparison.Ordinal)
               || name == "meta.json"
               || (name.StartsWith("lm", StringComparison.Ordinal) && name.EndsWith(".npz", StringComparison.Ordinal));
    }

    private static string FirstLine(string s)
    {
        var line = s.Replace("\r", "").Split('\n')[0];
        return line.Length > 200 ? line[..200] : line;
    }
}
