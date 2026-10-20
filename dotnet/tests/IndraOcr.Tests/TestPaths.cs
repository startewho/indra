using System.Text.Json;

namespace IndraOcr.Tests;

/// <summary>Locations of the golden data, artifacts and models used by parity tests.</summary>
internal static class TestPaths
{
    public static string Root { get; } = FindRoot();
    public static string Golden => Path.Combine(Root, "dotnet", "tests", "parity", "golden");
    public static string Artifacts => Path.Combine(Root, "dotnet", "tests", "parity", "artifacts");
    public static string? Models { get; } = FindModels();

    public static string Full(string relative) => Path.Combine(Root, relative.Replace('/', Path.DirectorySeparatorChar));

    private static string FindRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !Directory.Exists(Path.Combine(dir.FullName, "indra_ocr")))
        {
            dir = dir.Parent;
        }
        return dir?.FullName ?? throw new InvalidOperationException("repository root not found");
    }

    /// <summary>
    /// INDRA_OCR_MODEL_DIR, or the Python package's Hugging Face cache (the
    /// reference environment of docs/PORT_TO_CSHARP.md), or null (test skipped).
    /// </summary>
    private static string? FindModels()
    {
        string? env = Environment.GetEnvironmentVariable("INDRA_OCR_MODEL_DIR");
        if (!string.IsNullOrWhiteSpace(env))
        {
            return env;
        }
        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string snapshots = Path.Combine(home, ".cache", "huggingface", "hub", "models--datalab-to--indra-ocr", "snapshots");
        if (!Directory.Exists(snapshots))
        {
            return null;
        }
        foreach (string dir in Directory.GetDirectories(snapshots))
        {
            if (File.Exists(Path.Combine(dir, "det.onnx"))
                && File.Exists(Path.Combine(dir, "rec.onnx"))
                && File.Exists(Path.Combine(dir, "meta.json")))
            {
                return dir;
            }
        }
        return null;
    }

    public sealed record ArtifactPage(string Input, int Index, string Dir);

    public static IReadOnlyList<ArtifactPage> ArtifactPages()
    {
        string manifest = Path.Combine(Artifacts, "manifest.json");
        if (!File.Exists(manifest))
        {
            return Array.Empty<ArtifactPage>();
        }
        using var doc = JsonDocument.Parse(File.ReadAllText(manifest));
        return doc.RootElement.GetProperty("pages").EnumerateArray()
            .Select(p => new ArtifactPage(
                p.GetProperty("input").GetString()!,
                p.GetProperty("index").GetInt32(),
                p.GetProperty("dir").GetString()!))
            .ToList();
    }
}
