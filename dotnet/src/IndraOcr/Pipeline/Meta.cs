using System.Text.Json;

namespace IndraOcr.Pipeline;

/// <summary>meta.json: geometry, vocabularies and script routing.</summary>
public sealed class Meta
{
    public required string VocabJson { get; init; }
    public required Dictionary<string, string> VocabGroups { get; init; }
    public required int DetLongSide { get; init; }
    public required int RecHeight { get; init; }
    public required int RecWindow { get; init; }
    public required string[] ScriptGroups { get; init; }
    public JsonElement LmParams { get; init; }

    /// <summary>alpha, beta and gate of a recognizer's LM (meta.json "lm_params").</summary>
    public (double Alpha, double Beta, double Gate) LmParamsFor(string group)
    {
        double alpha = 0.5, beta = 1.5, gate = 0.9;
        if (LmParams.ValueKind == JsonValueKind.Object
            && LmParams.TryGetProperty(group, out var p))
        {
            if (p.TryGetProperty("alpha", out var a)) alpha = a.GetDouble();
            if (p.TryGetProperty("beta", out var b)) beta = b.GetDouble();
            if (p.TryGetProperty("gate", out var g)) gate = g.GetDouble();
        }
        return (alpha, beta, gate);
    }

    public static readonly string[] DefaultGroups = { "base", "cjk", "rtl", "other" };

    public static Meta Load(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path, System.Text.Encoding.UTF8));
        var root = doc.RootElement;
        var groups = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var prop in root.EnumerateObject())
        {
            if (prop.Name.StartsWith("vocab_", StringComparison.Ordinal))
            {
                groups[prop.Name["vocab_".Length..]] = prop.Value.GetString()!;
            }
        }
        var scriptGroups = root.TryGetProperty("script_groups", out var sg)
            ? sg.EnumerateArray().Select(e => e.GetString()!).ToArray()
            : DefaultGroups;
        return new Meta
        {
            VocabJson = root.GetProperty("vocab").GetString()!,
            VocabGroups = groups,
            DetLongSide = root.GetProperty("det_long_side").GetInt32(),
            RecHeight = root.GetProperty("rec_height").GetInt32(),
            RecWindow = root.TryGetProperty("rec_window", out var w) ? w.GetInt32() : 1,
            ScriptGroups = scriptGroups,
            LmParams = root.TryGetProperty("lm_params", out var lp) ? lp.Clone() : default,
        };
    }
}

/// <summary>One character position in a recognized line (id or char, and page x).</summary>
public readonly record struct Token(int Id, string? Char, float X, int Seq = -1)
{
    public static Token FromId(int id, float x) => new(id, null, x);

    public static Token FromChar(string c, float x, int seq = -1) => new(-1, c, x, seq);
}

/// <summary>A recognized line: text, mean CTC confidence and its tokens.</summary>
public sealed record RecLine(string Text, double Conf, Token[] Tokens)
{
    public static readonly RecLine Empty = new("", 0.0, Array.Empty<Token>());
}
