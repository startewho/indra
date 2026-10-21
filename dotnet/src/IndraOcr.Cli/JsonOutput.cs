using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;
using IndraOcr.Pipeline;

namespace IndraOcr.Cli;

/// <summary>The per-page JSON object the CLI writes with --format json.</summary>
public static class JsonOutput
{
    private static readonly JsonSerializerOptions Options = new()
    {
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        WriteIndented = false,
    };

    public static string Page(string file, int index, Page page)
    {
        var lines = new JsonArray();
        foreach (var line in page.Lines)
        {
            lines.Add(new JsonObject
            {
                ["box"] = NumberArray(line.Box),
                ["text"] = line.Text,
                ["conf"] = line.Conf,
            });
        }
        var blocks = new JsonArray();
        foreach (var block in page.Blocks)
        {
            blocks.Add(new JsonObject
            {
                ["label"] = block.Label,
                ["bbox"] = NumberArray(block.Bbox),
                ["text"] = block.Text,
            });
        }
        var timings = new JsonObject();
        foreach (var (key, value) in page.Timings)
        {
            timings[key] = value;
        }
        var obj = new JsonObject
        {
            ["file"] = file,
            ["page"] = index,
            ["source"] = page.Source,
            ["rotation"] = page.Rotation,
            ["skew"] = page.Skew,
            ["vertical"] = page.Vertical,
            ["note"] = page.Note,
            ["lines"] = lines,
            ["markdown"] = page.Markdown,
            ["blocks"] = blocks,
            ["timings"] = timings,
        };
        return obj.ToJsonString(Options);
    }

    private static JsonArray NumberArray(IEnumerable<float> values)
    {
        var arr = new JsonArray();
        foreach (float v in values)
        {
            // The pipeline rounds boxes to one decimal; storing them as float
            // widens 213.4 to 213.39999389648438 in JSON, so round again here.
            arr.Add(Math.Round((double)v, 1));
        }
        return arr;
    }
}
