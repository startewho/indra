using System.Reflection;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;

namespace IndraOcr.Viewer;

/// <summary>The local HTTP app: the same routes and JSON the Python viewer serves.</summary>
public static class ViewerApp
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    public static WebApplication Build(ViewerEngine engine, ViewerOptions options)
    {
        var builder = WebApplication.CreateSlimBuilder();
        // The Python viewer is quiet: only its own page line and errors are printed.
        builder.Logging.ClearProviders();
        builder.WebHost.UseUrls($"http://127.0.0.1:{options.Port}");
        var app = builder.Build();

        string html = EmbeddedText("index.html");
        byte[] logo = EmbeddedBytes("logo.png");

        static string Text(JsonNode node) => node.ToJsonString(JsonOptions);

        static IResult Error(Exception e)
            => Results.Text(Text(new JsonObject { ["error"] = e.Message }), "application/json", statusCode: 400);

        app.MapGet("/", () => Results.Text(html, "text/html; charset=utf-8"));
        app.MapGet("/index.html", () => Results.Text(html, "text/html; charset=utf-8"));
        app.MapGet("/info", () => Results.Text(Text(engine.Info()), "application/json"));
        app.MapGet("/logo.png", () => Results.Bytes(logo, "image/png"));
        app.MapPost("/upload", async (HttpRequest request) =>
        {
            try
            {
                return Results.Text(Text(engine.Upload(await ReadBody(request))), "application/json");
            }
            catch (Exception e)
            {
                return Error(e);
            }
        });
        app.MapPost("/page", async (HttpRequest request) =>
        {
            try
            {
                using var doc = await JsonDocument.ParseAsync(request.Body);
                var root = doc.RootElement;
                string id = root.GetProperty("id").GetString()!;
                int n = root.GetProperty("n").GetInt32();
                var opts = root.TryGetProperty("opts", out var o) ? o : default;
                return Results.Text(Text(engine.Page(id, n, opts)), "application/json");
            }
            catch (Exception e)
            {
                return Error(e);
            }
        });
        return app;
    }

    private static async Task<byte[]> ReadBody(HttpRequest request)
    {
        using var ms = new MemoryStream();
        await request.Body.CopyToAsync(ms);
        return ms.ToArray();
    }

    private static Stream Resource(string name)
    {
        var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(name);
        if (stream is null)
        {
            throw new InvalidOperationException($"missing embedded resource {name}");
        }
        return stream;
    }

    private static string EmbeddedText(string name)
    {
        using var stream = Resource(name);
        using var reader = new StreamReader(stream, System.Text.Encoding.UTF8);
        return reader.ReadToEnd();
    }

    private static byte[] EmbeddedBytes(string name)
    {
        using var stream = Resource(name);
        using var ms = new MemoryStream();
        stream.CopyTo(ms);
        return ms.ToArray();
    }
}
