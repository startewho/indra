using System.Text;
using System.Text.Json;
using IndraOcr.Viewer;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;

namespace IndraOcr.Tests;

/// <summary>One engine for all viewer tests (model loading and the warm-up page).</summary>
public sealed class ViewerFixture : IDisposable
{
    private ViewerEngine? _engine;

    public ViewerEngine? Engine => _engine ??= TestPaths.Models is null
        ? null
        : new ViewerEngine(TestPaths.Models, threads: 1, new ViewerOptions().Defaults());

    public void Dispose() => _engine?.Dispose();
}

/// <summary>S7 parity: the viewer engine and HTTP routes (viewer.py).</summary>
public class ViewerTests : IClassFixture<ViewerFixture>
{
    private readonly ViewerFixture _fixture;

    public ViewerTests(ViewerFixture fixture)
    {
        _fixture = fixture;
    }

    private static string ScanPath() => TestPaths.Full("examples/samples/scan.png");

    private static string PdfPath(string name)
        => Path.Combine(TestPaths.Root, "dotnet", "tests", "parity", "pdfs", name);

    private static JsonElement Options(string json = """{"mode":"auto","math":false,"furniture":false,"image_ocr":true,"dpi":200}""")
        => JsonDocument.Parse(json).RootElement.Clone();

    [SkippableFact]
    public void InfoReportsTheVersionAndDefaults()
    {
        Skip.If(_fixture.Engine is null, "models not found");
        var info = _fixture.Engine!.Info();
        Assert.Equal(IndraOcrInfo.Version, info["version"]!.GetValue<string>());
        Assert.False(info["tesseract"]!.GetValue<bool>());
        Assert.Equal("auto", info["defaults"]!["mode"]!.GetValue<string>());
        Assert.Equal(20, info["defaults"]!["max_pages"]!.GetValue<int>());
    }

    [SkippableFact]
    public void AnImageUploadsAndRendersOnePage()
    {
        Skip.If(_fixture.Engine is null, "models not found");
        var upload = _fixture.Engine!.Upload(File.ReadAllBytes(ScanPath()));
        string id = upload["id"]!.GetValue<string>();
        Assert.Equal(12, id.Length);
        Assert.Equal(1, upload["pages"]!.GetValue<int>());

        var rec = _fixture.Engine.Page(id, 0, Options());
        Assert.Equal(1, rec["page"]!.GetValue<int>());
        Assert.True(rec["w"]!.GetValue<int>() > 0);
        Assert.True(rec["h"]!.GetValue<int>() > 0);
        Assert.StartsWith("data:image/jpeg;base64,", rec["img"]!.GetValue<string>());
        Assert.Equal("ocr", rec["source"]!.GetValue<string>());
        Assert.Equal(36, rec["lines"]!.AsArray().Count);
        Assert.Equal(9, rec["blocks"]!.AsArray().Count);
        Assert.False(string.IsNullOrEmpty(rec["markdown"]!.GetValue<string>()));
        Assert.Equal(0, rec["json"]!["page"]!.GetValue<int>());
        Assert.Equal("ocr", rec["json"]!["source"]!.GetValue<string>());
        Assert.True(rec["timings"]!["total_ms"]!.GetValue<double>() > 0);
        Assert.True(rec["timings"]!["det_ms"]!.GetValue<double>() > 0);
        // The first line starts at 213.4, 224.3 in page pixels; the image is
        // scaled to at most 1600 px, and so are the boxes.
        var box = rec["lines"]![0]!["box"]!.AsArray();
        Assert.True(box[0]!.GetValue<double>() > 0);
        Assert.True(box[2]!.GetValue<double>() > box[0]!.GetValue<double>());
    }

    [SkippableFact]
    public void APdfUploadsAndShowsItsRotatedPageUpright()
    {
        Skip.If(_fixture.Engine is null, "models not found");
        var upload = _fixture.Engine!.Upload(File.ReadAllBytes(PdfPath("digital.pdf")));
        string id = upload["id"]!.GetValue<string>();
        Assert.Equal(2, upload["pages"]!.GetValue<int>());

        var first = _fixture.Engine.Page(id, 0, Options("""{"mode":"auto"}"""));
        Assert.Equal("text", first["source"]!.GetValue<string>());
        Assert.Equal(0, first["rotation"]!.GetValue<int>());
        Assert.Equal(5, first["lines"]!.AsArray().Count);

        var second = _fixture.Engine.Page(id, 1, Options("""{"mode":"auto"}"""));
        Assert.Equal(270, second["rotation"]!.GetValue<int>());
        Assert.Equal("text", second["source"]!.GetValue<string>());
        Assert.Equal(2, second["lines"]!.AsArray().Count);
        Assert.True(second["w"]!.GetValue<int>() > 0);
    }

    [SkippableFact]
    public void BadOptionsAndFilesAreReported()
    {
        Skip.If(_fixture.Engine is null, "models not found");
        var upload = _fixture.Engine!.Upload(File.ReadAllBytes(ScanPath()));
        string id = upload["id"]!.GetValue<string>();

        var mode = Assert.Throws<ArgumentException>(
            () => _fixture.Engine.Page(id, 0, Options("""{"mode":"nope"}""")));
        Assert.Contains("mode must be one of", mode.Message);

        Assert.Throws<InvalidOperationException>(
            () => _fixture.Engine.Page("does-not-exist", 0, Options()));

        var bad = Assert.Throws<InvalidOperationException>(
            () => _fixture.Engine.Upload(Encoding.ASCII.GetBytes("not a document")));
        Assert.Contains("not a PDF or a readable image", bad.Message);
    }

    [SkippableFact]
    public async Task HttpRoutesServeThePageAndApi()
    {
        Skip.If(_fixture.Engine is null, "models not found");
        var options = new ViewerOptions { Port = 0, Threads = 1, ModelDir = TestPaths.Models };
        var app = ViewerApp.Build(_fixture.Engine!, options);
        await app.StartAsync();
        try
        {
            var server = app.Services.GetRequiredService<IServer>();
            string address = server.Features.Get<IServerAddressesFeature>()!.Addresses.First();
            using var http = new HttpClient { BaseAddress = new Uri(address) };

            string home = await http.GetStringAsync("/");
            Assert.Contains("Indra OCR viewer", home);
            Assert.Contains("/logo.png", home);

            using var info = JsonDocument.Parse(await http.GetStringAsync("/info"));
            Assert.Equal(IndraOcrInfo.Version, info.RootElement.GetProperty("version").GetString());
            Assert.False(info.RootElement.GetProperty("tesseract").GetBoolean());

            var logo = await http.GetByteArrayAsync("/logo.png");
            Assert.True(logo.Length > 100);
            Assert.Equal(0x89, logo[0]);
            Assert.Equal((byte)'P', logo[1]);

            using var uploadResponse = await http.PostAsync(
                "/upload", new ByteArrayContent(File.ReadAllBytes(ScanPath())));
            uploadResponse.EnsureSuccessStatusCode();
            using var upload = JsonDocument.Parse(await uploadResponse.Content.ReadAsStringAsync());
            string id = upload.RootElement.GetProperty("id").GetString()!;
            Assert.Equal(1, upload.RootElement.GetProperty("pages").GetInt32());

            string body = "{\"id\":\"" + id + "\",\"n\":0,\"opts\":{\"mode\":\"auto\"}}";
            using var pageResponse = await http.PostAsync(
                "/page", new StringContent(body, Encoding.UTF8, "application/json"));
            pageResponse.EnsureSuccessStatusCode();
            using var page = JsonDocument.Parse(await pageResponse.Content.ReadAsStringAsync());
            Assert.Equal(1, page.RootElement.GetProperty("page").GetInt32());
            Assert.Equal(36, page.RootElement.GetProperty("lines").GetArrayLength());
            Assert.StartsWith("data:image/jpeg;base64,",
                page.RootElement.GetProperty("img").GetString());

            using var bad = await http.PostAsync(
                "/page", new StringContent("{\"id\":\"" + id + "\",\"n\":0,\"opts\":{\"mode\":\"x\"}}",
                    Encoding.UTF8, "application/json"));
            Assert.Equal(System.Net.HttpStatusCode.BadRequest, bad.StatusCode);
            Assert.Contains("mode must be one of", await bad.Content.ReadAsStringAsync());
        }
        finally
        {
            await app.StopAsync();
            await app.DisposeAsync();
        }
    }

    [Fact]
    public void CommandLineOptionsAreParsed()
    {
        var o = ViewerOptions.Parse(new[]
        {
            "--port", "9999", "--threads", "2", "--mode", "text", "--no-image-ocr",
            "--dpi", "150", "--max-pages", "3", "--keep-furniture", "--math",
        });
        Assert.Equal(9999, o.Port);
        Assert.Equal(2, o.Threads);
        Assert.Equal("text", o.Mode);
        Assert.False(o.ImageOcr);
        Assert.Equal(150, o.Dpi);
        Assert.Equal(3, o.MaxPages);
        Assert.True(o.Furniture);
        Assert.True(o.Math);

        Assert.Throws<UsageException>(() => ViewerOptions.Parse(new[] { "--mode", "nope" }));
        Assert.Throws<UsageException>(() => ViewerOptions.Parse(new[] { "--unknown" }));
        Assert.True(ViewerOptions.Parse(new[] { "--version" }).Version);
    }
}
