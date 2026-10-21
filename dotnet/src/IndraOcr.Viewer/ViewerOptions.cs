using System.Globalization;
using System.Text.Json.Nodes;

namespace IndraOcr.Viewer;

/// <summary>A command line usage error (exit code 2, like argparse).</summary>
public sealed class UsageException : Exception
{
    public UsageException(string message) : base(message)
    {
    }
}

/// <summary>viewer.py command line options.</summary>
public sealed class ViewerOptions
{
    public string? ModelDir { get; set; }
    public string Mode { get; set; } = "auto";
    public bool ImageOcr { get; set; } = true;
    public int Port { get; set; } = 8765;
    public int Threads { get; set; } = 4;
    public double Dpi { get; set; } = 200.0;
    public int MaxPages { get; set; } = 20;
    public bool Furniture { get; set; }
    public bool Math { get; set; }
    public bool Tesseract { get; set; }
    public string TesseractLang { get; set; } = "eng";
    public bool Version { get; set; }
    public bool Help { get; set; }

    public static ViewerOptions Parse(string[] argv)
    {
        var o = new ViewerOptions();
        for (int i = 0; i < argv.Length; i++)
        {
            string arg = argv[i];
            string name = arg;
            string? inlineValue = null;
            int eq = arg.IndexOf('=');
            if (eq > 0)
            {
                name = arg[..eq];
                inlineValue = arg[(eq + 1)..];
            }
            string Value()
            {
                if (inlineValue is not null)
                {
                    return inlineValue;
                }
                if (i + 1 >= argv.Length)
                {
                    throw new UsageException($"argument {name}: expected one argument");
                }
                return argv[++i];
            }
            switch (name)
            {
                case "-h":
                case "--help":
                    o.Help = true;
                    return o;
                case "--version":
                    o.Version = true;
                    return o;
                case "--model-dir":
                    o.ModelDir = Value();
                    break;
                case "--mode":
                {
                    string v = Value();
                    if (v is not ("auto" or "ocr" or "text"))
                    {
                        throw new UsageException(
                            $"argument {name}: invalid choice: '{v}' (choose from auto, ocr, text)");
                    }
                    o.Mode = v;
                    break;
                }
                case "--no-image-ocr":
                    o.ImageOcr = false;
                    break;
                case "--port":
                    o.Port = ParseInt(name, Value());
                    break;
                case "--threads":
                    o.Threads = ParseInt(name, Value());
                    break;
                case "--dpi":
                    o.Dpi = ParseDouble(name, Value());
                    break;
                case "--max-pages":
                    o.MaxPages = ParseInt(name, Value());
                    break;
                case "--keep-furniture":
                    o.Furniture = true;
                    break;
                case "--math":
                    o.Math = true;
                    break;
                case "--tesseract":
                    o.Tesseract = true;
                    break;
                case "--tesseract-lang":
                    o.TesseractLang = Value();
                    break;
                default:
                    throw new UsageException($"unrecognized arguments: {arg}");
            }
        }
        return o;
    }

    private static int ParseInt(string name, string value)
        => int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int v)
            ? v
            : throw new UsageException($"argument {name}: invalid int value: '{value}'");

    private static double ParseDouble(string name, string value)
        => double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out double v)
            ? v
            : throw new UsageException($"argument {name}: invalid float value: '{value}'");

    /// <summary>The initial pipeline options the page starts with (/info).</summary>
    public JsonObject Defaults() => new()
    {
        ["mode"] = Mode,
        ["image_ocr"] = ImageOcr,
        ["math"] = Math,
        ["furniture"] = Furniture,
        ["dpi"] = Dpi,
        ["max_pages"] = MaxPages,
    };
}
