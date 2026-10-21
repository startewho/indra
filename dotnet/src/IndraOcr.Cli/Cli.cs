using System.Globalization;
using System.Text;
using IndraOcr.Models;
using IndraOcr.Parallel;

namespace IndraOcr.Cli;

/// <summary>
/// cli.py: PDFs or images in, markdown (or text, or JSON lines) out.  Pages run
/// in parallel on all the cores the process may use (--cores to limit).
/// </summary>
public sealed class Cli
{
    private const string Prog = "indra-ocr";

    private readonly TextWriter _out;
    private readonly TextWriter _err;

    /// <summary>Writers default to the console; Program passes explicit UTF-8 writers.</summary>
    public Cli(TextWriter? stdout = null, TextWriter? stderr = null)
    {
        _out = stdout ?? Console.Out;
        _err = stderr ?? Console.Error;
    }

    private sealed class Options
    {
        public List<string> Inputs = new();
        public string? Out;
        public string Format = "md";
        public string Mode = "auto";
        public bool NoImageOcr;
        public string? ModelDir;
        public int Cores;
        public double Dpi = 200.0;
        public int MaxPages;
        public string? Password;
        public bool Math;
        public bool KeepFurniture;
        public bool Timings;
        public bool Version;
        public bool Help;
    }

    public int Run(string[] argv)
    {
        try
        {
            return RunCore(argv);
        }
        catch (ModelException e)
        {
            _err.WriteLine($"{Prog}: {e.Message}");
            return 1;
        }
        catch (IOException)
        {
            // The reader stopped early (indra-ocr x.pdf | head): no traceback.
            return 1;
        }
    }

    private int RunCore(string[] argv)
    {
        Options? options = Parse(argv, out int code);
        if (options is null)
        {
            return code;
        }
        if (options.Help)
        {
            PrintHelp();
            return 0;
        }
        if (options.Version)
        {
            _out.WriteLine($"{Prog} {IndraOcrInfo.Version}");
            return 0;
        }
        if (options.Inputs.Count == 0)
        {
            _err.WriteLine($"{Prog}: error: the following arguments are required: inputs");
            PrintUsage(_err);
            return 2;
        }

        if (options.Out is not null)
        {
            Directory.CreateDirectory(options.Out);
        }
        string ext = options.Format switch
        {
            "md" => ".md",
            "text" => ".txt",
            _ => ".jsonl",
        };
        string separator = options.Format == "json" ? "\n" : "\n\n";
        var names = new HashSet<string>(StringComparer.Ordinal);
        int failed = 0;

        var results = PageRunner.ConvertMany(
            options.Inputs,
            cores: options.Cores,
            maxPages: options.MaxPages,
            password: options.Password,
            modelDir: options.ModelDir,
            dpi: options.Dpi,
            mode: options.Mode,
            imageOcr: !options.NoImageOcr,
            furniture: options.KeepFurniture,
            math: options.Math);

        foreach (var file in results)
        {
            if (file.Error is not null)
            {
                _err.WriteLine($"{Prog}: {file.Error.Message}");
                failed++;
                continue;
            }
            var chunks = new List<string>();
            foreach (var (i, page) in file.Pages)
            {
                if (page.Source == "error")
                {
                    _err.WriteLine($"{Prog}: {file.Path} page {i + 1}: {page.Note}");
                }
                chunks.Add(options.Format switch
                {
                    "json" => JsonOutput.Page(file.Path, i, page),
                    "md" => page.Markdown ?? page.Text,
                    _ => page.Text,
                });
                if (options.Timings)
                {
                    double total = page.Timings.GetValueOrDefault("total_ms");
                    _err.WriteLine(
                        $"{file.Path} page {i + 1}: {page.Source}, " +
                        $"{total.ToString("0", CultureInfo.InvariantCulture)} ms {page.Note}");
                }
            }
            string body = WithPlatformNewlines(string.Join(separator, chunks) + "\n");
            if (options.Out is not null)
            {
                string stem = Path.GetFileNameWithoutExtension(file.Path);
                string name = stem;
                int k = 1;
                while (!names.Add(name))
                {
                    k++;
                    name = $"{stem}_{k}";
                }
                File.WriteAllText(Path.Combine(options.Out, name + ext), body, new UTF8Encoding(false));
            }
            else
            {
                _out.Write(body);
            }
        }

        if (failed > 0)
        {
            _err.WriteLine($"{Prog}: {failed} of {options.Inputs.Count} inputs failed");
        }
        return failed > 0 ? 1 : 0;
    }

    /// <summary>
    /// Python's text-mode stdout and file writes translate "\n" to os.linesep;
    /// match that so redirected output is byte-identical on Windows too.
    /// </summary>
    private static string WithPlatformNewlines(string text)
        => Environment.NewLine == "\n"
            ? text
            : text.Replace("\r\n", "\n").Replace("\n", Environment.NewLine);

    private Options? Parse(string[] argv, out int code)
    {
        code = 0;
        var o = new Options();
        bool positionalOnly = false;
        for (int i = 0; i < argv.Length; i++)
        {
            string arg = argv[i];
            if (positionalOnly || !arg.StartsWith("-", StringComparison.Ordinal) || arg == "-")
            {
                o.Inputs.Add(arg);
                continue;
            }
            if (arg == "--")
            {
                positionalOnly = true;
                continue;
            }
            string name = arg;
            string? inlineValue = null;
            int eq = arg.IndexOf('=');
            if (eq > 0)
            {
                name = arg[..eq];
                inlineValue = arg[(eq + 1)..];
            }
            string? Value()
            {
                if (inlineValue is not null)
                {
                    return inlineValue;
                }
                if (i + 1 < argv.Length)
                {
                    return argv[++i];
                }
                return null;
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
                case "--out":
                    o.Out = Value();
                    if (o.Out is null) return MissingValue(name, out code);
                    break;
                case "--format":
                {
                    string? v = Value();
                    if (v is null) return MissingValue(name, out code);
                    if (v is not ("md" or "text" or "json"))
                    {
                        return BadChoice(name, v, "md, text, json", out code);
                    }
                    o.Format = v;
                    break;
                }
                case "--mode":
                {
                    string? v = Value();
                    if (v is null) return MissingValue(name, out code);
                    if (v is not ("auto" or "ocr" or "text"))
                    {
                        return BadChoice(name, v, "auto, ocr, text", out code);
                    }
                    o.Mode = v;
                    break;
                }
                case "--no-image-ocr":
                    o.NoImageOcr = true;
                    break;
                case "--model-dir":
                    o.ModelDir = Value();
                    if (o.ModelDir is null) return MissingValue(name, out code);
                    break;
                case "--cores":
                case "--threads":
                {
                    string? v = Value();
                    if (v is null) return MissingValue(name, out code);
                    if (!int.TryParse(v, NumberStyles.Integer, CultureInfo.InvariantCulture, out o.Cores))
                    {
                        return BadInt(name, v, out code);
                    }
                    break;
                }
                case "--dpi":
                {
                    string? v = Value();
                    if (v is null) return MissingValue(name, out code);
                    if (!double.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out o.Dpi))
                    {
                        return BadFloat(name, v, out code);
                    }
                    break;
                }
                case "--max-pages":
                {
                    string? v = Value();
                    if (v is null) return MissingValue(name, out code);
                    if (!int.TryParse(v, NumberStyles.Integer, CultureInfo.InvariantCulture, out o.MaxPages))
                    {
                        return BadInt(name, v, out code);
                    }
                    if (o.MaxPages < 0)
                    {
                        _err.WriteLine(
                            $"{Prog}: error: argument --max-pages: must be 0 (all pages) or more, not {o.MaxPages}");
                        code = 2;
                        return null;
                    }
                    break;
                }
                case "--password":
                    o.Password = Value();
                    if (o.Password is null) return MissingValue(name, out code);
                    break;
                case "--math":
                    o.Math = true;
                    break;
                case "--keep-furniture":
                    o.KeepFurniture = true;
                    break;
                case "--timings":
                    o.Timings = true;
                    break;
                default:
                    _err.WriteLine($"{Prog}: error: unrecognized arguments: {arg}");
                    code = 2;
                    return null;
            }
        }
        return o;
    }

    private Options? MissingValue(string name, out int code)
    {
        _err.WriteLine($"{Prog}: error: argument {name}: expected one argument");
        code = 2;
        return null;
    }

    private Options? BadChoice(string name, string value, string choices, out int code)
    {
        _err.WriteLine($"{Prog}: error: argument {name}: invalid choice: '{value}' (choose from {choices})");
        code = 2;
        return null;
    }

    private Options? BadInt(string name, string value, out int code)
    {
        _err.WriteLine($"{Prog}: error: argument {name}: invalid int value: '{value}'");
        code = 2;
        return null;
    }

    private Options? BadFloat(string name, string value, out int code)
    {
        _err.WriteLine($"{Prog}: error: argument {name}: invalid float value: '{value}'");
        code = 2;
        return null;
    }

    private static void PrintUsage(TextWriter writer)
    {
        writer.WriteLine($"usage: {Prog} [-h] [--version] [--out OUT] [--format {{md,text,json}}]");
        writer.WriteLine("                 [--mode {auto,ocr,text}] [--no-image-ocr] [--model-dir MODEL_DIR]");
        writer.WriteLine("                 [--cores CORES] [--dpi DPI] [--max-pages MAX_PAGES]");
        writer.WriteLine("                 [--password PASSWORD] [--math] [--keep-furniture] [--timings]");
        writer.WriteLine("                 inputs [inputs ...]");
    }

    private void PrintHelp()
    {
        PrintUsage(_out);
        _out.WriteLine();
        _out.WriteLine("PDFs or images in, markdown (or text, or JSON lines) out.");
        _out.WriteLine();
        _out.WriteLine("positional arguments:");
        _out.WriteLine("  inputs                PDF or image files");
        _out.WriteLine();
        _out.WriteLine("options:");
        _out.WriteLine("  -h, --help            show this help message and exit");
        _out.WriteLine("  --version             show the version and exit");
        _out.WriteLine("  --out OUT             write one file per input here");
        _out.WriteLine("  --format {md,text,json}");
        _out.WriteLine("  --mode {auto,ocr,text}");
        _out.WriteLine("  --no-image-ocr        do not OCR text inside images on text-layer pages");
        _out.WriteLine("  --model-dir MODEL_DIR folder with the models (default: download)");
        _out.WriteLine("  --cores CORES         cores to use (default: all this process may run on)");
        _out.WriteLine("  --dpi DPI             render resolution for PDFs (default: 200)");
        _out.WriteLine("  --max-pages MAX_PAGES read at most this many pages of each file (0: all)");
        _out.WriteLine("  --password PASSWORD   password of encrypted PDFs");
        _out.WriteLine("  --math                write math as LaTeX ($...$)");
        _out.WriteLine("  --keep-furniture      keep page headers and footers");
        _out.WriteLine("  --timings             print per-page timings to stderr");
    }
}
