using System.Text;

namespace IndraOcr.Cli;

/// <summary>Command line entry point: the exit code is 0, or 1 when an input failed.</summary>
public static class Program
{
    public static int Main(string[] args)
    {
        // Python needs PYTHONUTF8=1 for the same reason: the Windows console
        // default (GBK here) cannot encode Korean or CJK output.  Explicit UTF-8
        // writers also keep redirected output (indra-ocr x.pdf > out.md) UTF-8.
        var utf8 = new UTF8Encoding(false);
        try
        {
            Console.OutputEncoding = utf8;
        }
        catch (IOException)
        {
            // No console (a service, a closed handle).
        }
        using var stdout = new StreamWriter(Console.OpenStandardOutput(), utf8) { AutoFlush = false };
        using var stderr = new StreamWriter(Console.OpenStandardError(), utf8) { AutoFlush = false };
        int code = new Cli(stdout, stderr).Run(args);
        stdout.Flush();
        stderr.Flush();
        return code;
    }
}
