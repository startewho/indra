using System.Text.Json;
using System.Text;

namespace IndraOcr.Vocab;

/// <summary>
/// Token vocabulary of a CTC recognizer (vocab.py).  Class 0 is the blank;
/// class i + 1 is chars[i].  A token is one character, or one LaTeX command
/// inside math.
/// </summary>
public sealed class Vocab
{
    private readonly Dictionary<string, int> _index;

    public Vocab(IEnumerable<string> chars)
    {
        Chars = chars.ToList();
        _index = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < Chars.Count; i++)
        {
            _index[Chars[i]] = i + 1;
        }
        HasCommands = Chars.Any(c => c.Length > 1);
        Jamo = _index.ContainsKey("\u1100");
    }

    public IReadOnlyList<string> Chars { get; }
    public IReadOnlyDictionary<string, int> Index => _index;
    public bool HasCommands { get; }
    public bool Jamo { get; }
    public int NClasses => Chars.Count + 1;

    public static Vocab FromJson(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        if (root.TryGetProperty("blank", out var blank) && blank.GetInt32() != 0)
        {
            throw new ArgumentException("only blank=0 vocabularies are supported");
        }
        var chars = root.GetProperty("chars").EnumerateArray().Select(e => e.GetString()!).ToList();
        return new Vocab(chars);
    }

    /// <summary>Token ids -> text (CTC collapse already applied by the model).</summary>
    public string Decode(IReadOnlyList<int> ids)
    {
        var toks = new List<string>(ids.Count);
        foreach (int id in ids)
        {
            if (id > 0 && id <= Chars.Count)
            {
                toks.Add(Chars[id - 1]);
            }
        }
        if (Jamo)
        {
            return NormalizeForm(string.Concat(toks), NormalizationForm.FormC);
        }
        if (!HasCommands)
        {
            return string.Concat(toks);
        }
        var sb = new StringBuilder();
        for (int k = 0; k < toks.Count; k++)
        {
            string t = toks[k];
            sb.Append(t);
            // "\alpha" + "x" must not read back as "\alphax"
            string next = k + 1 < toks.Count ? toks[k + 1] : "";
            if (t.Length > 2 && t[0] == '\\' && t[1..].All(char.IsAsciiLetter)
                && next.Length > 0 && char.IsAsciiLetter(next[0]))
            {
                sb.Append(' ');
            }
        }
        return sb.ToString();
    }

    private static string NormalizeForm(string s, NormalizationForm form) => s.Normalize(form);
}
