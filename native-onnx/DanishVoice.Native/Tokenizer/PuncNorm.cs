namespace DanishVoice.Native.Tokenizer;

/// <summary>Faithful port of chatterbox punc_norm (text cleanup before tokenizing).</summary>
internal static class PuncNorm
{
    private static readonly (string Old, string New)[] Replacements =
    [
        ("...", ", "), ("…", ", "), (":", ","), (" - ", ", "), (";", ", "),
        ("—", "-"), ("–", "-"), (" ,", ","),
        ("“", "\""), ("”", "\""), ("‘", "'"), ("’", "'"),
    ];

    private static readonly char[] SentenceEnders =
        ['.', '!', '?', '-', ',', '、', '，', '。', '？', '！'];

    public static string Apply(string text)
    {
        if (text.Length == 0)
        {
            return "You need to add some text for me to talk.";
        }

        if (char.IsLower(text[0]))
        {
            text = char.ToUpperInvariant(text[0]) + text[1..];
        }

        // collapse whitespace runs
        text = string.Join(" ", text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));

        foreach (var (oldSeq, newSeq) in Replacements)
        {
            text = text.Replace(oldSeq, newSeq);
        }

        text = text.TrimEnd(' ');
        if (!Array.Exists(SentenceEnders, p => text.EndsWith(p)))
        {
            text += ".";
        }
        return text;
    }
}
