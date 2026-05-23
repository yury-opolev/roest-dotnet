using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace DanishVoice.Native.Tokenizer;

/// <summary>
/// Faithful C# port of chatterbox MTLTokenizer for language_id="da".
///
/// Pipeline (matches the Python encode):
///   1. preprocess: lowercase + Unicode NFKD
///   2. prepend "[da]" language token
///   3. replace ' ' with "[SPACE]"
///   4. HF tokenizers pipeline: split on special tokens, Whitespace
///      pre-tokenize the rest, BPE-encode each word, map to ids.
///
/// The underlying model is BPE with empty continuing-subword-prefix /
/// end-of-word-suffix, fuse_unk=false, no normalizer/post-processor.
/// </summary>
internal sealed partial class MtlTokenizer
{
    private readonly Dictionary<string, int> vocab;
    private readonly Dictionary<(string, string), int> mergeRanks;
    private readonly List<(string Content, int Id)> specialTokens; // sorted longest-first
    private readonly int unkId;

    [GeneratedRegex(@"\w+|[^\w\s]+")]
    private static partial Regex WhitespacePreTokenizer();

    public MtlTokenizer(string tokenizerJsonPath)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(tokenizerJsonPath));
        var root = doc.RootElement;
        var model = root.GetProperty("model");

        this.vocab = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var kv in model.GetProperty("vocab").EnumerateObject())
        {
            this.vocab[kv.Name] = kv.Value.GetInt32();
        }

        this.mergeRanks = [];
        var rank = 0;
        foreach (var m in model.GetProperty("merges").EnumerateArray())
        {
            var pair = m.GetString()!;
            var sp = pair.Split(' ');
            this.mergeRanks[(sp[0], sp[1])] = rank++;
        }

        this.specialTokens = [];
        foreach (var t in root.GetProperty("added_tokens").EnumerateArray())
        {
            this.specialTokens.Add((t.GetProperty("content").GetString()!, t.GetProperty("id").GetInt32()));
        }
        // longest content first so the scanner prefers the longest match
        this.specialTokens.Sort((a, b) => b.Content.Length.CompareTo(a.Content.Length));

        this.unkId = this.vocab.GetValueOrDefault("[UNK]", 1);
    }

    public int[] Encode(string rawText, string languageId = "da")
    {
        var pre = Preprocess(rawText);
        var text = $"[{languageId.ToLowerInvariant()}]{pre}".Replace(" ", "[SPACE]");

        var ids = new List<int>();
        foreach (var (isSpecial, segment, specialId) in SplitOnSpecials(text))
        {
            if (isSpecial)
            {
                ids.Add(specialId);
                continue;
            }
            foreach (Match word in WhitespacePreTokenizer().Matches(segment))
            {
                EncodeWord(word.Value, ids);
            }
        }
        return [.. ids];
    }

    private static string Preprocess(string raw)
    {
        return raw.ToLowerInvariant().Normalize(NormalizationForm.FormKD);
    }

    private IEnumerable<(bool IsSpecial, string Segment, int SpecialId)> SplitOnSpecials(string text)
    {
        var i = 0;
        var buf = new StringBuilder();
        while (i < text.Length)
        {
            var matched = false;
            foreach (var (content, id) in this.specialTokens)
            {
                if (content.Length > 0 && string.CompareOrdinal(text, i, content, 0, content.Length) == 0)
                {
                    if (buf.Length > 0)
                    {
                        yield return (false, buf.ToString(), -1);
                        buf.Clear();
                    }
                    yield return (true, content, id);
                    i += content.Length;
                    matched = true;
                    break;
                }
            }
            if (!matched)
            {
                buf.Append(text[i]);
                i++;
            }
        }
        if (buf.Length > 0)
        {
            yield return (false, buf.ToString(), -1);
        }
    }

    private void EncodeWord(string word, List<int> ids)
    {
        // BPE in HF operates on code points, not grapheme clusters.
        var symbols = new List<string>();
        foreach (var ch in EnumerateCodePoints(word))
        {
            symbols.Add(ch);
        }

        // greedily merge the best-ranked adjacent pair until none remain
        while (symbols.Count > 1)
        {
            var bestRank = int.MaxValue;
            var bestIdx = -1;
            for (var k = 0; k < symbols.Count - 1; k++)
            {
                if (this.mergeRanks.TryGetValue((symbols[k], symbols[k + 1]), out var r) && r < bestRank)
                {
                    bestRank = r;
                    bestIdx = k;
                }
            }
            if (bestIdx < 0)
            {
                break;
            }
            symbols[bestIdx] = symbols[bestIdx] + symbols[bestIdx + 1];
            symbols.RemoveAt(bestIdx + 1);
        }

        foreach (var sym in symbols)
        {
            ids.Add(this.vocab.TryGetValue(sym, out var id) ? id : this.unkId);
        }
    }

    private static IEnumerable<string> EnumerateCodePoints(string s)
    {
        for (var i = 0; i < s.Length;)
        {
            if (char.IsHighSurrogate(s[i]) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1]))
            {
                yield return s.Substring(i, 2);
                i += 2;
            }
            else
            {
                yield return s[i].ToString();
                i++;
            }
        }
    }
}
