# Sentence-Level Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a sentence-level streaming API (`SynthesizeStreamingAsync` / `SynthesizeAsync` yielding 16-bit PCM `byte[]` per sentence) to the `DanishVoice.Native` library and release it as v0.2.1.

**Architecture:** Split text into sentences (new Danish `SentenceChunker`), synthesize each via the existing pipeline (now with a per-voice conditioning cache), convert float samples to 16-bit LE PCM, and emit chunks through a bounded-channel look-ahead producer — mirroring cortex's `KokoroTextToSpeech`. The library stays standalone (BCL types only).

**Tech Stack:** C# / .NET 10, `Microsoft.ML.OnnxRuntime`, `System.Threading.Channels`, xUnit.

**Spec:** `docs/superpowers/specs/2026-05-24-sentence-streaming-design.md`

---

## File Structure & Task Dependency Graph

| File | Task | Responsibility |
|------|------|----------------|
| `DanishVoice.Native.Tests/` (new project) | 1 | xUnit test host; `InternalsVisibleTo` target |
| `AudioPcm.cs` (new) | 2 | `FloatToPcm16` conversion |
| `WavWriter.cs` (modify) | 2 | reuse `FloatToPcm16` |
| `SentenceChunker.cs` (new) | 3 | Danish sentence splitting |
| `VoiceConditioning.cs` (new) + `SynthPipeline.cs` (modify) | 4 | per-voice conditioning cache |
| `PcmSentenceStream.cs` (new) | 5a | look-ahead channel orchestration (model-free) |
| `DanishVoiceTts.cs` (modify) | 5b | public streaming API + gate |
| `DanishVoice.Native.Cli/Program.cs` (modify) | 6 | `synth-stream` subcommand + parity check |
| `*.csproj`, `README.md`, `RUNTIME.md`, release notes | 7 | version bump + docs |

**Dependencies / parallelization:**
- **Task 1** first (creates test project + `InternalsVisibleTo`; shared scaffolding).
- **Wave 1 — parallel after Task 1:** Task 2, Task 3, Task 4 (no file overlap).
- **Task 5a** needs Task 2.
- **Task 5b** needs Tasks 3, 4, 5a.
- **Wave 3 — parallel after 5b:** Task 6, Task 7.

---

## Task 1: Create xUnit test project

**Files:**
- Create: `native-onnx/DanishVoice.Native.Tests/DanishVoice.Native.Tests.csproj`
- Create: `native-onnx/DanishVoice.Native.Tests/SanityTests.cs`
- Modify: `native-onnx/DanishVoice.Native/DanishVoice.Native.csproj` (add `InternalsVisibleTo`)
- Modify: `danish-voice.sln`

- [ ] **Step 1: Create the test project file**

Create `native-onnx/DanishVoice.Native.Tests/DanishVoice.Native.Tests.csproj`:

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <TargetFramework>net10.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
    <IsPackable>false</IsPackable>
    <IsTestProject>true</IsTestProject>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="Microsoft.NET.Test.Sdk" Version="17.12.0" />
    <PackageReference Include="xunit" Version="2.9.2" />
    <PackageReference Include="xunit.runner.visualstudio" Version="3.0.0" />
  </ItemGroup>

  <ItemGroup>
    <ProjectReference Include="..\DanishVoice.Native\DanishVoice.Native.csproj" />
  </ItemGroup>

</Project>
```

- [ ] **Step 2: Add the sanity test**

Create `native-onnx/DanishVoice.Native.Tests/SanityTests.cs`:

```csharp
namespace DanishVoice.Native.Tests;

public class SanityTests
{
    [Fact]
    public void TestHostRuns()
    {
        Assert.True(true);
    }
}
```

- [ ] **Step 3: Expose internals to the test project**

In `native-onnx/DanishVoice.Native/DanishVoice.Native.csproj`, add a second
`InternalsVisibleTo` next to the existing one:

```xml
  <ItemGroup>
    <InternalsVisibleTo Include="DanishVoice.Native.Cli" />
    <InternalsVisibleTo Include="DanishVoice.Native.Tests" />
  </ItemGroup>
```

- [ ] **Step 4: Add the project to the solution**

Run: `dotnet sln danish-voice.sln add native-onnx/DanishVoice.Native.Tests/DanishVoice.Native.Tests.csproj`
Expected: "Project ... added to the solution."

- [ ] **Step 5: Run the test to verify the host works**

Run: `dotnet test native-onnx/DanishVoice.Native.Tests/DanishVoice.Native.Tests.csproj`
Expected: PASS, 1 test passed.

- [ ] **Step 6: Commit**

```bash
git add native-onnx/DanishVoice.Native.Tests danish-voice.sln native-onnx/DanishVoice.Native/DanishVoice.Native.csproj
git commit -m "test: add DanishVoice.Native.Tests xUnit project"
```

---

## Task 2: `AudioPcm.FloatToPcm16` + WavWriter refactor

**Files:**
- Create: `native-onnx/DanishVoice.Native/AudioPcm.cs`
- Modify: `native-onnx/DanishVoice.Native/WavWriter.cs`
- Test: `native-onnx/DanishVoice.Native.Tests/AudioPcmTests.cs`

**Depends on:** Task 1.

- [ ] **Step 1: Write the failing tests**

Create `native-onnx/DanishVoice.Native.Tests/AudioPcmTests.cs`:

```csharp
namespace DanishVoice.Native.Tests;

public class AudioPcmTests
{
    [Fact]
    public void FloatToPcm16_ProducesTwoBytesPerSample()
    {
        var bytes = AudioPcm.FloatToPcm16(new[] { 0f, 0f, 0f });
        Assert.Equal(6, bytes.Length);
    }

    [Fact]
    public void FloatToPcm16_EmptyInputProducesEmptyOutput()
    {
        var bytes = AudioPcm.FloatToPcm16(ReadOnlySpan<float>.Empty);
        Assert.Empty(bytes);
    }

    [Fact]
    public void FloatToPcm16_ZeroIsZeroLittleEndian()
    {
        var bytes = AudioPcm.FloatToPcm16(new[] { 0f });
        Assert.Equal(new byte[] { 0x00, 0x00 }, bytes);
    }

    [Fact]
    public void FloatToPcm16_ClampsAboveOneToMaxShort()
    {
        var bytes = AudioPcm.FloatToPcm16(new[] { 2.0f });
        // 32767 = 0x7FFF, little-endian
        Assert.Equal(new byte[] { 0xFF, 0x7F }, bytes);
    }

    [Fact]
    public void FloatToPcm16_ClampsBelowMinusOneToMinShort()
    {
        var bytes = AudioPcm.FloatToPcm16(new[] { -2.0f });
        // -32768 = 0x8000, little-endian
        Assert.Equal(new byte[] { 0x00, 0x80 }, bytes);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `dotnet test native-onnx/DanishVoice.Native.Tests --filter AudioPcmTests`
Expected: FAIL — `AudioPcm` does not exist (compile error).

- [ ] **Step 3: Implement `AudioPcm`**

Create `native-onnx/DanishVoice.Native/AudioPcm.cs`:

```csharp
using System.Buffers.Binary;

namespace DanishVoice.Native;

/// <summary>Conversions between float samples and 16-bit PCM bytes.</summary>
internal static class AudioPcm
{
    /// <summary>
    /// Convert mono float samples in [-1, 1] to little-endian 16-bit PCM bytes.
    /// Output is always 2 bytes per sample (sample-aligned). Matches the int16
    /// quantization previously inlined in <see cref="WavWriter"/>.
    /// </summary>
    public static byte[] FloatToPcm16(ReadOnlySpan<float> samples)
    {
        var result = new byte[samples.Length * 2];
        for (var i = 0; i < samples.Length; i++)
        {
            var value = (short)Math.Clamp(samples[i] * 32767f, -32768f, 32767f);
            BinaryPrimitives.WriteInt16LittleEndian(result.AsSpan(i * 2), value);
        }
        return result;
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `dotnet test native-onnx/DanishVoice.Native.Tests --filter AudioPcmTests`
Expected: PASS, 5 tests.

- [ ] **Step 5: Refactor `WavWriter` to reuse `FloatToPcm16`**

Replace the body of `native-onnx/DanishVoice.Native/WavWriter.cs` `Write` so the
sample loop is replaced by the shared conversion:

```csharp
namespace DanishVoice.Native;

/// <summary>Writes mono float samples as a 16-bit PCM WAV file.</summary>
public static class WavWriter
{
    public static void Write(string path, float[] samples, int sampleRate)
    {
        var pcm = AudioPcm.FloatToPcm16(samples);
        using var fs = new FileStream(path, FileMode.Create);
        using var bw = new BinaryWriter(fs);
        var n = samples.Length;
        var byteRate = sampleRate * 2;
        bw.Write("RIFF"u8.ToArray());
        bw.Write(36 + n * 2);
        bw.Write("WAVE"u8.ToArray());
        bw.Write("fmt "u8.ToArray());
        bw.Write(16);
        bw.Write((short)1);          // PCM
        bw.Write((short)1);          // mono
        bw.Write(sampleRate);
        bw.Write(byteRate);
        bw.Write((short)2);          // block align
        bw.Write((short)16);         // bits per sample
        bw.Write("data"u8.ToArray());
        bw.Write(n * 2);
        bw.Write(pcm);
    }
}
```

- [ ] **Step 6: Build to verify the refactor compiles**

Run: `dotnet build native-onnx/DanishVoice.Native/DanishVoice.Native.csproj`
Expected: Build succeeded.

- [ ] **Step 7: Commit**

```bash
git add native-onnx/DanishVoice.Native/AudioPcm.cs native-onnx/DanishVoice.Native/WavWriter.cs native-onnx/DanishVoice.Native.Tests/AudioPcmTests.cs
git commit -m "feat: add AudioPcm.FloatToPcm16; WavWriter reuses it"
```

---

## Task 3: `SentenceChunker` (Danish sentence splitter)

**Files:**
- Create: `native-onnx/DanishVoice.Native/SentenceChunker.cs`
- Test: `native-onnx/DanishVoice.Native.Tests/SentenceChunkerTests.cs`

**Depends on:** Task 1.

- [ ] **Step 1: Write the failing tests**

Create `native-onnx/DanishVoice.Native.Tests/SentenceChunkerTests.cs`:

```csharp
namespace DanishVoice.Native.Tests;

public class SentenceChunkerTests
{
    [Fact]
    public void Split_EmptyInput_ReturnsEmpty()
    {
        Assert.Empty(SentenceChunker.Split(""));
        Assert.Empty(SentenceChunker.Split("   "));
    }

    [Fact]
    public void Split_ThreeSentences_ReturnsThree()
    {
        var chunks = SentenceChunker.Split("Hej. Hvordan går det? Godt!");
        Assert.Equal(3, chunks.Count);
        Assert.Equal("Hej.", chunks[0]);
        Assert.Equal("Hvordan går det?", chunks[1]);
        Assert.Equal("Godt!", chunks[2]);
    }

    [Fact]
    public void Split_DanishAbbreviation_DoesNotBreak()
    {
        // "bl.a." (blandt andet) must not end a sentence.
        var chunks = SentenceChunker.Split("Vi har bl.a. æbler. Færdig.");
        Assert.Equal(2, chunks.Count);
        Assert.Equal("Vi har bl.a. æbler.", chunks[0]);
        Assert.Equal("Færdig.", chunks[1]);
    }

    [Fact]
    public void Split_SingleInitials_DoNotBreak()
    {
        var chunks = SentenceChunker.Split("H. C. Andersen skrev eventyr.");
        Assert.Single(chunks);
    }

    [Fact]
    public void Split_AddsTerminalPunctuationWhenMissing()
    {
        var chunks = SentenceChunker.Split("Ingen punktum her");
        Assert.Single(chunks);
        Assert.Equal("Ingen punktum her.", chunks[0]);
    }

    [Fact]
    public void Split_OverlongSentence_SplitsAtClauseBoundary()
    {
        var clause = new string('a', 400);
        var text = clause + ", " + clause + ".";  // 802 chars, one sentence
        var chunks = SentenceChunker.Split(text, maxChunkChars: 600);
        Assert.True(chunks.Count >= 2);
        Assert.All(chunks, c => Assert.True(c.Length <= 600));
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `dotnet test native-onnx/DanishVoice.Native.Tests --filter SentenceChunkerTests`
Expected: FAIL — `SentenceChunker` does not exist.

- [ ] **Step 3: Implement `SentenceChunker`**

Create `native-onnx/DanishVoice.Native/SentenceChunker.cs` (Danish-tuned port of
the consumer project's chunker; default ceiling 600):

```csharp
using System.Text;

namespace DanishVoice.Native;

/// <summary>
/// Splits text into sentence-sized chunks for streaming TTS synthesis.
/// Primary split: sentence boundaries (. ! ? …) with Danish abbreviation handling.
/// Fallback for overlong sentences: clause boundaries (, ; — –), then words.
/// </summary>
internal static class SentenceChunker
{
    /// <summary>Default ceiling — sentences at or below this go through unsplit.</summary>
    public const int DefaultMaxChunkChars = 600;

    private static readonly HashSet<string> abbreviations = new(StringComparer.OrdinalIgnoreCase)
    {
        // Danish common abbreviations
        "f.eks", "bl.a", "osv", "dvs", "mfl", "hhv", "ca", "nr", "stk", "kr",
        "pga", "ift", "iht", "mht", "fx", "m.m", "evt", "inkl", "ekskl",
        "o.lign", "o.a", "jf", "vha", "tlf",
        // Titles
        "hr", "fr", "frk", "dr", "prof", "adj", "cand",
    };

    public static IReadOnlyList<string> Split(string text, int maxChunkChars = DefaultMaxChunkChars)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return [];
        }

        text = NormalizeWhitespace(text);

        var sentences = SplitOnSentenceBoundaries(text);
        var result = new List<string>();

        foreach (var sentence in sentences)
        {
            var trimmed = sentence.Trim();
            if (trimmed.Length == 0)
            {
                continue;
            }

            if (trimmed.Length <= maxChunkChars)
            {
                result.Add(EnsureTerminalPunctuation(trimmed));
            }
            else
            {
                var terminator = ExtractTerminator(trimmed);
                result.AddRange(SplitLongSentence(trimmed, maxChunkChars, terminator));
            }
        }

        return result;
    }

    private static List<string> SplitOnSentenceBoundaries(string text)
    {
        var sentences = new List<string>();
        var current = new StringBuilder();

        for (var i = 0; i < text.Length; i++)
        {
            var ch = text[i];
            current.Append(ch);

            if (ch is not ('.' or '!' or '?'))
            {
                continue;
            }

            // Ellipsis (... or ..) — treat as continuation, not a boundary.
            if (ch == '.' && ((i + 1 < text.Length && text[i + 1] == '.') || (i > 0 && text[i - 1] == '.')))
            {
                continue;
            }

            // Known abbreviation or single initial before the dot.
            if (ch == '.' && IsAbbreviation(text, i))
            {
                continue;
            }

            // Intra-abbreviation dot: next char is a lowercase letter.
            if (ch == '.' && i + 1 < text.Length && char.IsLower(text[i + 1]))
            {
                continue;
            }

            if (i + 1 >= text.Length || IsFollowedByNewSentence(text, i + 1))
            {
                sentences.Add(current.ToString());
                current.Clear();
            }
        }

        if (current.Length > 0)
        {
            sentences.Add(current.ToString());
        }

        return sentences;
    }

    private static bool IsFollowedByNewSentence(string text, int startIndex)
    {
        var i = startIndex;
        while (i < text.Length && char.IsWhiteSpace(text[i]))
        {
            i++;
        }

        if (i >= text.Length)
        {
            return true;
        }

        var next = text[i];
        return char.IsLetter(next) || char.IsDigit(next) || next is '"' or '\'' or '(' or '[' or '—' or '-';
    }

    private static bool IsAbbreviation(string text, int dotIndex)
    {
        var wordStart = dotIndex - 1;
        while (wordStart >= 0 && (char.IsLetter(text[wordStart]) || text[wordStart] == '.'))
        {
            wordStart--;
        }

        wordStart++;

        if (wordStart >= dotIndex)
        {
            return false;
        }

        var word = text[wordStart..dotIndex].TrimEnd('.');

        if (abbreviations.Contains(word))
        {
            return true;
        }

        // Single uppercase letter + dot → initial (e.g. "H. C. Andersen").
        if (word.Length == 1 && char.IsUpper(word[0]))
        {
            return true;
        }

        return false;
    }

    private static List<string> SplitLongSentence(string sentence, int maxChunkChars, char terminator)
    {
        var chunks = new List<string>();
        var remaining = sentence.AsSpan();

        while (remaining.Length > maxChunkChars)
        {
            var splitIndex = FindSecondarySplitPoint(remaining, maxChunkChars);
            var isForceSplit = splitIndex <= 0;
            if (isForceSplit)
            {
                splitIndex = maxChunkChars;
                while (splitIndex > maxChunkChars / 2 && !char.IsWhiteSpace(remaining[splitIndex]))
                {
                    splitIndex--;
                }

                if (splitIndex <= maxChunkChars / 2)
                {
                    splitIndex = maxChunkChars;
                }
            }

            var chunk = remaining[..splitIndex].ToString().Trim();
            if (chunk.Length > 0)
            {
                chunks.Add(EnsureContinuingPunctuation(chunk));
            }

            remaining = remaining[splitIndex..].TrimStart();
        }

        if (remaining.Length > 0)
        {
            var last = remaining.ToString().Trim();
            if (last.Length > 0)
            {
                chunks.Add(ApplyTerminator(last, terminator));
            }
        }

        return chunks;
    }

    private static int FindSecondarySplitPoint(ReadOnlySpan<char> text, int maxChunkChars)
    {
        for (var i = Math.Min(maxChunkChars, text.Length) - 1; i > maxChunkChars / 3; i--)
        {
            if (text[i] is ',' or ';' or '—' or '–')
            {
                return i + 1;
            }
        }

        return -1;
    }

    private static string EnsureTerminalPunctuation(string text)
    {
        if (text.Length == 0)
        {
            return text;
        }

        var lastChar = text[^1];
        if (lastChar is '.' or '!' or '?' or '…')
        {
            return text;
        }

        if (lastChar is ',' or ';' or ':' or '—' or '–')
        {
            return text[..^1] + ".";
        }

        return text + ".";
    }

    private static char ExtractTerminator(string sentence)
    {
        if (sentence.Length == 0)
        {
            return '.';
        }

        var lastChar = sentence[^1];
        return lastChar switch
        {
            '.' or '!' or '?' or '…' => lastChar,
            _ => '.',
        };
    }

    private static string EnsureContinuingPunctuation(string text)
    {
        if (text.Length == 0)
        {
            return text;
        }

        var lastChar = text[^1];
        if (lastChar is ',' or ';' or '—' or '–')
        {
            return text;
        }

        if (lastChar is ':')
        {
            return text[..^1] + ",";
        }

        if (lastChar is '.' or '!' or '?' or '…')
        {
            return text[..^1] + ",";
        }

        return text + ",";
    }

    private static string ApplyTerminator(string text, char terminator)
    {
        if (text.Length == 0)
        {
            return text;
        }

        var lastChar = text[^1];
        if (lastChar == terminator || lastChar is '…')
        {
            return text;
        }

        if (lastChar is '.' or '!' or '?' or ',' or ';' or ':' or '—' or '–')
        {
            return text[..^1] + terminator;
        }

        return text + terminator;
    }

    private static string NormalizeWhitespace(string text)
    {
        var sb = new StringBuilder(text.Length);
        var prevWasSpace = false;
        foreach (var ch in text)
        {
            if (char.IsWhiteSpace(ch))
            {
                if (!prevWasSpace)
                {
                    sb.Append(' ');
                    prevWasSpace = true;
                }
            }
            else
            {
                sb.Append(ch);
                prevWasSpace = false;
            }
        }

        return sb.ToString().Trim();
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `dotnet test native-onnx/DanishVoice.Native.Tests --filter SentenceChunkerTests`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add native-onnx/DanishVoice.Native/SentenceChunker.cs native-onnx/DanishVoice.Native.Tests/SentenceChunkerTests.cs
git commit -m "feat: add Danish SentenceChunker"
```

---

## Task 4: Per-voice conditioning cache

**Files:**
- Create: `native-onnx/DanishVoice.Native/VoiceConditioning.cs`
- Modify: `native-onnx/DanishVoice.Native/SynthPipeline.cs`

**Depends on:** Task 1 (build only — this is a behavior-preserving refactor; its
correctness is verified by the build and by the model-dependent CLI in Task 6).

- [ ] **Step 1: Create the `VoiceConditioning` record**

Create `native-onnx/DanishVoice.Native/VoiceConditioning.cs`:

```csharp
namespace DanishVoice.Native;

/// <summary>
/// Per-voice conditioning tensors loaded once from the refs bundle and reused
/// across sentences. Independent of the text being synthesized.
/// </summary>
internal sealed record VoiceConditioning(
    float[] CondEmb,
    int LenCond,
    float[] PromptFeat,
    float[] Xvector,
    int MelLen1,
    int[] PromptToken);
```

- [ ] **Step 2: Add a conditioning cache + loader to `SynthPipeline`**

In `native-onnx/DanishVoice.Native/SynthPipeline.cs`, add a using and a cache
field near the other fields:

```csharp
using System.Collections.Concurrent;
```

```csharp
    private readonly ConcurrentDictionary<string, VoiceConditioning> condCache = new();
```

Add a loader method (lifts the per-voice loading currently inline in `Synth`):

```csharp
    private VoiceConditioning GetConditioning(string voice)
    {
        return this.condCache.GetOrAdd(voice, v =>
        {
            var condEmb = TensorIo.Load(this.refsDir, $"cond_{v}_t3_cond_emb");
            var lenCond = TensorIo.Shape($"cond_{v}_t3_cond_emb")[1];
            var promptFeat = TensorIo.Load(this.refsDir, $"cond_{v}_prompt_feat");
            var xvector = TensorIo.Load(this.refsDir, $"cond_{v}_xvector");
            using var meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(this.refsDir, $"cond_{v}_meta.json")));
            var melLen1 = meta.RootElement.GetProperty("mel_len1").GetInt32();
            using var ptDoc = JsonDocument.Parse(File.ReadAllText(Path.Combine(this.refsDir, $"cond_{v}_prompt_token.json")));
            var promptToken = ptDoc.RootElement[0].EnumerateArray().Select(e => e.GetInt32()).ToArray();
            return new VoiceConditioning(condEmb, lenCond, promptFeat, xvector, melLen1, promptToken);
        });
    }
```

- [ ] **Step 3: Rewrite `Synth` to use the cache**

Replace the body of `Synth` so step 2 reads from `GetConditioning` instead of
loading from disk. The method signature is unchanged:

```csharp
    public float[] Synth(string text, string voice, int maxNewTokens = 600, int seed = 1234)
    {
        // 1. text prep
        var norm = PuncNorm.Apply(text);
        var ids = this.tokenizer.Encode(norm, "da");
        var textTokens = new int[ids.Length + 2];
        textTokens[0] = this.startTextToken;
        Array.Copy(ids, 0, textTokens, 1, ids.Length);
        textTokens[^1] = this.stopTextToken;

        // 2. per-voice conditioning (cached; independent of text)
        var cond = this.GetConditioning(voice);

        // 3. T3 greedy decode
        var speech = this.t3.Generate(cond.CondEmb, cond.LenCond, textTokens, maxNewTokens);
        var speechList = speech.ToList();
        while (speechList.Count > 0 && speechList[^1] == this.stopSpeechToken)
        {
            speechList.RemoveAt(speechList.Count - 1);
        }

        // 4. flow: token = prompt_token ++ speech
        var tokenConcat = new int[cond.PromptToken.Length + speechList.Count];
        Array.Copy(cond.PromptToken, tokenConcat, cond.PromptToken.Length);
        for (var i = 0; i < speechList.Count; i++)
        {
            tokenConcat[cond.PromptToken.Length + i] = speechList[i];
        }

        var t2 = 2 * tokenConcat.Length;
        var z = GaussianNoise(80 * t2, seed);
        var flowTrace = this.flow.Run(tokenConcat, cond.PromptFeat, cond.MelLen1, cond.Xvector, z);

        // 5. vocoder
        var outFrames = flowTrace.T2 - cond.MelLen1;
        var voc = this.vocoder.Run(flowTrace.MelOut, 80, outFrames);
        return voc.Wav;
    }
```

Remove the now-unused per-call `Console.WriteLine` progress lines if present
(the `T3 decoding...` / `T3 produced...` logs) so streaming many sentences does
not spam stdout. Leave all other behavior identical.

- [ ] **Step 4: Build to verify it compiles and the refactor is clean**

Run: `dotnet build native-onnx/DanishVoice.Native/DanishVoice.Native.csproj`
Expected: Build succeeded, no warnings about unused fields.

- [ ] **Step 5: Run the full test suite (no regressions in model-free tests)**

Run: `dotnet test native-onnx/DanishVoice.Native.Tests`
Expected: PASS (all existing tests still green).

- [ ] **Step 6: Commit**

```bash
git add native-onnx/DanishVoice.Native/VoiceConditioning.cs native-onnx/DanishVoice.Native/SynthPipeline.cs
git commit -m "refactor: cache per-voice conditioning in SynthPipeline"
```

---

## Task 5a: Look-ahead channel orchestration (`PcmSentenceStream`)

**Files:**
- Create: `native-onnx/DanishVoice.Native/PcmSentenceStream.cs`
- Test: `native-onnx/DanishVoice.Native.Tests/PcmSentenceStreamTests.cs`

**Depends on:** Task 2 (`AudioPcm`). Model-free and fully unit-testable via a fake
synth delegate.

- [ ] **Step 1: Write the failing tests**

Create `native-onnx/DanishVoice.Native.Tests/PcmSentenceStreamTests.cs`:

```csharp
namespace DanishVoice.Native.Tests;

public class PcmSentenceStreamTests
{
    // Fake synth: returns N float samples where N = sentence length, so chunk
    // byte length is 2 * sentence.Length — distinguishable per sentence.
    private static float[] FakeSynth(string sentence, CancellationToken ct) =>
        new float[sentence.Length];

    [Fact]
    public async Task StreamAsync_YieldsOneChunkPerSentenceInOrder()
    {
        string[] sentences = ["ab", "cde", "f"];
        var chunks = new List<byte[]>();
        await foreach (var c in PcmSentenceStream.StreamAsync(sentences, FakeSynth))
        {
            chunks.Add(c);
        }

        Assert.Equal(3, chunks.Count);
        Assert.Equal(4, chunks[0].Length);  // "ab" -> 2 samples -> 4 bytes
        Assert.Equal(6, chunks[1].Length);  // "cde" -> 3 samples -> 6 bytes
        Assert.Equal(2, chunks[2].Length);  // "f" -> 1 sample -> 2 bytes
    }

    [Fact]
    public async Task StreamAsync_EmptyList_YieldsNothing()
    {
        var chunks = new List<byte[]>();
        await foreach (var c in PcmSentenceStream.StreamAsync([], FakeSynth))
        {
            chunks.Add(c);
        }

        Assert.Empty(chunks);
    }

    [Fact]
    public async Task StreamAsync_ProducerException_SurfacesToConsumer()
    {
        string[] sentences = ["ok", "boom", "never"];
        float[] Synth(string s, CancellationToken ct) =>
            s == "boom" ? throw new InvalidOperationException("synth failed") : new float[s.Length];

        var chunks = new List<byte[]>();
        var ex = await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await foreach (var c in PcmSentenceStream.StreamAsync(sentences, Synth))
            {
                chunks.Add(c);
            }
        });

        Assert.Equal("synth failed", ex.Message);
        Assert.Single(chunks); // only the first ("ok") made it through
    }

    [Fact]
    public async Task StreamAsync_PreCancelledToken_DoesNotComplete()
    {
        string[] sentences = ["a", "b"];
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (var c in PcmSentenceStream.StreamAsync(sentences, FakeSynth, cancellationToken: cts.Token))
            {
            }
        });
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `dotnet test native-onnx/DanishVoice.Native.Tests --filter PcmSentenceStreamTests`
Expected: FAIL — `PcmSentenceStream` does not exist.

- [ ] **Step 3: Implement `PcmSentenceStream`**

Create `native-onnx/DanishVoice.Native/PcmSentenceStream.cs`:

```csharp
using System.Runtime.CompilerServices;
using System.Threading.Channels;

namespace DanishVoice.Native;

/// <summary>
/// Drives per-sentence synthesis with look-ahead: a background producer
/// synthesizes sentences sequentially and pushes PCM chunks through a bounded
/// channel, so the next sentence synthesizes while the caller consumes the
/// current one. Mirrors the consumer project's KokoroTextToSpeech streaming.
/// </summary>
internal static class PcmSentenceStream
{
    public static async IAsyncEnumerable<byte[]> StreamAsync(
        IReadOnlyList<string> sentences,
        Func<string, CancellationToken, float[]> synthSentence,
        int channelCapacity = 3,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var channel = Channel.CreateBounded<byte[]>(new BoundedChannelOptions(channelCapacity)
        {
            SingleReader = true,
            SingleWriter = true,
            FullMode = BoundedChannelFullMode.Wait,
        });

        var producer = Task.Run(async () =>
        {
            try
            {
                foreach (var sentence in sentences)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var samples = synthSentence(sentence, cancellationToken);
                    var pcm = AudioPcm.FloatToPcm16(samples);
                    if (pcm.Length > 0)
                    {
                        await channel.Writer.WriteAsync(pcm, cancellationToken).ConfigureAwait(false);
                    }
                }

                channel.Writer.TryComplete();
            }
            catch (Exception ex)
            {
                channel.Writer.TryComplete(ex);
            }
        }, cancellationToken);

        try
        {
            await foreach (var chunk in channel.Reader.ReadAllAsync(cancellationToken).ConfigureAwait(false))
            {
                yield return chunk;
            }
        }
        finally
        {
            channel.Writer.TryComplete();
            // Observe producer faults / cancellation; swallow to avoid masking
            // an exception already surfaced by ReadAllAsync.
            try
            {
                await producer.ConfigureAwait(false);
            }
            catch
            {
                // Already surfaced to the consumer via the channel completion.
            }
        }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `dotnet test native-onnx/DanishVoice.Native.Tests --filter PcmSentenceStreamTests`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add native-onnx/DanishVoice.Native/PcmSentenceStream.cs native-onnx/DanishVoice.Native.Tests/PcmSentenceStreamTests.cs
git commit -m "feat: add PcmSentenceStream look-ahead channel orchestration"
```

---

## Task 5b: Public streaming API on `DanishVoiceTts`

**Files:**
- Modify: `native-onnx/DanishVoice.Native/DanishVoiceTts.cs`

**Depends on:** Tasks 3 (`SentenceChunker`), 4 (cached `Synth`), 5a
(`PcmSentenceStream`). Thin wiring; verified by build + Task 6 (model).

- [ ] **Step 1: Add usings, a synthesis gate, and the streaming methods**

In `native-onnx/DanishVoice.Native/DanishVoiceTts.cs`, add at the top:

```csharp
using System.Runtime.CompilerServices;
```

Add a gate field next to `pipeline`:

```csharp
    private readonly SemaphoreSlim gate = new(1, 1);
```

Add the two methods after `SynthesizeToWav`:

```csharp
    /// <summary>
    /// Stream synthesis sentence by sentence. Splits <paramref name="text"/> into
    /// sentences and yields one 16-bit little-endian PCM chunk (24 kHz mono) per
    /// sentence as it is ready — first audio after the first sentence.
    /// </summary>
    public async IAsyncEnumerable<byte[]> SynthesizeStreamingAsync(
        string text, string voice = "mic", int maxNewTokens = 600, int seed = 1234,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var sentences = SentenceChunker.Split(text);

        await this.gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            await foreach (var pcm in PcmSentenceStream.StreamAsync(
                sentences,
                (sentence, _) => this.pipeline.Synth(sentence, voice, maxNewTokens, seed),
                cancellationToken: cancellationToken).ConfigureAwait(false))
            {
                yield return pcm;
            }
        }
        finally
        {
            this.gate.Release();
        }
    }

    /// <summary>
    /// Synthesize the full <paramref name="text"/> and return one concatenated
    /// 16-bit PCM (24 kHz mono) buffer. Convenience wrapper over the stream.
    /// </summary>
    public async Task<byte[]> SynthesizeAsync(
        string text, string voice = "mic", int maxNewTokens = 600, int seed = 1234,
        CancellationToken cancellationToken = default)
    {
        using var ms = new MemoryStream();
        await foreach (var pcm in this.SynthesizeStreamingAsync(text, voice, maxNewTokens, seed, cancellationToken).ConfigureAwait(false))
        {
            ms.Write(pcm);
        }
        return ms.ToArray();
    }
```

- [ ] **Step 2: Dispose the gate**

Update `Dispose` to also dispose the gate:

```csharp
    public void Dispose()
    {
        this.pipeline.Dispose();
        this.gate.Dispose();
    }
```

- [ ] **Step 3: Build to verify it compiles**

Run: `dotnet build native-onnx/DanishVoice.Native/DanishVoice.Native.csproj`
Expected: Build succeeded.

- [ ] **Step 4: Run the full model-free suite**

Run: `dotnet test native-onnx/DanishVoice.Native.Tests`
Expected: PASS (15 tests across the suite).

- [ ] **Step 5: Commit**

```bash
git add native-onnx/DanishVoice.Native/DanishVoiceTts.cs
git commit -m "feat: add SynthesizeStreamingAsync/SynthesizeAsync to DanishVoiceTts"
```

---

## Task 6: CLI `synth-stream` subcommand + model parity check

**Files:**
- Modify: `native-onnx/DanishVoice.Native.Cli/Program.cs`

**Depends on:** Task 5b. **Model-dependent:** the parity run needs the local
runtime bundle (the `onnx_models/` + `refs/` folder). The build is verifiable in
any environment; the actual run is performed where the bundle exists (ask the
operator for the bundle path).

- [ ] **Step 1: Add the `synth-stream` subcommand**

In `native-onnx/DanishVoice.Native.Cli/Program.cs`, add a new block immediately
after the existing `synth` block (after its `return;`):

```csharp
// Streaming synthesis CLI:
//   dotnet run -- synth-stream <root> <voice> <outDir> [--cuda] <text...>
// Writes chunk_000.wav, chunk_001.wav, ... plus combined.wav. When the input is
// a single sentence, asserts that streamed PCM == one-shot Synthesize PCM.
if (args.Length >= 5 && args[0] == "synth-stream")
{
    var streamRoot = args[1];
    var streamVoice = args[2];
    var outDir = args[3];
    var streamRest = args[4..];
    var streamProvider = ExecutionProvider.Cpu;
    if (streamRest.Length > 0 && streamRest[0] == "--cuda")
    {
        streamProvider = ExecutionProvider.Cuda;
        streamRest = streamRest[1..];
    }
    var streamText = string.Join(' ', streamRest);
    Directory.CreateDirectory(outDir);

    using var streamTts = new DanishVoiceTts(streamRoot, streamProvider);
    Console.WriteLine($"Streaming [{streamVoice}] on {streamTts.ActiveProvider}: {streamText}");

    var sw = System.Diagnostics.Stopwatch.StartNew();
    var idx = 0;
    var combined = new List<byte>();
    await foreach (var pcm in streamTts.SynthesizeStreamingAsync(streamText, streamVoice))
    {
        if (idx == 0)
        {
            Console.WriteLine($"  first chunk after {sw.Elapsed.TotalSeconds:F1}s ({pcm.Length} bytes)");
        }
        var chunkSamples = new float[pcm.Length / 2];
        for (var i = 0; i < chunkSamples.Length; i++)
        {
            chunkSamples[i] = BitConverter.ToInt16(pcm, i * 2) / 32767f;
        }
        WavWriter.Write(Path.Combine(outDir, $"chunk_{idx:D3}.wav"), chunkSamples, DanishVoiceTts.SampleRate);
        combined.AddRange(pcm);
        idx++;
    }
    sw.Stop();

    var combinedArr = combined.ToArray();
    var combinedSamples = new float[combinedArr.Length / 2];
    for (var i = 0; i < combinedSamples.Length; i++)
    {
        combinedSamples[i] = BitConverter.ToInt16(combinedArr, i * 2) / 32767f;
    }
    WavWriter.Write(Path.Combine(outDir, "combined.wav"), combinedSamples, DanishVoiceTts.SampleRate);
    Console.WriteLine($"Wrote {idx} chunk(s) + combined.wav to {outDir} in {sw.Elapsed.TotalSeconds:F1}s");

    // Parity check on single-sentence input: streamed PCM must equal one-shot.
    if (idx == 1)
    {
        var oneShot = streamTts.Synthesize(streamText, streamVoice);
        var oneShotPcm = AudioPcm.FloatToPcm16(oneShot);  // CLI has InternalsVisibleTo
        var match = combinedArr.AsSpan().SequenceEqual(oneShotPcm);
        Console.WriteLine($"  single-sentence parity (stream == Synthesize): {(match ? "PASS" : "FAIL")}");
    }
    return;
}
```

- [ ] **Step 2: Make `Main` async if needed**

The CLI uses top-level statements. `await foreach` requires the entry point to be
async, which top-level statements support automatically once an `await` is
present. Confirm the file compiles with the new `await`.

Run: `dotnet build native-onnx/DanishVoice.Native.Cli/DanishVoice.Native.Cli.csproj`
Expected: Build succeeded.

- [ ] **Step 3: Run the streaming CLI against the local bundle (operator step)**

Run (substitute the real bundle path):
`dotnet run --project native-onnx/DanishVoice.Native.Cli -c Release -- synth-stream <BUNDLE_PATH> mic out_stream "Hej. Hvordan går det i dag? Det er en dejlig dag."`
Expected: 3 chunks written, "first chunk after …s" printed before total time.

Run the single-sentence parity check:
`dotnet run --project native-onnx/DanishVoice.Native.Cli -c Release -- synth-stream <BUNDLE_PATH> mic out_one "Hej, hvordan går det i dag?"`
Expected: "single-sentence parity (stream == Synthesize): PASS".

- [ ] **Step 4: Commit**

```bash
git add native-onnx/DanishVoice.Native.Cli/Program.cs
git commit -m "feat: add synth-stream CLI subcommand with parity check"
```

---

## Task 7: Version bump + docs + release notes

**Files:**
- Modify: `native-onnx/DanishVoice.Native/DanishVoice.Native.csproj`
- Modify: `README.md`
- Modify: `native-onnx/RUNTIME.md`
- Create: `release-artifacts/RELEASE_NOTES_v0.2.1.md`

**Depends on:** Task 5b (so docs match the final API).

- [ ] **Step 1: Bump the library version**

In `native-onnx/DanishVoice.Native/DanishVoice.Native.csproj`, change:

```xml
    <Version>0.2.0</Version>
```
to:
```xml
    <Version>0.2.1</Version>
```

- [ ] **Step 2: Add a streaming snippet to README**

In `README.md`, under "### Use it" (after the existing `SynthesizeToWav`
example), add:

````markdown
Streaming (first audio after the first sentence — yields 16-bit PCM per sentence):

```csharp
await foreach (byte[] pcmChunk in tts.SynthesizeStreamingAsync(
    "Hej. Hvordan går det i dag? Det er en dejlig dag.", "mic"))
{
    // pcmChunk: 24 kHz mono 16-bit little-endian PCM for one sentence
    audioSink.Write(pcmChunk);
}

// or one concatenated PCM buffer:
byte[] pcm = await tts.SynthesizeAsync("…", "mic");
```
````

- [ ] **Step 3: Note bundle reuse in RUNTIME.md**

In `native-onnx/RUNTIME.md`, add a short note (near the top or in a version
section):

```markdown
> **v0.2.1** is a library-only release: it reuses the **v0.2.0** runtime bundle
> unchanged (no new ONNX graphs). If you already have the v0.2.0 bundle, you do
> not need to re-download anything.
```

- [ ] **Step 4: Write the release notes**

Create `release-artifacts/RELEASE_NOTES_v0.2.1.md`:

```markdown
# DanishVoice.Native v0.2.1

Library-only release. **Reuses the v0.2.0 runtime bundle unchanged** — no new
ONNX graphs, nothing to re-download.

## New

- **Sentence-level streaming API** on `DanishVoiceTts`:
  - `IAsyncEnumerable<byte[]> SynthesizeStreamingAsync(text, voice, …, ct)` —
    splits text into sentences and yields one 16-bit LE PCM chunk (24 kHz mono)
    per sentence as it is ready. First audio arrives after the first sentence.
  - `Task<byte[]> SynthesizeAsync(text, voice, …, ct)` — convenience wrapper
    returning one concatenated PCM buffer.
- A look-ahead bounded-channel producer synthesizes the next sentence while the
  caller consumes the current one.

## Improved

- Per-voice conditioning tensors are now cached and reused across sentences
  (previously reloaded from disk on every call).

## Unchanged

- `float[] Synthesize(...)` and `SynthesizeToWav(...)` behave exactly as before.
- Greedy decoding; 24 kHz mono output.
```

- [ ] **Step 5: Build the whole solution**

Run: `dotnet build danish-voice.sln`
Expected: Build succeeded.

- [ ] **Step 6: Commit**

```bash
git add native-onnx/DanishVoice.Native/DanishVoice.Native.csproj README.md native-onnx/RUNTIME.md release-artifacts/RELEASE_NOTES_v0.2.1.md
git commit -m "release: bump DanishVoice.Native to 0.2.1; streaming docs"
```

---

## Final verification (after all tasks)

- [ ] `dotnet build danish-voice.sln` — succeeds.
- [ ] `dotnet test native-onnx/DanishVoice.Native.Tests` — all green (~15 tests).
- [ ] Operator runs the Task 6 CLI parity check against the local bundle — PASS.
- [ ] superpowers code review on the branch before merge.
