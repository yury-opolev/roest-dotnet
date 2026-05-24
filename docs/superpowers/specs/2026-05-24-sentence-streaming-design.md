# Sentence-level streaming for `DanishVoice.Native` (v0.2.1)

**Date:** 2026-05-24
**Status:** Approved design — ready for implementation plan
**Scope:** roest-dotnet library only. Cortex rewiring is a separate follow-up.

## Goal

Give the native in-process TTS library a **sentence-level streaming API** so the
first audio is available after the first sentence is synthesized, instead of
waiting for the whole utterance. The contract mirrors the consumer project's
existing TTS abstraction (`Cortex.Contained.Speech.ITtsProvider`) so a future
native-backed provider in cortex can drop its Docker/HTTP sidecar and call this
library in-process.

### Why sentence-level

- It is the practical win: time-to-first-audio drops from "whole text" to "one
  sentence" with a small, well-bounded change.
- It aligns with the model's own need to keep T3 input length bounded
  (`maxNewTokens = 600`).
- Intra-sentence frame streaming (chunked/causal flow + vocoder) is materially
  harder and is explicitly out of scope.

## Reference: the contract we are matching

From the consumer repo (`cortex`), the canonical TTS provider interface:

```csharp
Task<byte[]>             SynthesizeAsync(string text, string voiceName, CancellationToken ct);
IAsyncEnumerable<byte[]> SynthesizeStreamingAsync(string text, string voiceName, CancellationToken ct);
AudioFormat OutputFormat { get; }   // (SampleRate, Channels, BitsPerSample)
bool        SupportsStreaming { get; }
```

Key facts established by reading the cortex code:

- **Chunks are raw little-endian 16-bit mono PCM `byte[]`** — no WAV header —
  one (or more) per sentence. Downstream resamples each chunk independently, so
  every chunk's byte count must stay sample-aligned (even).
- Cortex's Røst provider reports `OutputFormat = AudioFormat.Kokoro` =
  **24 kHz mono 16-bit**, which exactly matches `DanishVoiceTts.SampleRate =
  24000` — no resampling at the source.
- Cortex's `KokoroTextToSpeech` streams via a **bounded `Channel` + look-ahead**
  worker; the Røst sidecar streams server-side. We mirror the Kokoro pattern
  in-process.

**Dependency direction:** roest-dotnet is the lower-level library and must NOT
depend on cortex. The new API uses BCL types only (`IAsyncEnumerable<byte[]>`,
`string voice`, `CancellationToken`). Cortex later writes a thin provider that
delegates to it.

## Public API (additions to `DanishVoiceTts`)

The existing `float[] Synthesize(...)` and `void SynthesizeToWav(...)` are
unchanged. Two methods are added:

```csharp
/// Yields one 16-bit LE PCM (24 kHz mono) byte[] per sentence, as each is ready.
public IAsyncEnumerable<byte[]> SynthesizeStreamingAsync(
    string text, string voice = "mic",
    int maxNewTokens = 600, int seed = 1234,
    CancellationToken cancellationToken = default);

/// Convenience: drains the stream into one concatenated PCM buffer.
public Task<byte[]> SynthesizeAsync(
    string text, string voice = "mic",
    int maxNewTokens = 600, int seed = 1234,
    CancellationToken cancellationToken = default);
```

- `float[] Synthesize` stays the "raw samples" path; PCM `byte[]` is the
  streaming / cortex-facing path.
- The method shapes match `ITtsProvider` 1:1 (modulo the extra `maxNewTokens` /
  `seed` knobs, which the cortex provider can leave at defaults), so the cortex
  native provider becomes a near pass-through.

## Components

### 1. `SentenceChunker` (new, internal static)

A Danish-tuned port of cortex's `SentenceChunker`:

- Normalize whitespace.
- Primary split on sentence boundaries `.` `!` `?` (and `…`) with:
  - ellipsis handling (`...`/`..` treated as continuation),
  - abbreviation handling using a **Danish** abbreviation set,
  - single-uppercase-initial guard (e.g. "H. C. Andersen"),
  - lowercase-follows guard (intra-abbreviation dots).
- Overlong sentences (`> maxChunkChars`, default **600** — matching cortex)
  fall back to clause boundaries (`,` `;` `—` `–`), then to word boundaries,
  then to a hard split. The overlong fallback also protects the model's bounded
  T3 length.
- Returns `IReadOnlyList<string>`; empty/whitespace input → empty list.

Danish abbreviation set (initial): `f.eks, bl.a, osv, dvs, mfl, hhv, ca, nr,
stk, kr, pga, ift, iht, mht, fx, m.m, evt, inkl, ekskl, hr, fr, frk, dr, prof`.

Each resulting chunk continues to flow through the existing `PuncNorm.Apply`
inside the per-sentence synth path (unchanged behavior).

### 2. Per-voice conditioning cache (refactor in `SynthPipeline`)

Today `Synth` reloads conditioning tensors (`cond_emb`, `prompt_feat`,
`xvector`, `mel_len1`, `prompt_token`) from disk on **every** call. For N
sentences that is N redundant disk loads.

- Introduce a `VoiceConditioning` record holding the loaded per-voice tensors
  and scalars.
- Cache it per voice in a `Dictionary<string, VoiceConditioning>` on the
  pipeline (loaded once on first use of a voice).
- Split `Synth` into: resolve-conditioning (cached) →
  `SynthSentence(VoiceConditioning cond, string sentence, int maxNewTokens,
  int seed) → float[]` performing the existing steps (punc_norm → tokenize →
  T3 → flow → vocoder).
- The existing one-shot `Synth(text, voice, ...)` routes through the same
  cached path, preserving current behavior for `Synthesize`.

### 3. Streaming engine (look-ahead via bounded channel)

`SynthesizeStreamingAsync`:

1. Split `text` into sentences via `SentenceChunker`.
2. Create a bounded `Channel<byte[]>` (capacity ~3, `SingleReader`,
   `SingleWriter`, `FullMode = Wait`).
3. Background producer (`Task.Run`) iterates sentences:
   `cancellationToken.ThrowIfCancellationRequested()` →
   `SynthSentence(cond, sentence, maxNewTokens, seed)` (blocking CPU on the
   worker thread) → `AudioPcm.FloatToPcm16(...)` → `await writer.WriteAsync`.
   On normal finish `writer.TryComplete()`; on exception
   `writer.TryComplete(ex)` so the consumer observes the fault.
4. Consumer `await foreach`-es `channel.Reader.ReadAllAsync(ct)` and yields each
   chunk.

Effect: sentence N+1 synthesizes while the caller consumes N (look-ahead
buffering, not parallel inference); first audio after sentence 1.

A `SemaphoreSlim` gate on the instance serializes synthesis (the ONNX sessions
are not concurrency-safe). It is acquired by the producer for the duration of a
streaming run and is also used by the one-shot synth paths, mirroring cortex's
`KokoroTextToSpeech` gate. `SynthesizeAsync` (batch) drains
`SynthesizeStreamingAsync` into a single buffer.

### 4. `AudioPcm.FloatToPcm16` (new, internal static helper)

`internal static byte[] FloatToPcm16(ReadOnlySpan<float> samples)` — little-endian,
clamped to `[-1, 1]`, output always even-length (2 bytes/sample), so chunks are
inherently sample-aligned. `WavWriter` is refactored to reuse this conversion
instead of its inline clamp loop.

## Determinism

The same `seed` is applied to every sentence — deterministic and reproducible.
The Gaussian-noise length differs per sentence (it scales with token count), so
each sentence's noise differs naturally.

**Expected non-equivalence:** streamed per-sentence output is NOT bit-identical
to one-shot whole-text synthesis, because splitting changes the T3 token
sequence per chunk. This is expected behavior, not a regression. The valid
parity check is single-sentence streaming vs `Synthesize` of that same sentence.

## Verification

### Unit tests (new xUnit project — repo's first)

Pure logic, no model required (fast):

- `SentenceChunker`: Danish abbreviation cases (`f.eks.`, `bl.a.`, `osv.`),
  multi-sentence splitting, ellipsis continuation, single-initial guard,
  overlong sentence falling back to clause then word boundaries, empty input.
- `AudioPcm.FloatToPcm16`: even-length output, clamping at ±1, endianness, a
  known sample → known bytes.

### Model-dependent verification (CLI subcommand)

New `synth-stream` subcommand in `DanishVoice.Native.Cli` (uses the local
runtime bundle):

- chunk count == sentence count for a known multi-sentence Danish paragraph,
- every chunk non-empty and even-length,
- **single-sentence streaming == `Synthesize`** of that sentence (converted to
  PCM) — the correct parity check,
- optionally writes per-sentence WAVs and a concatenated WAV for listening.

## Versioning & docs

- Bump `DanishVoice.Native.csproj` `<Version>` `0.2.0` → **0.2.1**.
- **Runtime bundle is unchanged** — no new ONNX graphs. v0.2.1 is a
  library-only release that reuses the existing v0.2.0 runtime bundle. Call this
  out in the release notes so consumers do not re-download.
- README: add a streaming usage snippet alongside the existing example.
- RUNTIME.md: note that v0.2.1 reuses the v0.2.0 bundle.
- Add `release-artifacts/RELEASE_NOTES_v0.2.1.md`.

## Out of scope

- Intra-sentence (frame-level) streaming.
- Rewiring cortex's `RoestDanishTtsProvider` to the native library (separate
  follow-up in the cortex repo).
- Any change to the ONNX graphs or the runtime bundle.
- Temperature sampling / prosody changes (still greedy decode).

## File-level change summary

| File | Change |
|------|--------|
| `native-onnx/DanishVoice.Native/DanishVoiceTts.cs` | Add `SynthesizeStreamingAsync`, `SynthesizeAsync`; gate. |
| `native-onnx/DanishVoice.Native/SynthPipeline.cs` | Conditioning cache; `SynthSentence`; streaming producer. |
| `native-onnx/DanishVoice.Native/SentenceChunker.cs` | New: Danish sentence splitter. |
| `native-onnx/DanishVoice.Native/VoiceConditioning.cs` | New: cached per-voice tensors record. |
| `native-onnx/DanishVoice.Native/AudioPcm.cs` | New: `FloatToPcm16`. |
| `native-onnx/DanishVoice.Native/WavWriter.cs` | Reuse `AudioPcm.FloatToPcm16`. |
| `native-onnx/DanishVoice.Native/DanishVoice.Native.csproj` | Version → 0.2.1. |
| `native-onnx/DanishVoice.Native.Cli/Program.cs` | New `synth-stream` subcommand. |
| `native-onnx/DanishVoice.Native.Tests/` (new) | xUnit: chunker + PCM tests. |
| `README.md`, `native-onnx/RUNTIME.md` | Streaming docs; bundle-reuse note. |
| `release-artifacts/RELEASE_NOTES_v0.2.1.md` (new) | Release notes. |
