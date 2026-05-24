# roest-dotnet

Run **CoRal Røst-v3** Danish text-to-speech in .NET. Two ways:

1. **Native, in-process (no Python):** the `DanishVoice.Native` library runs the
   full model — T3 (Llama) → S3Gen flow → HiFT vocoder — via ONNX Runtime.
2. **Containerised server:** a Dockerised FastAPI server runs the original
   PyTorch model, with a thin .NET HTTP client. This is the highest-quality
   reference path and supports arbitrary text out of the box.

Two voices: **Mic** (female) and **Nic** (male). Output is 24 kHz mono.

> Built on [CoRal Røst-v3-Chatterbox-500m](https://huggingface.co/CoRal-project/roest-v3-chatterbox-500m)
> (MOS 4.23/5), a Danish fine-tune of ResembleAI Chatterbox Multilingual.
> **Model weights are OpenRAIL; this repo's code is MIT.** See
> [`NOTICE.md`](NOTICE.md) and [`LICENSE`](LICENSE).

---

## Option 1 — Native .NET library (in-process, no Python)

The model is fully reimplemented to run in .NET via `Microsoft.ML.OnnxRuntime`:
the BPE tokenizer, the T3 autoregressive loop (with KV-cache), the S3Gen flow
(conformer + flow-matching ODE), and the HiFT vocoder (NSF/STFT/iSTFT as C#
DSP). Every component is verified for numerical parity against the PyTorch
reference. See [`native-onnx/FINDINGS.md`](native-onnx/FINDINGS.md) for the full
conversion story and parity results.

### Get the runtime

1. Add the `DanishVoice.Native` library (NuGet `.nupkg` on the
   [Releases](https://github.com/yury-opolev/roest-dotnet/releases) page, or
   project-reference `native-onnx/DanishVoice.Native`).
2. Download the runtime model bundle from the same release (two zips, split for
   GitHub's 2 GiB asset limit) and unzip **both into one folder**, e.g.
   `C:\models\roest-dotnet` — they merge into `onnx_models/` + `refs/`.
   Details in [`native-onnx/RUNTIME.md`](native-onnx/RUNTIME.md).

### Use it

```csharp
using DanishVoice.Native;

using var tts = new DanishVoiceTts(@"C:\models\roest-dotnet", ExecutionProvider.Cpu);
tts.SynthesizeToWav("Hej, hvordan går det i dag?", "mic", "out.wav");   // 24 kHz mono
// or: float[] samples = tts.Synthesize("...", "nic");
```

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

CLI:

```
dotnet run --project native-onnx/DanishVoice.Native.Cli -c Release -- \
    synth C:\models\roest-dotnet mic out.wav [--cuda] "Din tekst her"
```

### Execution provider & performance

- **CPU** by default — correct everywhere, but slow (~seconds per sentence).
- **CUDA** via `ExecutionProvider.Cuda` (the library uses
  `Microsoft.ML.OnnxRuntime.Gpu`). Requires **CUDA 12.x + cuDNN 9** runtime
  libraries on the host; it falls back to CPU automatically if they're missing.
- T3 uses a **KV-cache** (prefill once, then constant-cost decode steps).

### Current limitations (native path)

- **Greedy decoding** — deterministic, but flatter prosody than the temperature
  sampling used by the container path. (Output is perceptually equivalent
  Danish; not bit-identical to PyTorch, which is itself stochastic.)
- The multilingual **alignment-stream analyzer** (extra anti-hallucination / EOS
  robustness) is not yet ported; greedy stops on natural EOS for normal text.
- No Perth watermark (unlike upstream Chatterbox).

---

## Option 2 — Containerised server + .NET client

Highest quality, supports arbitrary text and both voices via temperature
sampling. The model runs in PyTorch inside Docker; a thin .NET 10 CLI calls it
over HTTP.

### Prerequisites

- Docker Desktop (WSL2 backend on Windows)
- NVIDIA GPU + drivers + Container Toolkit (enable GPU in Docker Desktop)
- .NET 10 SDK
- ~5 GB disk; first boot downloads ~1 GB of weights into the `hf_cache` volume

### Run

```powershell
# one-liner: build CLI, start the container, wait for the model, synthesize
.\scripts\run.ps1
.\scripts\run.ps1 -Text "God morgen!" -Voice nic -Out morning.wav

# or manually
docker compose up -d
dotnet run --project src/DanishVoice.Cli -- "Hej, hvordan går det?"
```

CPU fallback and GPU notes are in [`docs/`](docs/) and the compose file.

---

## Repository layout

```
src/DanishVoice.Cli/        .NET HTTP client for the container server
server/                     Dockerised FastAPI + Røst-v3 (PyTorch)
docker-compose.yml          one-command server boot
native-onnx/
  DanishVoice.Native/       the native .NET TTS library (public DanishVoiceTts)
  DanishVoice.Native.Cli/   parity-test harness + native synth CLI
  export/                   Python scripts that export the ONNX graphs + tensors
  RUNTIME.md                how to consume the release runtime bundle
  FINDINGS.md               full ONNX conversion + parity write-up
scripts/                    run.ps1 (container), build-release-bundle.ps1
docs/                       research, datasets, fine-tuning paths, integration
```

## Regenerating the model artifacts

The ONNX graphs and reference tensors are produced from the upstream model by
`native-onnx/export/*.py` (run inside the `danish-voice-tts` Docker image) and
packed by `scripts/build-release-bundle.ps1`. They are distributed via GitHub
Releases, not committed to git.

## License & attribution

- **Code:** MIT — see [`LICENSE`](LICENSE).
- **Model weights** (release assets): derived from CoRal **Røst-v3**, licensed
  **OpenRAIL** (use-based restrictions); base Chatterbox is MIT. © Alexandra
  Institute / CoRal project and Resemble AI. See [`NOTICE.md`](NOTICE.md).
