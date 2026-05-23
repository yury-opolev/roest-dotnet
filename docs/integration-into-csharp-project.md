# Using Røst-v3 Danish TTS from Another C# Project

Considerations for consuming this spike as a Danish text-to-speech generator
inside a separate C# application, where the use case is a **real-time chat**
and GPU acceleration is wanted.

## The key insight: native C# would not be faster

The latency of a single synthesis is dominated by the **GPU model inference**
(~1–2 s per sentence on an RTX 4070). That cost is the same no matter how the
model is invoked, because the actual compute is PyTorch/CUDA kernels either
way. A "native C#" implementation does **not** speed up generation — it would
only remove a ~1–5 ms localhost HTTP hop from a ~1500 ms operation.

So the choice between native and containerized is about **deployment
footprint** (removing the Python/Docker dependency), not performance.

## Option comparison

| Option | Real-time latency | Effort | Removes Python/Docker? | Risk |
|---|---|---|---|---|
| **Container + local HTTP API** (current) | generation-bound (~1–2 s/sentence; ms transport) | done | no | low |
| Python.NET (embed CPython in the C# process) | same generation cost, no HTTP hop | medium | no (ships Python+torch in-proc) | medium — fragile packaging, GIL/threading |
| TorchSharp (libtorch in .NET) | same generation cost | high | partially (no Python, still libtorch native libs) | high — must re-port tokenizer, sampling loop, codec |
| Native ONNX (`Microsoft.ML.OnnxRuntime`) | possibly marginally faster | very high (multi-week) | yes | very high — S3 flow-matching codec + Perth watermark have custom ops that may not export cleanly |

### Why native (ONNX/TorchSharp) is not recommended

To port Røst-v3 to pure .NET you would have to reimplement, in C#:

1. The **grapheme MTL tokenizer** (`grapheme_mtl_merged_expanded_v1.json`).
2. The **T3 Llama-0.5B backbone** with its autoregressive token-sampling loop
   (temperature / top_p / min_p / repetition-penalty), including KV-cache.
3. The **S3Gen codec**: an x-vector speaker encoder, a Matcha-style
   flow-matching decoder, and a HiFi-GAN vocoder — the riskiest part to
   export, with custom ops.
4. The **Perth neural watermark** stage.

Estimate: multiple weeks, with a genuine chance the codec needs partial
reimplementation. For zero generation-speed benefit, this only makes sense
if eliminating Docker is a hard, non-negotiable requirement. If it ever
becomes one, the better native target is a **Piper VITS** Danish fine-tune
(all-standard ops, clean ONNX export) rather than fighting the Chatterbox
codec — see [`piper-finetune-guide.md`](piper-finetune-guide.md). That trades
some quality for true single-binary native deployment.

## Recommended approach: container API + a reusable C# client

Keep the GPU container running as the inference engine; integrate via a small
HTTP client. Concretely:

1. **Extract a `DanishVoice.Client` class library** from the existing CLI
   (`src/DanishVoice.Cli/Program.cs` already contains the HTTP call). The
   consuming project references it and calls:

   ```csharp
   var client = new DanishVoiceClient("http://localhost:8000");
   byte[] wav = await client.SynthesizeAsync("Hej, hvordan går det?", Voice.Mic);
   ```

2. **Keep the model warm.** Already handled — the server loads the model once
   at startup and the container runs with `restart: unless-stopped`. No
   per-request load cost.

3. **Run the container alongside the app.** On the same machine the localhost
   HTTP overhead is negligible compared to generation time.

### Latency budget for chat

- Time-to-first-audio today ≈ full sentence generation (~1–2 s).
- Transport (localhost HTTP) ≈ single-digit ms — not worth optimizing.
- The real lever for "instant" feel is **streaming** (below), not transport
  or native code.

## The real real-time lever: streaming

Today the server returns the whole WAV after the full sentence is generated.
For a chat that should feel instant, switch to **chunked/streaming
generation** so audio starts playing within a few hundred ms:

- There is a community streaming fork of Chatterbox
  (`davidbrowne17/chatterbox-streaming`) referenced by the CoRal model card.
- Server side: stream audio chunks as they are produced (HTTP chunked
  response, Server-Sent Events, or a WebSocket).
- Client side: feed chunks into an audio output buffer as they arrive.
- Also split long assistant messages **per sentence** and pipeline them — the
  model caps at short inputs anyway, and per-sentence streaming overlaps
  generation with playback.

This is a meaningful but bounded piece of work, and it is where the
real-time experience actually comes from.

## GPU notes for the target machine

- The container already uses CUDA 12.4 and claims all GPUs via the
  `deploy.resources` block in `docker-compose.yml`.
- Requirements on the host: NVIDIA driver + Container Toolkit (Docker Desktop
  on Windows handles this with GPU support enabled).
- An RTX 4070 (8 GB) comfortably holds the 0.5 B model with room to spare.
- If the chat needs concurrency (multiple simultaneous requests), the current
  server is synchronous/single-flight; add a small request queue or run
  multiple workers — but validate that one GPU's throughput meets demand
  first.

## Summary recommendation

1. Use the **container API** — it is built, GPU-accelerated, and native C#
   would not be faster.
2. Wrap access in a small **`DanishVoice.Client`** library for the consuming
   project.
3. If "real-time" must feel instant, invest in **streaming**, not a native
   port.
4. Only pursue native .NET if removing Docker is a hard requirement — and if
   so, target a Piper VITS fine-tune, not a Chatterbox port.
