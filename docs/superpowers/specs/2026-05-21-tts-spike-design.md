# Danish TTS Spike — Design

**Date:** 2026-05-21
**Goal:** stand up a working end-to-end Danish text-to-speech pipeline that a
.NET 10 application can call, using the best open-source Danish model
(CoRal Røst-v3-Chatterbox-500m, MOS 4.23/5, includes male and female voices).

## Decisions made during brainstorming

1. **Use CoRal Røst-v3 for quality.** Vanilla Chatterbox Multilingual is the
   fallback if its OpenRAIL licence becomes a blocker; switching is one model
   name in the server config. Out of scope for the spike.
2. **Don't try to port the model to native .NET right now.** Conversion is
   technically possible (Llama LM exportable to ONNX, eSpeak phonemizer
   P/Invokable) but the S3 neural audio codec and Perth watermark contain
   custom ops that make it a multi-week project with non-trivial risk.
   Document this in the README and revisit only if single-binary native
   deployment becomes a real requirement.
3. **Containerise the Python model behind an HTTP API.** The C# CLI stays
   fully native .NET 10 and depends on nothing but a running server.
4. **Danish-only for the spike.** Røst-v3 is the model; the API does not
   expose a language parameter. (The architecture allows swapping in vanilla
   Chatterbox Multilingual later if multilingual is wanted — that change is
   confined to the Python server.)

## Architecture

```
+----------------------------+         HTTP            +----------------------------+
|  DanishVoice.Cli (.NET 10) |  ----------------->     |  TTS server (Docker)       |
|                            |  POST /synthesize       |  FastAPI + uvicorn         |
|  - parses args             |  { text, voice }        |  + chatterbox-tts          |
|  - HttpClient              |  <- audio/wav           |  + torch (CPU or CUDA)     |
|  - writes WAV to disk      |                         |  + HF cache volume         |
+----------------------------+                         +----------------------------+
```

Two independent units, each with a single responsibility and a stable
HTTP boundary between them.

### Repo layout

```
danish-voice/
  docs/                              # already populated (research notes)
  src/
    DanishVoice.Cli/
      DanishVoice.Cli.csproj         # net10.0
      Program.cs                     # arg parsing + HTTP call + WAV save
  server/
    Dockerfile                       # python:3.11-slim base, torch CPU build
    app.py                           # FastAPI app, ~60 LOC
    requirements.txt                 # chatterbox-tts, fastapi, uvicorn, etc.
  docker-compose.yml                 # one-command boot, volume for HF cache
  danish-voice.sln
  README.md                          # how to run, "is it native" answer
  .gitignore                         # ignore bin/, obj/, __pycache__, *.wav, hf_cache/
```

### HTTP API (minimal)

| Method | Path | Request | Response | Notes |
|---|---|---|---|---|
| `POST` | `/synthesize` | JSON `{ "text": "...", "voice": "mic" \| "nic" }` | `audio/wav` (binary) | Default voice when omitted: `mic`. |
| `GET` | `/voices` | — | JSON `["mic","nic"]` | Static list for the spike. |
| `GET` | `/health` | — | `200 OK` + `{ "model_loaded": bool }` | Used by Docker healthcheck. |

Errors: 400 on empty text or unknown voice; 503 while the model is still
loading on cold start.

### C# CLI (minimal)

```
DanishVoice.Cli <text>
                [--voice mic|nic]               # default mic
                [--out <path>]                  # default out.wav
                [--server <url>]                # default http://localhost:8000
```

Single file, no DI, no logging framework — `System.Net.Http.HttpClient` and
`System.IO.File.WriteAllBytesAsync`. Exits non-zero with a readable message
on connection failure or non-2xx response.

### Server internals

- `chatterbox-tts` is heavy to import (~5 s) and the model is ~1 GB on disk.
  Load once at FastAPI startup; serve subsequent requests off the warm model.
- The 2 predefined voices (Mic, Nic) come from the model package; no extra
  files needed.
- Generation is synchronous PyTorch inference; FastAPI's threadpool offloads
  it from the event loop. Good enough for a single-user spike; no batching.
- Output is written to an in-memory `BytesIO`, returned as `Response(content=...,
  media_type="audio/wav")`. No temp files.

### Container

- Base: `python:3.11-slim` (small, has `apt`).
- System deps: `libsndfile1`, `ffmpeg` (for audio I/O), `espeak-ng` (Danish
  phonemes, useful even though Chatterbox does its own tokenization — leaving
  it in costs ~10 MB and avoids surprises).
- Python deps: pinned in `requirements.txt`.
- HuggingFace cache (`/root/.cache/huggingface`) lives on a named Docker
  volume so the 500 M weights survive `docker compose down`/`up`.
- CPU-only torch wheel by default; document the swap to CUDA in the README.

## Error handling

- Server: catch unknown voices and empty text at request-validation time
  (Pydantic). Wrap the model call in try/except, return 500 with a short
  message on failure. No stack traces over the wire.
- CLI: catch `HttpRequestException` (server down / network), report the URL
  the user tried. Non-2xx responses → print the server's error body and
  exit with code 1.

## Testing

Spike-level — no automated test suite. Manual smoke test:

1. `docker compose up -d`
2. `curl http://localhost:8000/health` → 200 with `model_loaded: true`
   (may take ~30 s on first run while the model downloads).
3. `dotnet run --project src/DanishVoice.Cli -- "Hej, hvordan går det?"`
   → produces `out.wav`, both Mic and Nic playable.
4. Same on Windows host + Docker Desktop (the user's environment).

## Explicitly out of scope

- Auth, HTTPS, rate limiting.
- Streaming responses (TTS streaming would be a nice next step).
- Zero-shot voice-cloning endpoint (architecture supports it; not in the spike).
- GPU autodetect, mixed-precision, batching, request queue.
- Multilingual operation (would require switching to vanilla Chatterbox).
- Native .NET execution (covered above — not a spike).
- Packaging the CLI as a single self-contained exe.
- Telemetry, structured logging.
- An automated test suite.

## Open questions deferred to implementation

- CPU inference speed of Røst-v3 on a laptop — if it's painfully slow (>30 s
  for a sentence), README will recommend the CUDA path.
- Whether the first-run model download is reasonable to do inside
  `docker compose up` or whether to pre-bake the weights into the image. Will
  decide after measuring; default is "download on first run".
