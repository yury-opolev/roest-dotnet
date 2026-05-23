# danish-voice

Spike: high-quality Danish text-to-speech callable from a .NET 10 console
application. Uses [CoRal Røst-v3-Chatterbox-500m](https://huggingface.co/CoRal-project/roest-v3-chatterbox-500m)
(MOS 4.23/5, includes male and female Danish voices) running in a Docker
container, with a thin C# CLI talking to it over HTTP.

Background research and the design rationale live under [`docs/`](docs/).

## Architecture

```
┌────────────────────────────┐    HTTP     ┌────────────────────────────┐
│  DanishVoice.Cli (.NET 10) │ ──────────► │  TTS server (Docker)       │
│  - parses CLI args         │  POST       │  FastAPI + Chatterbox      │
│  - HttpClient              │  /synthesize│  + Røst-v3 model           │
│  - writes WAV to disk      │ ◄────────── │                            │
└────────────────────────────┘  audio/wav  └────────────────────────────┘
```

## Why a sidecar instead of "native .NET"

The model is a 500 M-parameter Llama-based audio LM plus a custom neural
codec (Resemble's S3) plus a Perth watermark stage — all PyTorch, no ONNX
export published. Porting it to pure `Microsoft.ML.OnnxRuntime` is a
realistic engineering project (estimate: 2–4 weeks, with the codec carrying
most of the risk) but well beyond a spike. The containerised sidecar gets
you the quality today; the C# side is fully native .NET 10 and depends only
on a running HTTP endpoint, so swapping the engine later (custom fine-tune,
ONNX-ified Piper, etc.) is a server-only change.

## Prerequisites

- **Docker Desktop** (with WSL2 backend on Windows)
- **NVIDIA GPU + drivers + Container Toolkit.** Docker Desktop on Windows
  picks up the GPU automatically once "Enable GPU support" is on in
  Settings → Resources → WSL Integration / GPU and the NVIDIA driver on the
  host is recent. Verify with `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`.
  See "CPU fallback" below if no GPU is available.
- **.NET 10 SDK**
- **~5 GB free disk** for the model, CUDA libs and Python deps. First boot
  downloads ~1 GB of weights from Hugging Face into a named volume
  (`hf_cache`); subsequent starts are fast.

## Run — the one-liner

A PowerShell script wraps the whole flow: build the CLI, build+start the
container, wait for the model to load, then synthesize a test sentence.

```powershell
.\scripts\run.ps1
# -> writes out.wav with the default voice (mic = female) saying
#    "Hej, hvordan går det?"
```

Useful flags:

```powershell
.\scripts\run.ps1 -Text "God morgen!" -Voice nic -Out morning.wav
.\scripts\run.ps1 -NoSynth                  # boot only, don't synthesize
.\scripts\run.ps1 -SkipBuild -SkipDocker    # just synthesize against a running stack
.\scripts\run.ps1 -HealthTimeoutSeconds 900 # longer wait on slow networks
```

## Run — manually if you prefer

```powershell
# 1. Start the TTS server (first run downloads the model — wait ~1–3 minutes).
docker compose up -d --build
docker compose logs -f tts   # watch until you see "Model loaded"

# 2. Sanity check
curl http://localhost:8000/health

# 3. Synthesize from the CLI
dotnet run --project src/DanishVoice.Cli -- "Hej, hvordan går det?"
# -> writes out.wav in the current directory
Start-Process out.wav        # play it
```

## CLI reference

```
DanishVoice.Cli <text> [--voice mic|nic] [--out <path>] [--server <url>]
```

| Flag | Default | Meaning |
|---|---|---|
| `<text>` | — | Danish text to synthesize (required, positional) |
| `--voice` | `mic` | Speaker. `mic` = female, `nic` = male (per CoRal-TTS dataset metadata) |
| `--out` | `out.wav` | Output file path |
| `--server` | `http://localhost:8000` | TTS server base URL |

Exit codes: `0` success, `1` bad usage, `2` server unreachable, `3` server
returned non-2xx.

## HTTP API

| Method | Path | Body | Response |
|---|---|---|---|
| `POST` | `/synthesize` | `{ "text": "...", "voice": "mic" \| "nic" }` | `audio/wav` |
| `GET` | `/voices` | — | `["mic","nic"]` |
| `GET` | `/health` | — | `{ "model_loaded": bool, "device": "cpu"\|"cuda", "model": "..." }` |

## CPU fallback

The repo defaults to **GPU (CUDA 12.4)**. If you do not have an NVIDIA GPU
exposed to Docker, revert to CPU with these edits:

1. `server/Dockerfile`:
   - Replace `FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` with
     `FROM python:3.11-slim`.
   - Drop the `python3 python3-venv python3-pip` line from `apt install`.
   - Remove the venv setup (`python3 -m venv /opt/venv` and the
     `PATH=/opt/venv/bin:$PATH` env var).
   - Change `--extra-index-url https://download.pytorch.org/whl/cu124` to
     `https://download.pytorch.org/whl/cpu`.
2. `docker-compose.yml`:
   - Delete the `deploy:` block.

Expect ~10–30 s per sentence on a laptop CPU vs. sub-second on a discrete
GPU.

## Known limitations (it's a spike)

- Synchronous generation, no batching, no streaming.
- No auth, no HTTPS — bind to `localhost` or run behind a reverse proxy.
- Voice cloning (zero-shot from a reference clip) is supported by the model
  but not yet exposed by the API.
- The exact keyword used by Chatterbox to select between Mic and Nic isn't
  documented on the model card. `server/app.py` tries `speaker_id`, then
  `speaker`, then falls back to the default speaker — so the request will
  always succeed, but voice differentiation may need a tweak once the model
  is up and we can inspect the library.

## Repo layout

```
src/DanishVoice.Cli/          .NET 10 console — HttpClient + arg parser
server/                       FastAPI + Chatterbox, containerised
  Dockerfile
  app.py
  requirements.txt
docker-compose.yml            One-command boot
docs/                         Research notes, datasets, fine-tuning paths
docs/superpowers/specs/       Design document for this spike
```
