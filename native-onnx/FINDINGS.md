# Native ONNX-in-C# Feasibility — Findings

Spike to determine whether Røst-v3 (Chatterbox Multilingual fine-tune) can be
converted to ONNX and run **in-process in C#** via `Microsoft.ML.OnnxRuntime`,
removing the Python/Docker dependency.

All probes ran inside the existing `danish-voice-tts:spike` image against the
cached model, mounting only this `native-onnx/` folder. No existing project
files were touched. Scripts are in `export/`, raw logs are the `probe_*.txt`
files, exported graphs are in `onnx_models/`.

## Verdict: FEASIBLE, but a substantial build (≈1–2+ weeks)

It is **not impossible** — the opposite of what we feared. Four of the five
neural networks export to ONNX with the stock exporter, including the
make-or-break autoregressive transformer. The one blocker (the vocoder) has a
known, deterministic workaround. The remaining cost is real C# engineering and
numerical-parity debugging, not research risk.

## Export feasibility matrix

| Component | Params | ONNX export | Notes |
|---|---|---|---|
| VoiceEncoder (LSTM) | 1.4 M | ✅ OK (5.8 MB) | speaker embed for T3 conditioning |
| **T3 Llama core** (`tfmr`) | 503 M | ✅ OK (2.0 GB fp32) | the autoregressive backbone — traces cleanly |
| Conformer encoder (flow) | 38 M | ✅ OK (179 MB) | flow-path encoder |
| CFM flow decoder | 71 M | ✅ OK (302 MB) | flow-matching decoder (ODE step) |
| HiFT vocoder | 21 M | ❌ blocked | see below |

### Why the vocoder is blocked (and the fix)

Two export-hostile pieces, both in `chatterbox/models/s3gen/hifigan.py`:

1. **NSF harmonic-source generator** — builds the F0 harmonic matrix with an
   in-place loop (`F_mat[:, i:i+1, :] = ...`). The dynamo exporter rejects this
   ("cannot mutate tensors with frozen storage").
2. **iSTFT** — the final synthesis step uses complex types; the legacy
   exporter errors with "STFT does not currently support complex types".

Both are **deterministic DSP** (no learned weights in the iSTFT; the NSF source
is a closed-form harmonic generator). Fix = split the vocoder:

- **ONNX part:** the conv network (`conv_pre`, `ups`, `resblocks`,
  `conv_post`) → magnitude + phase spectrogram.
- **C# part:** NSF harmonic-source generation from F0, and the iSTFT
  overlap-add. ~100 lines of portable DSP, verified for parity.

## What the full C# build still requires

Export feasibility ≠ done. The real work, in rough effort order:

1. **T3 autoregressive loop in C#** (largest piece):
   - Re-export `tfmr` with **past-key-value I/O** (the probe exported a single
     forward without KV cache; proper incremental decode needs cache in/out —
     standard Llama-to-ONNX work, e.g. via `optimum`, but the T3 wrapper is
     custom).
   - Reimplement T3's input assembly: `text_emb` + `speech_emb` +
     learned position embeddings, the `cond_enc` Perceiver speaker
     conditioning, BOS/EOS handling.
   - Reimplement the sampling loop (temperature, top_p, min_p,
     repetition_penalty) and `speech_head` logits in C#.
2. **Flow path in C#:** conformer encoder → CFM decoder run inside an
   ODE-solver loop (10 timesteps), plus CFG (the model uses cfg_weight=0.5,
   i.e. two-pass conditional/unconditional).
3. **Vocoder split** (task above).
4. **Grapheme tokenizer in C#** (`grapheme_mtl_merged_expanded_v1.json`) —
   cleanly native, no ONNX.
5. **Precompute Mic/Nic conditioning offline** (CAMPPlus speaker encoder +
   S3Tokenizer + VoiceEncoder) and ship the resulting tensors. This keeps
   those three networks out of the realtime C# path entirely.
6. **End-to-end numerical-parity testing** against the container reference at
   each stage — the part most likely to consume time.

## Important caveats

- **No speed win.** As established in
  [`../docs/integration-into-csharp-project.md`](../docs/integration-into-csharp-project.md),
  native ONNX does not make generation faster — the GPU inference cost is the
  same. The only payoff here is **removing the Python/Docker dependency** so
  the model runs in-process in the C# app.
- **Model size.** T3 alone is ~2 GB in fp32. Expect ~2.5–3 GB of ONNX assets
  total (or roughly half that with fp16/int8 quantization, which also helps
  load time and VRAM — worth doing).
- **Watermark dropped.** The Perth watermark stage is excluded from the native
  pipeline for this experiment.
- **Maintenance.** A hand-ported pipeline must be re-validated whenever the
  upstream model/library changes. The container path has none of this burden.

## STATUS: full native C# TTS — arbitrary text, both voices ✅

The native C# pipeline now synthesizes **any** Danish text in **Mic or Nic**
with no Python at inference time:

```
dotnet run -c Release -- synth <native-onnx-dir> <mic|nic> <out.wav> "Din tekst her"
```

`SynthPipeline.cs` chains: punc_norm -> MTL tokenizer (+ start/stop text tokens)
-> T3 greedy autoregressive loop -> drop stop-speech token -> flow (input_embed +
conformer + encoder_proj + speaker affine + cond + CFM ODE) -> HiFT vocoder ->
WAV. Per-voice conditioning (T3 cond_emb, flow prompt_token/prompt_feat/xvector)
is precomputed for Mic & Nic in `refs/cond_<voice>_*` (the VE/CAMPPlus/
S3Tokenizer networks are NOT needed at runtime). Verified: "God morgen,
Danmark." (a sentence never used in any reference) renders in both voices,
~2 s of audio in ~21 s on CPU.

Current limitations: greedy decoding (deterministic but flatter than temperature
sampling); CPU execution (CUDA EP makes it fast); T3 alignment analyzer not yet
ported (greedy stops fine on natural EOS for normal sentences). These are
quality polish, not correctness gaps.

### KV-cache (v0.2)

The T3 decode loop now uses a KV cache instead of re-running the full sequence
each step (O(n) instead of O(n^2)). Two extra graphs exported via the dynamo
exporter (`native-onnx/export/15_t3_kvcache_export.py`):

- `t3_prefill.onnx` — inputs_embeds (B,L,D) -> hidden + present K/V per layer.
- `t3_decode.onnx` — inputs_embeds (B,1,D) + past K/V -> hidden + present K/V.

30 layers x (K,V), 16 KV heads, head_dim 64. `T3Model` auto-uses these when
present and falls back to `t3_backbone.onnx` otherwise. Verified: greedy token
parity 8/8 identical to the no-cache path; PyTorch decode-step parity 4.8e-6.
Note: each graph embeds its own ~2 GB weight copy (the dynamo exporter
duplicates them) — dedupe to a shared external-data file before shipping if
release size matters.

## (earlier) end-to-end verification

The full chain — tokenizer -> T3 -> flow -> vocoder -> WAV — is assembled and
runs in C# (`DanishVoice.Native`, Test 8), producing `native_csharp_out.wav`.
Stage-by-stage verification against a single deterministic Python run:

- T3 greedy tokens: exact (8/8)
- flow mu (input_embed + conformer + encoder_proj): 2e-4
- flow speaker embedding (affine): 3e-8 (exact)
- flow cond assembly: 0.0 (exact)
- vocoder input mel: mean 9e-4
- final WAV: produced; differs from a *stochastic-NSF* Python reference only by
  the inherent NSF random phase/noise (perceptually identical). With NSF made
  deterministic on both sides the wav matches at CPU/GPU-numerics level.

Remaining for production polish (not blockers to "runs natively"):
- T3 alignment-stream analyzer (better EOS/anti-hallucination) — needs attention
  export; greedy loop works without it.
- KV-cache T3 export + CUDA execution provider for speed.
- Offline conditioning precompute pipeline for arbitrary new voices (current
  Mic conditioning is captured/dumped).

## Build progress (verified in C#)

The native build is underway in `DanishVoice.Native/`. Components proven with
numerical parity against the PyTorch/CUDA reference (run `dotnet run -c Release`
in that project; references are in `refs/`, ONNX graphs in `onnx_models/`):

| Component | Where | Parity vs Python | Status |
|---|---|---|---|
| MTL tokenizer (BPE, Danish) | `Tokenizer/MtlTokenizer.cs` | **exact** (9/9 sentences) | done |
| HiFT vocoder: f0_predictor (ONNX) | `Vocoder.cs` | max 3.4e-4 | done |
| HiFT vocoder: NSF source (C#) | `Dsp/NsfSource.cs` | **exact** (0.0) | done |
| HiFT vocoder: STFT (C#) | `Dsp/ShortTimeFourier.cs` | 7.7e-10 | done |
| HiFT vocoder: full waveform | `Vocoder.cs` | mean 1e-4 (CPU/GPU conv numerics) | done |
| T3 backbone (Llama 500M) + speech_head | `t3_backbone.onnx` + C# matmul | hidden 5e-6, logits 1e-5 | verified |
| T3 autoregressive loop (CFG + processors) | `T3/T3Model.cs` | **exact (8/8 greedy tokens)** | done (analyzer pending) |
| CFM flow decoder (ODE solver, z input) | `cfm_decoder_z.onnx` | mean 2e-3 (CPU/GPU over 10 ODE steps) | verified |
| Conformer encoder | `conformer_encoder.onnx` | exported (single ONNX call) | pending wiring |

**Every model component now runs in C#.** The CFM was re-exported with the noise
`z` as an explicit input (`cfm_decoder_z.onnx`) to make the stochastic ODE
solver deterministic for parity; supply the same z as Python to reproduce, or a
fresh random z for production. Linear weights (input_embedding, spk affine,
encoder_proj) and all flow references are dumped in `refs/`.

The T3 loop reproduces the Python greedy token sequence exactly: input
construction (cond_emb + CFG text/speech embeddings + learned pos), CFG combine,
repetition-penalty/temperature/min_p/top_p, backbone, speech_head, and the
autoregressive feedback are all verified. Remaining for T3: the alignment-stream
analyzer (needs attention export) and switching to multinomial+CUDA for prod.

### Key technical decisions / findings during the build

- **Vocoder split**: HiFT can't export whole (NSF in-place loop + iSTFT complex
  types). Solution: f0_predictor + conv stack to ONNX; NSF source generation,
  forward STFT, and iSTFT done in C# DSP. n_fft=16/hop=4 so a direct DFT is fine.
- **T3 backbone export**: the legacy ONNX exporter fails on `aten::diff` (in
  transformers' causal-mask prep). The **dynamo exporter** (`dynamo=True`,
  needs `onnxscript`) works, with a dynamic sequence-length dim. Weights are
  stored externally as `t3_backbone.onnx.data` (~2 GB) — ORT auto-loads it.
- **Full-sequence (no KV cache)**: the backbone is exported stateless
  (inputs_embeds -> last_hidden_state) and re-run per step. Correct but O(n^2);
  KV-cache export is a later optimization.
- **T3 sampling is stochastic** (`torch.multinomial`), so bit-exact parity is
  impossible by nature — Python varies run-to-run. Parity strategy: match the
  deterministic logits path exactly (done: 1e-5), then sample from the same
  distribution (use greedy for reproducible end-to-end checks).
- **Speaker conditioning is offline**: VE/CAMPPlus/S3Tokenizer only build the
  per-voice conditioning (Mic/Nic). Precompute once in Python, ship tensors.
  Keeps three networks out of the realtime C# path. (VE's ONNX export also had
  a parity mismatch — another reason to keep it offline.)

### Remaining work to end-to-end (all components individually verified)

Every component is proven in C#. What's left is assembly + a couple of pieces:

0. **Re-export `conformer_encoder.onnx` with dynamic axes + named inputs**
   (it was exported during probing with a fixed shape, so it rejects other
   lengths — `[ErrorCode:InvalidArgument] ... onnx::MatMul_0`). Same one-line
   fix as the other graphs: set `input_names`/`output_names` + `dynamic_axes`
   for the length dim. The graph itself is correct (it ran during probing).
1. **Flow assembly in C#** (`Flow*.cs`): input_embedding(concat(prompt_token,
   speech_tokens))*mask -> `conformer_encoder.onnx` -> encoder_proj (linear) =
   mu -> build cond (zeros with prompt_feat in first mel_len1 frames) -> spk
   affine (normalize xvector + linear) = spks -> `cfm_decoder_z.onnx`(mu, mask,
   spks, cond, z) -> mel[:, :, mel_len1:]. All weights/refs in `refs/flow_*`.
   The CFM and (by export) conformer are verified; this is wiring + the two
   linears + cond assembly.
2. **Offline conditioning precompute** for Mic/Nic: run prepare_conditionals in
   Python once per voice and dump t3 cond_emb (have it for the test sentence),
   flow prompt_token, prompt_feat, and the CAMPPlus xvector. These are
   per-voice constants the C# app loads.
3. **Wire** tokenizer -> T3 -> flow -> vocoder; compare end-to-end audio to the
   Python reference (expect perceptual equivalence, not bit-exact, due to
   stochastic T3 sampling + CFM noise).
4. **T3 alignment-stream analyzer** (production quality): suppresses early EOS &
   catches hallucinations using attention maps from heads (12,15),(13,11),(9,2).
   Needs those attentions exported (or a simplified EOS-suppression heuristic).
   The greedy loop works without it; quality/robustness benefits from it.
5. Switch ORT execution provider to **CUDA** for performance (also tightens the
   CPU/GPU numerical deltas seen in the vocoder conv and CFM ODE).

### How to run the parity harness

`cd DanishVoice.Native && dotnet run -c Release -- <native-onnx-dir> <t3_steps>`
Tests 1-6 cover VE (offline, mismatch noted), vocoder, tokenizer, T3 backbone,
T3 greedy loop, and the CFM decoder.

## Recommendation

The native path is viable and the experiment de-risked it. But it is a
multi-component reimplementation whose only benefit is dropping Docker — with
no latency improvement. Decide based on how hard the "no Python/Docker in the
target app" requirement is:

- **If in-process C# is a hard requirement** → proceed with the full build,
  starting with the vocoder split and the T3 KV-cache export (the two
  highest-risk remaining pieces), gating on parity at each stage.
- **If Docker is acceptable** → the existing container + HTTP API remains the
  better engineering choice; this native path is not worth the maintenance
  cost for zero speed gain.
