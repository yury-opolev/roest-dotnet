# Native Danish TTS runtime (ONNX) — reuse in other .NET projects

This bundle (a GitHub Release asset, `danish-voice-native-runtime.zip`) contains
everything the native C# pipeline needs to synthesize Danish speech with **no
Python** at inference time: the ONNX graphs + reference tensors.

> The weights derive from CoRal **Røst-v3** (OpenRAIL); base Chatterbox is MIT.
> See `NOTICE.md`. By using these assets you accept the OpenRAIL use restrictions.

> **v0.2.1** is a library-only release: it reuses the **v0.2.0** runtime bundle
> unchanged (no new ONNX graphs). If you already have the v0.2.0 bundle, you do
> not need to re-download anything.

## Contents

```
onnx_models/   t3_backbone.onnx (+ .data), conformer_encoder_dyn.onnx,
               cfm_decoder_z.onnx, voc_f0_predictor.onnx, voc_conv_stack.onnx
refs/          tokenizer + configs + embedding tables + per-voice (Mic/Nic)
               conditioning tensors
```

## Use it from another project

1. Download **both** release zips (split to stay under GitHub's 2 GiB asset
   limit) and unzip **both into the same folder**, e.g. `C:\models\roest-dotnet`:
   - `roest-dotnet-runtime-part1-t3.zip` — the T3 backbone weights
   - `roest-dotnet-runtime-part2.zip` — everything else
   They merge into `onnx_models/` + `refs/` (so `t3_backbone.onnx` and
   `t3_backbone.onnx.data` end up side by side).
2. Reference the `DanishVoice.Native` project (or copy its `*.cs`), which depends
   only on `Microsoft.ML.OnnxRuntime`.
3. Synthesize:

   ```csharp
   using var tts = new SynthPipeline(
       onnxDir: @"C:\models\danish-voice\onnx_models",
       refsDir: @"C:\models\danish-voice\refs");
   float[] wav = tts.Synth("Hej, hvordan går det i dag?", voice: "mic"); // 24 kHz mono
   ```

   Or via the CLI:
   ```
   dotnet run -c Release -- synth C:\models\danish-voice mic out.wav "Din tekst her"
   ```

Voices: `mic` (female), `nic` (male).

## Notes / current limitations

- Decoding is **greedy** (deterministic; flatter prosody than temperature
  sampling). Swapping in multinomial sampling is a small change in `T3Model`.
- The default execution provider is **CPU** (~10× slower than real time). For
  speed, switch the ONNX sessions to the **CUDA** execution provider
  (`Microsoft.ML.OnnxRuntime.Gpu` + `SessionOptions.MakeSessionOptionWithCudaProvider`);
  `OnnxModel` already accepts a `useCuda` flag.
- The T3 alignment-stream analyzer (anti-hallucination / EOS robustness) is not
  ported; greedy stops on natural EOS for normal sentences.
- Output is 24 kHz mono. No Perth watermark (unlike upstream Chatterbox).

## Regenerating the artifacts

The bundle is produced from the upstream model by `native-onnx/export/*.py`
(run inside the `danish-voice-tts` Docker image) and zipped by
`scripts/build-release-bundle.ps1`. See `native-onnx/FINDINGS.md` for the full
conversion story and parity results.
