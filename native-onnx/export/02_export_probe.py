"""Feasibility probe: capture real sub-module inputs during a generate() call,
then attempt torch.onnx.export on each component and report a GO/NO-GO matrix.

This is deliberately fast and shallow — the point is to find which components
export cleanly and which throw (custom ops, iSTFT, dynamic control flow),
BEFORE investing days in any single one.

Writes:
  /work/probe_report.txt        human-readable matrix + errors
  /work/onnx_models/*.onnx      whatever managed to export
  /work/captured_shapes.txt     real input shapes/dtypes per sub-module
"""

from __future__ import annotations

import os
import sys
import traceback

import torch

WORK = "/work"
OUT = "/work/onnx_models"
os.makedirs(OUT, exist_ok=True)
_lines: list[str] = []


def log(*a):
    m = " ".join(str(x) for x in a)
    print(m, flush=True)
    _lines.append(m)


def desc(x):
    if isinstance(x, torch.Tensor):
        return f"Tensor{tuple(x.shape)}:{x.dtype}"
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__}[{', '.join(desc(i) for i in x)}]"
    if isinstance(x, dict):
        return "{" + ", ".join(f"{k}:{desc(v)}" for k, v in x.items()) + "}"
    return f"{type(x).__name__}={x!r}"[:80]


def main():
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from huggingface_hub import snapshot_download

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"torch {torch.__version__} device={device}")

    model_dir = snapshot_download(
        repo_id="CoRal-project/roest-v3-chatterbox-500m",
        token=os.getenv("HF_TOKEN") or None,
        allow_patterns=["*.safetensors", "*.json", "*.pt", "*.wav"],
    )
    model = ChatterboxMultilingualTTS.from_local(model_dir, device=device)

    captured: dict[str, dict] = {}

    def make_hook(name):
        def hook(module, args, kwargs):
            if name not in captured:  # keep only the first call
                captured[name] = {"args": args, "kwargs": kwargs}
        return hook

    targets = {
        "ve": model.ve,
        "s3gen_flow": model.s3gen.flow,
        "s3gen_mel2wav": model.s3gen.mel2wav,
        "s3gen_speaker_encoder": model.s3gen.speaker_encoder,
    }
    handles = [m.register_forward_pre_hook(make_hook(n), with_kwargs=True)
               for n, m in targets.items()]

    log("\n--- running one generate() to capture real inputs ---")
    ref = os.path.join(model_dir, "_ref_mic.wav")
    try:
        wav = model.generate("Hej, hvordan går det i dag?", language_id="da",
                             audio_prompt_path=ref if os.path.exists(ref) else None)
        log(f"generate ok, wav shape {tuple(wav.shape)}")
    except Exception:
        log("generate raised (may still have captured some inputs):")
        log(traceback.format_exc())
    for h in handles:
        h.remove()

    log("\n--- captured sub-module inputs ---")
    for name, cap in captured.items():
        log(f"\n[{name}]")
        log("  args:", desc(cap["args"]))
        log("  kwargs:", desc(cap["kwargs"]))

    # Attempt exports. Each in its own try so one failure doesn't stop others.
    log("\n" + "#" * 70 + "\nEXPORT ATTEMPTS\n" + "#" * 70)
    results = {}

    def try_export(name, module, args, kwargs=None, opset=17):
        kwargs = kwargs or {}
        path = os.path.join(OUT, f"{name}.onnx")
        try:
            module.eval()
            with torch.no_grad():
                # torch.onnx.export takes positional args as a tuple; kwargs via
                # the trailing dict trick is fragile, so we wrap in a lambda module.
                if kwargs:
                    class Wrap(torch.nn.Module):
                        def __init__(s):
                            super().__init__()
                            s.m = module
                        def forward(s, *a):
                            return s.m(*a, **kwargs)
                    export_mod = Wrap()
                else:
                    export_mod = module
                torch.onnx.export(
                    export_mod, tuple(args), path,
                    opset_version=opset, do_constant_folding=True,
                    dynamo=False,
                )
            sz = os.path.getsize(path) / 1e6
            results[name] = f"OK  ({sz:.1f} MB)"
            log(f"\n[{name}] EXPORT OK -> {path} ({sz:.1f} MB)")
        except Exception as ex:
            results[name] = f"FAIL: {type(ex).__name__}: {str(ex)[:200]}"
            log(f"\n[{name}] EXPORT FAILED:")
            log(traceback.format_exc()[-1500:])

    # VoiceEncoder: forward(mels)
    if "ve" in captured:
        a = captured["ve"]["args"]
        try_export("voice_encoder", model.ve, a)

    # CAMPPlus speaker encoder
    if "s3gen_speaker_encoder" in captured:
        a = captured["s3gen_speaker_encoder"]["args"]
        try_export("speaker_encoder_campplus", model.s3gen.speaker_encoder, a)

    # HiFT vocoder (mel2wav) — the high-risk iSTFT/NSF one
    if "s3gen_mel2wav" in captured:
        a = captured["s3gen_mel2wav"]["args"]
        kw = captured["s3gen_mel2wav"]["kwargs"]
        try_export("hift_vocoder", model.s3gen.mel2wav, a, kw)

    # Flow (Conformer + CFM) — high risk
    if "s3gen_flow" in captured:
        a = captured["s3gen_flow"]["args"]
        kw = captured["s3gen_flow"]["kwargs"]
        try_export("s3gen_flow", model.s3gen.flow, a, kw)

    log("\n" + "#" * 70 + "\nSUMMARY MATRIX\n" + "#" * 70)
    for n in ("voice_encoder", "speaker_encoder_campplus", "hift_vocoder", "s3gen_flow"):
        log(f"  {n:28s} {results.get(n, 'not attempted (no captured input)')}")
    log("\nNOTE: T3 (autoregressive Llama) is probed separately — it needs a")
    log("decode-step wrapper, not a single forward call.")


if __name__ == "__main__":
    try:
        main()
    finally:
        with open(os.path.join(WORK, "probe_report.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(_lines))
