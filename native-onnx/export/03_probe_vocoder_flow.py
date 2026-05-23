"""Decisive probe: export the HiFT vocoder (iSTFT/NSF — classic ONNX blocker)
and the CFM flow decoder. Capture their real inputs by monkeypatching the
custom inference methods, then attempt torch.onnx.export.

GO/NO-GO signal for the whole native-ONNX path lives here.
"""

from __future__ import annotations

import os
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
    s3 = model.s3gen

    cap = {}
    orig_hift = s3.mel2wav.inference

    def patched_hift(*args, **kwargs):
        cap["hift"] = {"args": args, "kwargs": {k: v for k, v in kwargs.items()}}
        return orig_hift(*args, **kwargs)
    s3.mel2wav.inference = patched_hift

    # capture flow decoder input too (CausalConditionalCFM forward)
    flow_dec = s3.flow.decoder
    orig_dec_forward = flow_dec.forward

    def patched_dec(*args, **kwargs):
        if "flow_dec" not in cap:
            cap["flow_dec"] = {"args": args, "kwargs": {k: v for k, v in kwargs.items()}}
        return orig_dec_forward(*args, **kwargs)
    flow_dec.forward = patched_dec

    log("\n--- generate() to capture vocoder + flow-decoder inputs ---")
    ref = os.path.join(model_dir, "_ref_mic.wav")
    wav = model.generate("Hej, hvordan går det i dag?", language_id="da",
                        audio_prompt_path=ref if os.path.exists(ref) else None)
    log("generate ok, wav", tuple(wav.shape))
    s3.mel2wav.inference = orig_hift
    flow_dec.forward = orig_dec_forward

    for k, v in cap.items():
        log(f"\n[{k}] args:", desc(v["args"]), " kwargs:", desc(v["kwargs"]))

    results = {}

    def try_export(name, mod, args, opset=17):
        path = os.path.join(OUT, f"{name}.onnx")
        try:
            mod.eval()
            with torch.no_grad():
                torch.onnx.export(mod, tuple(args), path, opset_version=opset,
                                  do_constant_folding=True, dynamo=False)
            sz = os.path.getsize(path) / 1e6
            results[name] = f"OK ({sz:.1f} MB)"
            log(f"\n[{name}] EXPORT OK -> {sz:.1f} MB")
        except Exception as ex:
            results[name] = f"FAIL: {type(ex).__name__}: {str(ex)[:160]}"
            log(f"\n[{name}] EXPORT FAILED:\n", traceback.format_exc()[-1800:])

    # ---- HiFT vocoder ----
    if "hift" in cap:
        a = cap["hift"]["args"]
        kw = cap["hift"]["kwargs"]
        speech_feat = kw.get("speech_feat", a[0] if a else None)
        cache_source = kw.get("cache_source", None)
        if cache_source is None:
            cache_source = torch.zeros(1, 1, 0, device=device)

        class HiftWrap(torch.nn.Module):
            def __init__(s, m):
                super().__init__()
                s.m = m
            def forward(s, speech_feat, cache_source):
                out = s.m.inference(speech_feat=speech_feat, cache_source=cache_source)
                return out[0] if isinstance(out, (tuple, list)) else out
        log("\nattempting HiFT vocoder export; speech_feat", desc(speech_feat))
        try_export("hift_vocoder", HiftWrap(s3.mel2wav), (speech_feat, cache_source))

    # ---- CFM flow decoder (single forward) ----
    if "flow_dec" in cap:
        a = cap["flow_dec"]["args"]
        kw = cap["flow_dec"]["kwargs"]
        if kw:
            class DecWrap(torch.nn.Module):
                def __init__(s, m, kw):
                    super().__init__()
                    s.m = m
                    s.kw = kw
                def forward(s, *args):
                    return s.m(*args, **s.kw)
            try_export("cfm_decoder", DecWrap(flow_dec, kw), a)
        else:
            try_export("cfm_decoder", flow_dec, a)

    log("\n" + "#" * 60 + "\nMATRIX\n" + "#" * 60)
    for n in ("hift_vocoder", "cfm_decoder"):
        log(f"  {n:16s} {results.get(n, 'not attempted')}")


if __name__ == "__main__":
    try:
        main()
    finally:
        with open(os.path.join(WORK, "probe_vocoder_flow.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(_lines))
