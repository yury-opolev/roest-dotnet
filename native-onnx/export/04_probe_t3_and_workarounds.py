"""Probe the remaining unknowns:
  (A) HiFT vocoder via the dynamo exporter (does it handle iSTFT?)
  (B) T3 Llama core (tfmr) single decode step export — the make-or-break
      autoregressive component.
  (C) UpsampleConformerEncoder (flow.encoder) — the other half of the flow path.

Captures real inputs via hooks during a generate() call, then attempts export.
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
        return f"T{tuple(x.shape)}:{str(x.dtype).replace('torch.','')}"
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__}[{', '.join(desc(i) for i in x[:4])}{',...' if len(x)>4 else ''}]"
    if isinstance(x, dict):
        return "{" + ", ".join(f"{k}:{desc(v)}" for k, v in x.items()) + "}"
    return f"{type(x).__name__}={x!r}"[:60]


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
    results = {}

    cap = {}

    def hook(name, store_first=True):
        def _h(mod, args, kwargs):
            if store_first and name in cap:
                return
            cap[name] = {"args": args, "kwargs": kwargs}
        return _h

    tfmr = model.t3.tfmr                 # LlamaModel
    conf_enc = model.s3gen.flow.encoder  # UpsampleConformerEncoder
    h1 = tfmr.register_forward_pre_hook(hook("tfmr"), with_kwargs=True)
    h2 = conf_enc.register_forward_pre_hook(hook("conf_enc"), with_kwargs=True)

    # also grab a vocoder input for the dynamo retry
    s3 = model.s3gen
    orig_hift = s3.mel2wav.inference
    def patched(*a, **k):
        cap.setdefault("hift", {"args": a, "kwargs": dict(k)})
        return orig_hift(*a, **k)
    s3.mel2wav.inference = patched

    log("\n--- generate() to capture inputs ---")
    ref = os.path.join(model_dir, "_ref_mic.wav")
    wav = model.generate("Hej, hvordan går det i dag?", language_id="da",
                        audio_prompt_path=ref if os.path.exists(ref) else None)
    log("generate ok, wav", tuple(wav.shape))
    h1.remove(); h2.remove(); s3.mel2wav.inference = orig_hift

    for k, v in cap.items():
        log(f"\n[{k}] args:", desc(v["args"]), "kwargs:", desc(v["kwargs"]))

    # ---------- (A) HiFT vocoder via dynamo exporter ----------
    if "hift" in cap:
        kw = cap["hift"]["kwargs"]
        a = cap["hift"]["args"]
        speech_feat = kw.get("speech_feat", a[0] if a else None)
        cache_source = kw.get("cache_source", torch.zeros(1, 1, 0, device=device))

        class HiftWrap(torch.nn.Module):
            def __init__(s, m):
                super().__init__(); s.m = m
            def forward(s, speech_feat, cache_source):
                out = s.m.inference(speech_feat=speech_feat, cache_source=cache_source)
                return out[0] if isinstance(out, (tuple, list)) else out

        path = os.path.join(OUT, "hift_vocoder_dynamo.onnx")
        try:
            with torch.no_grad():
                torch.onnx.export(HiftWrap(s3.mel2wav), (speech_feat, cache_source),
                                  path, dynamo=True)
            results["hift_dynamo"] = f"OK ({os.path.getsize(path)/1e6:.1f} MB)"
            log("\n[hift_dynamo] EXPORT OK")
        except Exception as ex:
            results["hift_dynamo"] = f"FAIL: {type(ex).__name__}: {str(ex)[:160]}"
            log("\n[hift_dynamo] FAILED:\n", traceback.format_exc()[-1500:])

    # ---------- (B) T3 Llama core single decode step ----------
    if "tfmr" in cap:
        kw = dict(cap["tfmr"]["kwargs"])
        a = cap["tfmr"]["args"]
        log("\nT3 tfmr capture — args:", desc(a), "kwargs keys:", list(kw.keys()))
        # Build a minimal single-step export: inputs_embeds + attention_mask.
        inputs_embeds = kw.get("inputs_embeds")
        if inputs_embeds is None and a:
            inputs_embeds = a[0]
        if inputs_embeds is not None:
            # single-token decode step
            step_embed = inputs_embeds[:, -1:, :].contiguous()

            class TfmrStep(torch.nn.Module):
                def __init__(s, m):
                    super().__init__(); s.m = m
                def forward(s, inputs_embeds):
                    out = s.m(inputs_embeds=inputs_embeds, use_cache=True, return_dict=True)
                    return out.last_hidden_state

            path = os.path.join(OUT, "t3_tfmr_step.onnx")
            try:
                with torch.no_grad():
                    torch.onnx.export(TfmrStep(tfmr), (step_embed,), path,
                                      opset_version=17, do_constant_folding=True,
                                      dynamo=False)
                results["t3_tfmr_legacy"] = f"OK ({os.path.getsize(path)/1e6:.1f} MB)"
                log("\n[t3_tfmr_legacy] EXPORT OK")
            except Exception as ex:
                results["t3_tfmr_legacy"] = f"FAIL: {type(ex).__name__}: {str(ex)[:160]}"
                log("\n[t3_tfmr_legacy] FAILED:\n", traceback.format_exc()[-1200:])
                # retry with dynamo
                try:
                    with torch.no_grad():
                        torch.onnx.export(TfmrStep(tfmr), (step_embed,),
                                          path.replace(".onnx", "_dynamo.onnx"), dynamo=True)
                    results["t3_tfmr_dynamo"] = "OK"
                    log("[t3_tfmr_dynamo] EXPORT OK")
                except Exception as ex2:
                    results["t3_tfmr_dynamo"] = f"FAIL: {type(ex2).__name__}: {str(ex2)[:160]}"
                    log("[t3_tfmr_dynamo] FAILED:\n", traceback.format_exc()[-1000:])

    # ---------- (C) Conformer encoder ----------
    if "conf_enc" in cap:
        a = cap["conf_enc"]["args"]
        kw = cap["conf_enc"]["kwargs"]
        class EncWrap(torch.nn.Module):
            def __init__(s, m, kw):
                super().__init__(); s.m = m; s.kw = kw
            def forward(s, *args):
                out = s.m(*args, **s.kw)
                return out[0] if isinstance(out, (tuple, list)) else out
        path = os.path.join(OUT, "conformer_encoder.onnx")
        try:
            with torch.no_grad():
                torch.onnx.export(EncWrap(conf_enc, kw) if kw else conf_enc, tuple(a),
                                  path, opset_version=17, do_constant_folding=True, dynamo=False)
            results["conformer_encoder"] = f"OK ({os.path.getsize(path)/1e6:.1f} MB)"
            log("\n[conformer_encoder] EXPORT OK")
        except Exception as ex:
            results["conformer_encoder"] = f"FAIL: {type(ex).__name__}: {str(ex)[:160]}"
            log("\n[conformer_encoder] FAILED:\n", traceback.format_exc()[-1200:])

    log("\n" + "#" * 60 + "\nMATRIX\n" + "#" * 60)
    for k, v in results.items():
        log(f"  {k:22s} {v}")


if __name__ == "__main__":
    try:
        main()
    finally:
        with open(os.path.join(WORK, "probe_t3.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(_lines))
