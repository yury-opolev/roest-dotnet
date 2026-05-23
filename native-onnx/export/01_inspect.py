"""Inspect Røst-v3 / Chatterbox Multilingual component architectures.

Runs inside the danish-voice-tts:spike image (torch + chatterbox + cached
model). Goal: understand what we must export to ONNX and how the
autoregressive loop is structured. Writes a report to /work/inspect_report.txt.

Touches no existing project files — only reads the HF cache and writes into
the mounted /work (native-onnx) folder.
"""

from __future__ import annotations

import os
import sys

import torch

REPORT = "/work/inspect_report.txt"
_lines: list[str] = []


def log(*args):
    msg = " ".join(str(a) for a in args)
    print(msg, flush=True)
    _lines.append(msg)


def summarize_module(name: str, module: torch.nn.Module, max_depth: int = 2):
    log(f"\n{'='*70}\n{name}: {module.__class__.__module__}.{module.__class__.__name__}\n{'='*70}")
    n_params = sum(p.numel() for p in module.parameters())
    log(f"  total params: {n_params:,}")
    # top-level children with param counts
    for child_name, child in module.named_children():
        cp = sum(p.numel() for p in child.parameters())
        log(f"  - {child_name}: {child.__class__.__name__} ({cp:,} params)")
        if max_depth > 1:
            for gname, gchild in child.named_children():
                gp = sum(p.numel() for p in gchild.parameters())
                log(f"      - {gname}: {gchild.__class__.__name__} ({gp:,})")


def main():
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from huggingface_hub import snapshot_download

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"torch {torch.__version__}  device={device}  cuda_avail={torch.cuda.is_available()}")

    model_dir = snapshot_download(
        repo_id="CoRal-project/roest-v3-chatterbox-500m",
        token=os.getenv("HF_TOKEN") or None,
        allow_patterns=["*.safetensors", "*.json", "*.pt", "*.wav"],
    )
    log(f"model_dir={model_dir}")
    log("files:", sorted(os.listdir(model_dir)))

    model = ChatterboxMultilingualTTS.from_local(model_dir, device=device)
    log(f"\nmodel class: {type(model)}")
    log(f"model.sr = {model.sr}")
    log("model public attrs:", [a for a in dir(model) if not a.startswith("_")])

    # The three neural sub-models we care about.
    for attr in ("t3", "s3gen", "ve", "tokenizer", "conds"):
        obj = getattr(model, attr, None)
        if obj is None:
            log(f"\n[{attr}] -> None / missing")
            continue
        if isinstance(obj, torch.nn.Module):
            summarize_module(attr, obj)
        else:
            log(f"\n[{attr}] type={type(obj)} (not nn.Module)")
            log("   attrs:", [a for a in dir(obj) if not a.startswith("_")][:40])

    # Forward signatures — critical for export planning.
    import inspect as _inspect
    log("\n" + "#" * 70 + "\nFORWARD / GENERATE SIGNATURES\n" + "#" * 70)
    for attr in ("t3", "s3gen", "ve"):
        obj = getattr(model, attr, None)
        if isinstance(obj, torch.nn.Module):
            try:
                log(f"\n{attr}.forward{_inspect.signature(obj.forward)}")
            except (ValueError, TypeError) as ex:
                log(f"\n{attr}.forward signature unavailable: {ex}")
            # also list any inference/generate-style methods
            methods = [m for m in dir(obj)
                       if not m.startswith("_") and callable(getattr(obj, m))
                       and any(k in m.lower() for k in ("infer", "generate", "decode", "sample", "flow", "forward"))]
            log(f"   {attr} candidate methods: {methods}")

    try:
        log(f"\nmodel.generate{_inspect.signature(model.generate)}")
    except (ValueError, TypeError):
        pass

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines))
    log(f"\nWrote report to {REPORT}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        with open(REPORT, "w", encoding="utf-8") as f:
            f.write("\n".join(_lines))
            f.write("\n\nTRACEBACK:\n")
            f.write(traceback.format_exc())
        sys.exit(1)
