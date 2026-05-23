"""Retry HiFT vocoder export with the dynamo exporter (now that onnxscript is
installed at runtime). If this still fails on iSTFT, the fallback is to export
the vocoder up to the iSTFT and implement overlap-add in C#.
"""

from __future__ import annotations

import os
import traceback

import torch

WORK = "/work"
OUT = "/work/onnx_models"
_lines = []


def log(*a):
    m = " ".join(str(x) for x in a)
    print(m, flush=True)
    _lines.append(m)


def main():
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from huggingface_hub import snapshot_download

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = snapshot_download(
        repo_id="CoRal-project/roest-v3-chatterbox-500m",
        token=os.getenv("HF_TOKEN") or None,
        allow_patterns=["*.safetensors", "*.json", "*.pt", "*.wav"],
    )
    model = ChatterboxMultilingualTTS.from_local(model_dir, device=device)
    s3 = model.s3gen

    cap = {}
    orig = s3.mel2wav.inference
    def patched(*a, **k):
        cap.setdefault("hift", dict(k))
        return orig(*a, **k)
    s3.mel2wav.inference = patched
    ref = os.path.join(model_dir, "_ref_mic.wav")
    model.generate("Hej, hvordan går det?", language_id="da",
                  audio_prompt_path=ref if os.path.exists(ref) else None)
    s3.mel2wav.inference = orig

    speech_feat = cap["hift"]["speech_feat"]
    cache_source = cap["hift"].get("cache_source", torch.zeros(1, 1, 0, device=device))

    class HiftWrap(torch.nn.Module):
        def __init__(s, m):
            super().__init__(); s.m = m
        def forward(s, speech_feat, cache_source):
            out = s.m.inference(speech_feat=speech_feat, cache_source=cache_source)
            return out[0] if isinstance(out, (tuple, list)) else out

    path = os.path.join(OUT, "hift_vocoder_dynamo.onnx")
    log("trying dynamo export of full HiFT vocoder (incl. iSTFT)...")
    try:
        with torch.no_grad():
            torch.onnx.export(HiftWrap(s3.mel2wav), (speech_feat, cache_source),
                              path, dynamo=True, dynamic_shapes=None)
        log(f"DYNAMO EXPORT OK -> {os.path.getsize(path)/1e6:.1f} MB")
        log("RESULT: vocoder fully exportable via dynamo. No C# iSTFT needed.")
    except Exception as ex:
        log(f"DYNAMO EXPORT FAILED: {type(ex).__name__}: {str(ex)[:200]}")
        log(traceback.format_exc()[-1800:])
        log("RESULT: vocoder needs the 'export-minus-iSTFT + C# overlap-add' fallback.")


if __name__ == "__main__":
    try:
        main()
    finally:
        with open(os.path.join(WORK, "probe_vocoder_dynamo.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(_lines))
