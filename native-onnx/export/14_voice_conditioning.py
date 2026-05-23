"""Dump per-voice conditioning (Mic + Nic) so the native C# pipeline can render
arbitrary text in either voice without VE/CAMPPlus/S3Tokenizer at runtime.

For each voice: T3 cond_emb, flow prompt_token ids, prompt_feat, and the xvector.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

REF = "/work/refs"
shapes = {}


def dump(name, t):
    a = t.detach().cpu().numpy().astype(np.float32)
    a.tofile(os.path.join(REF, name + ".bin"))
    shapes[name] = list(a.shape)
    print(f"  {name} {a.shape}")


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

    for voice in ("mic", "nic"):
        ref = os.path.join(model_dir, f"_ref_{voice}.wav")
        if not os.path.exists(ref):
            print(f"!! missing {ref}; skipping {voice}")
            continue
        model.prepare_conditionals(ref)
        gen = model.conds.gen
        with torch.inference_mode():
            cond_emb = model.t3.prepare_conditioning(model.conds.t3)
        dump(f"cond_{voice}_t3_cond_emb", cond_emb)
        dump(f"cond_{voice}_prompt_feat", gen["prompt_feat"].float())
        dump(f"cond_{voice}_xvector", gen["embedding"].float())
        pt = gen["prompt_token"].detach().cpu().numpy().astype(np.int64)
        json.dump(pt.tolist(), open(os.path.join(REF, f"cond_{voice}_prompt_token.json"), "w"))
        meta = {
            "prompt_token_len": int(gen["prompt_token"].shape[-1]),
            "mel_len1": int(gen["prompt_feat"].shape[1]),
        }
        json.dump(meta, open(os.path.join(REF, f"cond_{voice}_meta.json"), "w"))
        print(f"  {voice}: prompt_token {pt.shape}, meta {meta}")

    # text-token specials
    json.dump({"start_text_token": 255, "stop_text_token": 0,
               "start_speech_token": int(model.t3.hp.start_speech_token),
               "stop_speech_token": int(model.t3.hp.stop_speech_token)},
              open(os.path.join(REF, "text_specials.json"), "w"))

    with open(os.path.join(REF, "shapes_cond.json"), "w") as f:
        json.dump(shapes, f, indent=2)
    print("DONE voice conditioning")


if __name__ == "__main__":
    main()
