"""Dump everything the C# T3 autoregressive loop needs + a deterministic greedy
reference token sequence (multinomial->argmax, alignment analyzer disabled) so
the C# loop's core (input construction, CFG, logits processors, backbone) can be
verified before adding the analyzer.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

REF = "/work/refs"
os.makedirs(REF, exist_ok=True)
shapes = {}


def dump(name, t):
    a = t.detach().cpu().numpy().astype(np.float32)
    a.tofile(os.path.join(REF, name + ".bin"))
    shapes[name] = list(a.shape)
    print(f"  {name} {a.shape}")


def main():
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from huggingface_hub import snapshot_download
    import chatterbox.models.t3.t3 as t3mod
    from chatterbox.models.t3.inference.alignment_stream_analyzer import AlignmentStreamAnalyzer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = snapshot_download(
        repo_id="CoRal-project/roest-v3-chatterbox-500m",
        token=os.getenv("HF_TOKEN") or None,
        allow_patterns=["*.safetensors", "*.json", "*.pt", "*.wav"],
    )
    model = ChatterboxMultilingualTTS.from_local(model_dir, device=device)
    t3 = model.t3

    # ---- capture t3.inference args during a real generate() ----
    cap = {}
    orig_inf = t3.inference

    def spy_inference(*args, **kwargs):
        cap["t3_cond"] = kwargs.get("t3_cond")
        cap["text_tokens"] = kwargs.get("text_tokens")
        return orig_inf(*args, **kwargs)
    t3.inference = spy_inference

    ref = os.path.join(model_dir, "_ref_mic.wav")
    model.generate("Hej, hvordan går det i dag?", language_id="da",
                  audio_prompt_path=ref if os.path.exists(ref) else None)
    t3.inference = orig_inf

    t3_cond = cap["t3_cond"]
    text_tokens = cap["text_tokens"]
    print("text_tokens", tuple(text_tokens.shape), text_tokens.dtype)

    # ---- cond_emb (per-voice, offline) ----
    with torch.no_grad():
        cond_emb = t3.prepare_conditioning(t3_cond)  # (B, len_cond, dim)
    dump("t3_cond_emb", cond_emb)

    # ---- embedding tables ----
    dump("t3_text_emb_weight", t3.text_emb.weight)
    dump("t3_speech_emb_weight", t3.speech_emb.weight)
    dump("t3_text_pos_emb_weight", t3.text_pos_emb.emb.weight)
    dump("t3_speech_pos_emb_weight", t3.speech_pos_emb.emb.weight)

    # text tokens (ids) to json
    tt = text_tokens.detach().cpu().numpy().astype(np.int64).tolist()
    json.dump(tt, open(os.path.join(REF, "t3_text_tokens.json"), "w"))
    print("text_tokens dumped (batch", len(tt), "len", len(tt[0]), ")")

    # ---- deterministic greedy reference (argmax, analyzer disabled) ----
    orig_multinomial = torch.multinomial
    torch.multinomial = lambda probs, num_samples=1, **kw: probs.argmax(dim=-1, keepdim=True)
    orig_step = AlignmentStreamAnalyzer.step
    AlignmentStreamAnalyzer.step = lambda self, logits, next_token=None: logits  # identity

    with torch.inference_mode():
        greedy_tokens = t3.inference(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            max_new_tokens=40,
            temperature=0.8, top_p=0.95, min_p=0.05,
            repetition_penalty=2.0, cfg_weight=0.5,
        )
    torch.multinomial = orig_multinomial
    AlignmentStreamAnalyzer.step = orig_step

    gt = greedy_tokens.detach().cpu().numpy().astype(np.int64).tolist()
    json.dump(gt, open(os.path.join(REF, "t3_greedy_tokens.json"), "w"))
    print("greedy tokens:", gt)

    with open(os.path.join(REF, "shapes_t3loop.json"), "w") as f:
        json.dump(shapes, f, indent=2)
    print("DONE t3 loop refs")


if __name__ == "__main__":
    main()
