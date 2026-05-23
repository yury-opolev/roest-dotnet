"""T3 backbone export + reference capture.

Captures the real prefill inputs_embeds and step-0 logits during a generate()
call, exports the Llama backbone (full-sequence, no KV cache for simplicity/
correctness), and dumps speech_head weights + references so the C# side can
verify the core T3 compute path (backbone + head -> logits) exactly.

The autoregressive sampling loop, CFG combine, logits processors, and alignment
analyzer are reimplemented in C# on top of this backbone graph.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

OUT = "/work/onnx_models"
REF = "/work/refs"
os.makedirs(OUT, exist_ok=True)
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = snapshot_download(
        repo_id="CoRal-project/roest-v3-chatterbox-500m",
        token=os.getenv("HF_TOKEN") or None,
        allow_patterns=["*.safetensors", "*.json", "*.pt", "*.wav"],
    )
    model = ChatterboxMultilingualTTS.from_local(model_dir, device=device)
    t3 = model.t3

    cap = {}
    # Hook the patched backbone's first forward to grab prefill inputs + logits.
    orig_call = {}

    # We patch after generate() builds patched_model; easier: hook t3.patched_model
    # by wrapping inference. Instead, monkeypatch T3HuggingfaceBackend.forward via
    # the instance once it exists. Simplest: run generate with a wrapper that
    # records the first patched_model call.
    import chatterbox.models.t3.inference.t3_hf_backend as backend_mod

    orig_forward = backend_mod.T3HuggingfaceBackend.forward
    state = {"n": 0}

    def spy_forward(self, *args, **kwargs):
        out = orig_forward(self, *args, **kwargs)
        if state["n"] == 0:
            ie = kwargs.get("inputs_embeds")
            cap["inputs_embeds"] = ie.detach().clone()
            cap["logits_full"] = out.logits.detach().clone()
        state["n"] += 1
        return out
    backend_mod.T3HuggingfaceBackend.forward = spy_forward

    ref = os.path.join(model_dir, "_ref_mic.wav")
    torch.manual_seed(0)
    model.generate("Hej, hvordan går det i dag?", language_id="da",
                  audio_prompt_path=ref if os.path.exists(ref) else None)
    backend_mod.T3HuggingfaceBackend.forward = orig_forward

    inputs_embeds = torch.from_numpy(
        cap["inputs_embeds"].cpu().numpy().astype(np.float32)).to(device)
    print("prefill inputs_embeds", tuple(inputs_embeds.shape))

    # ---- Reference: backbone hidden + speech_head logits (deterministic) ----
    with torch.no_grad():
        out = t3.tfmr(inputs_embeds=inputs_embeds, use_cache=False,
                      output_hidden_states=True, return_dict=True)
        hidden = out.hidden_states[-1]              # (2, L, 1024)
        last_hidden = hidden[:, -1, :]              # (2, 1024)
        logits = t3.speech_head(last_hidden)        # (2, V)

    dump("t3_inputs_embeds", inputs_embeds)
    dump("t3_hidden_last", hidden[:, -1, :])
    dump("t3_logits_step0", logits)
    dump("t3_speech_head_weight", t3.speech_head.weight)  # (V, 1024)

    # ---- Export backbone: inputs_embeds -> last_hidden_state ----
    class Backbone(torch.nn.Module):
        def __init__(s, tfmr):
            super().__init__()
            s.tfmr = tfmr
        def forward(s, inputs_embeds):
            o = s.tfmr(inputs_embeds=inputs_embeds, use_cache=False, return_dict=True)
            return o.last_hidden_state

    path = os.path.join(OUT, "t3_backbone.onnx")
    # Legacy exporter chokes on aten::diff in transformers' causal-mask prep;
    # use the dynamo exporter with a dynamic sequence-length dim.
    from torch.export import Dim
    seq = Dim("L", min=2, max=4096)
    torch.onnx.export(
        Backbone(t3.tfmr), (inputs_embeds,), path,
        input_names=["inputs_embeds"], output_names=["last_hidden_state"],
        dynamic_shapes={"inputs_embeds": {1: seq}},
        dynamo=True)
    print("backbone export OK", os.path.getsize(path) / 1e6, "MB")

    # config values C# needs
    cfg = {
        "start_speech_token": int(t3.hp.start_speech_token),
        "stop_speech_token": int(t3.hp.stop_speech_token),
        "speech_tokens_dict_size": int(t3.hp.speech_tokens_dict_size),
        "hidden_size": int(t3.cfg.hidden_size),
        "is_multilingual": bool(getattr(t3.hp, "is_multilingual", False)),
    }
    with open(os.path.join(REF, "t3_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    with open(os.path.join(REF, "shapes_t3.json"), "w") as f:
        json.dump(shapes, f, indent=2)
    print("DONE t3 backbone export; config:", cfg)


if __name__ == "__main__":
    main()
