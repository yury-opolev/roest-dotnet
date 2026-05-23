"""Re-export the conformer with dynamic axes + dump flow-assembly references and
a fully deterministic end-to-end reference (greedy T3 + fixed CFM noise +
analyzer disabled) so the assembled C# pipeline can be checked against a single
reproducible Python run.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

OUT = "/work/onnx_models"
REF = "/work/refs"
shapes = {}


def dump(name, t, dtype=np.float32):
    a = t.detach().cpu().numpy().astype(dtype)
    a.tofile(os.path.join(REF, name + ".bin"))
    shapes[name] = list(a.shape)
    print(f"  {name} {a.shape} {a.dtype}")


def main():
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from huggingface_hub import snapshot_download
    import chatterbox.models.s3gen.flow_matching as fm
    from chatterbox.models.t3.inference.alignment_stream_analyzer import AlignmentStreamAnalyzer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = snapshot_download(
        repo_id="CoRal-project/roest-v3-chatterbox-500m",
        token=os.getenv("HF_TOKEN") or None,
        allow_patterns=["*.safetensors", "*.json", "*.pt", "*.wav"],
    )
    model = ChatterboxMultilingualTTS.from_local(model_dir, device=device)
    flow = model.s3gen.flow
    ref = os.path.join(model_dir, "_ref_mic.wav")

    # ---- re-export conformer encoder with dynamic axes (synthetic input,
    #      BEFORE any inference_mode generate() to avoid tensor contamination) ----
    class Enc(torch.nn.Module):
        def __init__(s, e):
            super().__init__(); s.e = e
        def forward(s, xs, xs_lens):
            h, _ = s.e(xs, xs_lens)
            return h

    synth_in = torch.randn(1, 200, 512, device=device)
    synth_len = torch.tensor([200], device=device, dtype=torch.long)
    path = os.path.join(OUT, "conformer_encoder_dyn.onnx")
    torch.onnx.export(
        Enc(flow.encoder), (synth_in, synth_len), path, opset_version=17,
        input_names=["xs", "xs_lens"], output_names=["h"],
        dynamic_axes={"xs": {0: "B", 1: "T"}, "h": {0: "B", 1: "T2"}},
        do_constant_folding=True, dynamo=False)
    print("conformer dynamic export OK", os.path.getsize(path) / 1e6, "MB")

    # ---- now capture flow-assembly inputs during a generate() ----
    cap = {}
    def hook(name):
        def _h(m, a, k):
            cap.setdefault(name, (a, k))
        return _h
    h2 = flow.input_embedding.register_forward_pre_hook(hook("inemb"), with_kwargs=True)
    fcap = {}
    orig_inf = flow.inference
    def spy(*a, **k):
        fcap.update(k)
        return orig_inf(*a, **k)
    flow.inference = spy

    model.generate("Hej, hvordan går det i dag?", language_id="da",
                  audio_prompt_path=ref if os.path.exists(ref) else None)
    h2.remove(); flow.inference = orig_inf

    # raw concatenated token ids fed to input_embedding
    tok = cap["inemb"][0][0]
    dump("flow_token_concat", tok, dtype=np.int64)
    json.dump(tok.detach().cpu().numpy().astype(np.int64).tolist(),
              open(os.path.join(REF, "flow_token_concat.json"), "w"))
    # token_len for the conformer (length of the concatenated token sequence)
    tok_len = int(tok.shape[-1])
    json.dump({"token_len": tok_len}, open(os.path.join(REF, "flow_token_len.json"), "w"))
    print("  token_len", tok_len)

    # prompt_feat + mel_len1 for cond assembly
    if "prompt_feat" in fcap:
        dump("flow_prompt_feat", fcap["prompt_feat"].float())
        json.dump({"mel_len1": int(fcap["prompt_feat"].shape[1])},
                  open(os.path.join(REF, "flow_meta.json"), "w"))

    with open(os.path.join(REF, "shapes_flow2.json"), "w") as f:
        json.dump(shapes, f, indent=2)

    # ---- fully deterministic end-to-end reference wav ----
    torch.multinomial = lambda probs, num_samples=1, **kw: probs.argmax(dim=-1, keepdim=True)
    AlignmentStreamAnalyzer.step = lambda self, logits, next_token=None: logits
    # fix CFM noise: make randn_like deterministic via fixed seed each call
    orig_randn_like = torch.randn_like
    def fixed_randn_like(x, *a, **k):
        g = torch.Generator(device=x.device).manual_seed(1234)
        return torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype)
    torch.randn_like = fixed_randn_like

    with torch.inference_mode():
        wav = model.generate("Hej, hvordan går det i dag?", language_id="da",
                            audio_prompt_path=ref if os.path.exists(ref) else None)
    dump("e2e_wav_deterministic", wav.float())
    print("DONE flow assembly refs")


if __name__ == "__main__":
    main()
