"""Flow path export + references.

Re-exports the CFM decoder with the noise `z` as an explicit input (the original
forward samples z=randn internally, making it stochastic). Captures real flow
inputs/outputs during a generate() call and dumps linear weights + a fixed z so
the C# flow path can be verified for parity.

C# flow chain to build from these:
  input_embedding(concat(prompt_token, speech_tokens)) * mask
    -> conformer encoder (ONNX) -> encoder_proj (linear) = mu
    -> CFM solver (ONNX, with z input, 10 steps, CFG) -> mel[:, :, mel_len1:]
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
    flow = model.s3gen.flow

    cap = {}

    def hook(name):
        def _h(mod, args, kwargs):
            cap.setdefault(name, {"args": args, "kwargs": kwargs})
        return _h

    h_enc = flow.encoder.register_forward_pre_hook(hook("encoder"), with_kwargs=True)
    h_proj = flow.encoder_proj.register_forward_pre_hook(hook("proj"), with_kwargs=True)
    h_dec = flow.decoder.register_forward_pre_hook(hook("decoder"), with_kwargs=True)

    # capture flow.inference args (token, prompt_*, embedding) too
    orig_inf = flow.inference
    def spy(*a, **k):
        cap.setdefault("flow_inf", {"args": a, "kwargs": k})
        return orig_inf(*a, **k)
    flow.inference = spy

    ref = os.path.join(model_dir, "_ref_mic.wav")
    model.generate("Hej, hvordan går det i dag?", language_id="da",
                  audio_prompt_path=ref if os.path.exists(ref) else None)
    h_enc.remove(); h_proj.remove(); h_dec.remove(); flow.inference = orig_inf

    # ---- linear weights C# needs ----
    dump("flow_input_embedding_weight", flow.input_embedding.weight)        # (vocab, 512)
    dump("flow_spk_affine_weight", flow.spk_embed_affine_layer.weight)      # (80, xvec_dim)
    dump("flow_spk_affine_bias", flow.spk_embed_affine_layer.bias)          # (80,)
    dump("flow_encoder_proj_weight", flow.encoder_proj.weight)             # (80, 512)
    dump("flow_encoder_proj_bias", flow.encoder_proj.bias)                 # (80,)

    # ---- decoder inputs (real) ----
    dk = cap["decoder"]["kwargs"]
    mu = dk["mu"].float()
    mask = dk["mask"].float()
    spks = dk["spks"].float()
    cond = dk["cond"].float()
    print("decoder mu", tuple(mu.shape), "mask", tuple(mask.shape), "spks", tuple(spks.shape), "cond", tuple(cond.shape))

    # fixed deterministic z for parity
    torch.manual_seed(1234)
    z = torch.randn_like(mu)

    cfm = flow.decoder

    class CfmWithZ(torch.nn.Module):
        def __init__(s, c):
            super().__init__(); s.c = c
        def forward(s, mu, mask, spks, cond, z):
            t_span = torch.linspace(0, 1, 11, device=mu.device, dtype=mu.dtype)
            return s.c.solve_euler(z, t_span, mu, mask, spks, cond)

    wrapper = CfmWithZ(cfm)
    with torch.no_grad():
        mel_ref = wrapper(mu, mask, spks, cond, z)
    dump("flow_mu", mu); dump("flow_mask", mask); dump("flow_spks", spks)
    dump("flow_cond", cond); dump("flow_z", z); dump("flow_mel_out", mel_ref)

    # encoder I/O (for wiring the conformer ONNX correctly)
    ek = cap["encoder"]
    enc_args = ek["args"]
    print("encoder args:", [tuple(a.shape) if torch.is_tensor(a) else a for a in enc_args])
    if len(enc_args) >= 1 and torch.is_tensor(enc_args[0]):
        dump("flow_encoder_in", enc_args[0].float())
    # encoder_proj input = encoder output h
    pk = cap["proj"]["args"]
    if pk and torch.is_tensor(pk[0]):
        dump("flow_h", pk[0].float())

    # conditioning bundle (offline) — token/prompt for completeness
    fk = cap["flow_inf"]["kwargs"] if "flow_inf" in cap else {}
    if "embedding" in fk:
        dump("flow_xvector", fk["embedding"].float())

    # ---- export CFM with z input (on CPU to avoid 8GB GPU OOM) ----
    torch.cuda.empty_cache()
    cfm_cpu = cfm.to("cpu")
    wrapper = CfmWithZ(cfm_cpu)
    mu, mask, spks, cond, z = (t.cpu() for t in (mu, mask, spks, cond, z))
    path = os.path.join(OUT, "cfm_decoder_z.onnx")
    torch.onnx.export(
        wrapper, (mu, mask, spks, cond, z), path, opset_version=17,
        input_names=["mu", "mask", "spks", "cond", "z"], output_names=["mel"],
        dynamic_axes={"mu": {0: "B", 2: "T"}, "mask": {0: "B", 2: "T"},
                      "cond": {0: "B", 2: "T"}, "z": {0: "B", 2: "T"},
                      "spks": {0: "B"}, "mel": {0: "B", 2: "T"}},
        do_constant_folding=True, dynamo=False)
    print("CFM(z) export OK", os.path.getsize(path) / 1e6, "MB")

    with open(os.path.join(REF, "shapes_flow.json"), "w") as f:
        json.dump(shapes, f, indent=2)
    print("DONE flow export")


if __name__ == "__main__":
    main()
