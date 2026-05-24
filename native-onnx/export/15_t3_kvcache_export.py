"""Export the T3 Llama backbone with KV-cache as two graphs:
  - t3_prefill.onnx : inputs_embeds (B,L,D) -> hidden + present K/V per layer
  - t3_decode.onnx  : inputs_embeds (B,1,D) + past K/V -> hidden + present K/V

Then the C# decode loop runs prefill once and 1 token/step (constant cost)
instead of re-running the whole sequence (O(n^2)).

Verifies parity in torch (prefill + 1 decode step == no-cache full forward).
"""

from __future__ import annotations

import os
import numpy as np
import torch

OUT = "/work/onnx_models"
REF = "/work/refs"


def main():
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from huggingface_hub import snapshot_download
    from transformers.cache_utils import DynamicCache

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = snapshot_download(
        repo_id="CoRal-project/roest-v3-chatterbox-500m",
        token=os.getenv("HF_TOKEN") or None,
        allow_patterns=["*.safetensors", "*.json", "*.pt", "*.wav"],
    )
    model = ChatterboxMultilingualTTS.from_local(model_dir, device=device)
    tfmr = model.t3.tfmr
    cfg = tfmr.config
    n_layers = cfg.num_hidden_layers
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    print(f"layers={n_layers} kv_heads={n_kv} head_dim={head_dim} hidden={cfg.hidden_size}")

    # real prefill input from refs
    ie = np.fromfile(os.path.join(REF, "t3_inputs_embeds.bin"), dtype=np.float32).reshape(2, 62, cfg.hidden_size)
    inputs_embeds = torch.from_numpy(ie).to(device)

    def extract(cache):
        flat = []
        for layer in cache.layers:
            flat.append(layer.keys)
            flat.append(layer.values)
        return flat

    class Prefill(torch.nn.Module):
        def __init__(s, m):
            super().__init__(); s.m = m
        def forward(s, inputs_embeds):
            out = s.m(inputs_embeds=inputs_embeds, use_cache=True, return_dict=True)
            return (out.last_hidden_state, *extract(out.past_key_values))

    class Decode(torch.nn.Module):
        def __init__(s, m, n):
            super().__init__(); s.m = m; s.n = n
        def forward(s, inputs_embeds, *past_flat):
            cache = DynamicCache()
            for i in range(s.n):
                cache.update(past_flat[2 * i], past_flat[2 * i + 1], i)
            past_len = past_flat[0].shape[2]
            cache_position = torch.arange(past_len, past_len + inputs_embeds.shape[1], device=inputs_embeds.device)
            out = s.m(inputs_embeds=inputs_embeds, past_key_values=cache, use_cache=True,
                      cache_position=cache_position, return_dict=True)
            return (out.last_hidden_state, *extract(out.past_key_values))

    prefill = Prefill(tfmr).eval()
    decode = Decode(tfmr, n_layers).eval()

    # ---- torch parity: prefill + 1 decode == no-cache full forward ----
    with torch.no_grad():
        pf = prefill(inputs_embeds)
        hidden_pf, present = pf[0], pf[1:]
        # next token embed = reuse last position (just a parity probe)
        next_embed = inputs_embeds[:, -1:, :].contiguous()
        dec = decode(next_embed, *present)
        hidden_dec = dec[0]
        # reference: no-cache full forward of [prefill ++ next]
        full_in = torch.cat([inputs_embeds, next_embed], dim=1)
        ref = tfmr(inputs_embeds=full_in, use_cache=False, return_dict=True).last_hidden_state[:, -1:, :]
        diff = (hidden_dec - ref).abs().max().item()
        print(f"decode-step parity vs no-cache: max abs diff = {diff:.3e}")

    # ---- export prefill ----
    from torch.export import Dim
    Lp = Dim("L", min=2, max=4096)
    out_names = ["hidden"] + [f"{t}{i}" for i in range(n_layers) for t in ("present_k", "present_v")]
    kv_axes_pf = {nm: {2: Dim.AUTO} for nm in out_names if nm != "hidden"}
    pf_path = os.path.join(OUT, "t3_prefill.onnx")
    torch.onnx.export(
        prefill, (inputs_embeds,), pf_path,
        input_names=["inputs_embeds"], output_names=out_names,
        dynamic_shapes={"inputs_embeds": {1: Lp}},
        dynamo=True)
    print("prefill export OK", round(os.path.getsize(pf_path) / 1e6, 1), "MB")

    # ---- export decode ----
    # decode always processes exactly 1 new token (S=1 static); only the past
    # length is dynamic. Let the exporter auto-infer the past dim.
    present = [p.contiguous() for p in present]
    past_names = [f"{t}{i}" for i in range(n_layers) for t in ("past_k", "past_v")]
    dec_in_names = ["inputs_embeds"] + past_names
    dec_out_names = ["hidden"] + [f"{t}{i}" for i in range(n_layers) for t in ("present_k", "present_v")]
    # structure mirrors forward(inputs_embeds, *past_flat): inputs_embeds static,
    # each past tensor dynamic along the sequence (dim 2).
    dyn_tuple = ({}, tuple({2: Dim.AUTO} for _ in range(2 * n_layers)))
    dec_path = os.path.join(OUT, "t3_decode.onnx")
    torch.onnx.export(
        decode, (next_embed, *present), dec_path,
        input_names=dec_in_names, output_names=dec_out_names,
        dynamic_shapes=dyn_tuple,
        dynamo=True)
    print("decode export OK", round(os.path.getsize(dec_path) / 1e6, 1), "MB")
    print("DONE kvcache export")


if __name__ == "__main__":
    main()
