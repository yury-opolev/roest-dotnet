"""Vocoder split export (task #8).

The HiFT vocoder can't export whole (NSF in-place loop + iSTFT complex). Split:
  - ONNX graph A: f0_predictor            (speech_feat -> f0)
  - C# DSP:       f0 upsample + NSF source + forward STFT  -> s_stft
  - ONNX graph B: conv stack              (speech_feat, s_stft -> magnitude, phase)
  - C# DSP:       iSTFT(magnitude, phase) -> waveform -> clamp

This script exports A and B, dumps every constant the C# DSP needs, and saves
reference tensors at each boundary so the C# port can be checked for parity.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

OUT = "/work/onnx_models"
REF = "/work/refs"
os.makedirs(OUT, exist_ok=True)
os.makedirs(REF, exist_ok=True)


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
    voc = model.s3gen.mel2wav
    voc.eval()

    # ---- capture a real speech_feat (vocoder input) during generate() ----
    cap = {}
    orig = voc.inference
    def patched(*a, **k):
        cap.setdefault("speech_feat", k.get("speech_feat", a[0] if a else None))
        return orig(*a, **k)
    voc.inference = patched
    ref_wav_path = os.path.join(model_dir, "_ref_mic.wav")
    model.generate("Hej, hvordan går det i dag?", language_id="da",
                  audio_prompt_path=ref_wav_path if os.path.exists(ref_wav_path) else None)
    voc.inference = orig
    # Rebuild as a normal (non-inference-mode) tensor via numpy round-trip so it
    # can be traced by torch.onnx.export.
    speech_feat = torch.from_numpy(
        cap["speech_feat"].detach().cpu().numpy().astype(np.float32)).to(device)
    print("speech_feat", tuple(speech_feat.shape))

    n_fft = voc.istft_params["n_fft"]
    hop = voc.istft_params["hop_len"]

    # ---- reference full inference output (end-to-end target) ----
    with torch.inference_mode():
        full = voc.inference(speech_feat=speech_feat,
                             cache_source=torch.zeros(1, 1, 0, device=device))
        full_wav = full[0] if isinstance(full, (tuple, list)) else full

    # ---- reproduce the internal source pipeline to get reference s_stft ----
    # Use no_grad (not inference_mode) so the resulting tensors are normal and
    # usable as export example inputs.
    with torch.no_grad():
        f0 = voc.f0_predictor(speech_feat)
        s_up = voc.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = voc.m_source(s_up)
        s = s.transpose(1, 2)
        s_real, s_imag = voc._stft(s.squeeze(1))
        s_stft = torch.cat([s_real, s_imag], dim=1)
        # round-trip to guarantee normal tensors for export
        s_stft = torch.from_numpy(s_stft.cpu().numpy()).to(device)
    print("f0", tuple(f0.shape), "s", tuple(s.shape), "s_stft", tuple(s_stft.shape))

    # ============ ONNX graph A: f0_predictor ============
    class F0Net(torch.nn.Module):
        def __init__(s, m):
            super().__init__(); s.m = m
        def forward(s, speech_feat):
            return s.m(speech_feat)

    pathA = os.path.join(OUT, "voc_f0_predictor.onnx")
    torch.onnx.export(
        F0Net(voc.f0_predictor), (speech_feat,), pathA, opset_version=17,
        input_names=["speech_feat"], output_names=["f0"],
        dynamic_axes={"speech_feat": {0: "B", 2: "T"}, "f0": {0: "B", 1: "T"}},
        do_constant_folding=True, dynamo=False)
    print("graph A f0_predictor OK", os.path.getsize(pathA) / 1e6, "MB")

    # ============ ONNX graph B: conv stack ============
    class ConvStack(torch.nn.Module):
        def __init__(s, v):
            super().__init__(); s.v = v
        def forward(s, speech_feat, s_stft):
            v = s.v
            x = v.conv_pre(speech_feat)
            for i in range(v.num_upsamples):
                x = F.leaky_relu(x, v.lrelu_slope)
                x = v.ups[i](x)
                if i == v.num_upsamples - 1:
                    x = v.reflection_pad(x)
                si = v.source_downs[i](s_stft)
                si = v.source_resblocks[i](si)
                x = x + si
                xs = None
                for j in range(v.num_kernels):
                    blk = v.resblocks[i * v.num_kernels + j]
                    xs = blk(x) if xs is None else xs + blk(x)
                x = xs / v.num_kernels
            x = F.leaky_relu(x)
            x = v.conv_post(x)
            magnitude = torch.exp(x[:, : n_fft // 2 + 1, :])
            phase = torch.sin(x[:, n_fft // 2 + 1 :, :])
            return magnitude, phase

    pathB = os.path.join(OUT, "voc_conv_stack.onnx")
    torch.onnx.export(
        ConvStack(voc), (speech_feat, s_stft), pathB, opset_version=17,
        input_names=["speech_feat", "s_stft"], output_names=["magnitude", "phase"],
        dynamic_axes={"speech_feat": {0: "B", 2: "T"}, "s_stft": {0: "B", 2: "Ts"},
                      "magnitude": {0: "B", 2: "Tm"}, "phase": {0: "B", 2: "Tm"}},
        do_constant_folding=True, dynamo=False)
    print("graph B conv_stack OK", os.path.getsize(pathB) / 1e6, "MB")

    with torch.no_grad():
        mag_ref, phase_ref = ConvStack(voc)(speech_feat, s_stft)

    # ---- constants the C# DSP needs ----
    ms = voc.m_source
    sg = ms.l_sin_gen
    consts = {
        "n_fft": int(n_fft),
        "hop_len": int(hop),
        "sampling_rate": int(voc.sampling_rate),
        "harmonic_num": int(sg.harmonic_num),
        "sine_amp": float(sg.sine_amp),
        "noise_std": float(sg.noise_std),
        "voiced_threshold": float(sg.voiced_threshold),
        "f0_upsample_factor": int(np.prod(voc.f0_upsamp.scale_factor) if hasattr(voc.f0_upsamp, "scale_factor") else 0) or int(voc.f0_upsamp.scale_factor),
        "audio_limit": float(voc.audio_limit),
        "lrelu_slope": float(voc.lrelu_slope),
        "l_linear_weight": ms.l_linear.weight.detach().cpu().numpy().tolist(),  # [1, harmonic_num+1]
        "l_linear_bias": ms.l_linear.bias.detach().cpu().numpy().tolist(),       # [1]
    }
    with open(os.path.join(REF, "vocoder_consts.json"), "w") as f:
        json.dump(consts, f, indent=2)

    # hann window
    np.save(os.path.join(REF, "stft_window.npy"), voc.stft_window.detach().cpu().numpy())

    # reference tensors at every boundary
    np.savez(
        os.path.join(REF, "vocoder_refs.npz"),
        speech_feat=speech_feat.cpu().numpy(),
        f0=f0.cpu().numpy(),
        s=s.cpu().numpy(),
        s_stft=s_stft.cpu().numpy(),
        magnitude=mag_ref.cpu().numpy(),
        phase=phase_ref.cpu().numpy(),
        full_wav=full_wav.cpu().numpy(),
    )
    print("dumped consts + refs to", REF)
    print("DONE vocoder split export")


if __name__ == "__main__":
    main()
