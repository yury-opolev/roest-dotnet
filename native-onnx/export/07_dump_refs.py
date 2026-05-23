"""Dump deterministic reference tensors as flat float32 .bin + shapes.json so the
C# parity tests can read them without an npz/npy parser.

Makes the NSF source deterministic (zero random harmonic phase + zero noise) so
the vocoder chain is reproducible for exact parity. The stochastic terms are
perceptually negligible (tiny additive noise + harmonic phase offsets).
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
    from chatterbox.models.s3gen import hifigan
    from huggingface_hub import snapshot_download

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = snapshot_download(
        repo_id="CoRal-project/roest-v3-chatterbox-500m",
        token=os.getenv("HF_TOKEN") or None,
        allow_patterns=["*.safetensors", "*.json", "*.pt", "*.wav"],
    )
    model = ChatterboxMultilingualTTS.from_local(model_dir, device=device)

    # ---- Make NSF deterministic for parity ----
    def deterministic_sinegen_forward(self, f0):
        F_mat = torch.zeros((f0.size(0), self.harmonic_num + 1, f0.size(-1)), device=f0.device)
        for i in range(self.harmonic_num + 1):
            F_mat[:, i:i + 1, :] = f0 * (i + 1) / self.sampling_rate
        theta_mat = 2 * np.pi * (torch.cumsum(F_mat, dim=-1) % 1)
        phase_vec = torch.zeros((f0.size(0), self.harmonic_num + 1, 1), device=f0.device)  # ZERO
        sine_waves = self.sine_amp * torch.sin(theta_mat + phase_vec)
        uv = (f0 > self.voiced_threshold).float()
        sine_waves = sine_waves * uv  # NO additive noise
        return sine_waves, uv, sine_waves
    hifigan.SineGen.forward = deterministic_sinegen_forward

    # ---- Voice encoder reference (toolchain proof) ----
    print("voice encoder ref:")
    ve = model.ve
    torch.manual_seed(0)
    mels = torch.randn(22, 160, 40, device=device)
    with torch.no_grad():
        ve_out = ve(mels)
    dump("ve_input_mels", mels)
    dump("ve_output", ve_out)

    # ---- Vocoder chain reference (deterministic) ----
    print("vocoder refs:")
    voc = model.s3gen.mel2wav
    voc.eval()
    cap = {}
    orig = voc.inference
    def patched(*a, **k):
        cap.setdefault("speech_feat", k.get("speech_feat", a[0] if a else None))
        return orig(*a, **k)
    voc.inference = patched
    rp = os.path.join(model_dir, "_ref_mic.wav")
    model.generate("Hej, hvordan går det i dag?", language_id="da",
                  audio_prompt_path=rp if os.path.exists(rp) else None)
    voc.inference = orig

    speech_feat = torch.from_numpy(
        cap["speech_feat"].detach().cpu().numpy().astype(np.float32)).to(device)

    with torch.no_grad():
        f0 = voc.f0_predictor(speech_feat)
        s_up = voc.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = voc.m_source(s_up)
        s = s.transpose(1, 2)
        s_real, s_imag = voc._stft(s.squeeze(1))
        s_stft = torch.cat([s_real, s_imag], dim=1)
        # deterministic full inference
        full = voc.inference(speech_feat=speech_feat,
                             cache_source=torch.zeros(1, 1, 0, device=device))
        full_wav = full[0] if isinstance(full, (tuple, list)) else full

    dump("voc_speech_feat", speech_feat)
    dump("voc_f0", f0)
    dump("voc_s", s)
    dump("voc_s_stft", s_stft)
    dump("voc_full_wav", full_wav)

    with open(os.path.join(REF, "shapes.json"), "w") as f:
        json.dump(shapes, f, indent=2)
    print("wrote shapes.json")


if __name__ == "__main__":
    main()
