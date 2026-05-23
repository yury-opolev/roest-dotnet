"""Single fully-deterministic generate() that dumps EVERY stage consistently so
the assembled C# pipeline can be verified end to end against one Python run.

Determinism: multinomial->argmax, alignment analyzer disabled, and the CFM noise
z captured (so C# can load the exact same z). Everything below comes from the
same run, so the stages chain correctly.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

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
    from chatterbox.models.t3.inference.alignment_stream_analyzer import AlignmentStreamAnalyzer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = snapshot_download(
        repo_id="CoRal-project/roest-v3-chatterbox-500m",
        token=os.getenv("HF_TOKEN") or None,
        allow_patterns=["*.safetensors", "*.json", "*.pt", "*.wav"],
    )
    model = ChatterboxMultilingualTTS.from_local(model_dir, device=device)
    t3, flow, voc = model.t3, model.s3gen.flow, model.s3gen.mel2wav

    cap = {}

    # determinism
    torch.multinomial = lambda probs, num_samples=1, **kw: probs.argmax(dim=-1, keepdim=True)
    AlignmentStreamAnalyzer.step = lambda self, logits, next_token=None: logits

    # deterministic NSF (zero random phase + zero noise) so the vocoder waveform
    # is reproducible for an apples-to-apples comparison with the C# vocoder.
    import numpy as _np
    from chatterbox.models.s3gen import hifigan as _hifi
    def _det_sinegen(self, f0):
        F_mat = torch.zeros((f0.size(0), self.harmonic_num + 1, f0.size(-1)), device=f0.device)
        for i in range(self.harmonic_num + 1):
            F_mat[:, i:i + 1, :] = f0 * (i + 1) / self.sampling_rate
        theta = 2 * _np.pi * (torch.cumsum(F_mat, dim=-1) % 1)
        sine = self.sine_amp * torch.sin(theta)
        uv = (f0 > self.voiced_threshold).float()
        sine = sine * uv
        return sine, uv, sine
    _hifi.SineGen.forward = _det_sinegen

    # capture T3 prepared text tokens + cond
    orig_t3_inf = t3.inference
    def t3_spy(*a, **k):
        cap["text_tokens"] = k.get("text_tokens")
        cap["t3_cond_obj"] = k.get("t3_cond")
        r = orig_t3_inf(*a, **k)
        cap["speech_tokens"] = r
        return r
    t3.inference = t3_spy

    # capture flow inputs
    orig_flow_inf = flow.inference
    def flow_spy(*a, **k):
        cap["flow_kwargs"] = k
        return orig_flow_inf(*a, **k)
    flow.inference = flow_spy

    # capture concatenated token ids into input_embedding
    def inemb_hook(m, args, kwargs):
        cap.setdefault("token_concat", args[0])
    h_ie = flow.input_embedding.register_forward_pre_hook(inemb_hook, with_kwargs=True)

    # capture CFM solver inputs + the z it uses (patch solve_euler entry)
    import chatterbox.models.s3gen.flow_matching as fm
    orig_forward = fm.CausalConditionalCFM.forward
    def cfm_forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None,
                    noised_mels=None, meanflow=False):
        g = torch.Generator(device=mu.device).manual_seed(1234)
        z = torch.randn(mu.shape, generator=g, device=mu.device, dtype=mu.dtype)
        cap["cfm_mu"] = mu; cap["cfm_mask"] = mask; cap["cfm_spks"] = spks
        cap["cfm_cond"] = cond; cap["cfm_z"] = z
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        out = self.solve_euler(z, t_span, mu, mask, spks, cond, meanflow=meanflow)
        cap["cfm_out"] = out
        return out, None
    fm.CausalConditionalCFM.forward = cfm_forward

    # capture vocoder mel input
    orig_voc = voc.inference
    def voc_spy(*a, **k):
        cap["voc_speech_feat"] = k.get("speech_feat", a[0] if a else None)
        return orig_voc(*a, **k)
    voc.inference = voc_spy

    with torch.inference_mode():
        wav = model.generate("Hej, hvordan går det i dag?", language_id="da",
                            audio_prompt_path=os.path.join(model_dir, "_ref_mic.wav"))

    h_ie.remove()

    # ---- dump consistent stage tensors ----
    fk = cap["flow_kwargs"]
    with torch.inference_mode():
        cond_emb = t3.prepare_conditioning(cap["t3_cond_obj"])
    dump("e2e_cond_emb", cond_emb)
    json.dump(cap["text_tokens"].detach().cpu().numpy().astype(np.int64).tolist(),
              open(os.path.join(REF, "e2e_text_tokens.json"), "w"))
    json.dump(cap["speech_tokens"].detach().cpu().numpy().astype(np.int64).tolist(),
              open(os.path.join(REF, "e2e_speech_tokens.json"), "w"))
    json.dump(cap["token_concat"].detach().cpu().numpy().astype(np.int64).tolist(),
              open(os.path.join(REF, "e2e_token_concat.json"), "w"))
    dump("e2e_prompt_feat", fk["prompt_feat"].float())
    dump("e2e_xvector", fk["embedding"].float())
    json.dump({"prompt_token_len": int(fk["prompt_token"].shape[-1]),
               "mel_len1": int(fk["prompt_feat"].shape[1]),
               "n_speech_tokens": int(cap["speech_tokens"].shape[-1])},
              open(os.path.join(REF, "e2e_meta.json"), "w"))
    dump("e2e_cfm_mu", cap["cfm_mu"]); dump("e2e_cfm_mask", cap["cfm_mask"])
    dump("e2e_cfm_spks", cap["cfm_spks"]); dump("e2e_cfm_cond", cap["cfm_cond"])
    dump("e2e_cfm_z", cap["cfm_z"]); dump("e2e_cfm_out", cap["cfm_out"])
    dump("e2e_voc_speech_feat", cap["voc_speech_feat"].float())
    dump("e2e_wav", wav.float())

    with open(os.path.join(REF, "shapes_e2e.json"), "w") as f:
        json.dump(shapes, f, indent=2)
    print("DONE consistent e2e refs; wav", tuple(wav.shape))


if __name__ == "__main__":
    main()
