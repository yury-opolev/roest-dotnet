# Danish & Nordic TTS — Open-Source Landscape (May 2026)

Models are ranked by expected Danish quality and practicality for a project that
needs *both male and female* voices. License notes are abbreviated — verify before
shipping anything commercial.

---

## 1. CoRal Røst-v3-Chatterbox-500m  ⭐ recommended

- **What:** Danish fine-tune of ResembleAI's Chatterbox Multilingual, by the
  Alexandra Institute / CoRal project. State-of-the-art for Danish today.
- **Architecture:** 0.5 B Llama backbone, audio-token LM (same family as
  Chatterbox). Fine-tuned on 2 000+ hours of Danish on top of the
  500 k-hour multilingual base.
- **Speakers:** two pre-defined voices, **Mic** and **Nic** — one female and one
  male (the underlying CoRal-TTS dataset is exactly 1 F + 1 M, ~17 h each).
- **Voice cloning:** zero-shot from ~10 s of reference audio. So unlimited
  additional male/female voices are available without retraining.
- **Quality:** MOS **4.23 / 5** rated by 20 native Danish speakers.
- **License:** OpenRAIL (responsible-AI clauses; usable for most purposes,
  read the terms before commercial use).
- **Watermarking:** inherits Resemble's "Perth" neural watermark on every
  generated clip.
- **Hugging Face:** `CoRal-project/roest-v3-chatterbox-500m`
- **Quick start:**

  ```python
  import torchaudio as ta
  from chatterbox.mtl_tts import ChatterboxMultilingualTTS

  model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
  wav = model.generate("Hej, hvordan går det?", language_id="da")
  ta.save("out.wav", wav, model.sr)

  # voice clone
  wav = model.generate("Hej!", language_id="da",
                       audio_prompt_path="my_reference_10s.wav")
  ```

  Install: `pip install chatterbox-tts huggingface_hub torchaudio`

This is the model Emma's summary email missed — it is newer and clearly the
top recommendation.

---

## 2. Chatterbox Multilingual (vanilla)

- **What:** the base ResembleAI model that Røst-v3 is built from.
- **Languages:** 23 native, **Danish is one of them**.
- **License:** **MIT** — fully commercial-friendly. (The CoRal fine-tune above
  changes the license to OpenRAIL; using the base directly keeps you on MIT.)
- **Features:** zero-shot voice cloning, **emotion exaggeration control**
  (unique among open-source models).
- **Trade-off vs Røst-v3:** Danish quality is good but not as strong as the
  Danish-only fine-tune — Røst-v3 was rated MOS 4.23 specifically because of
  the extra 2 000 h of Danish data.
- **Hugging Face:** `ResembleAI/chatterbox-multilingual`
- **Use when:** the OpenRAIL terms of Røst-v3 are a problem and you can accept
  slightly weaker Danish prosody, or you need emotion control.

---

## 3. syvai/plapre-compact

- **What:** Danish-specific neural TTS, trained on the CoRal-TTS dataset, uses
  eSpeak-NG Danish phonemes as front-end.
- **Why interesting:** smaller / lighter than Chatterbox, Danish-only focus.
- **Limitation:** single voice profile out of the box (so doesn't solve the
  M+F requirement on its own).
- **Status:** model card is gated/requires HF auth — needs an account to inspect
  details.
- **Use when:** you need a small CPU-friendly Danish model and have one gender
  covered elsewhere.

---

## 4. Meta MMS-TTS (`facebook/mms-tts-dan`)

- **What:** VITS-based TTS in Meta's Massively Multilingual Speech project,
  covers 1 100+ languages including Danish.
- **Quality:** smoother and more natural prosody than Piper.
- **License:** **CC-BY-NC 4.0 — non-commercial only.** Hard blocker if this
  ever becomes a product.
- **Speakers:** single fixed speaker, no cloning, no style control.
- **Quick start:**

  ```python
  from transformers import VitsModel, AutoTokenizer
  import torch, scipy.io.wavfile

  model = VitsModel.from_pretrained("facebook/mms-tts-dan")
  tok = AutoTokenizer.from_pretrained("facebook/mms-tts-dan")
  inputs = tok("Hej, hvordan går det?", return_tensors="pt")
  with torch.no_grad():
      wav = model(**inputs).waveform
  scipy.io.wavfile.write("out.wav", model.config.sampling_rate,
                         wav.squeeze().numpy())
  ```

- **Use when:** quick prototype, research-only, single voice acceptable.

---

## 5. Piper TTS (`da_DK-talesyntese-medium`)

- **What:** Rhasspy's fast ONNX VITS runtime. Danish voice "Anders".
- **License:** MIT, model under permissive terms.
- **Speed:** real-time on CPU, even Raspberry Pi.
- **Quality:** *Emma already tested this and found it underwhelming.* OK as a
  fallback or for embedded/offline scenarios where Chatterbox is too heavy.
- **Speakers:** single male voice; no Danish female voice in Piper's official
  voice catalog.
- **Repo:** `github.com/rhasspy/piper`, models on `rhasspy/piper-voices`.

---

## 6. eSpeak-NG

- **What:** classic formant synthesizer. `apt install espeak-ng;
  espeak-ng -v da "Hej"`.
- **Quality:** robotic, formant-style — not in the same category as anything
  above.
- **License:** GPLv3.
- **Use when:** as a phonemizer front-end for other neural models (some
  pipelines need eSpeak's Danish phoneme dictionary) or as an offline last
  resort.

---

## Multilingual base models worth considering (no Danish out of the box)

These don't list Danish in their official language set, but are realistic
candidates if you want to **fine-tune your own** model (see
[`fine-tuning-paths.md`](fine-tuning-paths.md)).

| Model | Languages | License | Why interesting |
|-------|-----------|---------|-----------------|
| **F5-TTS** | multilingual, trained on 100 k h public data | research-friendly | Simplest fine-tuning loop of any 2026 model; fits in <24 GB VRAM. Flow-matching, fast. |
| **XTTS v2** (Coqui) | 17 langs — Dutch + German closest to Danish, **no Danish** | research/non-commercial-leaning | Mature voice-cloning. Fine-tuning a Danish variant on RTX 4090 ≈ 3–5 h. Community Danish forks exist but no official model. |
| **Fish-Speech S2 Pro** | 80+ langs | check repo | Dual-AR architecture, 10 M+ h training; very strong zero-shot cloning. |
| **OuteTTS v1.0** | Qwen3 backbone, cloning generalizes zero-shot | check repo | Pure LM approach, simple stack, cross-language cloning. |

---

## Sibling Nordic models — useful as warm-start

If we ever decide to train from scratch instead of fine-tuning Chatterbox,
the Norwegian and Swedish neighbours have good open material:

- **KBLab Piper-TTS Swedish** — VITS trained on NST Swedish (5 300 utterances,
  single speaker). HF: `KBLab/piper-tts-nst-swedish`. Confirms that NST
  data + Piper/VITS pipeline produces shippable quality.
- **NST Norwegian** — companion to NST Swedish/Danish from Språkbanken.
- **KTH (Stockholm) speech-synthesis research** — recent neural-HMM TTS papers,
  useful if we want attention-free architectures.

---

## Explicitly checked and confirmed **not** to support Danish

Coqui XTTS v2 (official), Silero, MaryTTS, Mimic 3, Bark, Tortoise, Kokoro,
ChatTTS, OpenVoice, Parler-TTS. Either English-centric or the supported
language list excludes Danish.
