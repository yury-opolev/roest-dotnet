# Piper / VITS Fine-tune from the KBLab Swedish Checkpoint

Concrete recipe transcribed and lightly reorganised from Emma's email
"Fine-tuning a TTS model for Danish - technical guide" (2026-05-21).

This is the **Path D** alternative to fine-tuning Chatterbox/F5-TTS
(see [`fine-tuning-paths.md`](fine-tuning-paths.md)). It produces a tiny
ONNX model that runs in real time on a Raspberry Pi, under a fully
permissive MIT license. The trade-off is quality: VITS at this scale will
not match CoRal Røst-v3, but it is the right pick when you need
small/offline/permissive and are OK with single-voice output per fine-tune.

For a male **and** female pair, train **two** Piper models — one per
speaker — using the corresponding split of CoRal-TTS.

---

## Why this path exists

A Nordic-language warm-start (Swedish from KBLab) closes the gap to Danish
much faster than starting from an English VITS. Norwegian/Swedish/Danish
share enough phonetic territory that the Swedish base only needs
re-phonemizing to Danish and a few thousand fine-tune steps to specialise.

---

## Phase 1 — Choose the base model

Emma's comparison, with the trade-offs:

| Option | Quality | Effort | License | Notes |
|---|---|---|---|---|
| **VITS via Piper** (recommended) | High | Low | MIT (Piper) | End-to-end phonemes→waveform via VAE+GAN. Fast fine-tune. ONNX export. |
| MMS-TTS (VITS variant) | High | Low | **CC-BY-NC 4.0** | Research-only — blocker for product use. |
| XTTS v2 (Coqui) | Very high, zero-shot cloning | High | CPML (non-commercial) | Danish + Swedish are NOT in the 17 languages. Extending requires retraining a head. Skip unless you need cloning. |
| FastSpeech 2 + HiFi-GAN | High, controllable | Higher | Mixed | Two-stage pipeline (text→mel→waveform). More moving parts. Pick if you want explicit duration/pitch control. |

**Recommendation:** start with **Piper fine-tuned from the KBLab `sv_SE-nst-medium`
checkpoint.** Fastest path to a working Danish voice with a permissive license.

---

## Phase 2 — Dataset preparation (≈70 % of the work)

### How much audio?

- **3–5 h:** viable for fine-tuning a single voice on a strong base.
- **10–25 h:** comfortable target for production-grade quality.
- **50 h+:** only needed when training mostly from scratch.

### Public Danish corpora to draw from

- **CoRal-TTS** — `huggingface.co/datasets/alexandrainst/coral-tts` — first
  choice (studio-grade, 1 F + 1 M, ~17 h each).
- **NST Danish** — Språkbanken / Nasjonalbiblioteket — large historical ASR
  corpus, usable for TTS after cleaning.
- **Common Voice (da)** — `commonvoice.mozilla.org/da` — crowdsourced,
  noisier but free.
- **DanPASS** — Danish Phonetically Annotated Spontaneous Speech (research).
- **LibriVox Danish audiobooks** — limited but public domain.

### Audio preprocessing

- Resample to **22 050 Hz mono WAV** (Piper default; 24 000 Hz also fine).
- Loudness-normalize to **−23 LUFS** (EBU R128). Tool:
  [`ffmpeg-normalize`](https://github.com/slhck/ffmpeg-normalize).
- Trim leading/trailing silence (`sox` or `librosa.effects.trim`).
- Segment into **2–10 s** clips. Longer breaks attention, shorter loses
  prosodic context.
- Drop clips with background noise, music, multiple speakers, or clipping.

```bash
# ffmpeg loudness + resample pass
ffmpeg -i input.wav -af loudnorm=I=-23:LRA=7:TP=-2 -ar 22050 -ac 1 output.wav
```

### Text normalisation

Numbers, dates, currency and abbreviations must be spelled out exactly as you
want them pronounced. `"25 kr."` → `"femogtyve kroner."`. There is no
perfect open-source Danish text normaliser — expect to write rules or borrow
from danspeech.

### Phonemization (the most Danish-specific step)

Use **eSpeak-NG with the `da` voice** to produce IPA. This single step is
what makes the difference between "Danish voice" and "Swedish reading Danish".

```bash
sudo apt install espeak-ng
echo "Hej, hvordan går det?" | espeak-ng -v da --ipa -q
# h ɛj ʋɔɐ̯ˈdæn ɡɒːɐ̯ deːˀ
```

For batch use, the `phonemizer` Python package with the eSpeak backend.

### Splits

90 % train, 5 % validation, 5 % test. **Do not split the same speaker or
recording session across train and val** — that leaks. Hold out at least
100 sentences for the final listening test.

---

## Phase 3 — Environment

### Hardware

- **24 GB VRAM ideal** (RTX 3090 / 4090 / A5000 / A100).
- **12 GB workable** (3060 / 4070) with reduced batch + gradient accumulation.
- CPU training is impractical.

### Cloud GPU if you don't own one

- **RunPod** — ~$0.40/h for RTX 3090, good default.
- **Vast.ai** — cheapest spot, less reliable.
- **Lambda Labs** — pricier but stable, A100 available.
- **Colab Pro+** — session limits hurt long training runs.

### Install

```bash
# System
sudo apt install espeak-ng ffmpeg sox

# Python
python -m venv venv && source venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

git clone https://github.com/rhasspy/piper.git
cd piper/src/python
pip install -e .

# Build the monotonic_align CUDA extension
cd piper_train/vits/monotonic_align
python setup.py build_ext --inplace
```

---

## Phase 4 — Fine-tune Piper

### Dataset layout (LJSpeech format)

```
my_dataset/
  metadata.csv     # pipe-separated: clip_id|raw_text|normalized_text
  wavs/
    clip_0001.wav
    clip_0002.wav
    ...
```

### Preprocess

```bash
python -m piper_train.preprocess \
  --language da \
  --input-dir my_dataset \
  --output-dir my_dataset_preprocessed \
  --dataset-format ljspeech \
  --sample-rate 22050 \
  --single-speaker
```

This phonemizes via eSpeak-NG `da` and writes `config.json` plus a
`dataset.jsonl` with phoneme IDs.

### Train from the Swedish checkpoint

Grab `sv_SE/nst/medium` from `huggingface.co/rhasspy/piper-voices`. You want
the **`.ckpt`** file, not the ONNX.

```bash
python -m piper_train \
  --dataset-dir my_dataset_preprocessed \
  --accelerator gpu --devices 1 \
  --batch-size 32 \
  --validation-split 0.0 \
  --num-test-examples 0 \
  --max_epochs 10000 \
  --resume_from_checkpoint sv_SE-nst-medium.ckpt \
  --checkpoint-epochs 1 \
  --precision 16 \
  --quality medium
```

**Gotcha:** the Swedish checkpoint's phoneme embedding table is sized for
Swedish phonemes. When loading into a Danish-phonemized run you'll likely
hit a size mismatch on `emb_t`. Drop or extend that tensor in the state
dict before loading — there are scripts for both approaches in the Piper
issue tracker.

### Monitor

```bash
tensorboard --logdir lightning_logs/
```

Watch mel reconstruction loss and the generator/discriminator balance.
**Listen to TensorBoard audio samples every few epochs** — your ear catches
issues the loss curves miss.

### Realistic timeline

50 k – 200 k steps. On a single RTX 3090 with a 5 h Danish dataset that is
roughly **2–7 days** of continuous training. Stop when validation samples
sound clean and stop improving.

### Export to ONNX

```bash
python -m piper_train.export_onnx \
  --checkpoint epoch=N-step=M.ckpt \
  --output-file my-da-voice.onnx
cp my_dataset_preprocessed/config.json my-da-voice.onnx.json
```

---

## Phase 5 — Evaluation

### Objective

- **MCD** (Mel Cepstral Distortion) — lower is better.
- **F0 RMSE** — pitch contour accuracy.
- **WER via Whisper large-v3** — transcribe the synthesized audio and
  compare to the input text; high WER ⇒ poor intelligibility.

### Subjective

- **MOS** — 1–5 scale, 10+ native Danish raters, blind A/B against reference.
- **CMOS** — comparative MOS, side-by-side.
- **NISQA** — automated MOS estimator (`github.com/gabrielmittag/NISQA`).
  Useful for quick iteration but trust real listeners for final judgement.

---

## Phase 6 — Deployment

```bash
piper --model my-da-voice.onnx \
      --output_file out.wav \
      <<< "Hej, hvordan går det i dag?"
```

Piper supports streaming inference and runs comfortably on a Raspberry Pi 4.
For a server, batch requests for throughput. Publish the model to HuggingFace
Hub with a card crediting KBLab's `sv_SE` base and the dataset sources.

---

## Common pitfalls

1. **Skipping the eSpeak language switch from Swedish to Danish.** The #1
   trap. Output sounds like a confused Swede.
2. **Noisy or mislabeled training clips.** One hour of clean data beats five
   hours of dirty data.
3. **Transcript errors.** Run a Whisper pass on your audio and diff against
   transcripts to surface mismatches.
4. **Overfitting on tiny datasets.** <3 h ⇒ the voice memorises phrases.
5. **Timbre drift over long training.** Save checkpoints often, listen — the
   lowest-loss checkpoint is rarely the best-sounding.
6. **Forgetting loudness normalisation.** Inconsistent levels confuse the
   model's gain handling.
7. **Phoneme embedding size mismatch when resuming from Swedish.** Strip or
   extend `emb_t` before loading.

---

## 2-week starter plan

### Week 1 — Data

- **Day 1:** Download CoRal-TTS + Common Voice da. Inspect, pick the
  cleanest subset.
- **Day 2:** Preprocessing pipeline — resample, loudness-normalize, trim,
  segment. End with `wavs/` + `metadata.csv`.
- **Day 3:** Text-normalisation rules for Danish numbers/dates/abbrev.
  Validate on 100 sentences.
- **Day 4:** Phonemize the whole dataset with eSpeak-NG `da`. Spot-check
  50 samples by ear (synthesize directly with `espeak-ng`).
- **Day 5:** Train/val/test splits. Whisper-transcribe the audio and align
  against transcripts to catch labelling errors.

### Week 2 — Training & evaluation

- **Day 6:** Provision a cloud GPU (RunPod RTX 3090 is a fine default).
  Install everything; build `monotonic_align`.
- **Day 7:** Download `sv_SE-nst-medium.ckpt`. Run preprocessing. Kick off
  training. Watch the first few hundred steps for crashes.
- **Days 8–11:** Let it train. TensorBoard daily, listen every couple of
  epochs.
- **Day 12:** Export best checkpoint to ONNX. Synthesize 100 held-out test
  sentences. Compute Whisper WER.
- **Day 13:** Quick listening test with ≥3 Danish speakers. Note systematic
  issues (phonemes, prosody, pace).
- **Day 14:** Decide: more training, more data, or adjust phonemization.
  Document everything.

After two weeks you should have a Danish Piper voice **at least** as good as
the existing `da_DK-talesyntese-medium`, and usually better on the parts of
Danish well-represented in your data. The third iteration is typically where
it starts sounding genuinely good — plan for that.

To cover **male + female**, repeat the whole training with the other
speaker subset of CoRal-TTS (and optionally name them after the speakers
in the dataset).

---

## Learning resources (Emma's list)

- Piper training guide: <https://github.com/rhasspy/piper/blob/master/TRAINING.md>
- VITS paper (Kim, Kong, Son 2021): <https://arxiv.org/abs/2106.06103>
- MMS paper (Pratap et al, Meta 2023): <https://arxiv.org/abs/2305.13516>
- FastSpeech 2 paper: <https://arxiv.org/abs/2006.04558>
- HiFi-GAN paper: <https://arxiv.org/abs/2010.05646>
- HuggingFace Audio Course (TTS in chapters 6–7): <https://huggingface.co/learn/audio-course>
- Coqui TTS docs (concepts still useful): <https://docs.coqui.ai>
- Edinburgh CSTR Speech-Synthesis course: <https://speech.zone/courses/speech-synthesis>
- KBLab blog (Swedish National Library, owners of the `sv_SE` Piper voice): <https://kb-labb.github.io>
- NISQA: <https://github.com/gabrielmittag/NISQA>
- Phonemizer: <https://github.com/bootphon/phonemizer>

---

## How this fits with the rest of the docs

- If you don't need to train at all → [`recommendation.md`](recommendation.md)
  (try Røst-v3 first).
- If you want a higher-quality fine-tune than Piper VITS can give → Path C
  in [`fine-tuning-paths.md`](fine-tuning-paths.md) (Chatterbox / F5-TTS on
  CoRal-TTS).
- This Piper recipe is the right path when **MIT licence + tiny ONNX +
  Raspberry-Pi inference** matters more than absolute MOS.
