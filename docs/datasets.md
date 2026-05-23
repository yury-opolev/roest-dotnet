# Open Danish & Nordic Speech Datasets

For two purposes:

1. **Fine-tuning** an existing multilingual model on Danish.
2. **Training from scratch** (last resort — only the CoRal team has done this
   recently at scale, and it took 2 000+ h of curated audio).

---

## Danish

### CoRal-TTS  ⭐ first choice for fine-tuning

- **Hugging Face:** `CoRal-project/coral-tts` (also mirrored at
  `alexandrainst/coral-tts`).
- **Hours:** ~34 h total — **17 h female + 17 h male** (speakers Mic and Nic).
- **Quality:** studio recordings by Nota (Danish Ministry of Culture).
- **Sample rate:** 44 100 Hz.
- **Samples:** 18 863 utterances.
- **Text sources:** sundhed.dk, borger.dk, bus-stop / station names,
  manually filtered Reddit comments, dates and times.
- **License:** **CC0 1.0** (public domain) — use freely, even commercially.
- **Funding:** Danish Innovation Fund.

This is by far the cleanest open Danish TTS corpus and is exactly designed
for the M+F use case.

### NST Danish ASR Database (16 kHz)

- **Source:** Språkbanken, Norwegian National Library
  (`nb.no/sprakbanken`). Mirrors on HF (`alexandrainst/nst-da`).
- **Hours:** ~390 h read-aloud speech.
- **Speakers:** 616, with metadata (age, gender, region, dialect).
- **License:** **CC0**.
- **Caveat:** designed for ASR — short utterances, 16 kHz, noisier conditions
  than a TTS studio corpus. Still usable for multi-speaker TTS training or
  as augmentation.

### FT Speech — Danish Parliament Speech Corpus

- ~1 800 h of parliamentary speech.
- Useful for ASR; for TTS it's harder (overlapping speech, podium acoustics)
  but the diversity of speakers is enormous.

### Mozilla Common Voice — Danish

- Crowdsourced, validated read speech.
- Smaller than the above for Danish but actively growing.
- CC0 license.
- A VITS Danish female model has already been published trained purely on
  Common Voice ("Danish female TTS Model vits Encoding Trained on cv Dataset
  at 22050 Hz").

---

## Nordic neighbours (useful for transfer learning)

### NST Swedish ASR (16 kHz)

- Same Språkbanken provenance. KBLab's Swedish Piper voice was trained on
  the TTS-style subset (5 300 utterances, single speaker).

### NST Norwegian ASR (16 kHz, reorganised)

- Companion corpus, same license.

### Why bother with these?

Danish, Swedish and Norwegian Bokmål share a lot of phonetic territory.
If we ever want to train a single Nordic multi-speaker base and then specialise
for Danish, the combined NST corpus is the natural starting point.

---

## Practical recipe for "male + female Danish"

The simplest path that satisfies the requirement, in order of effort:

1. Use **CoRal Røst-v3-Chatterbox-500m** directly — already has Mic (♀) and
   Nic (♂), no training needed.
2. Voice-clone with **Chatterbox Multilingual (MIT)** using two short reference
   clips, one male and one female.
3. Fine-tune **Chatterbox Multilingual** or **F5-TTS** on **CoRal-TTS**
   to produce a custom dual-speaker model — gives you a fully self-contained
   artifact under a permissive license.

See [`fine-tuning-paths.md`](fine-tuning-paths.md) for the concrete steps for
option 3.
