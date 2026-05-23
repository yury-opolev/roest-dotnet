# Fine-tuning Paths for Danish TTS

Three realistic paths, sorted by effort. Pick one based on quality vs.
control vs. licence trade-offs.

---

## Path A — No training, just use Røst-v3

**Effort:** 30 minutes.

CoRal already did the work. Download `CoRal-project/roest-v3-chatterbox-500m`,
generate with the two pre-existing speakers (Mic / Nic) or zero-shot clone
any other male/female reference. Done.

**Drawback:** OpenRAIL licence on the fine-tuned weights — verify usage
constraints before any commercial deployment.

---

## Path B — Voice-clone on the vanilla Chatterbox Multilingual (MIT)

**Effort:** a couple of hours, mostly to record/curate reference audio.

1. Pick or record ~10–30 s of clean Danish speech per target voice
   (one female reference, one male reference — e.g. from CoRal-TTS samples,
   which are CC0).
2. Call `ChatterboxMultilingualTTS.generate(..., language_id="da",
   audio_prompt_path=ref.wav)` for each.
3. Wrap into a small service that selects the reference clip by `voice=male`
   or `voice=female`.

**Pros:** stays on **MIT** licence end-to-end. No GPU training needed.
**Cons:** Danish prosody not quite as strong as Røst-v3, which trained on
2 000 h of Danish.

---

## Path C — Fine-tune Chatterbox Multilingual (or F5-TTS) on CoRal-TTS yourself

**Effort:** 1–2 days on a single A100 / RTX 4090, plus dataset prep.

This is essentially reproducing what CoRal did — but you control the
licence of the resulting weights (the base model is MIT, so the resulting
fine-tune can be released under MIT too if you do not adopt their OpenRAIL
addendum).

### High-level recipe

1. **Dataset prep:**
   - Pull `CoRal-project/coral-tts` from HF (~15 GB).
   - Optionally augment with Common Voice DA (more speakers, more variance).
   - Resample to the model's native rate (Chatterbox: 24 kHz typically).
   - Split into train/val by utterance id, *keeping speakers in train and val*
     so the model learns speaker conditioning rather than memorising.

2. **Choice of base model:**
   - **Chatterbox Multilingual** — best Danish baseline since it already
     handles Danish, fine-tune just refines prosody and adds dedicated
     speaker tokens for Mic/Nic. Recommended if you want CoRal-quality output.
   - **F5-TTS** — simpler training stack, flow-matching, fits in <24 GB VRAM.
     A reasonable fallback if Chatterbox's training pipeline is awkward.

3. **Training:**
   - Follow each project's fine-tune docs (Chatterbox provides a CLI;
     F5-TTS has a fine-tune script in its repo).
   - Expect ~3–5 hours of compute on a 4090 for an F5-TTS fine-tune, longer
     for Chatterbox.
   - Evaluate with a small MOS test on held-out Danish text. Aim for
     parity with Røst-v3 (MOS ≈ 4.2).

4. **Add more speakers later:**
   - With voice cloning enabled, you do not need to retrain to add new
     genders/voices — just curate a 10 s reference clip.

### Side note on XTTS v2

Coqui XTTS v2 (17 official languages) does **not** include Danish, but the
architecture supports adding languages via fine-tuning. Community Danish
forks exist with mixed results. Only worth pursuing if voice cloning quality
of XTTS specifically (which is excellent for the languages it ships with)
is preferred over Chatterbox's. Most evidence in 2026 points to Chatterbox
as the better starting point for Danish.

---

## Decision shortcut

| You care most about… | Pick |
|---|---|
| Best Danish quality, fastest | **Path A** (Røst-v3) |
| Permissive (MIT) licence | **Path B** (Chatterbox clone) or **Path C** with Chatterbox base |
| Owning the model artifact end-to-end | **Path C** |
| Tiny / CPU-only / Raspberry-Pi-class | Piper `da_DK-talesyntese-medium` or syvai/plapre-compact (accept lower quality) |
| Tiny MIT-licensed model trained on your own data | **Path D** — see [`piper-finetune-guide.md`](piper-finetune-guide.md): fine-tune Piper VITS from the KBLab Swedish checkpoint |

---

## Path D — Piper VITS, fine-tuned from KBLab Swedish

A separate concrete recipe (transcribed from Emma's 2026-05-21 email) lives
in [`piper-finetune-guide.md`](piper-finetune-guide.md). Summary:

- Start from `sv_SE-nst-medium.ckpt` (KBLab) — Nordic warm-start beats English.
- Phonemize Danish text with **eSpeak-NG `da`** (the most language-critical step).
- Train 50 k–200 k steps on ~5 h of CoRal-TTS audio per speaker (2–7 days on RTX 3090).
- Export to **ONNX** — runs in real time on a Raspberry Pi.
- For male **and** female, train two models (one per speaker subset).

Trade-off vs. Path C: smaller and fully MIT-licensed, but lower MOS ceiling
than Chatterbox-class models.
