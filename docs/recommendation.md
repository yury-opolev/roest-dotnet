# Recommendation

## Short version

1. **Try first:** `CoRal-project/roest-v3-chatterbox-500m` on Hugging Face.
   Danish state-of-the-art (MOS 4.23/5), built-in male **and** female voices
   (Mic, Nic), zero-shot cloning from 10 s reference. Probably the answer.

2. **If the OpenRAIL licence is a blocker** for commercial use:
   `ResembleAI/chatterbox-multilingual` (MIT) — Danish is one of its 23
   built-in languages; clone a male and a female reference once from any
   CC0 source (e.g. CoRal-TTS samples) and you have your two voices.

3. **If you want to own a fully custom model** with your own data and
   licence terms: fine-tune Chatterbox Multilingual or F5-TTS on
   `CoRal-project/coral-tts` (CC0, 1 F + 1 M, 34 h, studio-grade).

The model Emma's email recommended (`facebook/mms-tts-dan`) is non-commercial
only and single-voice, so it does not satisfy the M+F + quality + licence
requirements together. Piper's `da_DK-talesyntese-medium` is fast but
quality is — as you already found — underwhelming.

## Why CoRal Røst-v3 instead of MMS-TTS

| | **CoRal Røst-v3** | MMS-TTS-dan |
|---|---|---|
| Quality (MOS) | 4.23 / 5 (native rated) | not formally rated, generally <4 |
| Male + female voices | yes (Mic, Nic) | no — single fixed speaker |
| Voice cloning | yes, 10 s zero-shot | no |
| Licence | OpenRAIL | CC-BY-NC 4.0 (non-commercial) |
| Architecture | 500 M Llama-based audio LM | VITS |

## What to do this week

1. `pip install chatterbox-tts huggingface_hub torchaudio` and run a few
   sentences through Røst-v3 with both Mic and Nic. See if quality matches
   what the MOS suggests.
2. Read the OpenRAIL terms — decide whether the licence works for the
   intended product.
3. If yes → ship. If no → fall back to vanilla Chatterbox Multilingual (MIT)
   with two cloned references (Path B in
   [`fine-tuning-paths.md`](fine-tuning-paths.md)).
4. Only if both of those disappoint, start Path C (fine-tune on CoRal-TTS).
