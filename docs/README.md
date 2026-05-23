# Danish Voice — Research Notes

Goal: generate **high-quality** Danish speech from text, ideally with both **male and female** voices, using **open-source** components.

Documents in this folder:

- [`tts-landscape.md`](tts-landscape.md) — survey of every Danish/Nordic open-source TTS option (ranked) plus multilingual models that include Danish out of the box.
- [`datasets.md`](datasets.md) — open Danish & Nordic speech datasets that can be used to train or fine-tune a TTS model.
- [`fine-tuning-paths.md`](fine-tuning-paths.md) — concrete recipes for getting a quality male+female Danish voice, from "download a model" to "fine-tune your own".
- [`piper-finetune-guide.md`](piper-finetune-guide.md) — step-by-step Piper/VITS fine-tune from the KBLab Swedish checkpoint (Emma's 2026-05-21 technical guide, full version). Pick this when MIT licence + tiny ONNX + Pi-class inference matters more than absolute MOS.
- [`recommendation.md`](recommendation.md) — short version: what to try first, second, third.
- [`integration-into-csharp-project.md`](integration-into-csharp-project.md) — how to consume this TTS from another C# project (native vs container, why native isn't worth it, streaming for real-time chat).

## TL;DR

Best out-of-the-box quality for Danish (May 2026) is **CoRal Røst-v3-Chatterbox-500m**
(MOS 4.23/5 by native speakers, two pre-defined speakers — one male, one female — plus
zero-shot voice cloning from 10 s of reference audio). See
[`recommendation.md`](recommendation.md) for the full ranking and rationale.
