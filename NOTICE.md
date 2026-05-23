# Notices & Attribution

This project converts and redistributes a third-party speech model. If you use
this repository or its release artifacts, you must comply with the upstream
licenses below.

## Model weights (release assets: ONNX graphs + reference tensors)

The ONNX graphs and dumped reference tensors distributed via GitHub Releases are
derived from:

- **CoRal Røst-v3-Chatterbox-500m** — the Danish fine-tuned model.
  - License: **OpenRAIL** (Open Responsible AI License) — includes
    use-based behavioral restrictions. By using the weights you agree to those
    restrictions.
  - © The Alexandra Institute / the CoRal project.
  - Source: https://huggingface.co/CoRal-project/roest-v3-chatterbox-500m
  - Trained on the CoRal-TTS, NST, FT Speech and Nota datasets.

- **Chatterbox Multilingual** — the base model Røst-v3 was fine-tuned from.
  - License: **MIT**.
  - © Resemble AI.
  - Source: https://huggingface.co/ResembleAI/chatterbox

The conversion does **not** retrain or alter the model's behavior; it re-encodes
the same weights into ONNX and reimplements surrounding pre/post-processing
(tokenizer, NSF/STFT/iSTFT, sampling) to run the model natively in .NET.

## Third-party code referenced during the port

- **chatterbox-tts** (the `chatterbox` Python package), MIT, © Resemble AI —
  the reference implementation whose logic the C# port reproduces. Copies of
  individual source files were consulted locally but are not redistributed here.

## This repository's own code

The .NET code under `native-onnx/DanishVoice.Native/` and `src/`, the Python
export scripts under `native-onnx/export/`, the Docker server under `server/`,
and the docs are original to this project. (Add a code LICENSE of your choice —
e.g. MIT — at the repo root; note it does not override the OpenRAIL terms on the
model weights.)

## Watermarking

The original Chatterbox output includes Resemble AI's "Perth" neural watermark.
The native ONNX pipeline in this repo omits that stage. Consider the upstream
project's guidance on responsible use and watermarking.
