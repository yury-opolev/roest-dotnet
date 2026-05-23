"""Dump reference token-id sequences from the real MTLTokenizer (language_id='da')
for a range of Danish sentences, so the C# BPE reimplementation can be checked
for exact parity. Also dumps the post-preprocess string for each.
"""

from __future__ import annotations

import glob
import json
import os

REF = "/work/refs"
os.makedirs(REF, exist_ok=True)

SENTENCES = [
    "Hej, hvordan går det med dig i dag?",
    "Vejret er smukt i København, og solen skinner.",
    "Klokken er kvart over ti, og mødet starter om femten minutter.",
    "Jeg vil gerne bestille to kopper kaffe og et stykke kage, tak.",
    "Det var en lang og spændende rejse gennem det nordlige Jylland.",
    "Æbler, øl og frække ål på Bornholm.",
    "Det koster 25 kr. og 3,50 øre!",
    "ABC test 123... (parenteser) & 'citater'.",
    "Så  mange   mellemrum.",
]


def main():
    from chatterbox.models.tokenizers.tokenizer import MTLTokenizer

    vocab = glob.glob("/root/.cache/huggingface/hub/models--CoRal-project--roest-v3-chatterbox-500m/snapshots/*/grapheme_mtl_merged_expanded_v1.json")[0]
    tok = MTLTokenizer(vocab)

    out = []
    for s in SENTENCES:
        ids = tok.encode(s, language_id="da")
        pre = tok.preprocess_text(s, language_id="da")
        out.append({"text": s, "preprocessed": pre, "ids": ids})
        print(f"{s!r}\n  -> {ids}")

    with open(os.path.join(REF, "tokenizer_refs.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("wrote tokenizer_refs.json")


if __name__ == "__main__":
    main()
