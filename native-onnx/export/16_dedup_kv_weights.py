"""The prefill and decode graphs embed byte-identical weight files (~2 GB each).
Point both at a single shared external-data file to halve the runtime size.
"""

from __future__ import annotations

import os
import shutil

import onnx

OUT = "/work/onnx_models"
SHARED = "t3_kv_weights.data"


def repoint(model_file: str):
    path = os.path.join(OUT, model_file)
    m = onnx.load(path, load_external_data=False)
    n = 0
    for t in m.graph.initializer:
        for e in t.external_data:
            if e.key == "location":
                e.value = SHARED
                n += 1
    onnx.save(m, path)  # data not loaded -> external refs preserved
    print(f"{model_file}: repointed {n} tensors -> {SHARED}")


def main():
    repoint("t3_prefill.onnx")
    repoint("t3_decode.onnx")
    # keep one copy of the (identical) weights under the shared name
    shutil.move(os.path.join(OUT, "t3_prefill.onnx.data"), os.path.join(OUT, SHARED))
    os.remove(os.path.join(OUT, "t3_decode.onnx.data"))
    print("shared weights:", SHARED, os.path.getsize(os.path.join(OUT, SHARED)), "bytes")
    print("DONE dedup")


if __name__ == "__main__":
    main()
