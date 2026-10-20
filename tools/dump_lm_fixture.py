"""LM parity fixture: backoff log-probabilities and one beam-search case.

    python tools/dump_lm_fixture.py

Writes dotnet/tests/parity/lm/{fixture.json,blank.npy,topv.npy,topi.npy}.
The arrays are one line's per-step outputs from rec.onnx (examples/samples/
equations.png), so the C# beam search can be compared without running the model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from indra_ocr.line_mask import crop_lines, line_darkness  # noqa: E402
from indra_ocr.lm import NgramLM, _pack, beam_search  # noqa: E402
from indra_ocr.models import model_dir  # noqa: E402
from indra_ocr.pipeline import OrtOCR  # noqa: E402
from indra_ocr.rec_preproc import pad_batch, width_batches  # noqa: E402

OUT = ROOT / "dotnet" / "tests" / "parity" / "lm"
PAGE = ROOT / "examples" / "samples" / "equations.png"
LINE = 2


def line_arrays():
    """blank/topv/topi of one line, taken from the batch it is recognized in."""
    ocr = OrtOCR(model_dir(), threads=1)
    gray = cv2.imread(str(PAGE), cv2.IMREAD_GRAYSCALE)
    boxes = ocr.detect(gray)
    dark = line_darkness(gray, boxes[:, :4])
    crops = crop_lines(gray, boxes[:, :4], ocr.height, ocr.mask, dark=dark)
    tall = crop_lines(gray, boxes[:, :4], ocr.height, window=ocr.window, dark=dark)
    groups, sure = ocr.box_groups(gray, boxes)
    base = [j for j in range(len(boxes)) if groups[j] == "base" or not sure[j]]
    for part in width_batches([crops[j].shape[1] for j in base]):
        idx = [base[i] for i in part]
        if LINE not in idx:
            continue
        arr, widths = pad_batch([tall[j] for j in idx])
        res = ocr.rec.run(
            ["ids", "conf", "lengths", "blank", "topv", "topi"],
            {"x": arr, "widths": widths},
        )
        k = idx.index(LINE)
        n = int(res[2][k])
        return res[3][k, :n], res[4][k, :n], res[5][k, :n]
    raise RuntimeError("line not found in any batch")


def main() -> int:
    blank, topv, topi = line_arrays()
    lm = NgramLM(model_dir() / "lm.npz")

    probes = [
        ((), 5), ((), 55), ((55,), 64), ((55, 64), 79), ((55, 64, 79), 30),
        ((55, 64, 79, 30), 36), ((64, 79, 30, 36, 64), 19), ((92,), 79),
        ((5,), 55), ((55, 92), 79),
    ]
    ids, steps = beam_search(blank, topv, topi, lm, 0.5, 1.5, 8, cand_floor=3e-3)

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "blank.npy", blank)
    np.save(OUT / "topv.npy", topv)
    np.save(OUT / "topi.npy", topi)
    fixture = {
        "order": lm.order,
        "vocab_size": lm.vocab_size,
        "bits": lm.bits,
        "max_logp": lm.max_logp,
        "alpha": 0.5,
        "beta": 1.5,
        "beam_size": 8,
        "cand_floor": 3e-3,
        "probes": [
            {
                "history": list(h),
                "token": t,
                # The beam search looks values up by packed key, so that is the
                # reference (not the scalar lm.logp, which differs for a short
                # history padded with BOS).
                "logp": lm.logp_keys([(_pack(h, lm.bits) << lm.bits) | (t + 1)])[0],
            }
            for h, t in probes
        ],
        "beam": {"ids": [int(v) for v in ids], "steps": [int(v) for v in steps]},
    }
    (OUT / "fixture.json").write_text(json.dumps(fixture, indent=1), encoding="utf-8")
    print(f"wrote {OUT} (beam {len(ids)} token(s): {ids})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
