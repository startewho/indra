"""Stage artifacts from the Python reference implementation (S1-S2 parity).

    python tools/dump_artifacts.py --out dotnet/tests/parity/artifacts examples/samples/scan.png

For each page writes:

    <slug>/page-<n>/det_x.npy      detector input [H, W] float32 (ink map)
    <slug>/page-<n>/det_out.npy    detector output [C, h, w] float32
    <slug>/page-<n>/det_scale.npy  the scale page_to_input applied (float64)
    <slug>/page-<n>/boxes.npy      detect() boxes [N, 5] in page pixels
    <slug>/page-<n>/pictures.npy   picture boxes [P, 4] in page pixels
    <slug>/page-<n>/rec.json       per line: box, dark flag, group, sure, text, conf, ids
    <slug>/page-<n>/meta.json      det_long_side, rec_height, rec_window, rotations

The C# test reads the page image itself, so it has to reproduce the python
inputs, not just replay them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from indra_ocr.models import model_dir  # noqa: E402
from indra_ocr.pipeline import OrtOCR, load_pages  # noqa: E402
from indra_ocr.det_post import page_to_input  # noqa: E402
from indra_ocr.line_mask import crop_lines, line_darkness  # noqa: E402


def _hash(a) -> dict:
    import hashlib

    a = np.ascontiguousarray(a, np.float32)
    return {"shape": [int(v) for v in a.shape], "sha256": hashlib.sha256(a.tobytes()).hexdigest()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--dpi", type=float, default=200.0)
    a = ap.parse_args(argv)

    models = model_dir(a.model_dir)
    ocr = OrtOCR(models, threads=1)
    out_root = a.out
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {"pages": []}

    for path in a.inputs:
        for index, gray in load_pages(path, dpi=a.dpi):
            slug = f"{path.stem}-"
            import hashlib

            slug += hashlib.sha256(path.as_posix().encode("utf-8")).hexdigest()[:8]
            out = out_root / slug / f"page-{index:04d}"
            out.mkdir(parents=True, exist_ok=True)

            # Detector, as `OrtOCR.detect` runs it (including the adaptive pass).
            x, scale = page_to_input(gray, ocr.long_side)
            det_out = ocr.det.run(None, {"x": x[None, None]})[0][0]
            boxes, pictures = ocr.detect_all(gray)

            np.save(out / "det_x.npy", x)
            np.save(out / "det_out.npy", det_out)
            np.save(out / "det_scale.npy", np.float64(scale))
            np.save(out / "boxes.npy", boxes)
            np.save(out / "pictures.npy", pictures)

            # Recognition, with the same boxes, plus why each line went where.
            rec = ocr.recognize(gray, boxes)
            dark = (
                line_darkness(gray, boxes[:, :4]) if ocr.window > 1 else [None] * len(boxes)
            )
            crops = crop_lines(gray, boxes[:, :4], ocr.height, ocr.mask, dark=dark)
            tall = (
                crop_lines(
                    gray, boxes[:, :4], ocr.height, window=ocr.window, dark=dark
                )
                if ocr.window > 1
                else crops
            )
            groups, sure = ocr.box_groups(gray, boxes)
            lines = []
            for i, ((text, conf, toks), b) in enumerate(zip(rec, boxes)):
                lines.append(
                    {
                        "box": [float(v) for v in b[:4]],
                        "group": groups[i],
                        "sure": bool(sure[i]),
                        "dark": dark[i],
                        "crop": _hash(crops[i]),
                        "tall": _hash(tall[i]),
                        "text": text,
                        "conf": float(conf),
                        "ids": [int(t[0]) for t in toks if isinstance(t[0], int)],
                    }
                )
            (out / "rec.json").write_text(
                json.dumps(lines, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            (out / "meta.json").write_text(
                json.dumps(
                    {
                        "input": path.as_posix(),
                        "index": index,
                        "shape": [int(gray.shape[0]), int(gray.shape[1])],
                        "det_long_side": ocr.long_side,
                        "rec_height": ocr.height,
                        "rec_window": ocr.window,
                    },
                    indent=1,
                ),
                encoding="utf-8",
            )
            manifest["pages"].append(
                {"input": path.as_posix(), "index": index, "dir": f"{slug}/page-{index:04d}"}
            )
            print(f"{out}: {len(boxes)} line(s), {len(pictures)} picture(s)")
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
