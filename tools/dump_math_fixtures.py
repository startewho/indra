"""Math parity fixture: line classification and MathReader readings.

    python tools/dump_math_fixtures.py

Writes dotnet/tests/parity/math-fixtures.json for the images in IMAGES.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indra_ocr.line_mask import crop_lines, line_darkness  # noqa: E402
from indra_ocr.models import model_dir  # noqa: E402
from indra_ocr.pipeline import OrtOCR  # noqa: E402

OUT = ROOT / "dotnet" / "tests" / "parity" / "math-fixtures.json"
IMAGES = [
    "examples/samples/equations.png",
    "dotnet/tests/parity/samples/english.png",
    "examples/samples/scan.png",
]


def main() -> int:
    models = model_dir(math=True)
    ocr = OrtOCR(models, threads=1, math=True)
    assert ocr.math is not None
    pages = []
    for rel in IMAGES:
        path = ROOT / rel
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        boxes = ocr.detect(gray)
        dark = line_darkness(gray, boxes[:, :4])
        crops = crop_lines(gray, boxes[:, :4], ocr.height, ocr.mask, dark=dark)
        flags = ocr.math.flags(crops)
        math = [j for j, f in enumerate(flags) if f]
        readings = ocr.math.read(gray, boxes[math]) if math else []
        lines = []
        for i, b in enumerate(boxes):
            entry = {
                "box": [float(v) for v in b[:4]],
                "flag": bool(flags[i]),
            }
            if flags[i]:
                k = math.index(i)
                entry["text"], entry["conf"] = readings[k]
            lines.append(entry)
        pages.append({"input": rel, "lines": lines})
        print(f"{rel}: {len(boxes)} lines, {len(math)} math")
    OUT.write_text(json.dumps({"pages": pages}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
