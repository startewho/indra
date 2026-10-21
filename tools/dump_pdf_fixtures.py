"""PDF parity fixtures: render hashes and text-layer structures.

    python tools/dump_pdf_fixtures.py

Writes dotnet/tests/parity/pdf-fixtures.json.  Full page output is produced
separately by tools/dump_golden.py with --mode auto/text/ocr.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indra_ocr.textlayer import (  # noqa: E402
    extract_runs,
    page_turn,
    read_text_layer,
    vertical_layout,
)

PDFS = ROOT / "dotnet" / "tests" / "parity" / "pdfs"
OUT = ROOT / "dotnet" / "tests" / "parity" / "pdf-fixtures.json"
SCALES = [1.0, 200 / 72.0]


def render(page, scale):
    arr = page.render(scale=scale, grayscale=True).to_numpy()
    arr = np.ascontiguousarray(arr[..., 0] if arr.ndim == 3 else arr)
    return arr


def line_json(run) -> dict:
    return {
        "bbox": [float(v) for v in run.bbox],
        "text": run.text,
        "quarter": int(run.quarter),
        "seq": run.seq,
        "chars": [[c, [float(v) for v in b]] for c, b in run.chars],
    }


def layer_json(tl) -> dict:
    return {
        "size": [float(v) for v in tl.size],
        "trusted": bool(tl.trusted),
        "reason": tl.reason,
        "stats": _plain(tl.stats),
        "lines": [line_json(r) for r in tl.lines],
        "rotated": [line_json(r) for r in tl.rotated],
    }


def _plain(v):
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def main() -> int:
    out = {"pdfs": {}}
    for path in sorted(PDFS.glob("*.pdf")):
        if path.name == "locked.pdf":  # password tests only (this dump opens without one)
            continue
        doc = pdfium.PdfDocument(str(path))
        pages = []
        for i in range(len(doc)):
            page = doc[i]
            entry = {
                "index": i,
                "size": [float(v) for v in page.get_size()],
                "rotation": int(page.get_rotation()),
                "renders": [],
                "layers": {},
            }
            for scale in SCALES + [min(200 / 72.0, (100e6 / (page.get_width() * page.get_height())) ** 0.5)]:
                arr = render(page, scale)
                entry["renders"].append(
                    {
                        "scale": scale,
                        "shape": [int(v) for v in arr.shape],
                        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                    }
                )
            check = render(page, 1.0)
            tl0 = read_text_layer(page, check, 1.0)
            entry["layers"]["0"] = layer_json(tl0)
            k = page_turn(tl0.stats)
            if k:
                tl = read_text_layer(page, np.ascontiguousarray(np.rot90(check, -k)), 1.0, k)
                entry["layers"][str(k)] = layer_json(tl)
            if vertical_layout(tl0.stats):
                tl = read_text_layer(page, np.ascontiguousarray(np.rot90(check, 1)), 1.0, 0, 3)
                tl.stats["vertical"] = True
                entry["layers"]["vertical"] = layer_json(tl)
            entry["page_turn"] = int(k)
            pages.append(entry)
        out["pdfs"][path.name] = pages
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
