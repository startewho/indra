"""Pure-function fixtures for S3: bidi flipping, reading order and layout text.

    python tools/dump_stage_fixtures.py

Writes dotnet/tests/parity/stage/fixtures.json, compared by
dotnet/tests/IndraOcr.Tests/StageParityTests.cs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indra_ocr.layout_post import join_lines, table_html  # noqa: E402
from indra_ocr.order_post import (  # noqa: E402
    _cluster,
    decode_order,
    line_features,
    order_xycut,
    xycut_blocks,
)
from indra_ocr.rtl import flip  # noqa: E402

OUT = ROOT / "dotnet" / "tests" / "parity" / "stage"


def rtl_cases() -> list[dict]:
    texts = [
        "مرحبا بكم في هذا الاختبار",
        "שלום זהו טקסט בעברית",
        "Mixed: English 123 inside RTL text.",
        "قول (اختبار) كلام",
        '\u0627\u0644\u0639\u0631\u0628\u064a\u0629 "quoted" \u0627\u0644\u0646\u0635',
        "plain latin text, no bidi",
        "[abc] عربي [def]",
    ]
    out = []
    for t in texts:
        out.append(
            {
                "text": t,
                "auto": flip(t),
                "rtl": flip(t, base_rtl=True),
                "ltr": flip(t, base_rtl=False),
            }
        )
    return out


def boxes() -> tuple[np.ndarray, list[str]]:
    # Two columns, a header row and a table-like grid at the bottom.
    b, t = [], []

    def add(x0, y0, x1, y1, text):
        b.append([x0, y0, x1, y1])
        t.append(text)

    add(60, 40, 540, 80, "Document title")
    for i in range(4):
        add(60, 120 + i * 40, 460, 150 + i * 40, f"Left column line {i}")
    for i in range(3):
        add(580, 120 + i * 40, 940, 150 + i * 40, f"Right column line {i}")
    for r in range(3):
        for c in range(3):
            add(60 + c * 300, 800 + r * 60, 320 + c * 300, 850 + r * 60, f"cell {r}{c}")
    return np.array(b, np.float32), t


def order_cases() -> dict:
    bx, texts = boxes()
    feats = line_features(bx, (1000.0, 1000.0), texts)
    n = len(bx)
    # A deterministic pairwise order logit matrix with a clear block structure.
    pos = list(range(n))
    a = np.zeros((n, n), np.float32)
    for i in range(n):
        for j in range(n):
            if i != j:
                a[i, j] = 2.0 if pos[i] < pos[j] else -2.0
    # Same-block logits: three groups (title, the two column groups, the table).
    same = np.full((n, n), -1.5, np.float32)
    groups = [[0], list(range(1, 5)), list(range(5, 8)), list(range(8, n))]
    for g in groups:
        for i in g:
            for j in g:
                if i != j:
                    same[i, j] = 1.2
    return {
        "page": [1000.0, 1000.0],
        "boxes": bx.tolist(),
        "texts": texts,
        "features": feats.tolist(),
        "xycut": [int(v) for v in order_xycut(bx)],
        "blocks": [[int(v) for v in g] for g in xycut_blocks(bx)],
        "decode": [int(v) for v in decode_order(a)],
        "a": a.tolist(),
        "same": same.tolist(),
        "cluster": [[int(v) for v in g] for g in _cluster(same)],
    }


def layout_cases() -> dict:
    joins = [
        ["tumori-", "genesis"],
        ["这是一段", "中文文字"],
        ["Ends with a dash -", "continuation"],
        ["one", "", "two"],
        ["no hyphen-", "Next"],
    ]
    cells = []
    for r in range(3):
        for c in range(3):
            cells.append([60 + c * 300, 800 + r * 60, 320 + c * 300, 850 + r * 60])
    texts = ["Name", "Value", "Note", "alpha", "12.5", "first", "beta", "7", "second"]
    return {
        "join": [{"lines": j, "result": join_lines(j)} for j in joins],
        "table": {
            "boxes": cells,
            "texts": texts,
            "html": table_html(np.array(cells, np.float32), texts),
        },
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {"rtl": rtl_cases(), "order": order_cases(), "layout": layout_cases()}
    (OUT / "fixtures.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT / 'fixtures.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
