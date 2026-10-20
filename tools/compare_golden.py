"""Compare golden page JSONs (Python reference) with a C# port's output.

    python tools/compare_golden.py golden/ actual/ [--box-tol 0.5]

Tolerances (see docs/PORT_TO_CSHARP.md section 4):
  * text, markdown, source, note, rotation, vertical, block label: exact
  * box / bbox:          <= --box-tol pixels per coordinate
  * conf:                <= --conf-tol
  * skew:                <= --skew-tol degrees
Any other float is compared with --float-tol; anything else exactly.

Exit code 0 when every page is within tolerance, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

EXACT = {"text", "markdown", "source", "note", "rotation", "vertical", "label", "file"}
BOX = {"box", "bbox"}


def _norm(v):
    """JSON numbers that came back as int/float compare as floats."""
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def compare(exp, act, path: str, opts, diffs: list) -> None:
    exp, act = _norm(exp), _norm(act)
    if isinstance(exp, dict):
        if not isinstance(act, dict):
            diffs.append((path, exp, act, "type"))
            return
        for k in exp:
            if k not in act:
                diffs.append((f"{path}.{k}", exp[k], None, "missing"))
            else:
                compare(exp[k], act[k], f"{path}.{k}", opts, diffs)
        for k in act:
            if k not in exp:
                diffs.append((f"{path}.{k}", None, act[k], "extra"))
        return
    if isinstance(exp, list):
        if not isinstance(act, list) or len(exp) != len(act):
            diffs.append((path, f"list[{len(exp) if isinstance(exp, list) else '?'}]",
                          f"list[{len(act) if isinstance(act, list) else '?'}]", "length"))
            return
        for i, (e, a) in enumerate(zip(exp, act)):
            compare(e, a, f"{path}[{i}]", opts, diffs)
        return

    key = path.rsplit(".", 1)[-1].split("[", 1)[0]
    if isinstance(exp, str) or isinstance(act, str):
        if exp != act:
            diffs.append((path, exp, act, "text"))
        return
    if isinstance(exp, bool) or isinstance(act, bool) or exp is None or act is None:
        if exp != act:
            diffs.append((path, exp, act, "value"))
        return
    e, a = float(exp), float(act)
    if math.isnan(e) and math.isnan(a):
        return
    tol = opts.float_tol
    if key in BOX:
        tol = opts.box_tol
    elif key == "conf":
        tol = opts.conf_tol
    elif key == "skew":
        tol = opts.skew_tol
    if key in EXACT and e != a:
        diffs.append((path, e, a, "exact"))
    elif abs(e - a) > tol:
        diffs.append((path, e, a, f"tol {tol:g} (delta {abs(e - a):.6g})"))


class Options:
    box_tol = 0.5
    conf_tol = 1e-4
    skew_tol = 0.05
    float_tol = 1e-4


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("golden", type=Path)
    ap.add_argument("actual", type=Path)
    ap.add_argument("--box-tol", type=float, default=0.5)
    ap.add_argument("--conf-tol", type=float, default=1e-4)
    ap.add_argument("--skew-tol", type=float, default=0.05)
    ap.add_argument("--float-tol", type=float, default=1e-4)
    ap.add_argument("--max-diffs", type=int, default=40)
    a = ap.parse_args(argv)

    opts = Options()
    opts.box_tol, opts.conf_tol = a.box_tol, a.conf_tol
    opts.skew_tol, opts.float_tol = a.skew_tol, a.float_tol

    golden_pages = sorted(p for p in a.golden.rglob("page-*.json"))
    if not golden_pages:
        print(f"no golden pages under {a.golden}", file=sys.stderr)
        return 1
    failed_pages = 0
    total_diffs = 0
    for gp in golden_pages:
        rel = gp.relative_to(a.golden)
        apath = a.actual / rel
        if not apath.exists():
            print(f"MISSING {rel}")
            failed_pages += 1
            continue
        try:
            exp = json.loads(gp.read_text(encoding="utf-8"))
            act = json.loads(apath.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"BAD JSON {rel}: {e}")
            failed_pages += 1
            continue
        diffs: list = []
        compare(exp, act, "", opts, diffs)
        if diffs:
            failed_pages += 1
            total_diffs += len(diffs)
            print(f"DIFF {rel} ({len(diffs)} difference(s))")
            for path, e, v, kind in diffs[: a.max_diffs]:
                print(f"  {path or '.'}: expected {e!r} got {v!r} [{kind}]")
            if len(diffs) > a.max_diffs:
                print(f"  ... {len(diffs) - a.max_diffs} more")
        else:
            print(f"OK   {rel}")

    extra = sorted(
        p.relative_to(a.actual)
        for p in a.actual.rglob("page-*.json")
        if not (a.golden / p.relative_to(a.actual)).exists()
    )
    for p in extra:
        print(f"EXTRA {p}")
    print(
        f"\n{len(golden_pages) - failed_pages}/{len(golden_pages)} page(s) match"
        f"{f', {total_diffs} difference(s)' if total_diffs else ''}"
    )
    return 1 if failed_pages or extra else 0


if __name__ == "__main__":
    sys.exit(main())
