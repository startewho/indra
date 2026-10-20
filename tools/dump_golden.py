"""Golden outputs from the Python reference implementation, for C# parity tests.

    python tools/dump_golden.py --out golden/ examples/samples/scan.png

One JSON file per page (``<slug>/page-<n>.json``) plus ``manifest.json`` with the
input list, library versions and model file hashes.  Timings are left out: they
are not reproducible and are not part of the contract.

Stage-specific tensor dumps (``--artifacts``) are added at S2, when the C# port
starts comparing intermediate results.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import indra_ocr  # noqa: E402
from indra_ocr.models import model_dir  # noqa: E402


def _versions() -> dict:
    out = {"indra_ocr": indra_ocr.__version__, "python": sys.version.split()[0]}
    for name in ("numpy", "opencv-python-headless", "onnxruntime", "pypdfium2", "huggingface_hub"):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            pass
    return out


def _hashes(folder: Path) -> dict:
    out = {}
    for f in sorted(folder.glob("*")):
        if f.suffix in (".onnx", ".json", ".npz") and f.is_file():
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            out[f.name] = {"bytes": f.stat().st_size, "sha256": h.hexdigest()}
    return out


def slug_for(path: Path) -> str:
    """A stable, collision-free folder name for an input path.

    Hashes the path as written (not its absolute form), so the same command from
    the repository root gives the same folder on every machine.
    """
    h = hashlib.sha256(path.as_posix().encode("utf-8")).hexdigest()[:8]
    return f"{path.stem}-{h}"


def page_record(path: Path, index: int, page) -> dict:
    return {
        "file": path.as_posix(),
        "page": index,
        "source": page.source,
        "rotation": page.rotation,
        "skew": page.skew,
        "vertical": page.vertical,
        "note": page.note,
        "lines": [
            {"box": list(ln.box), "text": ln.text, "conf": ln.conf} for ln in page.lines
        ],
        "blocks": page.blocks,
        "markdown": page.markdown,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="golden output folder")
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--dpi", type=float, default=200.0)
    ap.add_argument("--mode", choices=["auto", "ocr", "text"], default="auto")
    ap.add_argument("--math", action="store_true")
    ap.add_argument("--keep-furniture", action="store_true")
    ap.add_argument("--max-pages", type=int, default=0)
    ap.add_argument("--password", default=None)
    a = ap.parse_args(argv)

    models = model_dir(a.model_dir, math=a.math)
    conv = indra_ocr.load(
        path=models, threads=1, dpi=a.dpi, mode=a.mode,
        furniture=a.keep_furniture, math=a.math,
    )

    a.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "versions": _versions(),
        "models": {"dir": str(models), "files": _hashes(Path(models))},
        "options": {
            "dpi": a.dpi, "mode": a.mode, "math": a.math,
            "furniture": a.keep_furniture, "max_pages": a.max_pages,
        },
        "inputs": [str(p) for p in a.inputs],
        "pages": {},
    }
    failures = 0
    for path in a.inputs:
        slug = slug_for(path)
        pages = []
        try:
            for i, page in conv.convert(path, max_pages=a.max_pages, password=a.password):
                rel = f"{slug}/page-{i:04d}.json"
                (a.out / rel).parent.mkdir(parents=True, exist_ok=True)
                (a.out / rel).write_text(
                    json.dumps(page_record(path, i, page), ensure_ascii=False, indent=1),
                    encoding="utf-8",
                )
                pages.append({"index": i, "record": rel})
                if page.source == "error":
                    failures += 1
                    print(f"error page: {path} #{i}: {page.note}", file=sys.stderr)
        except indra_ocr.InputError as e:
            print(f"input failed: {e}", file=sys.stderr)
            failures += 1
            manifest.setdefault("failed_inputs", []).append([str(path), str(e)])
            continue
        manifest["pages"][slug] = {"input": str(path), "pages": pages}

    (a.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    n = sum(len(v["pages"]) for v in manifest["pages"].values())
    print(f"wrote {n} page(s) to {a.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
