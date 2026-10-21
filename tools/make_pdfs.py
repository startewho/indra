"""Digital PDFs for S4 parity: text layers, rotation, sideways text, OCR layers.

    python tools/make_pdfs.py

Writes dotnet/tests/parity/pdfs/*.pdf.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as R

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dotnet" / "tests" / "parity" / "pdfs"


def add_text(doc, page, text, size, matrix, render_mode=None):
    obj = R.FPDFPageObj_NewTextObj(doc.raw, b"Helvetica", ctypes.c_float(size))
    buf = (ctypes.c_ushort * (len(text) + 1))(*map(ord, text), 0)
    R.FPDFText_SetText(obj, buf)
    R.FPDFPageObj_Transform(obj, *matrix)
    if render_mode is not None:
        R.FPDFTextObj_SetTextRenderMode(obj, render_mode)
    R.FPDFPage_InsertObject(page.raw, obj)


def digital():
    doc = pdfium.PdfDocument.new()
    p1 = doc.new_page(612, 792)
    add_text(doc, p1, "Quarterly report 2026", 20, (1, 0, 0, 1, 72, 700))
    for i, line in enumerate(
        [
            "Revenue grew by 12 percent in the third quarter.",
            "The pipeline holds 41 active accounts.",
            "Margins stayed flat at 38 percent.",
        ]
    ):
        add_text(doc, p1, line, 12, (1, 0, 0, 1, 72, 660 - i * 18))
    add_text(doc, p1, "Prepared by the finance team.", 10, (1, 0, 0, 1, 72, 120))
    R.FPDFPage_GenerateContent(p1.raw)

    p2 = doc.new_page(612, 792)
    p2.set_rotation(90)
    add_text(doc, p2, "Rotated page", 18, (1, 0, 0, 1, 72, 700))
    add_text(doc, p2, "This page is displayed a quarter turn clockwise.", 12, (1, 0, 0, 1, 72, 660))
    R.FPDFPage_GenerateContent(p2.raw)
    doc.save(str(OUT / "digital.pdf"))


def sideways():
    doc = pdfium.PdfDocument.new()
    page = doc.new_page(612, 792)
    for i, t in enumerate(["Table 2 Baseline characteristics", "Age 54 years", "Women 61 percent"]):
        add_text(doc, page, t, 12, (0, 1, -1, 0, 100 + 30 * i, 100))
    R.FPDFPage_GenerateContent(page.raw)
    doc.save(str(OUT / "sideways.pdf"))


def labels():
    doc = pdfium.PdfDocument.new()
    page = doc.new_page(612, 792)
    add_text(doc, page, "Upright body text", 12, (1, 0, 0, 1, 72, 700))
    add_text(doc, page, "Axis label", 12, (0, 1, -1, 0, 40, 300))
    R.FPDFPage_GenerateContent(page.raw)
    doc.save(str(OUT / "labels.pdf"))


def invisible():
    doc = pdfium.PdfDocument.new()
    page = doc.new_page(612, 792)
    add_text(doc, page, "Hidden OCR layer text", 12, (1, 0, 0, 1, 72, 700), render_mode=3)
    add_text(doc, page, "Second invisible line", 12, (1, 0, 0, 1, 72, 680), render_mode=3)
    R.FPDFPage_GenerateContent(page.raw)
    doc.save(str(OUT / "invisible.pdf"))


def hybrid():
    """An image with text plus a small digital caption: the OCR/hybrid path."""
    from PIL import Image

    src = Image.open(ROOT / "dotnet" / "tests" / "parity" / "samples" / "english.png").convert("L")
    src = src.crop((0, 0, src.width, src.height // 2))
    jpg = OUT / "hybrid.jpg"
    src.save(jpg, quality=90)
    Image.open(jpg).convert("RGB").save(OUT / "hybrid.pdf", resolution=150)

    doc = pdfium.PdfDocument(str(OUT / "hybrid.pdf"))
    page = doc[0]
    add_text(doc, page, "Digital caption line", 14, (1, 0, 0, 1, 40, page.get_height() - 40))
    R.FPDFPage_GenerateContent(page.raw)
    tmp = OUT / "hybrid-edit.pdf"
    doc.save(str(tmp))
    doc.close()
    tmp.replace(OUT / "hybrid.pdf")
    jpg.unlink(missing_ok=True)


def locked():
    """The same digital page, encrypted with the password "pw" (pikepdf)."""
    try:
        import pikepdf
    except ImportError:
        return
    src = OUT / "digital.pdf"
    with pikepdf.open(src) as pdf:
        pdf.save(OUT / "locked.pdf", encryption=pikepdf.Encryption(user="pw", owner="owner"))


def main(argv=None) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    digital()
    sideways()
    labels()
    invisible()
    hybrid()
    locked()
    for p in sorted(OUT.glob("*.pdf")):
        print(f"{p} {p.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
