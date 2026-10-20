"""Synthetic pages for parity tests: Latin, dark ground, rotated, tiny text, RTL, CJK.

    python tools/make_samples.py

Writes dotnet/tests/parity/samples/*.png (checked in, small) so the detector and
recognizer are exercised on more than the two shipped examples.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "dotnet" / "tests" / "parity" / "samples"
FONTS = Path(r"C:\Windows\Fonts")


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    for candidate in (name, "arial.ttf", "segoeui.ttf"):
        p = FONTS / candidate
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def page(w: int = 1000, h: int = 1400, bg: int = 255, fg: int = 20) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("L", (w, h), bg)
    return img, ImageDraw.Draw(img)


def english() -> Image.Image:
    img, d = page()
    f_title = font("arialbd.ttf", 44)
    f_body = font("arial.ttf", 24)
    f_small = font("arial.ttf", 18)
    d.text((70, 70), "Indra OCR port: parity check", font=f_title, fill=20)
    lines = [
        "This page exercises the detector and the Latin recognizer.",
        "The quick brown fox jumps over the lazy dog, 0123456789.",
        "A paragraph line that is long enough to be read as one line,",
        "and a second one under it with the same shape.",
        "",
        "1. Numbered item with a marker",
        "2. Another item with (parentheses) and symbols: $ % & @ #",
        "",
        "Table-like cells below:",
    ]
    y = 150
    for ln in lines:
        d.text((70, y), ln, font=f_body, fill=20)
        y += 38
    # ruled table
    x0, y0, x1, y1 = 70, y + 10, 830, y + 190
    for i in range(4):
        yy = y0 + i * 60
        d.line((x0, yy, x1, yy), fill=40, width=2)
    for x in (x0, 300, 560, x1):
        d.line((x, y0, x, y1), fill=40, width=2)
    cells = [
        ("Name", "Value", "Note"),
        ("alpha", "12.5", "first"),
        ("beta", "7", "second"),
    ]
    for r, row in enumerate(cells):
        for c, text in enumerate(row):
            d.text((x0 + 12 + c * 260, y0 + 16 + r * 60), text, font=f_small, fill=20)
    d.text((70, 1300), "Footnote: small print at the bottom of the page.", font=f_small, fill=30)
    return img


def dark() -> Image.Image:
    img, d = page(bg=28, fg=235)
    f_head = font("arialbd.ttf", 40)
    f_body = font("arial.ttf", 26)
    d.text((70, 90), "LIGHT TEXT ON A DARK GROUND", font=f_head, fill=240)
    y = 180
    for i in range(6):
        d.text((70, y), f"Line {i + 1}: white-on-black paragraph text for inversion.", font=f_body, fill=235)
        y += 44
    return img


def rotated() -> Image.Image:
    img = english()
    return img.rotate(90, expand=True, fillcolor=255)


def tiny() -> Image.Image:
    img, d = page(1200, 1600)
    f = font("arial.ttf", 11)
    y = 60
    for i in range(60):
        d.text((40, y), f"Tiny line {i:02d}: the adaptive pass must raise the detection resolution.", font=f, fill=25)
        y += 24
    return img


def rtl() -> Image.Image:
    img, d = page()
    f = font("arial.ttf", 30)
    lines = [
        "\u0645\u0631\u062d\u0628\u0627 \u0628\u0643\u0645 \u0641\u064a \u0647\u0630\u0627 \u0627\u0644\u0627\u062e\u062a\u0628\u0627\u0631",  # Arabic
        "\u0627\u0644\u0633\u0637\u0631 \u0627\u0644\u062b\u0627\u0646\u064a \u0645\u0646 \u0627\u0644\u0646\u0635 \u0627\u0644\u0639\u0631\u0628\u064a",
        "\u05e9\u05dc\u05d5\u05dd \u05d6\u05d4\u05d5 \u05d8\u05e7\u05e1\u05d8 \u05d1\u05e2\u05d1\u05e8\u05d9\u05ea",  # Hebrew
        "\u05e9\u05d5\u05e8\u05d4 \u05e9\u05e0\u05d9\u05d4 \u05e9\u05dc \u05d8\u05e7\u05e1\u05d8 \u05d1\u05e2\u05d1\u05e8\u05d9\u05ea",
        "Mixed: English 123 inside RTL text.",
    ]
    y = 120
    for ln in lines:
        d.text((700, y), ln, font=f, fill=20, anchor="ra")
        y += 60
    return img


def cjk() -> Image.Image:
    img, d = page()
    f = font("msyh.ttc", 34)
    lines = [
        "\u4e2d\u6587\u8bc6\u522b\u6d4b\u8bd5\u9875\u9762",  # 中文识别测试页面
        "\u8fd9\u662f\u4e00\u6bb5\u7528\u4e8e\u68c0\u67e5\u6392\u7248\u548c\u8bc6\u522b\u7684\u6587\u5b57\u3002",
        "\u7b2c\u4e8c\u884c\u6587\u5b57\uff0c\u5305\u542b\u6807\u70b9\u7b26\u53f7\uff1a\u3001\u3002\uff0c\uff1f\uff01",
    ]
    y = 100
    for ln in lines:
        d.text((70, y), ln, font=f, fill=20)
        y += 60
    return img


def vertical() -> Image.Image:
    """A CJK page turned a quarter turn clockwise: the orient model may call it
    vertical (class 4) or a quarter turn (class 3); either way the pipeline should
    read it back."""
    return cjk().rotate(-90, expand=True, fillcolor=255)


def skewed() -> Image.Image:
    """Seven degrees off level, to exercise the deskew search."""
    return english().rotate(7, expand=True, fillcolor=255)


def main(argv=None) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pages = {
        "english.png": english(),
        "dark.png": dark(),
        "rotated.png": rotated(),
        "tiny.png": tiny(),
        "rtl.png": rtl(),
        "cjk.png": cjk(),
        "vertical.png": vertical(),
        "skewed.png": skewed(),
    }
    for name, img in pages.items():
        p = OUT / name
        img.save(p)
        print(f"{p} {img.size[0]}x{img.size[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
