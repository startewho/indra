# Indra OCR — C# port

A 1:1 structural port of the Python package in this repository (`indra_ocr/`).
The plan, stage list and parity rules live in
[`docs/PORT_TO_CSHARP.md`](../docs/PORT_TO_CSHARP.md).
Usage guide (Chinese): [`USAGE.md`](USAGE.md).

```
dotnet/
  IndraOcr.sln
  src/IndraOcr/            # core library (one file per Python module)
    Numerics/              # Np, Unicode tables, .npy/.npz reader
    Imaging/               # OpenCV wrappers
    Models/                # model folder + Hugging Face download
    Vocab/                 # CTC vocabulary
    Pipeline/              # Meta, DetPost, RecPreproc, LineMask, Skew, OrtOcr, Assembly
    Ordering/              # line features, XY-cut, reading order, clustering
    Layout/                # blocks, tables, markdown
    Rtl/                   # Unicode bidi subset (flip)
    Lm/                    # n-gram LM + CTC beam search
    Math/                  # math line classifier + attention decoder
    Pdf/                   # PDFium P/Invoke, document/page/text page
    TextLayer/             # text-layer runs and trust checks
    Document/              # Converter: auto / ocr / text PDF modes
    Parallel/              # convert_many: pages across cores
  src/IndraOcr.Cli/        # indra-ocr equivalent
  src/IndraOcr.Viewer/     # indra-ocr-viewer equivalent
  src/IndraOcr.Desktop/    # Avalonia + Semi Avalonia desktop app
  tests/IndraOcr.Tests/    # xUnit: ported tests + parity tests
  tests/parity/            # golden data, artifacts, PDFs, fixtures
```

## Build and test

```shell
dotnet build
dotnet test
```

Requires the .NET 10 SDK (the .NET 11 SDK also builds `net10.0` targets).
Parity tests skip themselves when no model folder is found. They look at
`INDRA_OCR_MODEL_DIR` first, then at the Python package's Hugging Face cache
(`~/.cache/huggingface/hub/models--datalab-to--indra-ocr/snapshots/*`).

The core library downloads the models on first use, into
`%LOCALAPPDATA%/indra-ocr/models/datalab-to--indra-ocr/<revision>`. Set
`HF_ENDPOINT` to use a mirror (the default `huggingface.co` is not reachable from
every network), and `HF_TOKEN` for a private repository. Downloads use the plain
`resolve` URLs, not the Xet CAS endpoint, so mirrors work.

## Parity harness

Reference data always comes from the Python implementation, never from the port.

```shell
# from the repository root (needs the Python env from the plan, section 2.3)
python tools/make_samples.py                                   # synthetic pages
python tools/make_pdfs.py                                      # digital / rotated / hybrid / locked PDFs
python tools/dump_golden.py   --out dotnet/tests/parity/golden examples/samples/scan.png
python tools/dump_artifacts.py --out dotnet/tests/parity/artifacts <pages...>
python tools/dump_lm_fixture.py                                # LM + beam fixture
python tools/dump_stage_fixtures.py                            # bidi / order / layout
python tools/dump_math_fixtures.py                             # math flags + readings
python tools/dump_pdf_fixtures.py                              # render hashes + text layers
python tools/compare_golden.py dotnet/tests/parity/golden dotnet/tests/parity/actual
```

PDF page goldens are produced per mode:

```shell
python tools/dump_golden.py --mode auto --out dotnet/tests/parity/golden-pdf/auto dotnet/tests/parity/pdfs/*.pdf
python tools/dump_golden.py --mode text --out dotnet/tests/parity/golden-pdf/text dotnet/tests/parity/pdfs/*.pdf
python tools/dump_golden.py --mode ocr  --out dotnet/tests/parity/golden-pdf/ocr  dotnet/tests/parity/pdfs/*.pdf
```

`tests/parity/artifacts/` is gitignored (about 40 MB); regenerate it with
`dump_artifacts.py` after a fresh clone. The golden page JSON, PDFs, TIFF input
and small fixtures are checked in.

Rules: text / reading order / markdown / `note` / `source` must match exactly;
boxes within 0.5 px, confidences within 1e-4 (page lines are rounded to 4
decimals, so one display step is allowed), detector tensors within 1e-6, line
crops byte for byte, PDF renders by SHA-256.

## CLI

```shell
dotnet run --project src/IndraOcr.Cli -- paper.pdf                 # markdown to stdout
dotnet run --project src/IndraOcr.Cli -- *.pdf --out results/      # one .md per input
dotnet run --project src/IndraOcr.Cli -- scan.png --format json    # lines, boxes, layout
dotnet run --project src/IndraOcr.Cli -- paper.pdf --math --timings
```

Options mirror `indra-ocr`: `--out`, `--format md|text|json`, `--mode
auto|ocr|text`, `--no-image-ocr`, `--model-dir`, `--cores`, `--dpi`,
`--max-pages`, `--password`, `--math`, `--keep-furniture`, `--timings`,
`--version`. Exit codes: 0 ok, 1 an input or the models failed, 2 bad arguments.

`--format md` output is byte-identical to the Python CLI (UTF-8, platform line
endings); JSON matches field by field except `timings`. Pages of one file run in
parallel; files are processed one at a time, in input order.

What the tests cover (S1–S7):

- detector input/output/boxes, adaptive second pass
- line crops (32 px and 96 px window), CTC output, script routing, confidence
  competition, LM log-probabilities and beam search
- page orientation, deskew, reading order, layout blocks and markdown
- PDFium rendering, text-layer runs and trust verdicts, all three Converter
  modes, hybrid pages, passwords, multi-frame TIFF and input errors
- math line classification, attention decoding with KV cache, LaTeX output on
  whole pages, and the PDF text-layer math replacement
- CLI options, formats, output naming, exit codes, UTF-8/newlines, parallel page
  conversion and error isolation
- viewer engine and HTTP routes: upload/page/info, image and PDF pages, options
  and error responses
- desktop session: document open, per-page reads, model presence and JSON output

Not covered yet: the optional `mask.onnx`, det's picture channel, real vertical
pages, `table_heads=True`, a real PDF whose text-layer math lines the classifier
flags, and the viewer's Tesseract comparison.

## Viewer

```shell
dotnet run --project src/IndraOcr.Viewer -- --port 8765 --threads 4
# open http://localhost:8765
```

Drop a PDF or an image: each page shows lines, layout blocks and reading order
over the page, the rendered markdown, its source, per-line confidences, the JSON
the CLI writes, where the text came from, and the time of each stage. Pipeline
options are changed in the page and apply when the document is read again.

The routes and response fields match `viewer.py` (`/`, `/info`, `/upload`,
`/page`, `/logo.png`); the HTML/JS is the Python page, embedded as a resource.
Tesseract comparison is not implemented (`--tesseract` reports it as
unavailable).

## Desktop app

```shell
dotnet run --project src/IndraOcr.Desktop
# or: src/IndraOcr.Desktop/bin/Debug/net10.0/IndraOcr.Desktop.exe paper.pdf
```

Avalonia 12 + Semi Avalonia, laid out like the web viewer: file drop/choose on
the left, pipeline options, model folder + download, display toggles and run
stats; each page shows the image with line boxes, layout blocks, labels and
reading order, plus Preview / Markdown / Lines / JSON tabs. Hovering a line in
the list highlights its box. Pages appear as they finish.

The UI follows the system language (English or Chinese) and can be switched in
the top bar; the choice is remembered. While a document is being read, a
spinning overlay covers the window and blocks input. The model path can be
typed, chosen with a folder picker, or left empty to use
`INDRA_OCR_MODEL_DIR` / the cache; `Download` fetches or verifies the models
(including the math models when Math is on). Tesseract comparison is disabled.

### NativeAOT publish

The desktop app ships as a NativeAOT executable:

```shell
pwsh ./publish-desktop.ps1
# -> dist/desktop/IndraOcr.Desktop.exe
```

The script rebuilds first and then publishes with `--no-build`. Avalonia
compiles XAML from `CoreCompile`; an incremental `dotnet publish` can skip that
step and produce an AOT image without compiled XAML. Plain `dotnet publish` is
therefore a managed build, not an AOT one. NativeAOT needs the MSVC C++ build
tools (`link.exe`) and takes a few minutes; the output is about 165 MB because
ONNX Runtime, OpenCV, PDFium and Skia ship as native libraries. ILC reports two
trim warnings from unused `OpenCvSharp.Mat.Cast<T>`/`Get<T>` helpers; the paths
used by Indra OCR are AOT-safe and were verified with a real OCR run.


## Native dependencies

| Package | Purpose |
|---|---|
| `Microsoft.ML.OnnxRuntime` 1.31.0 | inference (same runtime as the Python package) |
| `OpenCvSharp4` + `OpenCvSharp4.runtime.win` 4.13.0 | OpenCV (matches `opencv-python-headless` 4.13) |
| `bblanchon.PDFium.Win32` **156.0.8076** | PDFium, pinned to the build `pypdfium2` bundles (pixel-exact renders) |
| `Avalonia` 12.1.4 + `Semi.Avalonia` 12.1.0.1 | desktop UI |
