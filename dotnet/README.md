# Indra OCR — C# port

A 1:1 structural port of the Python package in this repository (`indra_ocr/`).
The plan, stage list and parity rules live in
[`docs/PORT_TO_CSHARP.md`](../docs/PORT_TO_CSHARP.md).

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
  src/IndraOcr.Viewer/     # indra-ocr-viewer equivalent (S7)
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

What the tests cover (S1–S6):

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

Not covered yet: the optional `mask.onnx`, det's picture channel, real vertical
pages, `table_heads=True`, and a real PDF whose text-layer math lines the
classifier flags.

## Native dependencies

| Package | Purpose |
|---|---|
| `Microsoft.ML.OnnxRuntime` 1.31.0 | inference (same runtime as the Python package) |
| `OpenCvSharp4` + `OpenCvSharp4.runtime.win` 4.13.0 | OpenCV (matches `opencv-python-headless` 4.13) |
| `bblanchon.PDFium.Win32` **156.0.8076** | PDFium, pinned to the build `pypdfium2` bundles (pixel-exact renders) |
