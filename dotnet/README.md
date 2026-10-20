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
  src/IndraOcr.Cli/        # indra-ocr equivalent (S6)
  src/IndraOcr.Viewer/     # indra-ocr-viewer equivalent (S7)
  tests/IndraOcr.Tests/    # xUnit: ported tests + parity tests
  tests/parity/            # golden data, detector/recognizer artifacts, LM fixture
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
python tools/dump_golden.py   --out dotnet/tests/parity/golden examples/samples/scan.png
python tools/dump_artifacts.py --out dotnet/tests/parity/artifacts <pages...>
python tools/dump_lm_fixture.py                                # LM + beam fixture
python tools/compare_golden.py dotnet/tests/parity/golden dotnet/tests/parity/actual
```

`tests/parity/artifacts/` is gitignored (about 40 MB); regenerate it with
`dump_artifacts.py` after a fresh clone. The golden page JSON and the small LM
fixture are checked in.

Rules: text / reading order / markdown / `note` / `source` must match exactly;
boxes within 0.5 px, confidences within 1e-4, detector tensors within 1e-6, line
crops byte for byte.

What the current tests cover (S1–S3): detector input/output/boxes, adaptive
second pass, line crops (32 px and 96 px window), CTC output, script routing,
confidence competition, LM log-probabilities and beam search, page orientation
and deskew, reading order, layout blocks and markdown.

```shell
python tools/dump_stage_fixtures.py    # bidi / order / layout pure functions
```

## Native dependencies

| Package | Purpose |
|---|---|
| `Microsoft.ML.OnnxRuntime` 1.31.0 | inference (same runtime as the Python package) |
| `OpenCvSharp4` + `OpenCvSharp4.runtime.win` 4.13.0 | OpenCV (matches `opencv-python-headless` 4.13) |
| PDFium native binaries | added at S4 (`bblanchon.PDFium.Win32`) |
