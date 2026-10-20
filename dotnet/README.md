# Indra OCR — C# port

A 1:1 structural port of the Python package in this repository (`indra_ocr/`).
The plan, stage list and parity rules live in
[`docs/PORT_TO_CSHARP.md`](../docs/PORT_TO_CSHARP.md).

```
dotnet/
  IndraOcr.sln
  src/IndraOcr/            # core library (one file per Python module)
  src/IndraOcr.Cli/        # indra-ocr equivalent
  src/IndraOcr.Viewer/     # indra-ocr-viewer equivalent (S7)
  tests/IndraOcr.Tests/    # xUnit: ported from tests/
  tests/parity/golden/     # reference outputs from the Python implementation
```

## Build and test

```shell
dotnet build
dotnet test
```

Requires the .NET 10 SDK (the .NET 11 SDK also builds `net10.0` targets).

## Parity harness

Golden data is produced by the Python reference implementation, not by the port:

```shell
# once, from the repository root (needs the Python env of section "环境" in the plan)
python tools/dump_golden.py --out dotnet/tests/parity/golden examples/samples/scan.png

# compare the port's output (same folder layout) with the golden data
python tools/compare_golden.py dotnet/tests/parity/golden dotnet/tests/parity/actual
```

Rules: text / reading order / markdown / `note` / `source` must match exactly;
boxes within 0.5 px, confidences within 1e-4.

## Native dependencies

| Package | Purpose |
|---|---|
| `Microsoft.ML.OnnxRuntime` 1.31.0 | inference (same runtime as the Python package) |
| `OpenCvSharp4` + `OpenCvSharp4.runtime.win` 4.13.0 | OpenCV (matches `opencv-python-headless` 4.13) |
| PDFium native binaries | added at S4 (`bblanchon.PDFium.Win32`) |
