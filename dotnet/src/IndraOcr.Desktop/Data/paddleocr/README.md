# PaddleOCR models (preinstalled)

These files are PP-OCRv6 ONNX plus the lightweight PP-DocLayout-S layout model,
all Apache-2.0:

- pp_ocrv6_det.onnx
- pp_ocrv6_rec.onnx
- ppocrv6_dict.txt
- pp_doclayout_s.onnx


Source:
- https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx
- https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx
- https://huggingface.co/stefanj0/PP-DocLayout-S-ONNX


Original project: https://github.com/PaddlePaddle/PaddleOCR (Apache-2.0).

The desktop app bundles these files under Data/paddleocr; runtime downloads are
disabled. The textline orientation model corrects 180-degree lines. The layout model supplies paragraph/title/table/figure/header blocks;
its class labels are mapped onto Indra's existing block colors.