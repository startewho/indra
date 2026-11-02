# PaddleOCR models (preinstalled)

These files are an ONNX export of PaddleOCR PP-OCRv4, redistributed under
Apache-2.0:

- ch_PP-OCRv4_det_infer.onnx
- ch_PP-OCRv4_rec_infer.onnx
- ch_ppocr_mobile_v2.0_cls_infer.onnx

Source: https://huggingface.co/SWHL/RapidOCR (PP-OCRv4)
Original models: https://github.com/PaddlePaddle/PaddleOCR (Apache-2.0)
Conversion: Paddle2ONNX.

The desktop app bundles these files under Data/paddleocr; runtime downloads are
disabled.