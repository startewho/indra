# Indra OCR：Python → C# 迁移方案

> 结论：**可以移植，而且条件很好。** 全流程没有 PyTorch，模型是纯 ONNX，图像处理是 OpenCV，
> PDF 是 PDFium。这三样在 .NET 侧都有成熟的官方/准官方绑定，且底层是同一套 C++ 原生库，
> 因此逐位对齐（bit-level parity）是现实目标。真正的工作量不在算法，而在
> **numpy 语义补齐、Unicode 数据表、PDFium 文字层 P/Invoke、以及一致性验证体系**。

- 参考实现：本仓库 `indra_ocr` 1.1.2（约 4,240 行库代码 + 1,250 行测试）
- 目标：`.NET 10` (LTS)，C# 14
- 本文档只描述方案，不改动现有 Python 代码

### 已确认的决策

| 项 | 决定 |
|---|---|
| 范围 | **全量对齐**（S0–S7：含 PDF 文字层、LM、数学、多语言分流、并行、viewer） |
| 目标框架 | `net10.0` |
| 交付物 | 核心类库 + 命令行工具 + 本地 Web 查看器 |
| 结构策略 | **1:1 结构移植**，对齐完成后再做 C# 惯用化重构 |

---

## 1. 现状评估

### 1.1 依赖盘点

| Python 依赖 | 作用 | C# 对应 | 风险 |
|---|---|---|---|
| `numpy` | 所有数值/数组运算 | 自写 `Np` 工具集 + `OpenCvSharp.Mat`；矩阵乘法用 `MathNet.Numerics` | 中（语义细节多） |
| `opencv-python-headless` | resize/warpAffine/连通域/旋转 | `OpenCvSharp4` + `OpenCvSharp4.runtime.win`（同一 OpenCV C++ 库） | 低 |
| `onnxruntime` | 推理 | `Microsoft.ML.OnnxRuntime` 1.31 | 低 |
| `pypdfium2` | PDF 渲染 + 文字层 | 直接 P/Invoke `pdfium.dll`（`bblanchon.PDFium.Win32` 等原生包） | 中高（互操作细节） |
| `huggingface_hub` | 模型下载/缓存 | `HttpClient` + HF `resolve` 直链 + 本地缓存 | 低 |
| 标准库 `multiprocessing` | 多核并行 | `Parallel.ForEach` + 每线程一个 PDF 文档句柄 | 中（线程安全约束不同） |
| 标准库 `http.server` (viewer) | 本地 Web 查看器 | ASP.NET Core Minimal API（HTML/JS 原样复用） | 低（可后置） |

`pytesseract` 只用于 viewer 的对照功能，属可选依赖，可延后或不做。

### 1.2 模块清单与难度

难度：A 很简单 / B 简单 / C 中等 / D 难（需要专门设计与验证）

| 文件 | 行数 | 内容 | 难度 | 备注 |
|---|---:|---|:--:|---|
| `pipeline.py` | 1031 | 检测、识别、倾斜矫正、方向判定、组装、tidy、拆分 | D | 核心；含可变实例状态（`_pic`/`_page_rtl`）需改为局部值 |
| `viewer.py` | 648 | HTTP 服务 + 内嵌 HTML/JS | C | 可选；前端代码可原样搬 |
| `document.py` | 426 | PDF 文字层/OCR 混排决策 | D | 依赖 `textlayer` 与渲染 |
| `layout_post.py` | 381 | 块切分、表格 HTML、Markdown | C | 纯逻辑，易对齐 |
| `textlayer.py` | 363 | pdfium 字符级坐标、可信度判定 | D | P/Invoke + UTF-16 代理对 + 仿射矩阵 |
| `order_post.py` | 352 | 线特征、阅读序解码、聚类 | C | numpy 语义密集（argsort/matmul） |
| `vocab.py` | 292 | 词表、NFC、LaTeX token | B | 需要 `unicodedata` 表 |
| `lm.py` | 254 | npz n-gram LM + CTC beam search | D | `.npz` 读取 + uint64 打包 + 性能敏感 |
| `parallel.py` | 212 | 多进程调度 | C | 改为线程模型（见 §5.6） |
| `cli.py` | 189 | 命令行 | B | 参数语义照抄 |
| `rtl.py` | 162 | 简化的 UAX#9 双向文本 | C | 需要 bidi class 表 |
| `det_post.py` | 155 | 检测预处理/解码 | B | 连通域 + 加权 bincount |
| `line_mask.py` | 153 | CenterMask 行掩码 | B | 批量推理 |
| `math.py` | 145 | 数学公式分类 + 自回归解码 | C | KV-cache 输入名裁剪 |
| `rec_preproc.py` | 145 | 行裁剪规则 | B | resize 参数需逐一对齐 |
| `models.py` | 75 | 模型目录/下载 | B | 复用 HF 直链 |
| `__init__.py` | 49 | 门面 API | A | |

### 1.3 模型接口契约（移植时必须逐字对齐）

| 模型 | 输入 | 输出 |
|---|---|---|
| `det.onnx` | `x: float32[1,1,H,W]`（H/W 为 32 的倍数） | `[1,C,h,w]`；C≥3：core logit、上边界距离、下边界距离；C>3 时第 4 通道为图片概率 logit |
| `rec.onnx` / `rec_<group>.onnx` | `x: float32[B,1,32*window,W]`、`widths: int64[B]` | `ids`、`conf`、`lengths`；可选 `blank`、`topv`、`topi`（有 LM 时） |
| `script.onnx` | `x`、`widths` | `[B]`（旧版二分类）或 `[B,4]`（组分类） |
| `mask.onnx` | `x: float32[B,1,96,W]` | `[B,1,96,W]` 概率图 |
| `orient.onnx` | `x: float32[1,1,768,768]` | `[1,classes]` 概率 |
| `order.onnx` | `feats: float32[1,N,16]`、`mask: bool[1,N]` | 至少 1 个：两两顺序 logits；≥3 个：+块分类 logits +同块 logits；>3：+表格头 |
| `math_*.onnx` | encoder/decoder 分离，带 KV cache | 逐步 logits |

模型文件：`det.onnx`、`rec.onnx`（或 `rec.int8.onnx`）、`order.onnx`、`meta.json` 必需；
`mask.onnx`、`orient.onnx`、`script.onnx`、`rec_cjk/rtl/other.onnx`、`lm.npz`/`lm_<group>.npz`、
`math_*.onnx` 可选。`meta.json` 内含 `vocab`、`vocab_<group>`、`det_long_side`、`rec_height`、
`rec_window`、`script_groups`、`lm_params`。

---

## 2. 目标架构

### 2.1 方案对比

| 方案 | 做法 | 工期 | 优点 | 缺点 |
|---|---|---|---|---|
| **A. 原生全量移植（推荐）** | 逐模块 C# 重写，模型不变 | 4–6 周 | 真原生、无 Python 依赖、可 AOT、可进程内调用 | 工作量大，需严格一致性验证 |
| B. Python.NET 内嵌 | `pythonnet` 加载现有包 | 3–5 天 | 最快可用 | 仍需 Python 运行时，部署更重，启动慢，调试差 |
| C. 子进程调用 CLI | `Process.Start("indra-ocr")` | 1 天 | 极快 | 无进程内 API，序列化开销，错误处理弱 |

若目标只是「Windows 上交付一个 exe」，可先用 C 兜底；若目标是长期维护的 .NET 产品/服务，选 A。
后文按 A 展开。

### 2.2 项目结构

C# 代码放在本仓库的 `dotnet/` 下（便于与 Python 参考实现同仓库对照；将来可整体拆成独立仓库）。
Python 侧的 `indra_ocr/` 保持不动。

```
dotnet/
  IndraOcr.sln
  Directory.Build.props
  README.md
  src/
    IndraOcr/                     # 核心库（对应 indra_ocr/，一个模块一个文件）
      IndraOcr.csproj             # net10.0
      IndraOcrInfo.cs             # 版本与模型仓库信息
      Numerics/Np.cs              # numpy 语义补齐
      Numerics/UnicodeData.cs     # 生成的 bidi / EastAsianWidth 表
      Numerics/NpzReader.cs       # .npz/.npy 读取
      Imaging/ImageOps.cs         # resize/rot90/warpAffine/连通域封装
      Pdf/PdfiumNative.cs         # DllImport 声明
      Pdf/PdfDocument.cs          # 渲染、页面、文字层
      Models/ModelFiles.cs        # 定位/下载/校验
      Vocab/Vocab.cs
      Pipeline/OrtOcr.cs
      Pipeline/Page.cs, Line.cs
      TextLayer/TextLayer.cs
      Ordering/OrderPost.cs
      Layout/LayoutPost.cs
      Lm/NgramLm.cs, BeamSearch.cs
      Math/MathReader.cs
      Document/Converter.cs
      Parallel/PageRunner.cs
      Rtl/Bidi.cs
    IndraOcr.Cli/                 # indra-ocr 等价命令行
    IndraOcr.Viewer/              # Web 查看器：ASP.NET Core + 原 HTML/JS
  tests/
    IndraOcr.Tests/               # xUnit：现有 14 个测试文件的 1:1 移植
    parity/golden/                # Python 参考实现产出的黄金数据（入库）
    parity/actual/                # C# 输出（不入库，.gitignore）
```

Python 侧的辅助脚本（不入包）：

```
tools/
  gen_unicode_tables.py           # 从 Python unicodedata 生成 C# 表（S1）
  dump_golden.py                  # 参考实现 → 黄金 JSON（S0，已完成）
  compare_golden.py               # 黄金 vs C# 输出比对（S0，已完成）
```

**建议保持「一文件对一文件」的映射**先完成移植，再在第二阶段做 C# 惯用化重构。
理由：对齐阶段最大的敌人是「找不到对应关系」，结构一致能把排查成本降一个数量级。

---

### 2.3 环境搭建（S0 已完成，2026-10-20）

参考端已在本机跑通，命令与坑记录如下（C# 侧设计必须考虑同样的网络/编码问题）。

```shell
python -m venv .venv
# PyPI 直连超时；使用国内镜像安装（仅开发环境）
.venv\Scripts\python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple `
    numpy "opencv-python-headless==4.13.0.92" onnxruntime pypdfium2 `
    huggingface_hub pytest pillow pikepdf
.venv\Scripts\python -m pip install --no-deps -e .
```

实测版本（黄金数据 `manifest.json` 中亦有记录）：

| 组件 | Python 侧 | C# 侧 |
|---|---|---|
| numpy | 2.5.4 | `Np` 工具集 |
| OpenCV | `opencv-python-headless` **4.13.0.92** | `OpenCvSharp4` **4.13.0**（同一 C++ 版本，特意对齐） |
| ONNX Runtime | 1.31.0 | `Microsoft.ML.OnnxRuntime` 1.31.0 |
| pypdfium2 | 5.14.0 | PDFium 原生库（S4 接入） |
| Hugging Face Hub | 2.2.0 | 自写下载器 |

模型下载：

```shell
# huggingface.co 在本机不可直连；hf-mirror 可用
$env:HF_ENDPOINT = "https://hf-mirror.com"
# 新版 hub 默认走 Xet/CAS，镜像不代理 → 必须关闭，回退到普通 resolve 下载
$env:HF_HUB_DISABLE_XET = "1"
.venv\Scripts\python -c "from indra_ocr.models import model_dir; print(model_dir())"
```

- 缓存目录：`~/.cache/huggingface/hub/models--datalab-to--indra-ocr/snapshots/<rev>`
- 已下载（v1.1.0，约 76 MB）：`det.onnx`、`rec.onnx`、`order.onnx`、`meta.json`、`orient.onnx`、
  `script.onnx`、`rec_cjk/rtl/world.onnx`、`lm*.npz`
- 该版本**没有** `mask.onnx`（可选模型），数学模型需 `math=True` 才下载

**给 C# 实现的直接影响**（已写入 S1/S4 任务）：

1. 下载器要支持 `HF_ENDPOINT` 覆盖，并直连 `https://<endpoint>/{repo}/resolve/{rev}/{file}`，
   **不要**依赖 Xet；失败信息保持与原实现一致的 `ModelError` 文案。
2. Windows 控制台默认 GBK，Python 需 `PYTHONUTF8=1` 才不会在输出韩文时崩溃；
   C# CLI 已在入口设置 `Console.OutputEncoding = UTF8`。

验证结果：`pytest` **74 passed**；`indra-ocr examples/samples/scan.png` 输出 36 行、9 个 block。
黄金数据已生成 2 页（`dotnet/tests/parity/golden/`），比对器自检通过。

## 3. 分阶段计划

估算按 1 名熟悉 .NET + CV 的工程师计；含自测，不含需求评审。

| 阶段 | 内容 | 交付物 | 完成标准（DoD） | 人日 |
|---|---|---|---|---:|
| **S0 准备** ✅ | 建 Python 参考环境（装依赖、下载 76MB 模型）；建 C# solution；写黄金数据生成器与比对器 | `tools/dump_golden.py` + `compare_golden.py`；`dotnet/` 骨架；2 页黄金数据 | 同页两次运行结果一致；比对器自检通过；`dotnet build` 通过 | 1–2 |
| **S1 基础设施** ✅ | `Np`（argsort/searchsorted/bincount/percentile/logaddexp/cumsum/median/triu/matmul）、Unicode 表生成器、`NpzReader`、ONNX 会话封装、图像 IO（PNG/JPEG/TIFF 多帧） | `Numerics/`、`Imaging/`、`Models/`；Unicode 表；`tools/gen_unicode_tables.py` | 每个 helper 有 numpy 基准用例，浮点误差 ≤1e-6 | 2–3 |
| **S2 检测+识别** ✅ | `vocab`、`rec_preproc`、`det_post`、`line_mask`、`pipeline` 的检测/裁剪/识别；**LM beam search 提前到本阶段**（默认模型必用） | `OrtOcr.Detect/Recognize`、`NgramLm`、`BeamSearch` | 10 页样本（暗底/90°/270°/7°/微小字/中英阿希）：box ≤0.5px，文本/ids/路由 100% 一致，裁剪逐字节相同 | 3–4 |
| **S3 版面+输出** ✅ | `rtl`、`order_post`、`layout_post`、`tidy_lines`、`split_line`、`assemble`、Markdown/JSON、倾斜/转向 | `Page` 全字段 + Markdown；`OrtOcr.Read` | 10 页整页对齐：Markdown/lines/blocks/rotation/skew 100% 一致 | 3–4 |
| **S4 PDF** ✅ | PDFium P/Invoke、渲染（灰度/DPI/Rotate）、`textlayer`、`document` 的 auto/ocr/text 三模式、混合页与密码 | `Converter.Convert` | 渲染逐像素一致；6 页 PDF × auto/text/ocr 与参考端全字段一致 | 4–6 |
| **S5 数学** ✅ | `mathcls` + 编码器 + KV-cache 解码器、`Detokenize`、PDF 文字层数学行重读 | `MathReader` + pipeline/document 接入 | 3 页判定/读数一致；`equations.png` 整页 LaTeX 输出一致 | 2–4 |
| **S6 CLI+并行** ✅ | 参数、退出码、`--out` 命名规则、多核分页、错误隔离、UTF-8/换行 | `indra-ocr` 等价 CLI；`PageRunner` | md 与 Python **逐字节一致**；JSON 语义一致；cores=1/4 输出相同 | 2–3 |
| **S7 Viewer** ✅ | ASP.NET Core + 原 HTML/JS（`PAGE` 原样复用）、`/upload`/`/page`/`/info` | `IndraOcr.Viewer` | 页面与 JSON 字段一致；真实 HTTP 冒烟通过 | 2–3 |
| **S8 收尾** | 跨平台原生库、NuGet 打包、文档、性能基线 | 可发布的包 | 见 §4 验收 | 2–4 |

- 全量对齐（S0–S7）≈ **5 周**；加跨平台/性能打磨 ≈ 6 周。
- 可选提前验证点：S0–S3（≈2 周）即可跑通「图片/扫描 → Markdown」，可在此时先做一次内部演示，
  但交付范围仍以全量对齐为准。

### 3.1 第一周可直接开工的任务

1. `python -m venv .venv && pip install -e .[test]`，跑 `pytest` 与 `indra-ocr examples/samples/scan.png`，确认参考端可复现。
2. `dotnet new sln` + 三个项目 + xUnit；`Directory.Build.props` 统一 `net10.0`、`Nullable=enable`、`InvariantGlobalization=false`。
3. 写 `tools/dump-golden.py`：输入路径 → 每页输出 `{file,page,source,note,rotation,skew,vertical,lines[],blocks[],markdown,timings}`；
   加 `--artifacts` 再存中间张量（`det_in`, `det_boxes`, `rec_ids`, `order_logits`, `same_logits`），存为 `.npy`。
4. 写 `tools/compare.py`：按 §4 的容差报告差异，输出最小反例页。
5. 移植 `vocab.py` + `det_post.py`（这两块能立刻用黄金数据验证，且不依赖 PDF）。

### 3.2 进展记录（S1–S7 已完成，2026-10-21）

**S1 基础设施**（`dotnet/src/IndraOcr/`）：

| 文件 | 对应 Python | 验证 |
|---|---|---|
| `Numerics/Np.cs` | numpy 子集 | 8 个单元用例（median/percentile/searchsorted/logaddexp/bincount/cumsum/argsort/clip） |
| `Numerics/Unicode.cs` + `UnicodeTables.g.cs` | `unicodedata` 的 bidi/EAW | 生成器 + 19 个已知码位用例 |
| `Numerics/Npy.cs` | `np.load/.npz` | 读 `det_*.npy` 与 `lm.npz`，类型/形状正确 |
| `Imaging/ImageOps.cs` | 读图、resize、rot90、连通域 | rot90/ink 单元用例；resize 与连通域由检测对齐间接验证 |
| `Models/ModelFiles.cs` | `models.py` | 本地目录/`INDRA_OCR_MODEL_DIR`/HF 运行时下载（支持 `HF_ENDPOINT`，直连 resolve，不走 Xet） |
| `Pipeline/OrtUtil.cs` | `SessionOptions` | 同 Python：intra/int=1、ORT_ENABLE_ALL、无 CPU arena |

**S2 检测+识别**：

| 文件 | 对应 Python | 状态 |
|---|---|---|
| `Vocab/Vocab.cs` | `vocab.py`（推理部分） | 词表、NFC、LaTeX 空格规则 |
| `Pipeline/DetPost.cs` | `det_post.py` | 输入 1e-6、输出 1e-6、框 ≤0.5px（实测差异 <0.001px） |
| `Pipeline/RecPreproc.cs` | `rec_preproc.py` | 裁剪图与 Python **逐字节相同**（SHA-256） |
| `Pipeline/LineMask.cs` | `line_mask.py` | 未测（默认模型无 `mask.onnx`） |
| `Pipeline/Meta.cs` | `meta.json` | 词表/组/几何/LM 参数 |
| `Pipeline/OrtOcr.cs` | `pipeline.py` 检测/识别 | 含自适应二次检测、script 分流、高置信度竞争 |
| `Lm/NgramLm.cs` + `Lm/BeamSearch.cs` | `lm.py` | logp 与 beam 的 ids/steps 完全一致（提前完成，属 S5） |

**语料与结果**：10 页（扫描韩文信、公式页、英文版式、暗底反白、90° 旋转、270° 旋转、微小字、阿拉伯/希伯来、中文、7° 倾斜）。
`dotnet test` **36 passed**：

- 检测：输入/输出张量与 Python ≤1e-6，框 ≤0.5px（实测 ≤0.001px）
- 识别：文本、置信度、ids、分组、dark 标记 100% 一致；裁剪图 SHA-256 相同
- LM：10 个 logp 探针与 1 条 beam 用例完全一致
- 整页（S3）：`rotation`/`skew`/`vertical`/`note`、逐行文本/框/置信度、blocks、Markdown **全部一致**

**S3 版面+输出**：

| 文件 | 对应 Python | 状态 |
|---|---|---|
| `Rtl/Bidi.cs` | `rtl.py` | 7 组双向文本用例（自动/RTL/LTR、括号镜像） |
| `Ordering/OrderPost.cs` | `order_post.py` | 线特征、XY-cut、`decode_order`、平均链接聚类、块级排序 |
| `Layout/LayoutPost.cs` | `layout_post.py` | `join_lines`、表头/几何两种表格、`page_blocks`、`page_markdown` |
| `Pipeline/Assembly.cs` | `pipeline.tidy_lines/split_line/gutter_cuts` | 端到端覆盖 |
| `Pipeline/Skew.cs` | 倾斜估计、转向、`orient_input`、`unrotate_box` | 7° 页与 90°/270° 页覆盖 |
| `Pipeline/OrtOcr.Assembly.cs` | `reading_order`/`assemble`/`__call__` | 整页 10/10 对齐 |
| `Pipeline/Page.cs` | `Page`/`Line` | 全字段 |

纯函数另有 `tools/dump_stage_fixtures.py` 产出的 JSON 夹具（bidi、17 行版面的特征/XY-cut/聚类/排序、`join_lines`、`table_html`），3 个测试逐项对齐。

**S4 PDF**：

| 文件 | 对应 Python | 状态 |
|---|---|---|
| `Pdf/PdfiumNative.cs` | `pypdfium2.raw` | 直接用 P/Invoke；原生库固定 **156.0.8076**（与 pypdfium2 内置版本一致） |
| `Pdf/PdfDocument.cs` | `PdfDocument`/`PdfPage` | 打开/密码、尺寸/旋转、灰度渲染、文字页、对象遍历（含 Form 递归） |
| `TextLayer/TextLayer.cs` | `textlayer.py` | 字符坐标/角度、仿射、quarter 判断、run 切分、可信度判定 |
| `Document/Converter.cs` | `document.py` | `auto`/`ocr`/`text`、`image_ocr`、像素上限、密码、混合页、侧向 run |
| `tools/make_pdfs.py` | — | 数字页、`/Rotate 90`、侧向表格、不可见文字层、图像+文字混合页、加密页 |
| `tools/dump_pdf_fixtures.py` | — | 渲染 SHA-256（3 种 scale）、文字层结构、`page_turn` |

结果：

- PDFium 渲染与 pypdfium2 **逐像素相同**（SHA-256），三种 scale、含 `/Rotate 90`
- 文字层：run/字符框/文本/quarter/seq、`trusted`/`reason`、全部 `stats` 与 Python 一致
- 6 页 PDF（含混合页与加密页）× 3 种 mode 的整页输出与参考端全字段一致
- 混合页：`source=hybrid`，图像内的行由 OCR 补齐；`image_ocr=false` 时只保留文字层（1 行）
- 加密页：无密码报 `password-protected PDF`，`password="pw"` 可读
- 多帧 TIFF、`max_pages`、不可读文件、缺失文件均有测试

**S5 数学**：

| 文件 | 对应 Python | 状态 |
|---|---|---|
| `Math/MathReader.cs` | `math.py` | 分类器、编码器、3 层 8 头 KV-cache 解码器、提前停止、`Detokenize` |
| `Pipeline/OrtOcr.cs` | `recognize` 的 math 分支 | `math=True` 时按行分类并替换为 LaTeX |
| `Document/Converter.cs` | `_read_math` | 文字层数学行重读，保留内容序 |
| `tools/dump_math_fixtures.py` | — | 3 页的逐行判定与读数 |

结果：

- `equations.png` 38 行中 23 行判为数学；`english.png` / `scan.png` 各 1 行（低置信度，不替换）
- 分类结果与每行 `(text, conf)` 与 Python **完全一致**
- `equations.png` 整页 `--math` 的 Markdown（含 `$...$`）与参考端一致
- 文字层替换逻辑有函数级测试（构造 3 元组 token，验证 `$` + 置信度门槛、保留首个内容索引）
- 数学模型随 `ModelFiles.Resolve(math: true)` 下载并校验

**S6 CLI+并行**：

| 文件 | 对应 Python | 状态 |
|---|---|---|
| `Parallel/PageRunner.cs` | `parallel.py` | `convert_many`：输入顺序、错误隔离、分页并行、每线程独立 PDF 文档 |
| `src/IndraOcr.Cli/Cli.cs` | `cli.py` | 参数、格式、`--out`、退出码、stderr 文案 |
| `src/IndraOcr.Cli/JsonOutput.cs` | `--format json` | 与 Python 相同的字段与数值 |
| `src/IndraOcr.Cli/Program.cs` | — | 显式 UTF-8 输出；Windows 换行与 Python 一致 |

并行模型说明：Python 把所有文件的页放进一个进程池；本移植**逐文件**处理，文件内分页并行。这样保留“一个大 PDF 用满所有核”，多个小文件则串行（后续可改为有界并行 + 有序输出）。ONNX Runtime 会话共享；PDFium 非线程安全，每线程一个文档且调用串行化。

结果：

- `--format md` 与 Python CLI **逐字节相同**（UTF-8、Windows CRLF、末尾换行）
- `--format json` 除 `timings` 外字段语义完全一致（box 按 1 位小数输出）
- `multipage.pdf`（4 页）在 `cores=1` 与 `cores=4` 下输出相同，且与 Python 黄金数据一致
- 参数错误退出 2；输入失败/模型错误退出 1；`--out` 重名生成 `_2`
- 61 个测试全部通过

**S7 Viewer**：

| 文件 | 对应 Python | 状态 |
|---|---|---|
| `src/IndraOcr.Viewer/ViewerEngine.cs` | `viewer.Engine` | 上传、最近 5 个文档、按请求选项读单页、OCR 缓存与预热 |
| `src/IndraOcr.Viewer/ViewerApp.cs` | `make_handler` | 路由、JSON 错误 400、内嵌 HTML/logo |
| `src/IndraOcr.Viewer/ViewerOptions.cs` | `_main` | 端口、线程、dpi、页数、模式等参数 |
| `src/IndraOcr.Viewer/wwwroot/index.html` | `viewer.PAGE` | 原样复用；logo 改为 `/logo.png` |

结果：

- 真实进程冒烟：`GET /` 返回 22 KB HTML，`/logo.png` 可读；上传 `scan.png` 后读取第 1 页得到 36 行、9 blocks、JPEG data URI
- `/info` 的 `version/tesseract/defaults` 与参考一致
- 6 个 viewer 测试通过（引擎 + 真实 Kestrel HTTP）：图像/PDF、旋转页、非法 mode、坏文件、未知文档、HTTP 400
- Tesseract 对照未实现：`--tesseract` 会提示不可用，`/info.tesseract` 为 false
- 全量测试 **67 passed**

**桌面 GUI（Avalonia 12 + Semi Avalonia）**：

| 文件 | 内容 |
|---|---|
| `src/IndraOcr.Desktop/Session/OcrSession.cs` | 打开文档、按选项读单页、模型缓存/下载、JPEG + 行/块 + JSON |
| `src/IndraOcr.Desktop/Controls/PageOverlay.cs` | 页图叠加行框、版面块、标签、阅读序；行高亮 |
| `src/IndraOcr.Desktop/Controls/MarkdownRenderer.cs` | 与 Web 相同的 Markdown 子集（标题/段落/列表/HTML 表格/LaTeX） |
| `src/IndraOcr.Desktop/MainWindow.axaml(.cs)` | 仿 Web 布局：来源、管线、模型、显示开关、逐页卡片与 4 个标签页 |
| `src/IndraOcr.Desktop/ViewerOptions/Program` | 启动、`--version`、可选初始文件 |

结果：

- 真实进程截图验证：150% DPI 下 1942×1286 物理窗口，左侧 300 DIP 侧栏、右侧页卡与覆盖层正常
- 图像（scan.png）读取 36 行、9 blocks；PDF 旋转页、模型目录/下载、显示开关可用
- 4 个 session 测试 + 全量 **71 passed**
- 不支持 Tesseract 对照；一次只打开一个文档

**尚未覆盖（留给后续阶段）**：

- `mask.onnx` 的 CenterMask 路径（模型集里没有该文件）
- `det.onnx` 的图片通道（合成样本未触发图片区；`Picture` block 与 `![](picture-k)` 逻辑已移植，缺样本）
- 真正的竖排页（合成页被 orient 判为 270° 而不是 VERTICAL 类；竖排分支代码已移植，缺可触发样本）
- `table_heads=True` 的表头路径（Python 的公开 `load()` 未暴露该开关）
- 真实 PDF 文字层数学行（合成 Helvetica 页未被分类器命中；替换级测试已覆盖）
- 多文件之间的并行（当前逐文件 + 文件内分页并行）
- Viewer 的 Tesseract 对照



---

## 4. 一致性验证策略（本项目成败关键）

### 4.1 三层验证

1. **纯函数层**：`vocab`、`rtl.flip`、`order_xycut`、`_cluster`、`table_html`、`tidy_lines`、`det_post.decode`
   等无模型函数，直接对照 Python 输出（大量随机输入 + 边界用例）。
2. **中间产物层**：每页保存检测输入张量、检测输出框、每行裁剪图、CTC ids/置信度、order/layout logits、LM logp。
   C# 端同样落盘，**先比中间结果，再比最终结果**，定位到具体阶段。
3. **端到端层**：整页 Markdown / JSON 全字段比对。

### 4.2 容差

| 项 | 标准 |
|---|---|
| 文本、阅读序、blocks 标签、Markdown | **完全一致**（字符串相等） |
| 行框 x0/y0/x1/y1 | ≤0.5 px（数字版 PDF 页可放宽到 1 px） |
| 置信度、LM 分数 | ≤1e-4 |
| 中间张量（检测/分类 logits） | ≤1e-4，或余弦相似度 ≥0.9999 |
| `note`、`source`、`rotation`、`skew`、`vertical` | 一致（`skew` 允许 ±0.05°） |

差异必须**解释**（浮点误差 / 排序稳定性 / OpenCV 版本差异），不允许「看起来差不多」放过。

### 4.3 语料

- `examples/samples/*.png`（扫描件、公式）
- `tools/make_samples.py` 生成的合成页（英文版式、暗底、90°/270°、7° 倾斜、微小字、RTL、CJK）与测试生成的 PDF
- 多语言页面：拉丁/中文/日文/韩文/阿拉伯/希伯来/天城文/泰文各 ≥5 页
- 退化样本：暗底浅字、45° 倾斜、超宽横幅、>100MP 大页
- 若允许联网：OmniParseBench 子集（`bench/opb_indra.py` 同源页面）作为准确率回归（已确认可用）。

### 4.4 验收（Definition of Done）

- 全部语料：文本/顺序/Markdown 100% 一致；框与置信度在容差内。
- 现有 14 个测试文件的用例在 xUnit 全部通过（含被 `pytest` 标记的模型可选路径）。
- CLI：退出码、stderr 文案、`--out` 重名规则一致。
- 性能：暂不设门槛，记录同机同语料的吞吐/延迟基线即可（S8 再优化）。
- 无 Python 依赖；`dotnet publish -r win-x64` 可独立运行。

---

## 5. 关键技术难点与解法

### 5.1 numpy 语义（最容易「差一点点」的地方）

不要引入 NumSharp 作为全局数组类型（慢且语义不全），只做**薄封装**：

- `Mat`（OpenCvSharp，`CV_32FC1`/`CV_8UC1`）承载图像与张量；
- `Np` 静态类补齐：`ArgSort`（含稳定性说明）、`SearchSorted(side)`、`Bincount(weights)`、
  `Percentile(q, method="linear")`、`LogAddExp`、`CumSum`（按 numpy 顺序）、`Median`、
  `Triu`、`MatMul`（MathNet）、`Clip`、`Stack`、`Indices`、`Take`/`Ravel`（行主序）。
- 特别小心：
  - `np.argsort` 默认**非稳定**快速排序。`decode_order`、`width_batches` 里的并列值可能影响结果。
    → 先按稳定排序实现并跑黄金对比；若出现顺序差，再复刻 numpy 的 introsort 行为。
  - `np.percentile` 是 `(n-1)*q` 线性插值（`rec_preproc` 有手写直方图版本，需逐行照抄）。
  - `np.sum` 使用成对求和，顺序与朴素循环不同；`estimate_skew` 用 `int64` 累加后比较平方和，必须保持整型累加。
  - `np.bincount(weights=...)` 的标签顺序、`minlength` 行为。
- 所有数值统一 `float32` 推理 / `float64` 统计，与 Python 保持一致（不要「顺手优化」成 float32）。

### 5.2 Unicode 数据（.NET 缺 bidi 与 East Asian Width）

.NET 提供 `CharUnicodeInfo.GetUnicodeCategory` 和 `string.Normalize(NFC/NFD/NFKC)`，但没有
**bidi class**（`unicodedata.bidirectional`）和 **East Asian Width**（`east_asian_width`）。

解法：`tools/gen-unicode-tables.py` 用 Python 的 `unicodedata` 生成两份压缩区间表
（`R/AL/L/EN/AN/...`、`W/F/H/Na/A/N`），输出为 C# 源或嵌入资源。约 200 行生成器，
覆盖 Unicode 15/16 全表，体积几十 KB。`category` 直接用 .NET。
`canonical()` 的字符映射表是手写的，照抄即可。

### 5.3 `.npz` 与 n-gram LM

- `.npz` = ZIP 包 + `.npy`（128 字节头 + 原始数据，含 dtype/shape/fortran_order）。
  写一个 ~200 行 `NpzReader` 支持 `uint64/int64/float64/float32` 即可，不要引三方库。
- LM 用 `uint64` 打包键（每 token `bits` 位，历史长度在最高 3 位）。C# 里就是 `ulong` 位运算。
- beam search 的 `dict` 必须**保序**（Python 3.7+ 插入序）：用 `List<Entry>` + `Dictionary<Key,int>`
  自行实现，或 `OrderedDictionary`。`max()` 并列时取第一个，要复刻。
- 性能：`logp_keys` 是逐 step 批量查找，Python 用向量化 searchsorted；C# 用 `Span<ulong>.BinarySearch`
  或对排序表做一次 `Array.BinarySearch` 循环，注意缓存（`cache`/`ikcache` 上限 1e6）。

### 5.4 PDFium 互操作（最脆弱的一环）

`textlayer.py` 直接用了 pdfium 的 C API，C# 侧建议**自己写 P/Invoke**（比第三方包更可控）：

```
FPDF_InitLibrary / FPDF_DestroyLibrary
FPDF_LoadDocument(path, password) / FPDF_LoadMemDocument
FPDF_GetPageCount / FPDF_LoadPage / FPDF_ClosePage / FPDF_CloseDocument
FPDF_GetPageWidthF / FPDF_GetPageHeightF / FPDF_GetPageSizeByIndexF
FPDFPage_GetRotation / FPDFPage_SetRotation
FPDFBitmap_CreateEx(format=FPDFBitmap_Gray) / FPDFBitmap_FillRect / FPDFBitmap_GetBuffer / FPDFBitmap_Destroy
FPDF_RenderPageBitmap
FPDFText_LoadPage / FPDFText_ClosePage / FPDFText_CountChars
FPDFText_GetUnicode / FPDFText_GetCharAngle / FPDFText_GetCharBox
FPDF_PageToDevice
FPDFPage_CountObjects / FPDFPage_GetObject / FPDFPageObj_GetType
FPDFTextObj_GetTextRenderMode
FPDF_GetLastError
```

原生库：`bblanchon.PDFium.Win32` / `.Linux` / `.macOS`（NuGet 有 64 位包）。
注意：

- **渲染标志必须与 pypdfium2 对齐**（是否含 `FPDF_ANNOT`、是否 `FPDF_LCD_TEXT`）。
  做法：把同一页的灰度渲染结果与 Python 逐像素比对，作为 S4 的第一个验收点。
- pdfium **非线程安全**：每个线程独立 `FPDF_LoadDocument`，或用 `ThreadLocal` 缓存
  （等价于 Python 里每进程一个 `_doc`）。
- 字符坐标是 1/72 inch 点；`_device_affine` 依赖 `FPDF_PageToDevice`，注意 `ctypes.c_int`
  → `out int`，以及 `S=1000` 的精度放大技巧要保留。
- 页面大于 100MP 时要降 DPI（`page_scale`）。
- 密码错误要映射成 `InputError`（对应 `PdfiumError` 文案里含 "password"）。

### 5.5 渲染与图像细节

- `cv2.INTER_AREA` 与 `INTER_LINEAR` 的选择条件（放大/缩小、`resize_window` 的一半阈值）必须照抄。
- `np.rot90` → `Cv2.Rotate` 或 `Transpose+Flip`，注意 `ascontiguousarray` 的效果。
- `imreadmulti` 多帧 TIFF → `Cv2.ImReadMulti`。
- `cv2.connectedComponentsWithStats`：`det_post` 用 4 连通，`picture_boxes` 用 8 连通，别写反。
- `Cv2.SetNumThreads(threads)` 与 ORT 线程数要一起控制，避免超订。

### 5.6 并行模型（与 Python 不同）

Python 用「每核一个单线程进程」，是因为 pdfium 非线程安全。C# 里推荐：

- `OrtOcr` 内 `InferenceSession` **线程安全**，可全局共享（`intra_op=1`）；
- PDF 文档按线程缓存：`[ThreadStatic] static (path, doc)`，或用 `ConcurrentDictionary<path, doc>`
  + `lock`（pdfium 同一 doc 不可并发）；
- 用 `Parallel.ForEach`（`MaxDegreeOfParallelism = cores`）逐页调度，保持**输出按输入顺序**；
- 必须把 `pipeline.py` 里的可变实例状态 `self._pic`、`self._page_rtl` 改成**方法返回值/局部变量**，
  否则并行会串页。这是移植时的必要重构（Python 多进程模型掩盖了这个问题）。
- 内存：单页大扫描 + 批处理容易到 GB 级；`SessionOptions.EnableCpuMemArena = false` 要保留。

### 5.7 ONNX Runtime 细节

- `SessionOptions`: `IntraOpNumThreads=threads`、`InterOpNumThreads=1`、
  `GraphOptimizationLevel = ORT_ENABLE_ALL`、`EnableCpuMemArena=false`；
  provider 通过 `SessionOptions.AppendExecutionProvider_CPU()` 或 CUDA 包（可选）。
- 输入/输出名按 §1.3 固定；`mask` 输入是 `bool`，`widths` 是 `int64`，别用 `int32`。
- 数学解码器：导出时可能省略未使用输入（如 init 图不读 `mask`），
  所以每次调用前用 `session.InputMetadata` 过滤 feed（与 Python `self.inputs` 一致）。
- 批量形状：`pad_batch` 宽度向上取 8 的倍数；`width_batches` 的分批规则影响数值（padding 不影响结果，
  但影响速度），照抄即可。

### 5.8 其它易错点

- `html.escape(text, quote=False)` 只转义 `& < >`，.NET 的 `WebUtility.HtmlEncode` 还会转义引号 → 自己写。
- Python 正则 `fullmatch` → C# `\A(?:...)\z`；`re` 的 `\d` 默认匹配 Unicode 数字，C# 默认也如此，需确认。
- 字符串 `isalpha/isupper/islower/isdigit` → `char.IsLetter/IsUpper/IsLower/IsNumber`，
  注意 Python 的 `isdigit` 与 .NET `IsDigit` 在罕见字符上的差异（现有用法集中在 ASCII，风险低但要在测试里覆盖）。
- JSON 输出：字段顺序与原实现一致（`file,page,source,rotation,skew,vertical,note,lines,markdown,blocks,timings`），
  `float` 用 `"R"`/`InvariantCulture`，`ensure_ascii=False` → `JavaScriptEncoder.UnsafeRelaxedJsonEscaping`。
- `read_images` 对不可读文件抛 `InputError`；单页异常返回 `source="error"` 的空页，不能中断整篇文档。

---

## 6. 风险与对策

| 风险 | 影响 | 概率 | 对策 |
|---|---|:--:|---|
| numpy 细节不一致导致输出漂移 | 高 | 高 | 中间产物逐层比对；先做纯函数层验证 |
| OpenCV 版本与 `opencv-python-headless` 不同 | 中 | 中 | 固定 4.x 版本；像素级比对；必要时自行编译同版本运行时 |
| pdfium 渲染/文字层细节不一致 | 高 | 中 | 先做「渲染像素一致」验收，再做文字层；准备 PDFium 版本升级预案 |
| 排序/并列值导致阅读序不同 | 中 | 中 | 稳定排序 + 黄金对比；必要时复刻 numpy 排序 |
| 线程模型引入竞态 | 高 | 中 | 去除可变实例状态；每线程独立 doc；压力测试并发 100 页 |
| LM/数学路径性能不足 | 中 | 中 | 缓存、Span、避免装箱；`ArrayPool` 复用缓冲区 |
| 模型许可证（OpenRAIL-M）限制分发 | 高 | 中 | 不打包权重，保持「运行时下载」或要求本地目录（与 Python 行为一致） |

---

## 7. 许可证与合规

- 代码：Apache-2.0。C# 移植属衍生作品，需保留 `LICENSE`、版权与变更说明；建议在 README 标注
  「port of datalab-to/indra」。
- 权重：修改版 OpenRAIL-M，**不能**随意随包分发，也含竞争性使用限制。
  最稳妥做法：与原实现一致，运行时从 Hugging Face 下载（`datalab-to/indra-ocr`，tag `v1.1.0`），
  或由用户通过配置指向本地目录（`INDRA_OCR_MODEL_DIR` / `--model-dir`）。
- 若需商业授权，需联系 Datalab（原 README 已说明）。

---

## 8. 决策记录与待确认项

已确认（2026-10-20）：

1. **范围**：全量对齐（S0–S7）。
2. **目标框架**：`net10.0`。
3. **交付形态**：核心类库 + CLI + 本地 Web viewer。
4. **结构策略**：先 1:1 结构移植，对齐后再重构。
5. **验证语料**：允许联网，使用 OmniParseBench 子集做回归。
6. **模型分发**：从 Hugging Face 运行时下载（不打包权重）。
7. **性能**：暂不作为验收门槛，只记录基线；功能与输出一致性优先。

仍需确认（不阻塞开发）：

8. **平台**：只 Windows，还是 Windows + Linux 容器？macOS 需额外验证原生库。
9. **既有 Python 代码**：迁移期间是否继续维护/发版？是否需要双端输出并存的过渡期？

---

## 9. 里程碑检查清单（S0–S7）

S0 已完成：

- [x] Python 参考环境可复现：`pytest` **74 passed**，`indra-ocr examples/samples/scan.png` 输出稳定
- [x] 模型已缓存（`~/.cache/huggingface`，tag `v1.1.0`），版本与哈希写入 `manifest.json`
- [x] `tools/dump_golden.py` 产出逐页 JSON + manifest
- [x] `tools/compare_golden.py` 可报告差异（自检 2/2 通过）
- [x] C# solution（`dotnet/`）可 `dotnet build`，0 警告 0 错误
- [x] 依赖版本锁定：OpenCV 4.13 / ORT 1.31（两端一致）

S1 已完成：

- [x] `Np` 核心函数有 numpy 对照测试（median/percentile/searchsorted/logaddexp/bincount/cumsum/argsort/clip）
- [x] Unicode 表生成器产出并校验（bidi/EAW 已知码位用例）
- [x] `NpzReader` 能读 `lm.npz` 与 `.npy`，键/值类型正确
- [x] ONNX 会话在 C# 跑通 `det.onnx`，与 Python 同形状、同数值（**≤1e-6**）
- [x] `ImageOps` 的 `resize`/连通域由检测端到端验证；`rot90` 有单元用例

S2 已完成：

- [x] 8 页样本检测框与 Python 一致（≤0.5px，实测 ≤0.001px），含自适应二次检测
- [x] 识别文本/置信度/ids/分组/dark 全部一致
- [x] 行裁剪（32px 与 96px tall 窗口）SHA-256 与 Python 相同
- [x] `script.onnx` 分流与"不确定时双读取高置信"一致
- [x] LM logp 与 beam search 与 Python 完全一致
- [ ] `mask.onnx` CenterMask 路径（模型集无此文件，待有模型时验证）
- [ ] `det.onnx` 图片通道（合成样本未触发，待真实图文页）
- [ ] OmniParseBench 子集回归（S3 端到端已跑通，待接入）



S3 已完成：

- [x] `rtl.flip` 7 组用例（自动方向、RTL/LTR、镜像括号）与 Python 一致
- [x] `line_features`、`order_xycut`、`xycut_blocks`、`decode_order`、`_cluster` 与 Python 一致
- [x] `join_lines`、`table_html`（几何表格）与 Python 一致
- [x] 整页 10/10：`rotation`（含 90°/270°）、`skew`（7°）、`note`、lines、blocks、Markdown 全一致
- [x] `tidy_lines`/`split_line`/`gutter_cuts`/表格单元格拆分由整页用例覆盖
- [x] `orient.onnx` 转向与 `estimate_skew` 倾斜矫正接入
- [ ] 真实竖排页（合成页被 orient 判为 270°，VERTICAL 分支待样本）
- [ ] 图片区（`Picture` block / `![](picture-k)`）待真实图文页
- [ ] `table_heads=True` 路径（公开 API 未暴露）
S4 已完成：

- [x] PDFium P/Invoke（固定 156.0.8076，与 pypdfium2 同版本）
- [x] 灰度渲染与 pypdfium2 逐像素一致（3 种 scale，含 /Rotate 90）
- [x] 文字层 run/字符框/角度/quarter/可信度判定与 Python 一致
- [x] `Converter` 三模式：auto/text/ocr；6 页 PDF × 3 模式整页输出一致
- [x] 混合页（source=hybrid）与 `image_ocr=false`
- [x] 加密 PDF、密码错误、多帧 TIFF、max_pages、不可读/缺失文件
- [ ] `math=True` 的 PDF 数学行重读（S5）
- [ ] 多线程 PDF 文档缓存（S6，pdfium 非线程安全）
S5 已完成：

- [x] `MathReader`：分类器 flags、编码器、解码器 KV cache、提前停止、`Detokenize`
- [x] `OrtOcr(math=True)` 接入：math 行用 LaTeX 替换（`$...$` 且置信度 ≥0.9）
- [x] `Converter` 文字层数学行重读（`_read_math`）：保留 PDF 内容序
- [x] 3 页数学判定与读数与 Python 一致（equations 23 行、english 1、scan 1）
- [x] `equations.png` 整页 `--math` 输出与参考端一致（含 LaTeX）
- [x] 数学模型按需下载并校验（`math_meta.json` + 4 个 onnx）
- [ ] 真实 PDF 文字层数学行（分类器在合成 Helvetica 页未命中；函数级测试已覆盖替换逻辑）
S6 已完成：

- [x] `PageRunner.ConvertMany`：输入顺序、错误隔离、`max_pages`、密码
- [x] 并行分页：共享 ONNX 会话，每线程独立 PDF 文档；<4 页或 1 核时回到单页多线程
- [x] 同一 PDF 在 `cores=1` 与 `cores=4` 下输出一致，且与 Python 黄金数据一致
- [x] CLI 全部选项：`--out/--format/--mode/--no-image-ocr/--model-dir/--cores/--dpi/--max-pages/--password/--math/--keep-furniture/--timings/--version`
- [x] 退出码：0 / 1（输入失败、模型错误）/ 2（参数错误）；stderr 文案与参考一致
- [x] `--out` 重名 `_2`、`_3`；UTF-8 输出；Windows 换行与 Python 一致
- [x] `--format md` 与 Python 逐字节相同；`--format json` 字段语义相同（timings 除外）
S7 已完成：

- [x] `IndraOcr.Viewer`：ASP.NET Core Minimal API，绑定 `127.0.0.1`
- [x] 原 `viewer.py` 的 HTML/JS 原样复用（内嵌资源，logo 改为 `/logo.png`）
- [x] `/`、`/index.html`、`/info`、`/logo.png`、`POST /upload`、`POST /page`
- [x] 上传保留最近 5 个；PDF 临时文件；按请求选项读取单页（流式出页）
- [x] 页面 JSON 字段与 Python 一致：`page/w/h/img/lines/blocks/markdown/source/note/rotation/skew/vertical/json/timings`
- [x] 图像/PDF/旋转页、非法 mode、坏文件、未知文档、HTTP 错误 400
- [x] `--port/--threads/--dpi/--max-pages/--mode/--no-image-ocr/--keep-furniture/--math/--model-dir/--version`
- [ ] Tesseract 对照（C# viewer 不提供；`--tesseract` 会提示不可用）

桌面 GUI 已完成：

- [x] Avalonia 12 + Semi Avalonia，布局仿 Web 查看器
- [x] 页图覆盖层：行框、版面块、块标签、阅读序、行高亮
- [x] 逐页读取、4 个标签页（Preview/Markdown/Lines/JSON）
- [x] 模型目录输入/选择/下载（含 math）；首次读取自动下载
- [x] 来源/管线/DPI/最大页数/显示开关；真实文件与拖放打开
- [x] 高 DPI 适配（PerMonitorV2 manifest + 按可用屏幕收敛窗口）
- [x] 4 个 session 测试；全量 71 passed
- [ ] Tesseract 对照（与 Web viewer 相同，未实现）