# PaddleOCR 引擎接入设计（先设计，后实现）

状态：**A/B/C 已实施**。默认引擎仍是 Indra ONNX；桌面 GUI 已可在 Indra 与 PaddleOCR (PP-OCRv4 ONNX) 间切换，Paddle 模型随应用预置，不运行时下载。

---

## 1. 目标与非目标

### 目标

1. 在桌面 GUI、CLI、Viewer 中可以选择 OCR 引擎：
   - `Indra`（现有 ONNX 流程，默认）
   - `PaddleOCR`
2. 引擎切换后不重建整个应用，只重建 OCR 会话，并重新读取当前文档。
3. 保持现有 Indra 的输出、测试和 NativeAOT 发布不变。
4. 新引擎以插件方式加入核心库，不把 PaddleOCR 细节写进 UI。
5. 引擎能力不同（PDF 文字层、数学、版面、竖排等）时，UI 明确显示支持/不支持，不静默改变结果。

### 第一版非目标

- 不要求 PaddleOCR 的 Markdown/版面结果与 Indra 完全一致。
- 第一版不实现 PDF 文字层混合模式、数学公式、Tesseract 对照。
- 第一版不承诺 Android/iOS 的 PaddleOCR 引擎；移动端先继续使用 Indra。
- 不把 Python `paddleocr` 作为必需的运行时依赖。

---

## 2. 现状与耦合点

当前 `Converter` 直接依赖 `OrtOcr` 的低层方法，而不是一个“页面 OCR”接口：

```text
Converter ──> OrtOcr.Detect
          ├─> OrtOcr.Recognize
          ├─> OrtOcr.BoxGroups
          ├─> OrtOcr.Assemble
          ├─> OrtOcr.Vocab / Groups
          └─> OrtOcr.MathRec

PageRunner / ViewerEngine / OcrSession ──> new OrtOcr(...)
```

直接换掉 `OrtOcr` 会同时破坏：

- PDF 文字层校验（`Vocab`、`Groups`、`Detect`、`Recognize`、`BoxGroups`）
- 数学行重读（`MathRec`、`Height`）
- 阅读序 / 版面 / Markdown（`Assemble`）
- 并行分页（每个 worker 一个 `OrtOcr`）

因此不能把 `OrtOcr` 简单替换成 `PaddleOcr` 类；需要分两层：

1. **高层页面接口**：所有引擎都必须实现，输入灰度页，输出 `Page`。
2. **Indra 专属低层接口**：只有需要 PDF 文字层/数学的引擎实现；Paddle 第一版不实现。

---

## 3. 目标架构

```text
┌──────────────────────────── UI / CLI / Viewer ────────────────────────────┐
│  EngineBox: Indra / PaddleOCR                                              │
│  Settings: OcrEngineKind, model dir, engine options                       │
└───────────────────────────────────┬────────────────────────────────────────┘
                                    │ OcrEngineConfig
                                    ▼
                         ┌─────────────────────┐
                         │ IOcrEngineFactory    │
                         └──────────┬──────────┘
                                    │
             ┌──────────────────────┴──────────────────────┐
             ▼                                             ▼
   ┌─────────────────────┐                       ┌─────────────────────┐
   │ IndraOnnxEngine     │                       │ PaddleOnnxEngine    │
   │ (OrtOcr adapter)    │                       │ (det/rec/cls ONNX)  │
   └──────────┬──────────┘                       └──────────┬──────────┘
              │ IOcrEngine.Read(Mat) -> Page                │
              └──────────────────────┬──────────────────────┘
                                     ▼
                         ┌─────────────────────┐
                         │ DocumentPipeline     │
                         │ PDF render / images  │
                         │ result -> UI / JSON  │
                         └─────────────────────┘
```

核心原则：

- `Page` 仍是统一输出模型；UI 不感知具体引擎。
- `Converter` 通过 `IOcrEngine` 读页面；Indra 的文字层/数学能力通过可选接口访问。
- 引擎选择是 **应用级设置**，不是页面级设置；切换后清空 OCR 实例和页面缓存。
- 每个 worker 独立创建引擎实例，避免 Paddle/ORT 会话的线程安全问题。

---

## 4. 核心接口（建议实现）

### 4.1 引擎与配置

```csharp
public enum OcrEngineKind
{
    Indra,
    PaddleOcr,
}

public sealed record OcrEngineConfig(
    OcrEngineKind Kind,
    string ModelDir,
    int Threads,
    bool Math,
    bool Furniture,
    string? Revision = null);

public interface IOcrEngine : IDisposable
{
    OcrEngineKind Kind { get; }
    string DisplayName { get; }
    OcrCapabilities Capabilities { get; }

    /// <summary>灰度页 -> 行/块/Markdown/耗时。</summary>
    Page Read(OpenCvSharp.Mat gray, bool turn = true);
}

[Flags]
public enum OcrCapabilities
{
    None = 0,
    Images = 1 << 0,
    PdfRenderedPages = 1 << 1,
    PdfTextLayer = 1 << 2,
    Math = 1 << 3,
    LayoutBlocks = 1 << 4,
    Vertical = 1 << 5,
    ModelDownload = 1 << 6,
}
```

`IOcrEngine.Read` 是唯一的强制入口。PaddleOCR 第一版只需实现它。

### 4.2 Indra 专属能力

PDF 文字层、数学和精确版面仍由 Indra 的现有算法提供。不要让 Paddle 被迫实现这些低层方法：

```csharp
/// <summary>仅 Indra 引擎实现；Paddle 第一版不实现。</summary>
public interface IIndraOcrDetails
{
    IReadOnlyDictionary<string, int> BaseVocab { get; }
    IEnumerable<IReadOnlyDictionary<string, int>> GroupVocabs { get; }
    int LineHeight { get; }
    MathReader? MathRec { get; }

    float[] Detect(Mat gray);
    RecLine[] Recognize(Mat gray, float[] boxes, int batch = 64, bool vertical = false);
    (string[] Groups, bool[] Sure) BoxGroups(Mat gray, float[] boxes);
    Page Assemble(int width, int height, float[] boxes, RecLine[] lines,
                  Dictionary<string, double> timings, bool vertical);
}
```

`IndraOnnxEngine` 同时实现 `IOcrEngine` 和 `IIndraOcrDetails`，内部持有现有 `OrtOcr`。
`Converter` 的逻辑：

```csharp
if (_engine.Capabilities.HasFlag(OcrCapabilities.PdfTextLayer)
    && _engine is IIndraOcrDetails details)
{
    // 现有文字层/混合/数学路径
}
else
{
    // 高层的 PDF 渲染 -> _engine.Read(gray)
    // mode=text 明确报“不支持”，auto 退化为 ocr
}
```

这样第一版 Paddle 不做文字层，也不会让现有 Indra 路径退化。

### 4.3 工厂与模型提供者

```csharp
public interface IOcrEngineFactory
{
    IOcrEngine Create(OcrEngineConfig config);
    OcrEngineDescriptor Describe(OcrEngineKind kind);
}

public sealed record OcrEngineDescriptor(
    OcrEngineKind Kind,
    string DisplayName,
    OcrCapabilities Capabilities,
    IReadOnlyList<OcrOptionDescriptor> Options);

public interface IOcrModelProvider
{
    OcrEngineKind Kind { get; }
    bool IsPresent(string dir, OcrEngineConfig config);
    string Resolve(string? requestedDir, OcrEngineConfig config, bool allowDownload);
}
```

模型目录建议：

```text
Data/
  ...                       # 现有 Indra 模型，保持兼容
  paddleocr/
    PP-OCRv5/
      det.onnx
      rec.onnx
      cls.onnx
      dict.txt
      paddle_meta.json
```

- Indra 继续使用现有 `Data` 根目录，避免破坏已发布版本。
- Paddle 使用 `Data/paddleocr/...`，两套模型互不覆盖。
- 默认路径仍由应用目录 `Data` 派生；可用环境变量/命令行覆盖。

---

## 5. PaddleOCR 接入方式比较

| 方案 | 优点 | 缺点 | 结论 |
|---|---|---|---|
| **PaddleOCR ONNX + 现有 ORT** | 复用 `Microsoft.ML.OnnxRuntime`；可跨 Windows/Linux/macOS/Android；无 Paddle 运行时；容易配合现有 OpenCV | 需要移植 DB 后处理、透视裁剪、CTC 解码；模型需导出 ONNX | **推荐** |
| PaddleOCRSharp / Paddle Inference .NET | C# API 较直接，Paddle 官方后处理；桌面开发快 | 原生库平台有限；NativeAOT/Android/iOS 风险高；包体大；可能与现有 ORT 重复 | 作为桌面快速验证的备选 |
| Python `paddleocr` sidecar | 最快得到官方结果；支持模型最全 | 需要 Python/Paddle 环境和进程管理；破坏单文件 AOT；分发/授权复杂 | 不纳入产品主路线 |
| ONNX + PP-Structure | 后续可补版面/表格 | 模型和预处理更多 | 第二阶段 |

**推荐第一版：PaddleOCR ONNX**。

具体模型：

- PP-OCRv5 detection（DB）
- PP-OCRv5 recognition（SVTR/CTC）
- 可选文字方向分类 `cls`
- 语言模型按需选择（中文/英文/多语言）

模型来源和许可证必须在下载页/`Data/paddleocr/LICENSE` 中明确；PaddleOCR 代码通常为 Apache-2.0，具体模型卡以发布页为准。

---

## 6. Paddle 引擎内部设计

建议新增：

```text
src/IndraOcr/Paddle/
  PaddleOcrEngine.cs       # IOcrEngine 实现
  PaddleModelFiles.cs      # 模型下载/校验
  PaddleMeta.cs            # dict、输入尺寸、mean/std
  PaddleDetect.cs          # DB 预处理/后处理
  PaddleRecognize.cs       # 透视裁剪、resize、CTC 解码
  PaddleResultAdapter.cs   # Paddle line -> Page/Block/Markdown
```

### 数据流

```text
Mat gray
  -> det preprocessing (resize/normalize)
  -> det.onnx
  -> DB postprocess (score map -> boxes -> unclip)
  -> sort boxes (top-to-bottom / left-to-right)
  -> crop + perspective transform
  -> rec.onnx (batch)
  -> CTC decode + dict
  -> PaddleResultAdapter
  -> Page { Lines, Blocks, Markdown, Source="paddleocr", ... }
```

### 结果归一化

- Paddle 的框是四边形：先取 `minX/minY/maxX/maxY` 写入现有 `Line.Box`；后续可扩展 `Line.Polygon`。
- `Conf` 使用 Paddle recognition score。
- 空文本/低于阈值的行按引擎配置过滤。
- `Source` 写 `"paddleocr"`，UI 芯片显示 `PaddleOCR`，便于用户判断结果来源。
- 第一版 Markdown：按行 + 空行分段；不伪造 Indra 的 layout blocks。`Blocks` 可为空或一个 `Text` 块；UI 的“版面块”开关在无块时置灰。
- 不实现数学：`Capabilities.Math=false`；Math 开关禁用。
- 不实现 PDF 文字层：`Capabilities.PdfTextLayer=false`；PDF 一律渲染后 OCR，`mode=text` 报错，`auto` 退化为 `ocr`。

### 线程/性能

- 每个 `PageRunner` worker 一个 `PaddleOcrEngine`，内部持有独立 ORT sessions。
- Paddle det/rec 可复用一个 `InferenceSession`，但不要跨 worker 共享可变状态。
- 模型常驻内存；切换引擎时 `Dispose` 旧实例。
- 首次加载后做一次空白图 warm-up（与 Indra 一致）。

---

## 7. UI / CLI / Viewer 设计

### 桌面 GUI

在侧栏 `SOURCE` 上方新增：

```text
OCR ENGINE
[ Indra (ONNX)        ▼ ]
   Indra (ONNX)
   PaddleOCR
```

- 切换后清空页面、显示 Loading 遮罩、重新读取当前文档。
- 顶部页卡显示引擎徽标：`Indra` / `PaddleOCR`。
- 模型区跟随引擎：
  - Indra：现有 `Data` 路径、核心 + math
  - PaddleOCR：`Data/paddleocr` 路径、det/rec/cls 下载按钮
- 不支持的选项：
  - Paddle：禁用 `Math as LaTeX`、`Text layer` 模式、`版面块/块标签`（若第一版无 layout）、`Tesseract`
  - 不禁用会误导用户的选项，而是禁用并显示原因。
- 语言切换、Loading、模型下载仍复用现有机制。

### CLI

```text
--engine indra|paddleocr        (default: indra)
--paddle-model-dir DIR          (optional; default Data/paddleocr)
```

- `--engine paddleocr` 与 `--math` 冲突时明确报错。
- JSON 增加可选字段 `"engine": "paddleocr"`（保持旧字段不变）；Markdown 不承诺与 Python/Indra 一致。
- 默认值保持 `indra`，现有 parity 测试不受影响。

### Viewer

- `/info` 增加 `engines: ["indra","paddleocr"]`；`defaults.engine="indra"`。
- 上传/读取时接受 `engine` 选项；缺省 Indra。
- HTML 增加引擎下拉；不支持的选项置灰。

### Android

第一版仍只使用 Indra。Paddle ONNX 理论上可复用现有 ORT/OpenCV，但模型、包体和触屏选项需要单独验证，放到后续阶段。

---

## 8. 设置与迁移

新增一个轻量设置（沿用现有语言设置的做法，避免 AOT 反射序列化）：

```text
%LOCALAPPDATA%\indra-ocr\desktop-engine.txt
indra
```

- 读取失败/未知值回退 `indra`。
- Android 使用应用私有设置文件。
- 引擎切换只影响新会话，不迁移/修改现有模型文件。
- 首次选择 Paddle 时：
  - 如果 `Data/paddleocr` 不存在，显示“下载模型”提示；
  - 可自动下载或让用户点击下载，默认不静默消耗流量。

---

## 9. 分阶段实施计划

| 阶段 | 内容 | 完成标准 |
|---|---|---|
| **A. 抽象层** | 新增 `IOcrEngine`/`OcrCapabilities`/`IOcrEngineFactory`；`OrtOcr` 包装为 `IndraOnnxEngine`；`Converter` 改依赖接口 | 现有 73 测试全过；Indra 输出逐字节不变；GUI/CLI 行为不变 |
| **B. 设置与 UI 占位** | 引擎下拉、设置持久化、引擎徽标；Paddle 选项禁用 | 切换只重建会话；默认 Indra；无 Paddle 时明确提示 |
| **C. Paddle 图片 OCR** | 模型下载、det/rec ONNX、结果适配、桌面 GUI/CLI | 用固定样例图片完成 OCR；框/文本/置信度显示正常；异常/缺模型有中文提示 |
| **D. PDF 渲染页** | Paddle 的 PDF 渲染 + OCR；能力门控 | `auto/ocr` 可用；`text` 明确不支持；与 Indra 模式不混淆 |
| **E. 分发与测试** | 模型目录、许可证、AOT/包体、文档、可选 Android | 发布包可切换；Paddle 有独立回归样例；Indra 仍全绿 |

预计：A 2–3 天；B 1–2 天；C 5–10 天；D 2–4 天；E 2–4 天。具体取决于 Paddle ONNX 后处理移植和模型版本。

---

## 10. 风险与验收

### 风险

1. **Paddle ONNX 后处理**：DB 的 unclip、框排序、透视裁剪、CTC 字典必须与导出模型匹配，否则框/文本会明显偏差。
2. **模型版本漂移**：PP-OCRv5 的 det/rec 输入输出与旧版不同；需把版本、输入尺寸、mean/std 写入 `paddle_meta.json`。
3. **输出差异**：Paddle 的阅读序、分段和 Markdown 与 Indra 不一致；UI 必须标明引擎，不能声称 parity。
4. **PDF 文字层/数学**：Paddle 第一版没有对应实现；必须显式禁用，不能静默忽略用户勾选。
5. **NativeAOT**：ONNX 方案可以继续 AOT；PaddleSharp/Python 方案不能保持当前单文件 AOT 发布。
6. **包体/模型**：Paddle 模型应下载或单独放置，不默认塞进 256 MB 的桌面发布包；移动端更敏感。
7. **许可证**：PaddleOCR 代码与模型分别确认；在 `Data/paddleocr/LICENSE` 和文档中记录。

### 验收标准

- 引擎下拉可切换，切换后重新读取当前文档。
- Indra 默认路径、输出、73 个测试和 NativeAOT 发布不回归。
- Paddle 图片 OCR 至少覆盖中文、英文、旋转/低质量样例各一组。
- Paddle 缺模型、下载失败、内存不足、PDF `text` 模式都有明确错误。
- UI 不显示 Paddle 不支持的选项为“可用”。
- JSON 可区分 `engine`；Markdown/JSON 不混用旧 parity golden。

---

## 11. 需要确认的决策

建议先确认以下 4 项再进入编码：

1. **Paddle 接入方式**：推荐 PaddleOCR ONNX + 现有 ORT；如果优先最快出效果，可先用 PaddleOCRSharp 做桌面验证。
2. **PaddleOCR 版本与语言**：PP-OCRv5 中文/英文/多语言中的哪一组。
3. **第一版范围**：只图片 OCR（推荐），还是必须包含 PDF 渲染页。
4. **模型分发**：首次下载（推荐）还是随发布包携带。

默认建议：**ONNX + 桌面优先 + 图片/PDF 渲染页 + 首次下载模型**；Android 暂不改引擎。
