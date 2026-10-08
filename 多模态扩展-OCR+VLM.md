# WeKnow 扩展 #5：多模态文档处理（OCR + VLM）

> 在 RAGLite 内核之上自研的文档处理器。新增 `src/weknow/_multimodal.py`（672 行）+
> `MultimodalConfig` + 22 个单测（280 行）；一次独立提交。
> 目标：让**图片与扫描版 PDF** 与文本知识源共享同一套「三层切分 → 混合建库 → 重排」检索链路。

---

## 一、设计：为什么是「双阶段 + 可插拔」

一张图里有两类信息，单一手段都拿不全：

| 信息类型 | 例子 | 提取手段 |
|---|---|---|
| **图内文字**（渲染在像素里的字） | 截图、扫描合同、表格数字 | OCR |
| **非文字语义**（像素组合表达的含义） | 折线图趋势、架构图、流程图、照片 | 视觉语言模型（VLM）描述 |

因此处理链是两阶段并行后**拼装**，而不是二选一：

```
图片 bytes ──┬─→ OCR 引擎   ─→ ocr_text    ─┐
             │   (RapidOCR/Tesseract/注入)  ├─→ ImageExtraction.to_markdown()
             └─→ VLM 引擎   ─→ description ─┘         │
                 (LiteLLM 视觉模型/注入)              ▼
                                    "图内文字\n\n[Image (chart): 描述]"
                                                      │
                                                      ▼
                                  复用既有三层切分 / 多向量嵌入 / BM25+VSS 混合检索
```

关键设计点：

1. **双阶段独立降级**：OCR 失败（模型缺失、图片损坏、推理异常）只丢 OCR 那一路，
   描述照常产出；VLM 失败（网络、配额、模型不支持视觉）只丢描述那一路，OCR 文本照常保留。
   两处均以 `logger.warning` 记录原因。绝不因为一个模态失败而中止整篇文档。
2. **引擎可注入**：`ocr_engine` / `vlm` 既接受字符串（自动实例化），也接受实现了
   `OCREngine` / `VLMEngine` 协议的对象。这让单元测试完全离线、确定性，也方便接自研模型。
3. **本地优先、零新增必需依赖**：默认 OCR 走 RapidOCR 的 ONNX 模型（CPU、离线、含中英），
   VLM 走 LiteLLM（已是 RAGLite 核心依赖）；重依赖全部放在 `multimodal` extra 里惰性导入，
   不装就不用。
4. **PDF 按页栅格化**：`pymupdf` 把每页渲染成 PNG（默认 200 DPI，可配、可截断页数），
   之后与普通图片走同一条提取路径——**不需要为 PDF 单独维护一套分支**。
5. **多页并发提取**：VLM 阶段是网络延迟主导，逐页串行会浪费绝大部分墙钟时间。
   用 `ThreadPoolExecutor`（上限 `extraction_max_workers`，默认 4）并发，并按输入顺序返回结果。
6. **图像类型作为检索线索**：按文件名启发式标注 `chart / diagram / table / screenshot / logo / photo`，
   以 `[Image (chart): ...]` 形式写进 Markdown，既提升可读性，也让类型词进入 BM25 词表。

---

## 二、配置项（`MultimodalConfig`）

| 字段 | 默认 | 说明 |
|---|---|---|
| `ocr_engine` | `None`（等价 `"auto"`） | `"auto"` 依次探测 RapidOCR → Tesseract；`"none"` 关掉 OCR；也可传 `OCREngine` 实例 |
| `ocr_languages` | `("ch", "en")` | RapidOCR 语言请求 |
| `tesseract_languages` | `("eng",)` | Tesseract 语言包（中文需 `chi_sim`） |
| `include_ocr_text` | `True` | 是否保留 OCR 文本 |
| `vlm` | `None` | LiteLLM 视觉模型串（`openai/gpt-4o-mini`、`ollama/llava`…）；`None` = 只做 OCR；也可传 `VLMEngine` 实例 |
| `vlm_api_key` / `vlm_api_base` | `None` | 云端密钥 / 本地 OpenAI 兼容端点 |
| `vlm_prompt` | `DEFAULT_VLM_PROMPT` | 先要求逐字转录、再描述图表，并按图像语言作答 |
| `vlm_max_tokens` / `vlm_temperature` / `vlm_num_retries` | `512` / `0.0` / `3` | 采样与重试 |
| `describe_images` | `True` | 总开关，关掉后不构造 VLM 引擎 |
| `pdf_dpi` | `200` | PDF 栅格化分辨率 |
| `max_pdf_pages` | `None` | 页数上限（大报告先切前 N 页试跑） |
| `extraction_max_workers` | `4` | 多页/多图并发度 |

---

## 三、用法

**安装（可选）**

```bash
pip install 'weknow[multimodal]'   # rapidocr-onnxruntime + pymupdf + pillow + pytesseract
```

**A. 纯离线：只有 OCR（零网络、零密钥）**

```python
from weknow import Document, RAGLiteConfig, MultimodalConfig, insert_documents

config = RAGLiteConfig(
    document_processor=MultimodalConfig(ocr_engine="auto", vlm=None),
)
insert_documents([Document.from_path("scan.png", config=config)], config=config)
```

**B. OCR + VLM 描述（云端视觉模型）**

```python
config = RAGLiteConfig(
    document_processor=MultimodalConfig(
        ocr_engine="auto",
        vlm="openai/gpt-4o-mini",       # 任意 LiteLLM 视觉模型
        vlm_api_key="sk-...",           # 或走环境变量
    ),
)
insert_documents([Document.from_path("report.pdf", config=config)], config=config)
```

**C. OCR + VLM 描述（本地 OpenAI 兼容端点，如 Ollama）**

```python
MultimodalConfig(
    vlm="openai/llava",
    vlm_api_base="http://localhost:11434/v1",
    vlm_api_key="ollama",
)
```

**D. 只做描述、不做 OCR（照片/示意图场景）**

```python
MultimodalConfig(ocr_engine="none", vlm="openai/gpt-4o-mini")
```

**E. 单图快速提取（不入库，想看提取质量）**

```python
from weknow import extract_image, MultimodalConfig

extraction = extract_image(open("chart.png", "rb").read(), processor_config=MultimodalConfig())
print(extraction.ocr_text)      # 图内文字
print(extraction.description)   # 视觉模型描述
print(extraction.image_type)    # chart / diagram / ...
print(extraction.to_markdown()) # 实际入库的 Markdown
```

> ⚠️ 未配置多模态处理器时索引图片会**显式报错**（而不是把二进制当文本读入库），
> 错误信息里直接给出配置示例。

---

## 四、与检索链路的关系

多模态内容不是独立的旁路索引，而是**拼进同一份 `document.content`**，因此：

- 图内文字与图像描述一并进入 **三层切分**（SaT 句界 → DP chunklet → BIP 语义分块）；
- 一并进入 **多向量嵌入**（late chunking + 标题增强）与 **DuckDB FTS(BM25)** 索引；
- 检索时由既有的 **BM25 + VSS 混合召回 + RRF + FlashRank 重排 + chunk span 回拼** 处理，
  无需为多模态新增任何检索分支。

代价与边界（面试可主动说）：

- 图像描述与 OCR 文本在同一个 chunk 空间里竞争，**单一 chunk 可能图文混杂**；若要严格分列，
  需要在 metadata 上打 `modality` 标记再做过滤（当前 `Document` 的 metadata 机制已支持，未做默认分流）。
- 图像**区域级**（bbox）检索未实现——当前是「整图 → 一段文字」的粒度，WeKnora 的
  `PaddleOCR-VL + bbox_annotation_format` 才是区域级方案，属于下一步。

---

## 五、验证与实测状态

**已跑通（真实执行）**

| 项 | 结果 |
|---|---|
| 多模态单测 | **22/22 通过**，含真实 PDF（`tests/specrel.pdf`）经 `pymupdf` 栅格化出 PNG 的集成路径 |
| 覆盖点 | 双模态合并 / 仅 OCR / 仅描述 / 空提取 / OCR 失败降级 / VLM 失败降级 / 并发顺序 / 类型标注 / 扩展名路由 / `auto` 无引擎报错 / 注入引擎 / 图像无处理器报错 / `Document.from_path` 端到端 / 页数上限 / prompt 透传 |
| 相关回归 | `_retrieval_extensions`、`_markdown`、`_mistral_ocr`、split 系列共 65 passed（另 2 个失败为改动前既有：本机未装 `llama-cpp-python`） |
| 静态检查 | `ruff check` / `ruff format` / `mypy`（strict）对新增与改动文件全部干净 |

**尚未取数（需环境）**

| 待测项 | 前置条件 |
|---|---|
| OCR 字符准确率 | `pip install 'weknow[multimodal]'`（RapidOCR 首次会拉 ONNX 模型） |
| 含 VLM 的端到端 P50 / 首字延迟 | 视觉模型端点（云端 key 或本地 Ollama） |
| 图文问答准确率（改造 `gen_bench.py` 加图问） | 同上 |
| 多模态入库吞吐（页/秒） | 同上 |

---

## 六、面试预案

**Q：为什么不用现成的 Unstructured / PaddleOCR-VL 一体化方案？**
A：它们把解析和切分绑死了。RAGLite 的价值恰好在切分与混合检索，所以这里只做**解析侧的可插拔
补强**，产出标准 Markdown 后交回原链路——换 OCR 引擎或 VLM 不影响切分、嵌入、检索的任何一环。

**Q：一个模态失败为什么要吞掉异常？**
A：文档处理是批量的，单页失败不应该让整篇入库失败。而且两路信息高度互补：OCR 挂了描述还在，
描述挂了（比如自建端点抖动）OCR 文本还在。异常只降级、不中止，日志留痕——这是批处理管线
应有的失败语义。测试里两路失败各有专门用例。

**Q：并发提取会不会打乱页码顺序？**
A：不会。`extract_images` 按输入顺序收集 future 结果，顺序由输入决定，与完成顺序无关，
测试用 4 张不同图片断言了顺序。

**Q：这算 RAGLite 的能力吗？**
A：不算，这是二次开发扩展。RAGLite 原生只有 pdftext/Pandoc 的**文本**解析和一个基于 Mistral
云端 API 的 OCR 处理器；本地 OCR、VLM 描述、并发提取、类型标注、双路降级都是这次自研的。
