# WeKnow 完整教程（附源码）

> 本教程覆盖**本仓库**（`D:/2026/raglite-main`，代号 WeKnow）的全部能力：上游 RAGLite 1.1.1 的检索内核 + 本项目二次开发的 **8 项工程扩展**，每一节都给出 `src/raglite/` 里的真实源码摘录与可运行示例。
> 定位：读完能独立跑通「解析 → 切分 → 建库 → 增量 → 检索 → 重排 → 生成 → 评测」全链路，并说清每个设计决策与它的边界。
>
> 配套文档：`图增强检索-BEIR评测.md`（图增强的标准基准消融）、`图增强检索-AB评测.md`（自建题集的 A/B）、`复现指南-实测数据.md`（数字怎么来的）、`个人知识库RAG-简历.md`（简历口径）。
>
> **与 `RAGLite-完整教程.md` 的关系**：那份只覆盖上游 API 的快速上手，本教程是它的超集与替代，包含全部自研扩展与源码。

---

## 目录

- [0. 项目是什么](#0-项目是什么)
- [1. 安装与运行](#1-安装与运行)
- [2. 架构总览](#2-架构总览)
- [3. 五分钟最小示例](#3-五分钟最小示例)
- [4. 离线索引链](#4-离线索引链)
  - [4.1 解析：解析引擎注册表](#41-解析解析引擎注册表)
  - [4.2 多模态 OCR + VLM](#42-多模态-ocr--vlm)
  - [4.3 三层最优切分](#43-三层最优切分)
  - [4.4 多向量嵌入与混合建库](#44-多向量嵌入与混合建库)
  - [4.5 内容哈希增量索引](#45-内容哈希增量索引)
- [5. 在线检索链](#5-在线检索链)
  - [5.1 混合检索与 RRF](#51-混合检索与-rrf)
  - [5.2 查询意图路由](#52-查询意图路由)
  - [5.3 低置信度二次检索](#53-低置信度二次检索)
  - [5.4 分层缓存](#54-分层缓存)
  - [5.5 图增强检索](#55-图增强检索)
- [6. 生成与评测](#6-生成与评测)
- [7. 配置总表](#7-配置总表)
- [8. 性能修复与已知坑](#8-性能修复与已知坑)
- [9. 测试与代码质量](#9-测试与代码质量)
- [10. 附录：文件索引与复现命令](#10-附录文件索引与复现命令)

---

## 0. 项目是什么

WeKnow 是一个**进程内、可嵌入、零外部服务**的个人知识库 RAG 引擎：

- **检索内核**来自开源项目 [RAGLite](https://github.com/superlinear-ai/raglite) 1.1.1（DuckDB 单文件 + 原生 FTS/BM25 + VSS/HNSW）；
- **架构思路**参考腾讯 [WeKnora](https://github.com/Tencent/WeKnora)（文档解析引擎注册表、GraphRAG 的实体图/投影/查询期扩展）；
- **图增强的存储方案**参考 [LightRAG](https://github.com/HKUDS/LightRAG)（不引入 Neo4j，实体/关系直接落 DuckDB）。

一句话：**把异构文档解析、三层最优切分、混合检索、图增强、增量索引、评测全部收进一个单文件数据库 + 本地小模型就能跑通的引擎里。**

| 维度 | 选择 | 说明 |
|---|---|---|
| 向量库 | DuckDB VSS（HNSW） | 单文件、进程内、可嵌入；**没有独立向量数据库** |
| 全文检索 | DuckDB FTS（BM25） | 原生扩展 |
| 图存储 | 复用 DuckDB 表 | 无 Neo4j、无 APOC、零新服务 |
| Embedder | LiteLLM 任意 / llama.cpp 本地 | 默认 `bge-m3` GGUF |
| LLM | LiteLLM 任意 / llama.cpp 本地 | 默认 `Qwen3` GGUF |
| 重排 | FlashRank（Cross-Encoder） | `ms-marco-MiniLM-L-12-v2`（英）/ `MultiBERT`（其他） |
| 切分 | SaT ONNX + 动态规划 + 0-1 整数规划 | 三层最优切分 |
| 对外接口 | Python API + CLI + MCP Server | 可接 Claude Desktop |

### 8 项二次开发扩展一览

| # | 扩展 | 模块 | 解决的问题 |
|---|---|---|---|
| 1 | 解析引擎注册表 | `_parsers.py` | 28 类异构格式统一解析 + 外部引擎三级降级 |
| 2 | 多模态 OCR + VLM | `_multimodal.py` | 图片与扫描件里的文字与图表信息 |
| 3 | 低置信度二次检索 | `_iterative_search.py` | 首轮检索没把握时扩池重排 |
| 4 | 内容哈希增量索引 | `_incremental.py` | 只重建变更文档，避免全量重建 |
| 5 | 查询意图路由 | `_router.py` | 微秒级把查询分流到 BM25 / 混合 |
| 6 | 分层缓存 | `_cache.py` | L1 查询向量缓存 + TTL 结果缓存 |
| 7 | 生成端评估 | `_eval_generation.py` | 零依赖 ROUGE-L / BLEU |
| 8 | 图增强检索 | `_graph.py` | 实体图 + chunk 图投影作为 RRF 第三条加权路径 |

---

## 1. 安装与运行

### 1.1 用 pip（只用上游功能）

```bash
pip install raglite                 # 核心
pip install raglite[chainlit]       # 可选：ChatGPT 风格前端
pip install raglite[pandoc]         # 可选：pandoc 解析引擎
pip install raglite[ragas]          # 可选：RAGAS 评测
pip install 'raglite[multimodal]'   # 可选：OCR + VLM 多模态
```

### 1.2 本仓库（含扩展与基准脚本）

```bash
cd /d/2026/raglite-main
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple   # 清华源，避免官方源超时
uv sync --python 3.11                                             # 别用 3.13：numpy 1.26.4 无 wheel
```

> ⚠️ 本机 `.venv` 由 uv 创建（**没有 pip**），单独装包请用：
> ```bash
> UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv pip install <pkg>
> ```

### 1.3 两个必须知道的离线注意点

1. **`RAGLiteConfig` 的默认值会惰性导入 `llama_cpp`**。不显式传 `llm` / `embedder` 时，`default_factory` 触发 `import llama_cpp`，没装就 `ModuleNotFoundError`。**始终显式传字符串**（本仓库基准脚本都这么做）。
2. **默认 `reranker` 工厂会联网下载 MultiBERT 模型**。离线时务必传 `reranker=None`，否则首次构造直接 `ProxyError`。

---

## 2. 架构总览

### 2.1 数据流

```
┌──────────────────────────── 离线索引（离线/写入侧）────────────────────────────┐
  异构文档 ──▶ parse_document ──▶ document_to_markdown ──▶ split_sentences(SaT 神经句界)
  (PDF/DOCX/                     (ParserEngineRegistry)          │
   PPTX/XLSX/                                                     ▼
   EPUB/HTML/                                             split_chunklets(动态规划)
   CSV/图片...)                                                   │
                                                                  ▼
                                                        split_chunks(0-1 整数规划)
                                                                  │
                                                                  ▼
                                       embed_strings(多向量: late chunking + 标题增强)
                                                                  │
                                                                  ▼
                          DuckDB:  document / chunk / chunk_embedding(FLOAT[N])
                                   + FTS(BM25)  + VSS(HNSW)  + graph_* (可选图增强)
                                                                  │
                                    sync_documents(内容哈希比对，只重建变更) ◀── 增量入口
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────── 在线检索（在线/读取侧）────────────────────────────┐
   query ──▶ route_search_method(微秒级意图路由)                                  │
                │                                                                │
   ┌────────────┼─────────────┬──────────────┐                                   │
   ▼            ▼             ▼              ▼                                   │
 vector_search keyword_search graph_search ──▶ reciprocal_rank_fusion(RRF k=60)   │
 (VSS/HNSW)   (FTS/BM25)   (实体图+chunk图)  权重 0.75/0.25/0.1                  │
   └────────────┴─────────────┴──────────────┘         │                         │
                                                        ▼                         │
                              search_with_secondary_retrieval(低置信度二次检索)   │
                                                        │                         │
                                                        ▼                         │
                                   rerank_chunks_with_scores(FlashRank)           │
                                                        │                         │
                                                        ▼                         │
                              retrieve_chunk_spans(邻居扩展 + 连续分组)            │
                                                        │                         │
                                                        ▼                         │
                                    add_context ──▶ rag() ──▶ LLM 流式生成         │
                                                        │                         │
                                                        ▼                         │
                              evaluate_generation(ROUGE-L / BLEU)                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 模块地图

| 文件 | 职责 | 行数 |
|---|---|---|
| `_parsers.py` | **解析引擎注册表**：`(engine, file_type)→parser` + 三级回退 + 28 类格式 builtin | 1129 |
| `_multimodal.py` | **OCR + VLM** 双阶段图片/扫描件提取 | 672 |
| `_graph.py` | **图增强**：实体/关系抽取、PMI 加权、chunk 图投影、查询期扩散 | 1491 |
| `_search.py` | vector / keyword / hybrid / RRF / rerank / chunk span | 605 |
| `_config.py` | `RAGLiteConfig` / `GraphConfig` / `MultimodalConfig` | 277 |
| `_incremental.py` | **内容哈希增量索引**（`sync_documents` / `diff_documents`） | 216 |
| `_router.py` | **查询意图路由**（启发式，无 LLM） | 64 |
| `_cache.py` | **分层缓存**（TTL + 有界） | 116 |
| `_iterative_search.py` | **低置信度二次检索** | 121 |
| `_eval_generation.py` | **生成端评估**（ROUGE-L / BLEU，零依赖） | 139 |
| `_markdown.py` | 文档 → GitHub Flavored Markdown（接入注册表） | — |
| `_insert.py` | 入库编排（线程池 + 批量 flush） | 275 |
| `_database.py` | SQLModel 表定义 + 引擎创建（FTS/VSS 装载） | 745 |
| `_embed.py` | 多向量嵌入：late chunking + 标题增强 | 205 |
| `_rag.py` | `retrieve_context` / `add_context` / `rag` / `async_rag` | 590 |
| `_split_sentences.py` / `_split_chunklets.py` / `_split_chunks.py` | 三层切分 | 219 / 184 / 122 |

---

## 3. 五分钟最小示例

```python
from raglite import RAGLiteConfig, Document, insert_documents, rag

# 离线友好：显式给字符串模型标识 + 关掉会联网下载的重排器
config = RAGLiteConfig(
    db_url="duckdb:///weknow.db",
    llm="gpt-4o-mini",
    embedder="text-embedding-3-small",
    reranker=None,
)

docs = [Document.from_text(
    "# 狭义相对论\n光速不变，时间与空间随速度变化。E=mc^2。",
    author="Einstein", topic="physics", year=1905,
)]
insert_documents(docs, config=config)          # 自动完成：解析 → 切分 → 嵌入 → 建库

messages = [{"role": "user", "content": "E 和 mc^2 是什么关系？"}]
for update in rag(messages, config=config):     # 流式生成
    print(update, end="")
```

> ⚠️ 默认 `search_method` 是**纯向量** `_vector_search`，不是混合。要启用「混合 + 重排 + span」需显式切换（见 §5.1）。

---

## 4. 离线索引链

### 4.1 解析：解析引擎注册表

**目的**：把 28 类异构格式统一解析为 Markdown，并允许接入外部引擎（pandoc / markitdown / multimodal / mistral），且**任何外部引擎失败都回退到 builtin**——引擎是优化，不是正确性依赖。

**用法**：

```python
from pathlib import Path
from raglite import parse_document, parser_report, supported_file_types, RAGLiteConfig

# 1) 看每个引擎是否可用、覆盖哪些格式、缺依赖怎么装
import json; print(json.dumps(parser_report(), ensure_ascii=False, indent=2))

# 2) 看 builtin 覆盖的格式
print(sorted(supported_file_types()))

# 3) 解析单个文档（不指定 engine 时用 builtin）
parsed = parse_document(Path("report.docx"))
print(parsed.engine, parsed.file_type, len(parsed.content))
print([img.name for img in parsed.images])

# 4) 强制使用某个引擎（不可用/不支持/运行时报错 → 自动回退 builtin）
from raglite import MultimodalConfig
parsed = parse_document(Path("scan.pdf"), engine="multimodal",
                        config=RAGLiteConfig(document_processor=MultimodalConfig()))
```

**源码**（`src/raglite/_parsers.py`）—— 三级回退是本扩展的核心：

```python
class ParserEngineRegistry:
    """Map (engine, file type) pairs to parsers, falling back to the builtin engine."""

    def get_parser_class(self, engine: str | None, file_type: str) -> type[BaseParser]:
        """Return the parser class for (engine, file type), falling back to the builtin engine."""
        normalized = file_type.lower().lstrip(".")
        if engine and engine in self._engines and self.is_available(engine):
            parser_class = self._engines[engine].get(normalized)
            if parser_class is not None and parser_class.is_available():
                return parser_class
        builtin = self._engines.get(BUILTIN_ENGINE, {})
        parser_class = builtin.get(normalized)
        if parser_class is None:
            error_message = f"Unsupported file type: {file_type}"
            raise UnsupportedDocumentError(error_message)
        return parser_class

    def parse(self, doc_path, *, engine=None, config=None) -> ParsedDocument:
        """Parse `doc_path`, degrading to the builtin engine rather than failing."""
        requested = engine
        if requested and not self.is_available(requested):      # ① 引擎不可用 → 先降级
            requested = None
        parser = self.parser_for(doc_path, engine=requested)
        try:
            return parser.parse(doc_path, config=config)
        except (UnsupportedDocumentError, ParserFallbackError) as error:  # ② 不支持/运行失败
            if requested is None or requested == BUILTIN_ENGINE:
                raise
            fallback = self.parser_for(doc_path, engine=BUILTIN_ENGINE)   # 回退 builtin
            return fallback.parse(doc_path, config=config)
```

**关键设计**：
- builtin 覆盖 `md / pdf / html / csv / docx / pptx / xlsx / epub / 图片`，其中 **docx/pptx/xlsx/epub 用标准库 `zipfile` + `ElementTree` 直接解 OOXML/OCF 容器**，所以**零新增依赖**；
- `ImageParser` 会在没有多模态处理器时**抛错而不是把二进制图当文本读**（防止静默产出乱码）；
- 产物是 `ParsedDocument{content, images, engine, file_type}`，`_markdown.document_to_markdown` 根据 config 判断该用哪个引擎：

```python
def _preferred_parser_engine(doc_path: Path, config: RAGLiteConfig) -> str | None:
    if config.parser_engine:
        return config.parser_engine
    if isinstance(config.document_processor, MultimodalConfig) and is_multimodal_document(doc_path):
        return MULTIMODAL_ENGINE
    if isinstance(config.document_processor, MistralOCRConfig) and (
        doc_path.suffix.lower() in SUPPORTED_EXTENSIONS
    ):
        return MISTRAL_ENGINE
    return None   # None = 让注册表选，最终落到 builtin
```

### 4.2 多模态 OCR + VLM

**目的**：图片和扫描件里，OCR 拿得到「印在图像里的文字」，但拿不到「图表/示意图在讲什么」。用两个独立阶段互补，且都**可注入、可独立降级**。

**用法**：

```python
from raglite import RAGLiteConfig, MultimodalConfig, Document, insert_documents

# 全离线：RapidOCR（ONNX，本地）+ 不启用 VLM
cfg = RAGLiteConfig(
    db_url="duckdb:///weknow.db",
    llm="gpt-4o-mini", embedder="text-embedding-3-small", reranker=None,
    document_processor=MultimodalConfig(ocr_engine="auto", vlm=None),
)

# 带 VLM 描述（LiteLLM 视觉模型）
cfg = RAGLiteConfig(
    document_processor=MultimodalConfig(ocr_engine="auto", vlm="openai/gpt-4o-mini"),
)

docs = [Document.from_path("scanned-report.pdf")]
insert_documents(docs, config=cfg)     # PDF 会先逐页栅格化，再跑 OCR + VLM
```

**源码**（`src/raglite/_multimodal.py` 的引擎解析）—— 两阶段彼此独立降级：

```python
def resolve_ocr_engine(processor_config: MultimodalConfig) -> OCREngine:
    """Pick the OCR engine: explicit instance > "auto" first installed > none."""
    engine = processor_config.ocr_engine
    if isinstance(engine, OCREngine):        # 注入的测试替身/自定义实现
        return engine
    if engine is None or engine == "none":
        return NoOCREngine()
    # engine == "auto"
    if _module_is_available("rapidocr_onnxruntime"):
        return RapidOCREngine(languages=processor_config.ocr_languages)
    if _module_is_available("pytesseract"):
        return TesseractOCREngine(languages=processor_config.tesseract_languages)
    return NoOCREngine()                      # 都没有 → 静默降级为空 OCR


def resolve_vlm_engine(processor_config: MultimodalConfig) -> VLMEngine | None:
    vlm = processor_config.vlm
    if vlm is None:
        return None
    if isinstance(vlm, VLMEngine):
        return vlm
    return LiteLLMVLMEngine(model=vlm, ...)   # 字符串 → LiteLLM 视觉模型
```

**关键设计**：
- 默认提示词刻意要求「先逐字转写文本（弥补 OCR 漏字），再描述图表内容」，是**面向检索的标注**而非自由散文：

```python
DEFAULT_VLM_PROMPT = (
    "Transcribe all visible text in this image verbatim, then describe any charts, tables, "
    "diagrams, or other visual content. Be concise, factual, and complete. "
    "Answer in the same language as the text in the image."
)
```

- PDF 用 PyMuPDF 按 `pdf_dpi`（默认 200）逐页栅格化，`extraction_max_workers`（默认 4）并发提取；
- 两个引擎都通过 Protocol 注入，因此测试可完全离线（`StaticVLMEngine` / `NoOCREngine`）。

### 4.3 三层最优切分

**三层分别是**：

1. `split_sentences`——SaT（Segment any Text）**神经句界模型**（ONNX）预测句边界，再用 Markdown 标题边界约束；
2. `split_chunklets`——在句子上做**动态规划**最优子段划分；
3. `split_chunks`——以**去 discourse 向量的余弦相似度**为代价，用 `scipy.optimize.linprog` 解 **0-1 整数规划**做语义分块，Markdown 标题处加偏置。

**源码**（`_split_sentences.py`）—— 神经预测 + 已知标题边界覆盖：

```python
def split_sentences(doc, min_len=4, max_len=None,
                    boundary_probas: FloatVector | Callable = markdown_sentence_boundaries) -> list[str]:
    sat, sat_kwargs = _load_sat()
    predicted_probas = sat.predict_proba(doc, **sat_kwargs)           # 神经句界概率
    known_probas = boundary_probas(doc) if callable(boundary_probas) else boundary_probas
    probas = predicted_probas.copy()
    # Markdown 标题处已知的边界概率覆盖模型预测（NaN 处不覆盖）
    probas[np.isfinite(known_probas)] = known_probas[np.isfinite(known_probas)]
    ...
    probas[i : j - 1] = min_proba   # 空白字符永远跟在句尾，绝不留在句首
    probas[j - 1] = max_proba
    sentences = _split_sentences(doc, probas, min_len=min_len, max_len=None)
```

**源码**（`_split_chunks.py`）—— 代价函数是「去 discourse 向量后的余弦相似度」：

```python
# 去 discourse 向量：先剔除长度分位上的异常 chunklet，再减去均值方向
q15, q85 = np.quantile(chunklet_size, [0.15, 0.85])
nonoutlying_chunklets = (q15 <= chunklet_size) & (chunklet_size <= q85)
if np.any(nonoutlying_chunklets):
    discourse = np.mean(X[nonoutlying_chunklets, :], axis=0)
    ...
# 用 linprog 求解 0-1 整数规划，最小化「在何处切」的相似度代价
```

**实测**（96k 字符文档）：**SaT ~2.9s / chunklet DP 93ms / BIP 14ms**，714 句 → 198 chunklets → 58 chunks。
⚠️ SaT 段对机器负载极度敏感（空载 2.9s ↔ load=68 时 11.2s），引用该数字必须标注负载。

### 4.4 多向量嵌入与混合建库

**多向量表达**：每个 chunk 不只一条向量，而是多条子向量（**late chunking**：先对整段编码再按句切分 pooled 向量，并注入**上下文标题增强**，提升长文档局部匹配）。

**检索期聚合**：对同一 chunk 的多条子向量取 **L∞ 范数**（对子向量取 max 相似度）：

```python
# _search.py: 用 L∞ 范数把「多子向量」聚合成 chunk 级相似度
top_vectors = select(ChunkEmbedding.chunk_id, sim, dist).order_by(dist).limit(num_hits).subquery()
sim_norm = func.max(top_vectors.c.sim).label("sim_norm")     # max over 子向量
statement = (select(top_vectors.c.chunk_id, sim_norm)
             .group_by(top_vectors.c.chunk_id).order_by(sim_norm.desc()).limit(num_results))
```

**建库**：`documents` / `chunk` / `chunk_embedding(FLOAT[N])` 落在**同一个 DuckDB 文件**里；`chunk` 表上建 FTS(BM25) 索引，`chunk_embedding` 上建 VSS(HNSW) 索引。向量维度由 `ChunkEmbedding.set_embedding_dim()` 在运行时确定。

> ⚠️ **HNSW 命中与否取决于查询投影形状**（面试高频）。实测用真实 1536 维向量逐项 `EXPLAIN`：
> - 只投影距离 `dist` → `HNSW_INDEX_SCAN` ✅
> - 上游 `vector_search` 同时投影 `sim = 1.0 - dist` 与 `dist` → `SEQ_SCAN` ❌
>
> 即**投影列表里出现第二个距离表达式就会让规划器放弃 HNSW**。「索引存在 ≠ 索引被使用」。修法：去掉 `sim` 投影，改在 Python 侧 `1 - dist`。

### 4.5 内容哈希增量索引

**目的**：`Document.from_text` 默认用**内容哈希**做 `id`，所以内容变了 `id` 自动变。缺的是编排：比对数据库判断「新增 / 未变 / 变更」，删旧版、只插增量。

**用法**：

```python
from raglite import sync_documents, diff_documents

# 只比对、不动索引（返回 new / unchanged / changed）
report = diff_documents(docs, config=config)
print(report)

# 增量同步：按 filename 识别「同一文档的新版本」（热更新推荐）
report = sync_documents(docs, key="filename", config=config)
print(report["inserted"], report["unchanged"], report["deleted"])
# 图增强开启时，report["graph"] 会带回图层的增量构建报告
```

**源码**（`_incremental.py`）—— 核心判定与「任何索引变更都让缓存失效」：

```python
for document in documents:
    existing = existing_by_key.get(document_keys[document.id], [])
    if any(record.id == document.id for record in existing):
        report["unchanged"].append(document.id)      # 同内容哈希 → 跳过
        continue
    stale_document_ids.update(record.id for record in existing)  # 旧版本待删
    report["deleted"].extend(record.id for record in existing)
    documents_to_insert.append(document)
    report["inserted"].append(document.id)

if stale_document_ids:
    delete_documents(list(stale_document_ids), config=config, ...)
if documents_to_insert:
    insert_documents(documents_to_insert, max_workers=max_workers, config=config)

# 任何索引变更都让检索结果缓存失效
if report["inserted"] or report["deleted"]:
    get_retrieval_cache().clear()

# 图增强开启时，图层跟着同一份 delta 走
if config.graph.enabled and (report["inserted"] or report["deleted"]):
    prune_graph(config=config)
    report["graph"] = build_graph(document_ids=report["inserted"] or None, ..., config=config)
```

**关键设计**：`key` 决定「什么算同一文档」：
- `key="id"`：内容哈希即身份，变更文档看起来像新文档，需 `prune=True` 才删旧版；
- `key="filename"`（**热更新推荐**）：同名文件即同一文档，内容变了会被识别为「同一文档新哈希」，自动替换。

---

## 5. 在线检索链

### 5.1 混合检索与 RRF

**默认行为**：`RAGLiteConfig.search_method` 是**纯向量** `_vector_search`。要混合 + 重排 + span：

```python
from dataclasses import replace
from raglite import RAGLiteConfig, search_and_rerank_chunk_spans, hybrid_search

config = replace(config, search_method=search_and_rerank_chunk_spans)  # 检索+重排+span 一步到位
# 或仅混合检索（向量 + 关键词 + 可选图路）：
config = replace(config, search_method=hybrid_search)
```

**源码**（`_search.py`）—— 三条路径按权重融合，图路为空时等价于两路基线：

```python
def hybrid_search(query, *, num_results=3, oversample=2,
                  vector_search_weight=None, keyword_search_weight=None,
                  graph_search_weight=None, metadata_filter=None, config=None):
    config = config or RAGLiteConfig()
    vector_search_weight = config.hybrid_vector_weight if vector_search_weight is None else vector_search_weight
    keyword_search_weight = config.hybrid_keyword_weight if keyword_search_weight is None else keyword_search_weight
    graph_search_weight = config.hybrid_graph_weight if graph_search_weight is None else graph_search_weight

    vs_chunk_ids, _ = vector_search(query, num_results=oversample * num_results, ...)
    ks_chunk_ids, _ = keyword_search(query, num_results=oversample * num_results, ...)
    rankings, weights = [vs_chunk_ids, ks_chunk_ids], [vector_search_weight, keyword_search_weight]

    if config.graph.enabled:                                   # 第三条加权路径
        seen_chunk_ids = (list(dict.fromkeys(vs_chunk_ids + ks_chunk_ids))
                          if config.graph.deduplicate_seen_chunks else None)  # WeKnora filterSeenChunk
        gs_chunk_ids, _ = graph_search(query, num_results=oversample * num_results,
                                       exclude_chunk_ids=seen_chunk_ids, config=config)
        rankings.append(gs_chunk_ids); weights.append(graph_search_weight)

    chunk_ids, hybrid_score = reciprocal_rank_fusion(rankings, weights=weights)
    return chunk_ids[:num_results], hybrid_score[:num_results]
```

**RRF 融合**（`k=60`，权重可配，默认 0.75 / 0.25 / 0.1）：

```python
def reciprocal_rank_fusion(rankings, *, k=60, weights=None):
    if weights is None:
        weights = [1.0] * len(rankings)
    chunk_id_score = defaultdict(float)
    for ranking, weight in zip(rankings, weights, strict=True):
        for i, chunk_id in enumerate(ranking):
            chunk_id_score[chunk_id] += weight / (k + i)
    rrf_chunk_ids, rrf_score = zip(*sorted(chunk_id_score.items(), key=lambda x: x[1], reverse=True))
    return list(rrf_chunk_ids), list(rrf_score)
```

**重排降级**（`rerank_chunks_with_scores`）：重排器抛异常时**记录警告并退回检索顺序**，而不是让整次检索失败：

```python
if reranker:
    try:
        results = reranker.rank(query=query, docs=[str(chunk) for chunk in chunks])
    except Exception:
        logger.warning("Reranking failed; falling back to retrieval order.", exc_info=True)
    else:
        reranked_chunks = [chunks[result.doc_id] for result in results.results]
        rerank_scores = [float(result.score) for result in results.results]
        return reranked_chunks, rerank_scores
return chunks, None
```

### 5.2 查询意图路由

**目的**：短、关键词重（含引号短语、术语、编号）的查询更适合 BM25；自然语言问句才走混合。用**微秒级启发式**（无 LLM）分流。

**用法**：

```python
from dataclasses import replace
from raglite import route_search_method, routed_search, RAGLiteConfig

print(route_search_method('"exact phrase"').__name__)   # keyword_search
print(route_search_method("什么是图增强检索").__name__)  # hybrid_search

config = replace(config, search_router=True)             # 让管线自动用 routed_search
# 或直接把 routed_search 设为 search_method：replace(config, search_method=routed_search)
```

**源码**（`_router.py`）—— 三条启发式按序判定：

```python
def route_search_method(query: str, *, config=None) -> BasicSearchMethod:
    from raglite._search import hybrid_search, keyword_search
    if _QUOTED_PHRASE.search(query):          # ① 引号短语 = 精确匹配意图
        return keyword_search
    tokens = _tokenize(query)
    if not tokens:
        return hybrid_search
    stopword_ratio = sum(token in STOPWORDS for token in tokens) / len(tokens)
    if len(tokens) <= 6 and stopword_ratio <= 0.25:   # ② 短且虚词少 = 关键词型
        return keyword_search
    return hybrid_search                       # ③ 其余 = 混合
```

### 5.3 低置信度二次检索

**目的**：首轮重排后的**最高分低于置信阈值**，说明首轮候选池没抓住重点，于是**扩池（默认 2×）重排**，两轮结果按重排分合并取 top。

**用法**：

```python
from dataclasses import replace
from raglite import search_with_secondary_retrieval, RAGLiteConfig

config = replace(config, search_method=search_with_secondary_retrieval,
                 secondary_retrieval=True, rerank_confidence_threshold=0.3,
                 secondary_retrieval_multiplier=2)
```

**源码**（`_iterative_search.py`）—— 没有重排器时跳过（无置信信号可比）：

```python
chunk_ids, _ = search(query, num_results=oversample * num_results, ...)
reranked_chunks, rerank_scores = rerank_chunks_with_scores(query, chunk_ids, config=config)
ranked = {chunk.id: (rerank_scores[i] if rerank_scores else 0.0) for i, chunk in enumerate(reranked_chunks)}

if not config.secondary_retrieval or rerank_scores is None:   # 关闭 / 无信号 → 直接返回
    final_ids = list(ranked)[:num_results]
    return final_ids, [ranked[chunk_id] for chunk_id in final_ids]

top_scores = list(ranked.values())[:num_results]
if top_scores and max(top_scores) < config.rerank_confidence_threshold:   # 首轮不够自信
    expanded_ids, _ = search(query, num_results=oversample * num_results * config.secondary_retrieval_multiplier, ...)
    new_ids = [chunk_id for chunk_id in expanded_ids if chunk_id not in ranked]
    if new_ids:
        new_chunks, new_scores = rerank_chunks_with_scores(query, new_ids, config=config)
        for i, chunk in enumerate(new_chunks):
            ranked[chunk.id] = new_scores[i] if new_scores else 0.0

ranked_pairs = sorted(ranked.items(), key=lambda item: item[1], reverse=True)[:num_results]
return [chunk_id for chunk_id, _ in ranked_pairs], [score for _, score in ranked_pairs]
```

### 5.4 分层缓存

**目的**：**L1** 缓存「查询向量」（每查询最贵的前置步骤，且分页/重试/检索层 miss 会反复命中同一查询）；**L2** 缓存「检索结果」（带 TTL、有界、最久未用淘汰）。任何索引变更都会清空。

**L1 查询向量缓存**（`_search.py`）：

```python
QUERY_EMBEDDING_CACHE_MAX_SIZE = 128
_query_embedding_cache: dict[tuple[str, str], FloatVector] = {}

def _embed_query_cached(query: str, config: RAGLiteConfig) -> FloatVector:
    cache_key = (str(config.embedder), query)
    cached = _query_embedding_cache.get(cache_key)
    if cached is not None:
        return cached
    query_embedding = embed_strings([query], config=config)[0, :]
    if len(_query_embedding_cache) >= QUERY_EMBEDDING_CACHE_MAX_SIZE:
        _query_embedding_cache.pop(next(iter(_query_embedding_cache)))   # FIFO 淘汰
    _query_embedding_cache[cache_key] = query_embedding
    return query_embedding
```

**L2 结果缓存**（`_cache.py`）—— 键由「查询 + 参数 + 归一化后的元数据过滤」的 sha256 构成：

```python
@dataclass
class RetrievalCache:
    max_size: int = DEFAULT_CACHE_MAX_SIZE   # 256
    ttl: float = DEFAULT_CACHE_TTL           # 300s
    entries: dict[str, _CacheEntry] = field(default_factory=dict)

    def get(self, key: str):
        entry = self.entries.get(key)
        if entry is None:
            return None
        if time.monotonic() >= entry.expires_at:   # 过期即删
            del self.entries[key]
            return None
        return entry.value

    def set(self, key, value, *, ttl=None):
        if len(self.entries) >= self.max_size and key not in self.entries:
            oldest_key = min(self.entries, key=lambda k: self.entries[k].expires_at)  # LRU-by-expiry
            del self.entries[oldest_key]
        self.entries[key] = _CacheEntry(value=value, expires_at=time.monotonic() + (self.ttl if ttl is None else ttl))

    @staticmethod
    def make_key(query, *, num_results, method_name="hybrid_search", metadata_filter=None) -> str:
        normalized_filter = {k: sorted(v if isinstance(v, list) else [v])
                             for k, v in (metadata_filter or {}).items()}
        key_material = json.dumps({"query": query, "num_results": num_results,
                                   "method": method_name, "filter": normalized_filter},
                                  ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(key_material.encode()).hexdigest()
```

用法：`replace(config, search_method=cached_hybrid_search)`，或直接调 `get_retrieval_cache().clear()` 手动失效。

### 5.5 图增强检索

这是本仓库最重的一项扩展，也是**唯一一项默认关闭**的扩展。

**存储方案**：**不引入 Neo4j，不引入新依赖**——实体/关系/chunk 投影全部复用既有 DuckDB 表（`graph_entity` / `graph_relation` / `graph_entity_chunk` / `graph_chunk_relation` / `graph_chunk`），作为 RRF 的**第三条加权路径**参与重排前融合。

**用法**：

```python
from dataclasses import replace
from raglite import RAGLiteConfig, GraphConfig, build_graph, graph_search, query_knowledge_graph

config = replace(config, graph=GraphConfig(
    enabled=True,
    extractor=None,                 # None = 用 RAGLite 的 LLM 配置构建 LLM 抽取器
    pmi_weight=0.6, strength_weight=0.4, weight_scale=9.0,   # 边权 PMI 0.6 + strength 0.4，缩放到 1-10
    build_chunk_graph=True,         # 实体图 → chunk 图投影
    chunk_neighbor_limit=32,        # hub 实体最多追踪的 chunk 邻居数
))

build_graph(config=config)                  # 建图：抽取 → 写实体/关系 → PMI 加权 → chunk 图投影
ids, scores = graph_search("图增强怎么接进 RRF？", num_results=10, config=config)
result = query_knowledge_graph("图增强怎么接进 RRF？", config=config)   # 面向 agent 的工具
print(result.entities, result.relations)
```

**1) 边权 = PMI 与 strength 归一化后缩放到 1–10**（`_graph.py`）：

```python
def pointwise_mutual_information(joint_probability, source_probability, target_probability) -> float:
    """WeKnora 的 PMI: log2(P(x,y) / (P(x)*P(y)))，下钳到 0。"""
    if joint_probability <= 0 or source_probability <= 0 or target_probability <= 0:
        return 0.0
    return max(math.log2(joint_probability / (source_probability * target_probability)), 0.0)

def recompute_relation_weights(*, config=None) -> int:
    """weight = 1 + weight_scale * (pmi_weight * 归一化PMI + strength_weight * 归一化strength)"""
    ...
    combined = graph_config.pmi_weight * normalized_pmi + graph_config.strength_weight * normalized_strength
    scored_rows.append((source_id, target_id, 1.0 + graph_config.weight_scale * combined))
```

**2) 实体图 → chunk 图投影**（`build_chunk_relations`）：共享实体的两个 chunk 双向连边，权重累加关系权重，并记录端点实体度（`degree`）：

```python
for source_id, target_id, weight in relation_rows:
    source_chunks = chunks_by_entity.get(source_id, [])
    target_chunks = chunks_by_entity.get(target_id, [])
    degree = len(source_chunks) + len(target_chunks)
    for source_chunk_id in source_chunks[: graph_config.max_chunks_per_entity]:
        for target_chunk_id in target_chunks[: graph_config.max_chunks_per_entity]:
            if source_chunk_id == target_chunk_id:
                continue
            for key in ((source_chunk_id, target_chunk_id), (target_chunk_id, source_chunk_id)):
                stored_weight, stored_degree = edges.get(key, (0.0, 0))
                edges[key] = (stored_weight + weight, max(stored_degree, degree))
```

**3) 查询期：匹配种子实体 → 沿关系扩散 → 打分 chunk → 沿 chunk 图扩散 → 去重**（`graph_search`）：

```python
seed_ids = match_seed_entities(query, entities, graph_config)
if graph_config.query_extraction:                       # 可选：查询期也用抽取器抽实体
    seed_ids = list(dict.fromkeys(seed_ids + extract_query_entities(query, entities, config=config)))
seed_ids = seed_ids[: graph_config.seed_entity_limit]

entity_weights = dict.fromkeys(seed_ids, 1.0)           # 沿关系逐跳衰减
frontier = list(seed_ids)
for hop in range(max(0, graph_config.neighbor_hops)):
    decay = graph_config.neighbor_weight ** (hop + 1)
    for relation in relations:
        if relation.target_id in entity_weights or decay <= 0:
            continue
        entity_weights[relation.target_id] = decay
        next_frontier.append(relation.target_id)

chunk_scores = defaultdict(float)
for entity_id, chunk_id in links:
    chunk_scores[chunk_id] += entity_weights[entity_id]

scores = _expand_chunk_neighbors(dict(chunk_scores), session, graph_config)  # chunk 图跳数扩散
excluded = set(exclude_chunk_ids or ())                 # WeKnora filterSeenChunk 去重
ordered = [c for c in sorted(scores, key=lambda c: (-scores[c], c)) if c not in excluded]
```

**4) 为什么默认关闭**：项目对图增强做了标准基准（BEIR NFCorpus + SciFact）六臂消融，**结论不稳定、跨数据集不可移植**，所以 `GraphConfig.enabled=False` 是默认，且任何文档都不写图增强的「提升数字」。详见 §6.3。

---

## 6. 生成与评测

### 6.1 RAG 生成

**自适应**（让 LLM 自己决定检索与否，最多 `agentic_iterations` 次）：

```python
from raglite import rag
messages = [{"role": "user", "content": "狭义相对论讲了什么？"}]
chunk_spans = []
for update in rag(messages, on_retrieval=lambda x: chunk_spans.extend(x), config=config):
    print(update, end="")
```

**可编程**（手动控制每一步）：

```python
from raglite import retrieve_context, add_context, rag
chunk_spans = retrieve_context(user_prompt, num_chunks=5, metadata_filter={"author": "Einstein"}, config=config)
messages = [add_context(user_prompt, context=chunk_spans, config=config)]
for update in rag(messages, config=config):
    print(update, end="")
```

### 6.2 生成端评估：ROUGE-L / BLEU（零依赖）

**目的**：不给生成端加 `ragas` 这类重依赖，用标准库实现 **token 级、对 CJK 友好**的 ROUGE-L 与 BLEU。

**源码**（`_eval_generation.py`）—— CJK 逐字切分 + 最长公共子序列：

```python
_TOKEN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")     # 拉丁字母数字词 + 单个汉字

def rouge_l_f1(reference: str, hypothesis: str) -> float:
    reference_tokens, hypothesis_tokens = _tokenize(reference), _tokenize(hypothesis)
    if not reference_tokens or not hypothesis_tokens:
        return 0.0
    lcs = _lcs_length(reference_tokens, hypothesis_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(hypothesis_tokens)
    recall = lcs / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)

def bleu_n(reference: str, hypothesis: str, *, max_n: int = 4) -> float:
    """BLEU，add-one 平滑，带 brevity penalty。"""
    ...
    bp = min(0.0, 1 - len(reference_tokens) / len(hypothesis_tokens))
    return math.exp(bp + sum(log_precisions) / max_n)
```

用法（可传预生成答案，也可给 config 让它现场生成）：

```python
from raglite import evaluate_generation, rouge_l_f1, bleu_n
report = evaluate_generation(
    [("E=mc^2 的含义？", "质能等价，质量与能量可相互转换。")],
    answers=["质量与能量等价。"],   # 省略则需 config 里的 LLM 现场生成
)
print(report["mean_rouge_l"], report["mean_bleu"])
```

> ⚠️ 诚实口径：`_eval_generation.py` **有代码与单测，但仓库里没有真实分数产物**。简历/文档不得声称「已用 ROUGE-L/BLEU 得到 XX 分」。

### 6.3 检索端评测：BEIR 两数据集消融（主证据）

**方法**：用 `ir_datasets` 取标准基准，`ir_measures` 算指标（RR@10 / nDCG@10 / nDCG@100 / R@100），**配对 bootstrap（10000 次重采样，seed 20260929，95% CI）** 做显著性检验。评测脚本 `bench_beir_graph.py` 跑六臂：

| 臂 | 含义 |
|---|---|
| `bm25` | 纯 BM25 基线 |
| `rrf_w0.1` / `rrf_w0.3` / `rrf_w1` | 图路 RRF 权重 0.1 / 0.3 / 1.0 |
| `rrf_w0.3_dedup` | 权重 0.3 + `filterSeenChunk` 去重 |
| `graph_only` | 只用图路 |

**基线可信度**：脚本 `bm25` 臂与独立的纯 BM25 脚本**四位小数一致**（NFCorpus 0.3181 / SciFact 0.6730），并对齐 Anserini flat（NFCorpus 0.3218、SciFact 0.673）。

**结果（nDCG@10）**：

| 臂 | NFCorpus（3633 文档 / 323 查询） | SciFact（5183 文档 / 300 查询） |
|---|---|---|
| bm25 | **0.3181** | **0.6730** |
| rrf_w0.1 | 0.3172 | 0.6329 |
| rrf_w0.3 | 0.3052 | 0.5511 |
| rrf_w1 | 0.2503 | 0.3628 |
| rrf_w0.3_dedup | 0.3192 | 0.6730（= bm25，逐位相同） |
| graph_only | 0.1459 | 0.1105 |

**唯一稳定的结论**：
1. NFCorpus 上图路最多「打平」（R@100 有微小提升但 nDCG@10 略降）；
2. SciFact 上 **w=0.1 就已经显著掉精度**（nDCG@10 −0.0401），召回提升**不显著**（CI 触 0）；
3. `dedup` 在 SciFact 上把图路变成**恒等变换**（所有 delta 都是 `0.0000 [0,0]`）——因为它没带来新 chunk；
4. 因此**图路增益高度依赖语料与挂载的图**，跨数据集不可移植 → **默认关闭**。

> 完整报告与逐项 CI 见 `图增强检索-BEIR评测.md`。自建 16 题 A/B（`bench_graph_ab.py`）结论也是「单跳类三臂逐位相同、关系类净打平」，见 `图增强检索-AB评测.md`。
>
> 数字纪律：`bench_results.json` 是唯一落盘的延迟来源；`vector_search_top20_ms` 的 6273ms 是**连接级索引加载伪影，不可引用**；「rerank 预热后 85.8ms」在仓库里**找不到产物**，已弃用。

### 6.4 延迟基准

| 指标 | 实测值 | 口径 |
|---|---|---|
| 切分全链路 | ~3.0s（SaT 2.9 / DP 0.09 / BIP 0.014） | 96k 字符，mock embedding |
| HNSW 构建 | 26.6s（文件库）/ 14.0s（内存） | 10k×1536 |
| FlashRank 重排 | P50 923.1ms / P95 1042.5ms | 20 候选，**未做 warmup 剥离** |
| BM25 top-20 | P50 17.5ms / P95 25.5ms | 50 次真实查询 |
| RRF 融合 | P50 0.01ms | 200 次 |
| chunk span 组装 | P50 57.6ms | 8 命中 + 邻居 |

> 引用 923ms 时必须同时说明「未剥离 warmup」；本机负载不稳定（同脚本两次差 3.5–15 倍），重测需在空载机器上进行。

---

## 7. 配置总表

### `RAGLiteConfig`（检索/生成相关）

| 字段 | 默认 | 含义 |
|---|---|---|
| `db_url` | `duckdb:///~/.cache/raglite/raglite.db` | DuckDB / PostgreSQL 连接串 |
| `llm` / `embedder` | 本地 Qwen3 / bge-m3 GGUF | **务必显式传字符串**，避免惰性导入 llama_cpp |
| `chunk_max_size` | 2048 | 单 chunk 最大字符数 |
| `document_processor` | `None` | `MultimodalConfig` 或 `MistralOCRConfig` |
| `parser_engine` | `None` | 偏好引擎；`None` 让注册表选（落到 builtin） |
| `vector_search_multivector` | `True` | 多向量（late chunking） |
| `vector_search_query_adapter` | `True` | Query Adapter 修正 query 向量 |
| `search_method` | `_vector_search` | **默认纯向量**，非混合 |
| `hybrid_vector_weight` / `hybrid_keyword_weight` / `hybrid_graph_weight` | 0.75 / 0.25 / 0.1 | RRF 三路权重 |
| `secondary_retrieval` | `True` | 低置信度二次检索总开关 |
| `rerank_confidence_threshold` | 0.3 | 触发二次检索的重排分阈值 |
| `secondary_retrieval_multiplier` | 2 | 二次检索扩池倍数 |
| `search_router` | `False` | 启用查询意图路由 |
| `retrieval_cache_ttl` | 300.0 | L2 结果缓存 TTL（秒） |
| `retrieval_cache_max_size` | 256 | L2 结果缓存条目上限 |

### `GraphConfig`（图增强，默认关闭）

| 字段 | 默认 | 含义 |
|---|---|---|
| `enabled` | `False` | 图路是否参与混合检索 |
| `extractor` | `None` | `None`→LLM 抽取器；字符串→LiteLLM；实例→直接用 |
| `pmi_weight` / `strength_weight` / `weight_scale` | 0.6 / 0.4 / 9.0 | 边权公式，落在 1–10 |
| `build_chunk_graph` | `True` | 是否投影到 chunk 图 |
| `max_chunks_per_entity` | 64 | 单个实体最多贡献的 chunk 数（防 hub 爆炸） |
| `chunk_neighbor_hops` / `chunk_neighbor_weight` / `chunk_neighbor_limit` | 1 / 0.5 / 32 | chunk 图扩散 |
| `seed_entity_limit` / `neighbor_hops` / `neighbor_weight` | 8 / 1 / 0.5 | 实体扩散 |
| `query_extraction` | `False` | 查询期是否也调抽取器 |
| `deduplicate_seen_chunks` | `False` | WeKnora `filterSeenChunk` 去重 |

### `MultimodalConfig`（OCR + VLM）

| 字段 | 默认 | 含义 |
|---|---|---|
| `ocr_engine` | `None` | `"auto"` 首个已装引擎；`"none"` 关闭；实例直接用 |
| `ocr_languages` / `tesseract_languages` | `("ch","en")` / `("eng",)` | 语言包 |
| `vlm` | `None` | LiteLLM 模型标识 / 实例 / `None` 跳过 |
| `vlm_prompt` | 见 §4.2 | 面向检索的转写+描述提示词 |
| `vlm_max_tokens` / `vlm_temperature` / `vlm_num_retries` | 512 / 0.0 / 3 | VLM 参数 |
| `pdf_dpi` / `max_pdf_pages` | 200 / `None` | PDF 栅格化 |
| `extraction_max_workers` | 4 | 页级并发 |

---

## 8. 性能修复与已知坑

1. **HNSW 索引未被查询计划命中**（§4.4）：投影里多一个距离表达式就退化成顺序扫描。10k×1536 下精确扫描 ~50ms、召回 100%，ANN 收益要到 10⁵–10⁶ 才显著。
2. **`insert_documents` 入库慢 3 个数量级**：SQLAlchemy 逐行 flush + `FLOAT[1536]` 逐行参数绑定，单文档约 480s。修复路径：**Parquet 列式批量导入**（`fill_and_search.py` 已验证 10k 向量约 2.5 分钟）。
3. **所有「参数化多行插入」在 DuckDB 上是二次的**：图相关的批量写入统一走 `_staged_rows`——stdlib `csv` → `io.StringIO` → `read_csv` → 视图 → 集合式 `INSERT … SELECT`。
4. **`vector_search` 端到端 6.3s 是伪影**：连接级 HNSW 索引加载；固定单连接后稳态 46.5ms。**这个数字不能写进简历。**
5. **config 默认值惰性导入 `llama_cpp`**（§1.3 坑 1）。
6. **默认 reranker 工厂联网下载 MultiBERT**（§1.3 坑 2）；离线必须传 `reranker=None`。
7. **SaT 段负载敏感**（§4.3）：引用 2.9s 必须标注负载。
8. **DuckDB 同进程不能二次 attach 同一文件**；诊断脚本猴补丁 `create_database_engine` 会跳过 `set_embedding_dim()` → `dim=-1` → `FLOAT[-1]` 语法错。

---

## 9. 测试与代码质量

```bash
uv run poe lint     # ruff 检查 + mypy
uv run poe test     # pytest
```

- 自研扩展的**单测合计 100/100 通过**，`mypy` 干净；
- `tests/test_parsers.py`（+19）/ `tests/test_graph.py`（+33+10）/ `tests/test_multimodal.py` / `tests/test_retrieval_extensions.py` 覆盖 8 项扩展；
- 代码风格遵循 `AGENTS.md`：**S1 优先完整单词**（`max_capacity` 而非 `max_cap`）、**S2 优先内联而非中间变量**（有 ≥3 次引用的中间量除外）。

> ⚠️ 本机未装 `rapidocr-onnxruntime` / `pytesseract` / `llama-cpp-python`，所以 `_markdown`、`_extract`、`_split_sentences` 的部分测试在本机会失败——属**既有环境缺失**，非代码问题。

---

## 10. 附录：文件索引与复现命令

### 关键文件

| 文件 | 作用 |
|---|---|
| `src/raglite/_parsers.py` | 解析引擎注册表（扩展 #1） |
| `src/raglite/_multimodal.py` | OCR + VLM（扩展 #2） |
| `src/raglite/_iterative_search.py` | 二次检索（扩展 #3） |
| `src/raglite/_incremental.py` | 增量索引（扩展 #4） |
| `src/raglite/_router.py` | 意图路由（扩展 #5） |
| `src/raglite/_cache.py` | 分层缓存（扩展 #6） |
| `src/raglite/_eval_generation.py` | 生成端评估（扩展 #7） |
| `src/raglite/_graph.py` | 图增强（扩展 #8） |
| `bench_beir_graph.py` | BEIR 六臂消融 |
| `bench_graph_ab.py` | 自建题集 A/B |
| `bench_results.json` | 唯一落盘延迟来源 |

### 复现命令

```bash
# 标准基准图增强消融（两数据集，各 ~1h；写 JSON 后才打印，可中断续跑）
UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv run python bench_beir_graph.py

# 自建题集 A/B
uv run python bench_graph_ab.py

# 入库 10k 向量 + 检索延迟
uv run python fill_and_search.py
```

### 相关文档

- `图增强检索-BEIR评测.md`——图增强的标准基准消融（**主证据**，含逐项 CI）
- `图增强检索-AB评测.md`——自建 16 题 A/B
- `文档解析与GraphRAG-对齐WeKnora.md`——与 WeKnora 的差距→实现→验证
- `WeKnora对比-优化清单.md`——扩展清单与对齐进度
- `多模态扩展-OCR+VLM.md`——多模态设计与用法
- `复现指南-实测数据.md`——所有数字怎么来的

---

*本教程所有源码摘录均取自 `src/raglite/` 当前实现；所有性能/效果数字均标注口径，未取数的一律标注「未取数」，不做未经验证的声明。*
