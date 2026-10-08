# 文档解析与 GraphRAG：对齐 WeKnora

> 日期：2026-09-29
> 参考：WeKnora 源码解析教程（`docreader/` 解析层 + `graph.go` / `search_entity.go` 的 GraphRAG），WeKnora v0.7.x / v0.8.2
> 目标：把 RAGLite（`D:\2026\raglite-main`）的**文档解析**与**图检索**两块，按 WeKnora 的模块设计补齐
> 结论先行：**解析层已完整对齐（13 类格式 + 引擎路由 + builtin 回退）；GraphRAG 在"不改存储栈"的前提下补齐了六项机制，仍默认关闭**

---

## 一、差距总览

| 能力 | WeKnora | 对齐前的 RAGLite | 本次实现后 |
|---|---|---|---|
| 格式支持 | 13 类，`ParserEngineRegistry` 路由 | PDF + Markdown/TXT（pdftext + pandoc 兜底） | **28 类 builtin 可解析**，注册表路由 |
| 解析引擎 | builtin / markitdown / opendataloader 三引擎互补 | 无引擎概念，按后缀 if-else 分支 | builtin / pandoc / markitdown / multimodal / mistral 五引擎 |
| 引擎降级 | 请求引擎不可用 → 回退 builtin | 无 | 三级回退（无引擎 → 未安装 → 运行时抛错），均 `logger.warning` |
| 解析产物 | `Document{content, images}` | 只返回 str，图片丢失 | `ParsedDocument{content, images, metadata, engine, file_type}` |
| 边权 | `0.6*PMI + 0.4*Strength`，缩放 1–10 | 关系只记 `weight=1.0` | `recompute_relation_weights()` 同公式 |
| 图投影 | 实体图 → chunk-chunk 图（`buildChunkGraph`） | 无 | `GraphChunkRelation` 表 + `build_chunk_relations()` |
| 检索期扩展 | `PluginExtractEntity` + `PluginSearchEntity` | 仅词面匹配实体名 | `extract_query_entities()` + 跳数扩展 |
| 去重 | `filterSeenChunk` | 无 | `graph_search(exclude_chunk_ids=...)` |
| 图查询工具 | `query_knowledge_graph` agent 工具 | 无 | `query_knowledge_graph()` |
| 抽取约束 | `ExtractConfig`（实体/关系类型 + 示例注入 prompt） | 固定 prompt | `entity_types / relation_types / extraction_examples` |

---

## 二、文档解析：从 `if suffix == ...` 到 `ParserEngineRegistry`

### 2.1 对齐前的形态

`_markdown.document_to_markdown()` 里是一串后缀分支：`.pdf` 走 `pdftext`，`.md/.txt` 直接 `read_text()`，其余交给 `pypandoc`（没装 pandoc 就抛错）。问题有三个：**加一种格式要改主流程**、**引擎可用性没有探测与提示**、**图片在解析阶段就被丢弃**（多模态扩展只能在 `document_processor` 里另起炉灶）。

### 2.2 实现：`src/weknow/_parsers.py`（新建，1129 行）

WeKnora 的 `docreader/parser/registry.py` 是 `(engine, file_type) -> parser class` 的二维字典 + 回退。RAGLite 侧照此建模：

```python
class BaseParser:
    ENGINE: ClassVar[str]
    FILE_TYPES: ClassVar[frozenset[str]]
    UNAVAILABLE_HINT: ClassVar[str] = ""
    needs_config: ClassVar[bool] = False

    @classmethod
    def is_available(cls) -> bool: ...

    def parse(self, doc_path: Path, *, config: RAGLiteConfig | None = None) -> ParsedDocument: ...
```

注册表 `ParserEngineRegistry.get_parser_class(engine, file_type)` 的三级语义：

| 情况 | 行为 |
|---|---|
| 请求的引擎支持该类型且可用 | 返回该解析器 |
| 引擎未安装 / 不支持该类型 / `is_available()` 为假 | 回退 builtin |
| `parse()` 抛 `ParserFallbackError` / `UnsupportedDocumentError` | 回退 builtin + `logger.warning` |
| builtin 也不支持该类型 | 抛 `UnsupportedDocumentError` |

内置解析器（`BUILTIN_ENGINE = "builtin"`）：

| 解析器 | 覆盖 | 依赖 |
|---|---|---|
| `MarkdownParser` | md / markdown / mdown / txt / text / log / rst / org | 标准库 |
| `PDFParser` | pdf | `pdftext`（原 `_markdown` 逻辑迁入） |
| `HtmlParser` | html / htm / xhtml / mhtml | 标准库 `HTMLParser`（保留 h1–h6/列表/表格/链接/图片，丢弃 script/style/svg） |
| `CSVParser` | csv / tsv | 标准库 `csv` → Markdown 表格 |
| `DocxParser` | docx | 标准库 `zipfile` + `ElementTree`（OOXML：标题/段落/列表/表格 + 内嵌图） |
| `PPTXParser` | pptx | 同上（逐 slide 抽取文本） |
| `XLSXParser` | xlsx / xlsm | 同上（`sharedStrings.xml` + 逐 sheet） |
| `EPUBParser` | epub | 同上（OCF：`opf` spine 顺序 + Dublin Core 标题 + 内嵌图） |
| `ImageParser` | png/jpg/jpeg/webp/bmp/tif/tiff/gif | 走 `document_processor`（OCR/VLM） |

可选引擎：`PandocParser`（doc/docx/odt/rtf/epub/latex/ipynb/wiki…）、`MarkItDownParser`、`MultimodalParser`（图片 + 扫描 PDF）、`MistralParser`（Mistral OCR）。

**零新增依赖**是硬约束：docx/pptx/xlsx/epub 全部用标准库解 OOXML/OCF 容器（`zipfile` + `ElementTree`），不引入 `python-docx` / `openpyxl` / `ebooklib`。图片上限 `MAX_PARSED_IMAGES = 32`，避免一个 200 页 PDF 把内存吃穿。

### 2.3 接入方式

`document_to_markdown()` 现在只剩三行决策：

```python
engine = None if config is None else _preferred_parser_engine(doc_path, config)
return parse_document(doc_path, engine=engine, config=config).content
```

`_preferred_parser_engine()` 的优先级：`config.parser_engine`（显式指定）→ 配了 `MultimodalConfig` 且是多模态文档 → `MULTIMODAL_ENGINE` → 配了 `MistralOCRConfig` 且后缀在其 `SUPPORTED_EXTENSIONS` → `MISTRAL_ENGINE` → `None`（走 builtin）。**既有行为完全不变**：没配任何 OCR/VLM 的用户拿到的还是原来的 PDF/Markdown 路径。

新增的公开 API：`parse_document`、`default_registry`、`parser_report`、`supported_file_types`、`ParsedDocument`、`ParsedImage`、`BaseParser`、`ParserEngineRegistry`。`parser_report()` 可直接打印"哪个引擎可用、支持哪些类型、缺什么包"：

```
builtin     ✔ 28 类   -
multimodal  ✔ 9 类    （已装 PyMuPDF；完整能力需 pip install 'weknow[multimodal]'）
pandoc      ✘ 17 类   pip install 'weknow[pandoc]'
markitdown  ✘ 12 类   pip install markitdown
mistral     ✘ 10 类   需要 mistral-ocr extra + MISTRAL_API_KEY
```

---

## 三、GraphRAG：在 DuckDB 上补齐 WeKnora 的六件事

**前提（不改）**：存储仍跟随 LightRAG —— 实体/关系落在既有 DuckDB 表（`graph_entity` / `graph_entity_chunk` / `graph_relation` / `graph_chunk_relation`），**不引入 Neo4j、不引入 APOC、不新增服务**。WeKnora 的 GraphRAG 强依赖 Neo4j + APOC 且默认关闭，对一个 pip 装完即用的嵌入式引擎来说是硬伤；所以对齐的是它的**加权、投影、查询期扩展**这三层逻辑，而不是它的存储。

### 3.1 边权：`0.6 × PMI + 0.4 × Strength`（WeKnora `graph.go:534/548`）

| 步骤 | 实现 |
|---|---|
| 抽取器输出强度 | `ExtractedRelation.strength`（1–10），`normalized_strength()` clamp 到 0–1 |
| 强度累计 | `_write_graph` 改为累加 `observations` 与**均值**强度（增量建图不再覆盖旧值） |
| PMI | `pointwise_mutual_information(joint, source, target) = log2(P(x,y) / (P(x)·P(y)))` |
| 组合缩放 | `recompute_relation_weights()`：`weight = 1 + 9 × (0.6·norm_pmi + 0.4·norm_strength)` → 落在 1–10 |

`pmi_weight` / `strength_weight` / `weight_scale` 全部是 `GraphConfig` 字段，可调参。`build_graph()` 的返回值多了 `relations_scored`。

### 3.2 实体图 → chunk 图投影（WeKnora `buildChunkGraph`）

新增表 `GraphChunkRelation(source_chunk_id, target_chunk_id, weight, degree)`，由 `build_chunk_relations()` 全量重建（先 `sa_delete` 再写，双向存边）：两个 chunk 只要共享任一实体就连边，`weight` 累加它们之间所有实体关系的权重。查询期 `_expand_chunk_neighbors()` 按 `chunk_neighbor_hops` 跳数衰减（`chunk_neighbor_weight`）扩散分数——**这正是"chunk A 提到查询实体、chunk B 只提到相关实体也能被召回"的那条通路**。

### 3.3 查询期实体抽取（WeKnora `PluginExtractEntity`）

`extract_query_entities(query, entities, *, config)`：词面匹配之外，可选调用抽取器从 query 里抽实体（对应 `GraphConfig.query_extraction`），失败只 `warning` 返回空列表，**绝不因抽取失败拖垮检索**。种子集合 = 词面匹配 ∪ 抽取结果。

### 3.4 `filterSeenChunk` 去重（WeKnora `search_entity.go:42`）

`graph_search()` 新增 `exclude_chunk_ids` 参数；`hybrid_search()` 在 `GraphConfig.deduplicate_seen_chunks=True` 时把向量/BM25 已召回的 chunk 传进去，让第三路**只贡献新信息**而不是把前两路的结果复读一遍。默认 `False`（保持既有 A/B 结论的行为不变）。

### 3.5 `query_knowledge_graph()`（WeKnora 同名 agent 工具）

返回 `GraphQueryResult{query, entities, relations, chunk_ids}`，支持 `max_entities` 与 `neighbor_hops`。这是给 agent / 前端画图用的探索接口，**不参与检索路径**。

### 3.6 `ExtractConfig` 约束注入（WeKnora `ExtractConfig`）

`build_extraction_prompt()` 把 `entity_types` / `relation_types` / `extraction_examples` 追加到 prompt 末尾；默认三者为空（让模型自选），因为过度约束的 schema 会静默丢掉个人语料里真正重要的实体。

---

## 四、验证口径（全部实测，非估计）

| 项目 | 结果 |
|---|---|
| `tests/test_parsers.py` | **19 passed** |
| `tests/test_graph.py` | **43 passed**（含 10 个新增的 WeKnora 对齐用例） |
| 离线回归子集（parsers/graph/markdown/multimodal/mistral_ocr/metadata_filter/retrieval_extensions/split/import/lazy_llama） | **124 passed, 1 skipped** |
| `mypy src/weknow` | **Success: no issues found in 33 source files** |
| `ruff check`（本次改动的 9 个文件） | 与 HEAD 基线对比**无新增规则类别**（基线 52 → 当前 49，仅魔法值类略有增加，与既有 `_graph.py` 风格一致） |

`test_parsers.py` 的 fixture 是**真实的最小 OOXML/OCF 容器**（内联 1×1 PNG），不是 mock：docx 测标题/列表/表格/内嵌图，xlsx 测逐 sheet + sharedStrings，pptx 测逐 slide，epub 测 spine 顺序 + Dublin Core 标题，pdf 用仓库自带的 `specrel.pdf` 断言起首是 `# ON THE ELECTRODYNAMICS OF MOVING`。注册表回退三条路径（引擎无此类型 / 引擎未安装 / 引擎运行时抛 `ParserFallbackError`）各有独立用例。

**未在本次验证的环境限制**（既有问题，非本次改动引入）：`test_database` / `test_delete` / `test_query_adapter` / `test_extract` 需要联网下载模型或 `llama-cpp-python`，`test_split_sentences` 需要下载 SaT 模型——本机无外网/未装该包。

---

## 五、仍未对齐的部分（诚实清单）

1. **社区检测与全局检索**：WeKnora 自述是"轻量版 GraphRAG"，没有 Leiden 社区与 global search；RAGLite 同样没有。这是双方共同的缺口，不是差距。
2. **抽取质量**：PMI/强度/投影这些机制已就位，但**效果仍取决于抽取器**。之前三臂 A/B 的结论（单跳零退化、关系类净打平）用的是术语表 oracle 与离线共现；**用真实 LLM 抽取器重跑 A/B 仍未完成**，阻塞在 LLM 端点。因此：
   - `GraphConfig.enabled` **保持默认 `False`**；
   - 简历与文档中**不写任何图增强的效果数字**。
3. **解析层未做的**：`.doc`（旧二进制）/ `.xls` / `.pages` 等只由 pandoc/markitdown 引擎覆盖，builtin 不做；PPTX/XLSX 目前只取文本，不还原图表数值与批注。
4. **图片落地**：`ParsedDocument.images` 已被 `DocxParser`/`PPTXParser`/`EPUBParser` 携带，但只在配了 `document_processor` 时才会被消费（送 OCR/VLM）；未配置时仍然丢弃，与对齐前一致。

---

## 六、简历措辞（可辩护版）

- 文档解析：**「参考 WeKnora 的解析器注册表设计，把按后缀分支的文档解析重构为引擎注册表：内置 28 类格式（docx/pptx/xlsx/epub 用标准库直接解 OOXML/OCF 容器，零新增依赖），可选 pandoc / MarkItDown / 多模态 OCR / Mistral OCR 四类引擎，三级降级回退；统一输出 Markdown 正文 + 图片列表，供下游三层切分与混合检索复用」**
- GraphRAG：**「参考 WeKnora 的 GraphRAG 加权与检索期扩展机制（PMI×0.6 + 关系强度×0.4 边权、实体图投影为 chunk 图、查询期实体抽取与跳数扩散、跨路去重），在既有 DuckDB 存储上实现，零新增服务依赖；已做三臂 A/B 评测，结论为单跳零退化、关系类净打平，故保持默认关闭」**

> 注意：图增强部分参考的是 **WeKnora 的加权/投影/扩展机制**，存储仍明确写 LightRAG 风格（DuckDB 无 Neo4j）——答辩时不要说"实现了 WeKnora 的 GraphRAG"，会说不清为什么不依赖 Neo4j。
