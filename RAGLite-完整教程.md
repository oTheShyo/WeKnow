# RAGLite 完整教程

> 适用版本：本仓库 `superlinear-ai/raglite`（DuckDB 后端，Windows / Python 3.11.9 / DuckDB 1.5.5 实测）。
> 配套文档：`RAGLite-简历实测.md`（简历口径与数字）、`复现指南-实测数据.md`（如何复现基准）。
> 本教程所有 API 均取自 `src/raglite/__init__.py` 的公开导出，可直接照抄运行。

---

## 0. 这个项目解决什么问题

RAGLite 是一个**检索增强生成（RAG）工具包**，定位是「进程内、可嵌入、轻依赖」的 RAG 引擎：

- 数据库层：DuckDB（单文件）或 PostgreSQL，自带 **FTS（BM25 全文）+ VSS（向量 HNSW）** 扩展，无需常驻服务进程。
- 模型层：通过 LiteLLM 接任意 LLM / Embedder；也支持 **llama.cpp 本地量化模型**，实现零外部 API 依赖。
- 查询层：混合检索 + 重排 + 自适应检索 + Query Adapter，管线比「向量检索 + top-k」完整得多。

一句话：**它是一个能塞进单文件 DuckDB、用本地小模型就能跑通「解析 → 切分 → 入库 → 混合检索 → 重排 → 生成」全链路的 RAG 引擎。**

| 维度 | 选择 | 说明 |
|---|---|---|
| 向量库 | DuckDB VSS（HNSW） | 单文件、进程内、可嵌入 |
| 全文检索 | DuckDB FTS（BM25） | 原生扩展 |
| Embedder | LiteLLM 任意 / llama.cpp 本地 | 默认 `bge-m3` GGUF |
| LLM | LiteLLM 任意 / llama.cpp 本地 | 默认 `Qwen3` GGUF |
| 重排 | FlashRank（Cross-Encoder） | 默认 `ms-marco-MiniLM-L-12-v2`（英文）/ `MultiBERT`（其他） |
| 切分 | SaT ONNX + 动态规划 + 0-1 整数规划 | 三段最优切分 |
| 对外接口 | Python API + CLI + MCP Server | Claude Desktop 可直接连 |

---

## 1. 架构总览

```
                 ┌──────────────── 插入阶段（ingestion） ────────────────┐
  PDF/DOCX/文本 ──▶ document_to_markdown ──▶ split_sentences(SaT) ──▶ split_chunklets(DP)
                                                                          │
                                                                          ▼
                                                              split_chunks(BIP, 语义分块)
                                                                          │
                                                                          ▼
                                                   embed_strings(多向量: late chunking + 标题增强)
                                                                          │
                                                                          ▼
                                          DuckDB: document / chunk / chunk_embedding(FLOAT[1536])
                                                   + FTS 索引 + VSS HNSW 索引
                 └───────────────────────────────────────────────────────────┘

                 ┌──────────────── 查询阶段（retrieval） ───────────────────┐
   query ──▶ vector_search(DuckDB VSS) ┐                                     │
           keyword_search(DuckDB FTS) ┴─▶ reciprocal_rank_fusion(RRF k=60)   │
                                          │                                  │
                                          ▼                                  ▼
                                 rerank_chunks(FlashRank)        retrieve_chunk_spans(邻居扩展+分组)
                                          │                                  │
                                          └──────────▶ 拼接成上下文 chunk span ──▶ add_context ──▶ LLM 生成
                 └───────────────────────────────────────────────────────────┘

  旁路：update_query_adapter（正交 Procrustes 修正 query 向量，提升召回）
```

模块清单（按职责）：

| 文件 | 职责 |
|---|---|
| `_markdown.py` | PDF / 文档 → Markdown（pdftext + pypdfium2，可选 Mistral OCR） |
| `_split_sentences.py` | SaT 神经句界预测 + Markdown 标题边界约束 |
| `_split_chunklets.py` | chunklet 动态规划最优分区 |
| `_split_chunks.py` | 0-1 整数规划（BIP）语义分块，Markdown 标题偏置 |
| `_embed.py` | 多向量嵌入：late chunking + 上下文标题增强 |
| `_insert.py` | 入库编排（线程池 + 批量 flush） |
| `_search.py` | vector / keyword / hybrid / RRF / rerank / chunk span |
| `_rag.py` | `retrieve_context` / `add_context` / `rag` / `async_rag` |
| `_query_adapter.py` | 正交 Procrustes 闭式解修正 query 向量 |
| `_mcp.py` / `_cli.py` | MCP Server / 命令行入口 |

---

## 2. 安装

### 2.1 用 pip

```bash
pip install raglite                 # 核心
pip install raglite[chainlit]       # 可选：ChatGPT 风格前端
pip install raglite[pandoc]         # 可选：除 PDF 外的文档格式
pip install raglite[ragas]          # 可选：检索/生成评测
pip install mistralai               # 可选：Mistral OCR 文档处理
```

### 2.2 本仓库（含基准脚本）

```bash
cd /d/2026/raglite-main
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple   # 必须，否则连官方源易挂
uv sync --python 3.11                                             # 注意：别用 3.13，numpy 1.26.4 无 wheel
```

> ⚠️ **本机已就绪的资源，直接复用即可**（见 `复现指南-实测数据.md` 第 1 节）：Python 3.11.9 venv、SaT ONNX 模型（399 MB）、SaT tokenizer、FlashRank 模型、测试 PDF、现成 `bench_markdown.txt`、已建好的 `bench_data.db`。

---

## 3. 五分钟最小例子（远程模型）

如果你有 OpenAI 兼容的 API key，这是最短链路：

```python
from raglite import RAGLiteConfig, Document, insert_documents, rag

# 离线友好：默认 reranker 工厂会联网下载 MultiBERT，无网时显式置 None
config = RAGLiteConfig(
    db_url="duckdb:///raglite.db",
    llm="gpt-4o-mini",
    embedder="text-embedding-3-small",
    reranker=None,
)

docs = [Document.from_text(
    "# 狭义相对论\n光速不变，时间与空间随速度变化。E=mc²。",
    author="Einstein", topic="physics", year=1905,
)]
insert_documents(docs, config=config)

messages = [{"role": "user", "content": "E 和 mc² 是什么关系？"}]
for update in rag(messages, config=config):
    print(update, end="")
```

就这 4 步：配置 → 插入 → 提问 → 流式生成。RAGLite 自动完成切分、多向量嵌入、向量检索、上下文组装。

> ⚠️ 想启用**混合检索 + 重排**，见 §4.5：默认 `search_method` 是纯向量 `_vector_search`，不是混合。

---

## 4. 分步 API 教程

### 4.1 配置 `RAGLiteConfig`

```python
from raglite import RAGLiteConfig
from rerankers import Reranker

# 远程（API）
config_remote = RAGLiteConfig(
    db_url="duckdb:///raglite.db",
    llm="gpt-4o-mini",
    embedder="text-embedding-3-large",
    self_query=True,          # 让 LLM 自动生成元数据过滤
)

# 本地（llama.cpp 量化模型，可直接嵌入）
config_local = RAGLiteConfig(
    db_url="duckdb:///raglite.db",
    llm="llama-cpp-python/unsloth/Qwen3-8B-GGUF/*Q4_K_M.gguf@8192",
    embedder="llama-cpp-python/lm-kit/bge-m3-gguf/*F16.gguf@512",
)
```

关键配置项：

| 字段 | 默认 | 含义 |
|---|---|---|
| `db_url` | `duckdb:///~/.cache/raglite/raglite.db` | DuckDB 或 PostgreSQL 连接串 |
| `llm` | 本地 Qwen3 Q4（GPU 否则 4B） | 生成模型（LiteLLM 标识） |
| `embedder` | 本地 bge-m3 F16 | 嵌入模型 |
| `chunk_max_size` | 2048 | 单 chunk 最大字符数 |
| `vector_search_multivector` | True | 启用多向量（late chunking） |
| `vector_search_query_adapter` | True | 启用 Query Adapter 修正 |
| `reranker` | FlashRank en/other | 重排器（可换成 Cohere 等） |
| `self_query` | False | LLM 自动生成元数据过滤 |
| `agentic_iterations` | 3 | 自适应检索最多迭代次数 |

> ⚠️ **大坑 1：config 默认值会惰性导入 `llama_cpp`**。不传 `llm`/`embedder` 时，`default_factory` 触发对 `llama_cpp` 的导入；没装就会 `ModuleNotFoundError: No module named 'llama_cpp'`。**解决办法：始终显式传字符串**（如上）。本仓库基准脚本 `bench_resume.py` 的 `BASE_KWARGS = {"llm": "gpt-4o-mini", "embedder": "text-embedding-3-small"}` 就是这个原因。

> ⚠️ **大坑 2：默认 reranker 工厂会联网下载模型**。不传 `reranker` 时，默认工厂会实例化英文 `ms-marco-MiniLM-L-12-v2`（本机已缓存）**和**多语言 `ms-marco-MultiBERT-L-12`（未缓存），首次构造会去 HuggingFace 下载，无网环境直接 `ProxyError`。**离线时务必传 `reranker=None`**（基准脚本就是这么做的）。要启用重排又想离线，需先把 MultiBERT 模型手动缓存到 `user_data_dir("raglite")`。

> 📝 小提示：LiteLLM 初始化时可能尝试联网拉取 `model_prices_and_context_window.json`，超时后回退到本地备份，只是警告、不影响运行。

### 4.2 插入文档

```python
from pathlib import Path
from raglite import Document, insert_documents

# 从文件（PDF 等）
docs = [
    Document.from_path(Path("NVIDIA-report.pdf")),
    Document.from_path(Path("specrel.pdf")),
]

# 从文本 / Markdown（可带元数据，检索时可过滤）
content = "# 狭义相对论\nBy A. Einstein, 1905\n..."
docs = [Document.from_text(content, author="Einstein", topic="physics", year=1905)]

# 可选：用 LLM 自动补全元数据
from raglite import expand_document_metadata
from typing import Annotated
from pydantic import Field
fields = {
    "title": Annotated[str, Field(..., description="Document title.")],
    "author": Annotated[str, Field(..., description="Primary author.")],
}
docs = list(expand_document_metadata(docs, fields, config=config))

insert_documents(docs, config=config)
```

插入阶段 RAGLite 自动完成：Markdown 转换 → 三层切分 → 多向量嵌入 → 落库（FTS + VSS 索引）。

### 4.3 检索：向量 / 关键词 / 混合 / 过滤

```python
from raglite import vector_search, keyword_search, hybrid_search

query = "E=mc² 的物理含义"

# 纯向量
chunk_ids_v, scores_v = vector_search(query, num_results=20, config=config)
# 纯关键词（BM25）
chunk_ids_k, scores_k = keyword_search(query, num_results=20, config=config)
# 混合（RRF 融合，权重 0.75/0.25）
chunk_ids_h, scores_h = hybrid_search(query, num_results=20, config=config)

# 元数据过滤：只返回 topic="physics" 的 chunk
chunk_ids_f, _ = hybrid_search(
    query, num_results=20, metadata_filter={"topic": "physics"}, config=config,
)
# 多值过滤为 OR 语义：domain 含 "open" 或 "music"
chunk_ids_or, _ = hybrid_search(
    query, num_results=20, metadata_filter={"domain": ["open", "music"]}, config=config,
)
```

> 实测延迟（1k chunk / 10k×1536，单连接稳态）：向量 P50 46.5ms、BM25 17.5ms、RRF 0.01ms。详见 `RAGLite-简历实测.md`。

### 4.4 重排与 chunk span

```python
from raglite import retrieve_chunks, rerank_chunks, retrieve_chunk_spans

# 1) 取回 chunk 实体
chunks = retrieve_chunks(chunk_ids_h, config=config)

# 2) Cross-Encoder 重排，取前 5
chunks_reranked = rerank_chunks(query, chunks, config=config)[:5]

# 3) 按原文邻接关系扩展前后邻居、合并成连续 span（等价于 parent-child 回拼）
chunk_spans = retrieve_chunk_spans(chunks_reranked, config=config)
```

实测：FlashRank 重排 20 候选 P50 923ms；chunk span 组装 58ms。

### 4.5 RAG 生成

**自适应（让 LLM 自己决定是否检索）：**

```python
from raglite import rag

messages = [{"role": "user", "content": "狭义相对论讲了什么？"}]
chunk_spans = []
stream = rag(messages, on_retrieval=lambda x: chunk_spans.extend(x), config=config)
for update in stream:
    print(update, end="")
# 引用的文档
docs = [span.document for span in chunk_spans]
```

**可编程（手动控制每一步）：**

```python
from raglite import retrieve_context, add_context, rag

user_prompt = "狭义相对论讲了什么？"
chunk_spans = retrieve_context(
    user_prompt, num_chunks=5,
    metadata_filter={"author": "Einstein"},   # 可选
    config=config,
)
messages = [add_context(user_prompt, context=chunk_spans, config=config)]
stream = rag(messages, config=config)
for update in stream:
    print(update, end="")
```

> ⚠️ **关键事实**：`RAGLiteConfig` 的 `search_method` 默认值不是混合检索，而是纯向量 `_vector_search`（源码 `src/raglite/_config.py` 明确写了 `search_method = _vector_search`，并备注「TODO: 换成 search_and_rerank_chunk_spans」）。所以默认管线**只做向量检索、不做重排**。要启用混合 + 重排，必须显式切换：
>
> ```python
> from dataclasses import replace
> from raglite import hybrid_search, search_and_rerank_chunk_spans
> config = replace(config, search_method=search_and_rerank_chunk_spans)  # 推荐：检索+重排+span 一步到位
> # 或仅做混合向量/关键词检索：replace(config, search_method=hybrid_search)
> ```
>
> 也就是说，README 里「RAGLite applies hybrid search and reranking」的说法与当前代码默认行为不一致——以代码为准。

### 4.6 Query Adapter（提升召回）

```python
from raglite import insert_evals, update_query_adapter

insert_evals(num_evals=100, config=config)   # 自生成 (query, positive, negative) 三元组
update_query_adapter(config=config)          # 之后每次 vector_search 自动应用修正
```

原理（详见 §5.5）：从评测三元组构造目标向量，对 query/title 矩阵做 SVD 求正交 Procrustes 解，推理时零开销线性变换 query 向量。

### 4.7 评测（需 `raglite[ragas]`）

```python
from raglite import insert_evals, answer_evals, evaluate

insert_evals(num_evals=100, config=config)
answered = answer_evals(num_evals=10, config=config)
report = evaluate(answered, config=config)   # Ragas 指标
```

### 4.8 MCP Server（让 Claude Desktop 直接搜你的知识库）

```bash
# 本地模型
raglite --db-url duckdb:///raglite.db \
        --llm llama-cpp-python/unsloth/Qwen3-4B-GGUF/*Q4_K_M.gguf@8192 \
        --embedder llama-cpp-python/lm-kit/bge-m3-gguf/*F16.gguf@512 \
        mcp install

# 远程模型（key 放 .env）
export OPENAI_API_KEY=sk-...
raglite --llm gpt-4o-mini --embedder text-embedding-3-large mcp install
```

### 4.9 删除文档

```python
from raglite import delete_documents, delete_documents_by_metadata

delete_documents(["doc-id-1", "doc-id-2"], config=config)
delete_documents_by_metadata({"topic": "physics"}, config=config)
```

---

## 5. 核心机制深读（简历加分项）

### 5.1 三层最优切分

1. **`split_sentences`（SaT）**：用 ONNX 神经句界模型（segment-any-text）预测句边界，`stride=128, block_size=256, weighting="hat"`，再以 Markdown 标题边界约束。
2. **`split_chunklets`（动态规划）**：在句子基础上做最优子段划分（`max_size=2048`）。
3. **`split_chunks`（0-1 整数规划 BIP）**：以**去 discourse 向量的余弦相似度**为代价函数、`scipy linprog` 求解语义分块；Markdown 标题处加偏置（`partition_similarity[i-1] /= 4`）。

> 实测（96k 字符文档）：SaT ~2.9s / chunklet DP 93ms / BIP 14ms，714 句 → 198 chunklets → 58 chunks。**注意 SaT 段对机器负载极度敏感**（空载 2.9s ↔ load=68 时 11.2s）。

### 5.2 多向量 + late chunking

每个 chunk 不只一条向量，而是**多条子向量**（late chunking：先对整个 chunk 做 transformer 编码，再按句切分 pooled 向量；并注入**上下文标题增强**）。检索时对同一 chunk 的多条子向量取 **L∞ 范数聚合**（`max` over sub-vectors），保留长文档的局部匹配能力。

### 5.3 混合检索与 RRF

- 向量检索走 DuckDB VSS（HNSW，cosine）
- 关键词检索走 DuckDB FTS（BM25）
- 两者排名经 **Reciprocal Rank Fusion**（`k=60`）融合，默认权重 0.75/0.25
- 命中 chunk 再 `retrieve_chunk_spans` 按 `neighbors=(-1,1)` 扩展前后邻居、去重、分组为连续 span

### 5.4 自适应检索

`rag()` 先让 LLM 判断「是否需要检索、检索什么」，复杂问题才触发 `hybrid_search` + `rerank`，最多迭代 `agentic_iterations`（默认 3）次。

### 5.5 Query Adapter 数学

给定自生成评测三元组 `(query, positive, negative)`，构造目标向量 `t*`（非负最小二乘对偶），对 query 矩阵 `Q` 与 title 矩阵 `T` 做 SVD 求**正交 Procrustes** 闭式解 `A* = U Vᵀ`，推理时以 `A*` 零开销线性变换 query 向量，提升跨域召回。

---

## 6. 性能基准与复现

完整数字与分步复现命令见 **`复现指南-实测数据.md`**，要点：

| 指标 | 实测值 | 口径 |
|---|---|---|
| 切分全链路 | ~3.0s（SaT 2.9 / DP 0.09 / BIP 0.014） | 96k 字符，mock embedding |
| HNSW 构建 | 26.6s（文件库）/ 14.0s（内存） | 10k×1536 |
| 精确向量检索 top-20 | P50 46.5ms（内存）/ ~75ms（文件库） | 10k×1536 |
| BM25 top-20 | P50 17.5ms | 50 次真实查询 |
| RRF 融合 | P50 0.01ms | 200 次 |
| chunk span 组装 | P50 58ms | 8 命中 + 邻居 |
| FlashRank 重排 | P50 923ms | 20 候选，CPU ONNX |

---

## 7. 已知问题与坑（面试能讲）

1. **HNSW 索引未被查询计划命中**：`EXPLAIN` 验证执行计划是 `HASH_JOIN` + `ORDER_BY(array_distance(...))` 全量扫描，没有 `USING HNSW`。「索引存在 ≠ 索引被使用」。10k×1536 下精确扫描 ~50ms、召回 100%，ANN 收益要到 10⁵–10⁶ 才显著。
2. **`insert_documents` 入库慢 3 个数量级**：SQLAlchemy 逐行 flush + `FLOAT[1536]` 逐行参数绑定，单文档约 480s。修复路径：Parquet 列式批量导入（本仓库 `fill_and_search.py` 已验证 10k 向量约 2.5 分钟完成）。
3. **`vector_search` 端到端 ~6.3s 是伪影**：每查询连接级 HNSW 索引加载。改用 `SingletonThreadPool` 固定单连接后稳态为 46.5ms。**这个数字不能写进简历。**
4. **config 默认值惰性导入 llama_cpp**（见 §4.1）。
5. **SaT 段负载敏感**（见 §5.1）。

---

## 8. 简历表述建议

- 措辞用「**基于开源 RAGLite 的深度研读与二次开发**」，不要用「独立开发」。
- 只写**你亲手跑过、能解释边界**的数字，并标注测试机/规模（如「10k×1536 向量，单连接稳态」）。
- 生成链路（本地量化模型端到端延迟、评测集准确率）未实测前用占位符，面试被问到能说「待测」。
- 面试高频拷打：① HNSW 为什么没命中；② 精确扫描为什么可接受；③ ORM 入库为什么慢；④ RAGLite 不是你写的你做了什么 —— 答案分别对应 §7 的 1/1/2/4 条。

---

*相关文件：`RAGLite-简历实测.md`、`复现指南-实测数据.md`、`bench_resume.py`、`fill_and_search.py`。*
