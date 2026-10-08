# WeKnow 使用指南（本机可直接执行）

> 面向 `D:/2026/raglite-main`。所有命令都已按本机实际环境（Windows + Git Bash + 已建好的 `.venv`）核对过。
> 想了解原理与源码，看 `WeKnow-完整教程.md`；这份只讲**怎么跑起来**。

---

## 0. 本机环境现状（2026-10-08 实测）

| 项 | 状态 |
|---|---|
| `.venv` | ✅ Python 3.11.9，已装 `weknow`（可编辑安装） |
| 已装 | duckdb · sqlmodel · litellm · rerankers · flashrank · pymupdf · scipy · numpy · langdetect · mcp · **ir_datasets · ir_measures** |
| **未装** | `llama_cpp`（本地模型）· `rapidocr_onnxruntime` / `pytesseract`（OCR）· `mistralai`（Mistral OCR）· `chainlit`（前端）· `PIL` |
| 已有数据库 | `bench_data.db`（1 文档 / 1000 chunk / 10000 向量，10k×1536） |
| `.env` | ❌ 不存在，需要自己建 |
| LLM / Embedding 端点 | ⚠️ **本机当前没有可用的**——`CUSTOM_API_KEY` 有值但无 base URL 与模型名，不能盲猜 |

> 这意味着：**不开多模态、不接云端也能跑**的是检索内核（BM25 那条路完全离线）。要跑端到端 RAG，先按 §2 配端点。

---

## 1. 三条最短路径（按你有多少条件选一条）

| 路径 | 需要什么 | 能验证什么 | 耗时 |
|---|---|---|---|
| **A. 零配置自检** | 什么都不需要 | 解析 / 切分 / 检索内核是否可用 | < 1 分钟 |
| **B. 标准基准复现** | 什么都不需要（数据集自动下载） | 检索准确率、图增强消融 | 5 分钟 ~ 1 小时 |
| **C. 端到端 RAG** | 一个 OpenAI 兼容端点 | 完整「解析→检索→生成」 | 5 分钟 |

---

## 2. 路径 A：零配置自检（先做这个）

```bash
cd /d/2026/raglite-main

# ① 解析引擎可用性报告
.venv/Scripts/python.exe -c "
from weknow import parser_report, supported_file_types
import json
print(json.dumps(parser_report(), ensure_ascii=False, indent=2))
print('builtin 覆盖：', sorted(supported_file_types()))
"
```

预期（本机实测输出）：

```
builtin     | available = True  | types = 28
multimodal  | available = True  | types = 9     ← pymupdf 已装，栅格化可用（但 OCR 引擎未装）
pandoc      | available = False | types = 17
markitdown  | available = False | types = 12
mistral     | available = False | types = 11
```

> 注意 `multimodal` 显示 `available = True` 是因为**栅格化后端（pymupdf）已装**；但本机未装 `rapidocr-onnxruntime` / `pytesseract`，所以真正跑起来时 OCR 阶段会降级为空（返回空字符串），VLM 阶段需 `vlm=` 指定模型。要真正用 OCR 见 §6。

```bash
# ② 解析一个真实文档（用仓库自带的测试 PDF）
.venv/Scripts/python.exe -c "
from pathlib import Path
from weknow import parse_document
d = parse_document(Path('tests/NVIDIA-report.pdf'))
print('engine =', d.engine, '| file_type =', d.file_type)
print('前 200 字：', d.content[:200].replace(chr(10),' '))
"
```

```bash
# ③ 检索内核对自检：跑纯 BM25 基准（不需要任何模型，数据集自动下载）
cd /d/2026/raglite-main
.venv/Scripts/python.exe bench_beir.py beir/nfcorpus/test 2>&1 | tail -20
```

预期输出 nDCG@10 ≈ **0.3181**（与 Anserini 公开值 0.3218 对齐）。跑通这条说明**检索内核完好**。
> 脚本把数据集 ID 作为**位置参数**（`sys.argv[1]`），默认 `beir/nfcorpus/test`，没有 `--help`。

```bash
# ④ 看已建好的库长什么样
.venv/Scripts/python.exe -c "
import duckdb
con = duckdb.connect('bench_data.db', read_only=True)
print(con.execute('SHOW TABLES').fetchall())
print('chunk =', con.execute('SELECT count(*) FROM chunk').fetchone()[0])
"
```

---

## 3. 路径 B：标准基准复现（不需要模型）

```bash
# 两数据集图增强六臂消融（NFCorpus 约 40 分钟，SciFact 约 1 小时 47 分）
# 数据集 ID 同样是位置参数，默认 beir/nfcorpus/test
.venv/Scripts/python.exe bench_beir_graph.py beir/nfcorpus/test 2>&1 | tee beir_graph_nfcorpus.log
.venv/Scripts/python.exe bench_beir_graph.py beir/scifact/test  2>&1 | tee beir_scifact.log

# 结果落在 bench_beir_graph_beir_nfcorpus_test.json / ..._scifact_test.json
```

> 脚本设计为**先写 JSON 再打印**，所以中途 Ctrl-C 也不会丢已完成的结果；重跑会跳过已完成的臂。
> 复现出的数字应与已有产物一致（NFCorpus bm25 nDCG@10 = 0.3181、SciFact = 0.6730）。

---

## 4. 路径 C：端到端 RAG（需要端点）

### 4.1 配置端点（`.env`）

在仓库根目录创建 `.env`：

```bash
# .env
OPENAI_API_BASE=https://<你的端点>/v1
OPENAI_API_KEY=<你的 key>
RAG_LLM_MODEL=<模型名，如 qwen3-chat>
RAG_EMBED_MODEL=<嵌入模型名>
# 可选
# RAG_DB_URL=duckdb:///weknow.db
# RAG_RERANKER=   # 留空则关闭重排（离线推荐留空）
```

> ⚠️ **`llm` / `embedder` 必须显式给字符串**。`RAGLiteConfig` 的默认值会惰性导入 `llama_cpp`，不显式传就会 `ModuleNotFoundError`。
> ⚠️ **`reranker` 必须显式置 `None`**（离线时）。默认工厂会去 HuggingFace 下载 MultiBERT，无网直接 `ProxyError`。

### 4.2 自建最小索引并提问

```bash
.venv/Scripts/python.exe -c "
from weknow import RAGLiteConfig, Document, insert_documents, rag

config = RAGLiteConfig(
    db_url='duckdb:///weknow.db',
    llm='openai/' + __import__('os').environ['RAG_LLM_MODEL'],
    embedder='openai/' + __import__('os').environ['RAG_EMBED_MODEL'],
    reranker=None,          # 离线必须显式关闭
)

docs = [Document.from_text('# 狭义相对论\n光速不变，时间与空间随速度变化。E=mc^2。',
                           author='Einstein', topic='physics', year=1905)]
insert_documents(docs, config=config)
print('[insert] ok')

for tok in rag([{'role':'user','content':'E 和 mc^2 是什么关系？'}], config=config):
    print(tok, end='')
"
```

### 4.3 用仓库自带的生成基准脚本（更省事）

```bash
.venv/Scripts/python.exe gen_bench.py --check-env   # 先验证端点连通
.venv/Scripts/python.exe gen_bench.py               # 跑生成延迟基准
```

---

## 5. 各项扩展怎么开

| 扩展 | 开关 | 生效方式 |
|---|---|---|
| ① 解析注册表 | `parser_engine="pandoc"` 等，或不传走 builtin | 插入时自动生效 |
| ② 多模态 OCR+VLM | `document_processor=MultimodalConfig(...)` | 需先装引擎（见 §6） |
| ③ 二次检索 | `secondary_retrieval=True`（**默认开**） | 需重排器才有置信信号 |
| ④ 增量索引 | 调 `sync_documents(docs, key="filename", config=...)` | 替代 `insert_documents` |
| ⑤ 意图路由 | `search_router=True`（默认关） | 或 `search_method=routed_search` |
| ⑥ 分层缓存 | `search_method=cached_hybrid_search` | 索引变更自动失效 |
| ⑦ 生成端评估 | 调 `evaluate_generation(...)` | 需端点或传预生成答案 |
| ⑧ 图增强 | `graph=GraphConfig(enabled=True, ...)`（默认关） | 需先 `build_graph()` |

**重要**：默认 `search_method` 是**纯向量**，不是混合。要三路混合 + 重排：

```python
from dataclasses import replace
from weknow import search_and_rerank_chunk_spans

config = replace(config, search_method=search_and_rerank_chunk_spans)
```

### 常用组合示例

```python
# 混合检索 + 重排 + 二次检索 + 缓存 + 意图路由（推荐生产配置）
from dataclasses import replace
from weknow import RAGLiteConfig, cached_hybrid_search, routed_search

config = replace(
    RAGLiteConfig(db_url='duckdb:///weknow.db', llm='openai/gpt-4o-mini',
                  embedder='openai/text-embedding-3-small', reranker=None),
    search_method=routed_search,     # ⑤ 路由 → 内部再调混合检索
    search_router=True,
    secondary_retrieval=True,        # ③ 默认已开
    retrieval_cache_ttl=300.0,       # ⑥
)

# ④ 增量同步（替代全量 insert）
from weknow import Document, sync_documents
docs = [Document.from_path(p) for p in ['a.pdf', 'b.docx']]
print(sync_documents(docs, key='filename', config=config))
```

---

## 6. 按需安装可选依赖

```bash
# 本机 .venv 由 uv 创建（无 pip），必须用 uv pip
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

# 本地模型（零 API 跑全链路）
uv pip install llama-cpp-python

# 离线 OCR（多模态②的 OCR 阶段）
uv pip install rapidocr-onnxruntime

# 图片栅格化（②的 PDF 分页需要，本机已有 pymupdf，无需再装）
# uv pip install pymupdf

# Mistral OCR
uv pip install mistralai && export MISTRAL_API_KEY=...

# Chainlit 前端
uv pip install chainlit

# 全装
uv pip install 'weknow[multimodal]'
```

> ⚠️ 别用 `pip install`（本机 `.venv` 无 pip），也别装全局。

---

## 7. 命令行与 MCP

```bash
# 查看 CLI
.venv/Scripts/python.exe -m weknow --help
# 或装好后直接用入口脚本
.venv/Scripts/weknow.exe --help

# 装 MCP server 到 Claude Desktop（需已装 Claude 桌面版）
.venv/Scripts/weknow.exe \
  --db-url duckdb:///weknow.db \
  --llm openai/gpt-4o-mini \
  --embedder openai/text-embedding-3-small \
  mcp install

# Chainlit 前端（需装 chainlit）
.venv/Scripts/weknow.exe --db-url duckdb:///weknow.db --llm ... --embedder ... chainlit
```

---

## 8. 常见报错速查

| 报错 | 原因 | 解法 |
|---|---|---|
| `ModuleNotFoundError: No module named 'llama_cpp'` | 没显式传 `llm`/`embedder` | 显式传字符串（§4.1） |
| `ProxyError` / 下载 MultiBERT 失败 | 默认 reranker 联网 | 传 `reranker=None` |
| `dim=-1` / `FLOAT[-1]` 语法错 | 猴补丁跳过了 `set_embedding_dim()` | 先调 `ChunkEmbedding.set_embedding_dim(dim)` |
| OCR / VLM 报错但文档仍入库 | **这是设计如此**（双路独立降级） | 无需处理，查 warning 日志 |
| `no module named 'chainlit'` | 未装前端 extra | `uv pip install chainlit` |
| DuckDB 同进程二次 attach 同文件失败 | DuckDB 限制 | 换文件或复用连接 |

---

## 9. 一页速查

```bash
# 环境
cd /d/2026/raglite-main
.venv/Scripts/python.exe -V                    # 3.11.9

# 自检（无需模型）
.venv/Scripts/python.exe -c "from weknow import parser_report; print(parser_report())"
.venv/Scripts/python.exe bench_beir.py --dataset beir/nfcorpus/test

# 端到端（需 .env）
.venv/Scripts/python.exe gen_bench.py --check-env
.venv/Scripts/python.exe gen_bench.py

# 代码质量
uv run poe lint
uv run poe test
```

---

*相关文档：`WeKnow-完整教程.md`（原理+源码）、`复现指南-实测数据.md`（数字来源）、`图增强检索-BEIR评测.md`（评测方法与结论）。*
