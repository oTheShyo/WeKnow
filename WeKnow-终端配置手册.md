# WeKnow 终端配置与运行手册

> 在终端里从零到跑通的完整操作步骤。Windows + Git Bash（本机环境）。  
> 原理见 `WeKnow-完整教程.md`；能力清单见 `WeKnow-使用指南.md`。这份只管**敲什么命令**。

---

## 0. 前置：环境是否就绪（30 秒自检）

```bash
cd /d/2026/raglite-main

# Python 与包
.venv/Scripts/python.exe -V                          # 期望 3.11.9
.venv/Scripts/python.exe -c "import weknow; print('weknow ok')"

# CLI 是否可用
.venv/Scripts/weknow.exe --help
```

三条都通过就可以往下走。**注意用 `.venv/Scripts/python.exe`，不要用裸 `python`** —— 系统的 3.13 装了也没用。

---

## 1. 配置模型端点（`.env`）

### 1.1 为什么必须显式配

`RAGLiteConfig` 的 `llm` / `embedder` 默认值会**惰性导入 `llama_cpp`**，没装就报 `ModuleNotFoundError`。所以无论用哪家，都要显式给出模型字符串。

### 1.2 方案 A：云 API（LLM 与 embedding 同厂商）

仓库根目录创建 `.env`：

```bash
cd /d/2026/raglite-main
cat > .env << 'EOF'
# LLM 与 embedding 共用一个 OpenAI 兼容端点
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=sk-你的key
RAG_LLM_MODEL=qwen-plus
RAG_EMBED_MODEL=text-embedding-v3
RAG_DB_URL=duckdb:///weknow.db
EOF
```

### 1.3 方案 B：本地服务（零成本、零网络）

```bash
ollama pull qwen3:4b        # LLM
ollama pull bge-m3          # embedding
ollama serve                # 默认 127.0.0.1:11434
```

```bash
cat > .env << 'EOF'
OPENAI_API_BASE=http://127.0.0.1:11434/v1
OPENAI_API_KEY=ollama
RAG_LLM_MODEL=qwen3:4b
RAG_EMBED_MODEL=bge-m3
RAG_DB_URL=duckdb:///weknow.db
EOF
```

### 1.4 方案 C：只有 LLM、没有 embedding

这是最需要处理的情况。三条路：

| 路                       | 做法                                           | 代价           |
| ----------------------- | -------------------------------------------- | ------------ |
| C1 **补 embedding**（推荐）  | 去 LLM 厂商控制台开通 embedding，填进 `RAG_EMBED_MODEL` | 通常同 key 通用   |
| C2 **加本地 embedding 服务** | 照 1.3 起 ollama，把 `OPENAI_API_BASE` 指向它       | 需两个端点，见 §1.5 |
| C3 **纯 BM25 模式**        | 完全不建向量，见 §4                                  | 无语义召回        |

### 1.5 LLM 与 embedding 在不同端点

`RAGLiteConfig` 只有 `llm` / `embedder` 两个字符串字段，**没有独立的 base_url**。要分开指向时，靠 LiteLLM 的 provider 前缀 + 各自的环境变量：

```bash
cat > .env << 'EOF'
# LLM 走云
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=sk-xxx
# embedding 走本地（LiteLLM 的 ollama provider 不会被 OPENAI_API_BASE 影响）
RAG_LLM_MODEL=qwen-plus
RAG_EMBED_MODEL=ollama/bge-m3
OLLAMA_API_BASE=http://127.0.0.1:11434
EOF
```

> 关键点：`embedder="ollama/bge-m3"` 走的是 LiteLLM 的 ollama provider，读 `OLLAMA_API_BASE`，**不受 `OPENAI_API_BASE` 影响**。这是分开指向最省事的办法。

---

## 2. 验证端点连通（配完必做）

```bash
# 逐项检查环境变量
.venv/Scripts/python.exe gen_bench.py --check-env
```

缺变量会直接报错并列出缺哪个。然后直连端点确认：

```bash
# 列出可用模型
curl -s "$OPENAI_API_BASE/models" -H "Authorization: Bearer $OPENAI_API_KEY" | head -c 500
```

```bash
# 实测 embedding 是否可用（最关键，因为建库必须要有它）
.venv/Scripts/python.exe -c "
import os
from openai import OpenAI
c = OpenAI(base_url=os.environ['OPENAI_API_BASE'], api_key=os.environ['OPENAI_API_KEY'])
r = c.embeddings.create(model=os.environ['RAG_EMBED_MODEL'], input='测试')
print('embedding 维度 =', len(r.data[0].embedding))
"
```

维度打印出来就说明 embedder 可用。**记下这个维度**，建库后可以用它核对。

---

## 3. 在终端里跑检索（不需要端点）

```bash
# 解析自检
.venv/Scripts/python.exe -c "
from pathlib import Path
from weknow import parse_document, parser_report
import json
print(json.dumps(parser_report(), ensure_ascii=False, indent=2))
d = parse_document(Path('tests/NVIDIA-report.pdf'))
print('engine =', d.engine, '| chars =', len(d.content))
"
```

```bash
# 纯 BM25 基准（数据集是位置参数，没有 --dataset）
.venv/Scripts/python.exe bench_beir.py beir/nfcorpus/test 2>&1 | tail -15
```

预期 nDCG@10 ≈ **0.3181**。

```bash
# CLI 自带的 bench 子命令（用的是 ir_datasets + ir_measures）
.venv/Scripts/weknow.exe bench --dataset beir/nfcorpus/test --measure nDCG@10
```

---

## 4. 纯 BM25 模式（C3，没有 embedding 时）

不建向量，只用 FTS/BM25。**注意不能用 `insert_documents`**（它会强行走嵌入），要自己写库：

```bash
.venv/Scripts/python.exe -c "
from weknow import RAGLiteConfig, keyword_search, retrieve_chunks, add_context, rag
import os

# 库里已有 chunk 时才能这么用；建库见下方脚本
config = RAGLiteConfig(
    db_url='duckdb:///weknow.db',
    llm='openai/' + os.environ['RAG_LLM_MODEL'],
    reranker=None,
)
ids, scores = keyword_search('量子比特', num_results=5, config=config)
print('命中', len(ids), '条')
chunks = retrieve_chunks(ids, config=config)
if chunks:
    msgs = [add_context('什么是量子比特？', context=chunks, config=config)]
    for tok in rag(msgs, config=config):
        print(tok, end='')
"
```

推荐直接用现成的 BM25 建库脚本（`bench_beir.py` 里的 `BM25Evaluator` 就是完整范例）：

```bash
# 它会把数据集文档写进库、只建 FTS 索引、完全不碰 embedder
.venv/Scripts/python.exe bench_beir.py beir/nfcorpus/test
```

---

## 5. 完整端到端（端点齐了之后）

### 5.1 建库 + 提问（一行脚本）

```bash
.venv/Scripts/python.exe -c "
import os
from weknow import RAGLiteConfig, Document, insert_documents, rag

config = RAGLiteConfig(
    db_url=os.environ.get('RAG_DB_URL', 'duckdb:///weknow.db'),
    llm='openai/' + os.environ['RAG_LLM_MODEL'],
    embedder='openai/' + os.environ['RAG_EMBED_MODEL'],
    reranker=None,          # 离线必须显式关闭，否则联网下载 FlashRank 模型
)

docs = [Document.from_text('# 狭义相对论\n光速不变，时间与空间随速度变化。E=mc^2。',
                           author='Einstein', topic='physics', year=1905)]
insert_documents(docs, config=config)
print('[insert] ok')

for tok in rag([{'role': 'user', 'content': 'E 和 mc^2 是什么关系？'}], config=config):
    print(tok, end='')
"
```

### 5.2 启用混合检索 + 重排（默认是纯向量！）

```bash
.venv/Scripts/python.exe -c "
import os
from dataclasses import replace
from weknow import RAGLiteConfig, Document, insert_documents, rag, routed_search

config = replace(
    RAGLiteConfig(
        db_url='duckdb:///weknow.db',
        llm='openai/' + os.environ['RAG_LLM_MODEL'],
        embedder='openai/' + os.environ['RAG_EMBED_MODEL'],
        reranker=None,
    ),
    search_method=routed_search,   # 意图路由 → 内部调混合检索
    search_router=True,
)
for tok in rag([{'role': 'user', 'content': 'E 和 mc^2 是什么关系？'}], config=config):
    print(tok, end='')
"
```

### 5.3 生成延迟基准

```bash
.venv/Scripts/python.exe gen_bench.py --check-env     # 先验证
.venv/Scripts/python.exe gen_bench.py --runs 5        # 跑 5 次
```

### 5.4 增量更新索引

```bash
.venv/Scripts/python.exe -c "
import os
from pathlib import Path
from weknow import RAGLiteConfig, Document, sync_documents

config = RAGLiteConfig(db_url='duckdb:///weknow.db',
                       llm='openai/' + os.environ['RAG_LLM_MODEL'],
                       embedder='openai/' + os.environ['RAG_EMBED_MODEL'],
                       reranker=None)
docs = [Document.from_path(p) for p in ['a.pdf', 'b.docx']]
print(sync_documents(docs, key='filename', config=config))
"
```

---

## 6. CLI 与 MCP

```bash
# 顶层帮助（本机已修复：以前没装 llama_cpp 会直接崩）
.venv/Scripts/weknow.exe --help

# 用参数覆盖端点
.venv/Scripts/weknow.exe \
  --db-url duckdb:///weknow.db \
  --llm openai/qwen-plus \
  --embedder openai/text-embedding-v3 \
  bench --dataset beir/nfcorpus/test --measure nDCG@10

# 装 MCP server 到 Claude Desktop
.venv/Scripts/weknow.exe \
  --db-url duckdb:///weknow.db \
  --llm openai/qwen-plus \
  --embedder openai/text-embedding-v3 \
  mcp install

# Chainlit 前端（需先 uv pip install chainlit）
.venv/Scripts/weknow.exe --db-url duckdb:///weknow.db --llm openai/qwen-plus --embedder openai/text-embedding-v3 chainlit
```

> `.env` 里的 `RAGLITE_*` 前缀变量会被 CLI 自动读取（`SettingsConfigDict(env_prefix="RAGLITE_", env_file=".env")`）：
>
> ```bash
> echo 'RAGLITE_DB_URL=duckdb:///weknow.db' >> .env
> echo 'RAGLITE_LLM=openai/qwen-plus' >> .env
> echo 'RAGLITE_EMBEDDER=openai/text-embedding-v3' >> .env
> ```
>
> 三个都配好后，`weknow.exe` 不带参数即可。

---

## 7. 安装可选依赖（本机 `.venv` **没有 pip**）

必须用 uv：

```bash
cd /d/2026/raglite-main
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

uv pip install llama-cpp-python         # 本地 GGUF 模型
uv pip install rapidocr-onnxruntime     # 离线 OCR（多模态②的 OCR 阶段）
uv pip install mistralai                # Mistral OCR
uv pip install chainlit                 # Chainlit 前端
uv pip install 'weknow[multimodal]'    # 一次性装多模态
```

---

## 8. 终端报错速查

| 报错                            | 原因                                          | 命令                                                          |
| ----------------------------- | ------------------------------------------- | ----------------------------------------------------------- |
| `No module named 'llama_cpp'` | 没显式传 `llm`/`embedder`                       | 配 `.env`（§1）                                                |
| `ProxyError` / 下载失败           | 默认 reranker 联网                              | 传 `reranker=None`                                           |
| `KeyError: '--dataset'`       | `bench_beir.py` 用位置参数                       | `.venv/Scripts/python.exe bench_beir.py beir/nfcorpus/test` |
| `No module named 'pip'`       | 本机 venv 由 uv 建                              | 用 `uv pip install`                                          |
| `weknow.exe --help` 崩溃       | **已在本次修复**（`_cli.py` 不再实例化 `RAGLiteConfig`） | 拉最新代码                                                       |
| embedding 维度对不上已有库            | 换过 embedder                                 | 删库重建（`FLOAT[N]` 维度不可变）                                      |

---

## 9. 一页速查

```bash
cd /d/2026/raglite-main

# 自检
.venv/Scripts/weknow.exe --help
.venv/Scripts/python.exe -c "from weknow import parser_report; print(parser_report())"

# 配置
vi .env                                  # 按 §1 填
.venv/Scripts/python.exe gen_bench.py --check-env

# 无端点也能跑
.venv/Scripts/python.exe bench_beir.py beir/nfcorpus/test

# 端到端
.venv/Scripts/python.exe gen_bench.py --runs 5

# 代码质量
uv run poe lint && uv run poe test
```

---

*相关：`WeKnow-完整教程.md`（原理+源码）、`WeKnow-使用指南.md`（能力与开关）、`复现指南-实测数据.md`（数字来源）。*
