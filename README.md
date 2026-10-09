# WeKnow

> 一个面向**个人知识库**的检索增强生成（RAG）问答系统：文档进、答案出、引用可追溯。
> 检索内核基于开源项目 [RAGLite](https://github.com/superlinear-ai/raglite) **1.1.1** 的深度研读与二次开发，
> 架构设计参考腾讯 [WeKnora](https://github.com/Tencent/WeKnora) 与 [LightRAG](https://github.com/HKUDS/LightRAG) 的公开思路。

WeKnow = **RAGLite 的检索内核** + **8 项自研工程扩展** + **一个可离线运行的 Web 前端** + **单容器部署**。
不引入独立向量数据库（不是 FAISS / Milvus / Qdrant / Chroma），向量与关键词索引都放在关系型数据库里。

---

## 这是什么

| | 说明 |
|---|---|
| **输入** | PDF / DOCX / PPTX / XLSX / EPUB / Markdown / 图片 / 扫描件等 28 类格式（内置解析器；docx·pptx·xlsx·epub 用标准库解 OOXML/OCF，零额外依赖） |
| **处理** | 三层切分（句子 → 块元 → 块）→ 多向量 late-chunking 嵌入 → 入库 |
| **检索** | 向量路（HNSW）与关键词路（BM25）并行召回 → RRF 融合 → Cross-Encoder 重排 → 按原文邻接回拼 chunk span |
| **输出** | 流式回答 + 引用来源（可点回原文切片） |
| **存储** | **默认 DuckDB**（`vss` 扩展给 HNSW、`fts` 扩展给 BM25）；备选 **PostgreSQL + pgvector** |
| **入口** | FastAPI + SSE 单页前端（`weknow serve`），另有 CLI / MCP / Chainlit |
| **部署** | 单容器 Docker Compose，`.env` 集中配置，数据库落挂载卷 |

---

## 快速开始

需要 Python 3.10–3.13。推荐用 [uv](https://docs.astral.sh/uv/) 管理环境。

```bash
# 1. 装依赖（核心依赖都在 pyproject.toml 里，一条命令装齐）
uv sync

# 2. 配端点：复制模板，填上你的模型 Key
cp .env.example .env
#    必填两项：LLM 与 Embedder。两者可以来自完全不同的厂商与端点 ——
#    provider 前缀（openai/ dashscope/ ollama/ …）决定读哪个环境变量。

# 3. 起服务
uv run weknow serve            # → http://127.0.0.1:8000
```

打开浏览器即可：**知识库**标签拖动上传文档，**对话**标签提问并看到流式回答与引用来源，
**检索测试**标签直接看每一路的召回切片与得分。

> **首次运行会下载切分模型**：句子切分用 Segment any Text（`sat-1l-sm` 的 ONNX 权重约 **399 MB**）。
> 国内网络请保留 `.env` 里的 `HF_ENDPOINT=https://hf-mirror.com`；无网机器需提前离线预置到 Hugging Face 缓存。
>
> **配置只有一处**：`src/weknow/_dotenv.py` 的 `config_from_environ()`。所有入口（CLI / Web / Chainlit / MCP）都走它；
> 模型能力（上下文长度、向量维度）**首次运行自动探测并写回 `.env`**，不用手填。

### 常用命令

```bash
uv run weknow --help                 # 查看全部子命令与参数
uv run weknow serve --port 8080      # 起 Web 前端（WeKnora 风格）
uv run weknow mcp                    # 以 MCP server 方式暴露检索（供 Claude 等客户端连接）
uv run weknow chainlit               # 备选的 Chainlit 前端（需 uv sync --extra chainlit）
uv run weknow bench                  # 跑检索基准
```

按需装可选能力（**核心功能不需要**）：

```bash
uv sync --extra server               # FastAPI 前端（uv sync 默认不含，serve 需要它）
uv sync --extra chainlit             # Chainlit 前端
uv pip install 'weknow[multimodal]'  # 离线 OCR（RapidOCR）+ PDF 栅格化（PyMuPDF）
uv pip install 'weknow[pandoc]'      # Pandoc 解析引擎
uv pip install 'weknow[ragas]'       # RAGAS 端到端评测
```

---

## 架构一览

```
接口层    Web(SSE) · CLI · MCP server · Chainlit
             │
   ┌─────────┴─────────┐
在线检索（读）        离线索引（写）
 意图路由 → 缓存命中   │ 解析引擎注册表 → 三层切分
 ↓ 未命中             │      ↓
 三路召回：向量 / 关键词 / 图 │ 多向量嵌入（late chunking）
 ↓                    │      ↓
 RRF 融合 → 重排 → chunk span │ 内容 Hash 增量（只写变化的文档）
 ↓                    │
 低置信度二次检索      │
 └─────────┬─────────┘
存储层    DuckDB（HNSW + BM25） 或 PostgreSQL + pgvector
模型层    LiteLLM（任意 provider）· llama.cpp 本地量化 · CUDA/Metal 加速
横切      评测（RAGAS / 自研 ROUGE-L·BLEU / BEIR 基准）
```

**全部 8 项自研扩展**（每一项都有独立提交与单测）：

| # | 能力 | 入口 | 默认 |
|---|---|---|---|
| 1 | 文档解析引擎注册表（`(engine, file_type) → parser` + 三级回退，内置 28 类格式） | `_parsers.py` | 开 |
| 2 | 多模态 OCR + VLM（离线 ONNX + 视觉模型，双路独立降级） | `_multimodal.py` | 关 |
| 3 | 低置信度二次检索（rerank 阈值触发扩池重召回，两轮取优） | `_iterative_search.py` | 条件开（需挂重排器） |
| 4 | 内容 Hash 增量索引（只重写变化的文档） | `_incremental.py` | 开 |
| 5 | 查询意图路由（微秒级启发式，零额外 LLM 调用） | `_router.py` | 开 |
| 6 | 分层缓存（检索结果 TTL+LRU，入库/删除自动失效） | `_cache.py` | 开 |
| 7 | 生成端评估（零依赖 ROUGE-L / BLEU） | `_eval_generation.py` | 关 |
| 8 | 图增强检索（实体图 → chunk 图投影，作为 RRF 第三路） | `_graph.py` | **关**（见下） |

**图增强检索为什么默认关闭**：在 BEIR 两个官方数据集上做过 6 臂消融。NFCorpus 上图路低权重时召回有显著小增益
（R@100 **+0.0058**，95% CI [+0.0025, +0.0096]）但 top-10 精度与基线不可区分；**换到 SciFact 结论翻转**
（w=0.1 时 nDCG@10 **−0.0401**）。**收益不可跨语料移植**，因此默认关闭、不做泛化宣称。
关键词基线本身经过校准：BEIR NFCorpus nDCG@10 = **0.3178**、SciFact = **0.673**，与 Anserini 公开复现值对齐。

---

## 配置

**只有一个配置文件 `.env`**（已在 `.gitignore` 中）。`cp .env.example .env` 后按注释填即可，六个分组：
LLM 端点 / Embedder 端点 / 数据库 / 重排器 / 解析引擎 / 其它。

几个容易踩的点：

- **LLM 与 Embedder 可以不同端点、不同 Key** —— 靠 litellm 的 provider 前缀区分（`openai/` 读 `OPENAI_API_BASE`，`dashscope/` 读 `DASHSCOPE_API_KEY`）。
- **`RAG_RERANKER` 语义**：未设置 → 用默认重排器；**设为空 → 关闭重排**；设值 → 透传。离线环境建议关掉，因为默认重排器首次使用会联网下载权重。
- **换 Embedder 必须重建库**：向量列是定长数组 `FLOAT[N]`，维度入库后不可变。
- 交互式入口（CLI / Web / Chainlit）默认 `disable_default_reranker=True`，并开启模型能力自动探测。

---

## 部署

单容器生产部署（`.env` 经 `env_file` 注入，数据库落宿主卷）：

```bash
cp .env.example .env      # 填好端点
docker compose up -d --build
```

镜像内已固化 `HF_ENDPOINT=https://hf-mirror.com` 与 `LITELLM_LOCAL_MODEL_COST_MAP=True`。
细节与常见问题见 [`部署指南.md`](部署指南.md)。

---

## 文档

| 文档 | 内容 |
|---|---|
| [`WeKnow-使用指南.md`](WeKnow-使用指南.md) | 照着跑起来：三条路径选择、命令速查、常见报错 |
| [`WeKnow-终端配置手册.md`](WeKnow-终端配置手册.md) | 终端里从零到跑通的完整操作步骤（含端点配不通的三条出路） |
| [`WeKnow-完整教程.md`](WeKnow-完整教程.md) | 原理 + 逐项源码摘录 + 8 项扩展的实现细节 |
| [`WeKnow-简历-架构图版.md`](WeKnow-简历-架构图版.md) | 按架构六区块组织的项目描述与能力归属索引 |
| [`WeKnora对比-优化清单.md`](WeKnora对比-优化清单.md) | 与 WeKnora 的逐项对齐差距（含未对齐清单） |
| [`文档解析与GraphRAG-对齐WeKnora.md`](文档解析与GraphRAG-对齐WeKnora.md) | 解析注册表与 GraphRAG 六项机制的对齐记录 |
| [`图增强检索-BEIR评测.md`](图增强检索-BEIR评测.md) · [`图增强检索-AB评测.md`](图增强检索-AB评测.md) | 图路的基准评测与机制层 A/B |
| [`多模态扩展-OCR+VLM.md`](多模态扩展-OCR+VLM.md) | 离线 OCR + VLM 双阶段提取 |
| [`复现指南-实测数据.md`](复现指南-实测数据.md) | 性能数字的复现步骤与口径 |
| [`部署指南.md`](部署指南.md) | Docker 部署、API 一览、常见问题 |

---

## 上游归属与许可

检索内核（三层切分、多向量 late-chunking 嵌入、混合检索、RRF、重排、chunk span 回拼、自适应 RAG、
Query Adapter、DuckDB/PostgreSQL 存储、LiteLLM 模型层、MCP server）**全部来自上游 RAGLite 1.1.1**，
版权归 [Laurent Sorber / superlinear-ai](https://github.com/superlinear-ai/raglite) 所有，**未重写上游任何模块**。
本仓库在此基础上做的是上表 8 项扩展、Web 前端与部署形态。上游的英文 API 教程与完整能力清单见
[RAGLite README](https://github.com/superlinear-ai/raglite)。

许可证：**Mozilla Public License 2.0**（沿用上游）。
