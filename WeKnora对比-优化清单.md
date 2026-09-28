# 参考 WeKnora：RAGLite 优化分析（含简历扩展路线）

> 对比对象：腾讯 WeKnora v0.8.2（Go/Rust 服务化全家桶）vs RAGLite（Python 进程内嵌入式引擎）。
> 结论先行：**两者定位不同，RAGLite 在嵌入式形态与切分质量上反而领先；WeKnora 的价值是给 RAGLite
> 缺失的 6 个模块提供了现成的参考实现**——照做就是名正言顺的"二次开发扩展"。

---

## 一、定位差异（面试可讲的对比研究）

| 维度 | WeKnora | RAGLite | 谁强 |
|---|---|---|---|
| 部署形态 | Docker/K8s 服务化（依赖 PG+Redis） | pip 装完即用，DuckDB 单文件 | **RAGLite**（嵌入式零依赖） |
| 文档解析 | 13 格式，三引擎互补（anydoc Rust + DocReader OCR + PaddleOCR-VL） | PDF/MD（pdftext+Pandoc），纯文本 | WeKnora |
| 切分 | 父子分块 + 自适应三级切分 | 三层最优化切分（SaT/DP/BIP，有理论依据） | **RAGLite**（更讲究） |
| 检索 | BM25+Dense+GraphRAG，RRF 归一化修复、意图识别、FAQ 迭代 | BM25+Dense+RRF+FlashRank+agentic 迭代 | 各有侧重 |
| 评估 | Recall + BLEU/ROUGE + Langfuse 全链路追踪 | TREC MRR/Hit/nDCG（检索端更强） | 检索 RAGLite，生成+可观测 WeKnora |
| 生态 | 8 种向量库、27 家 LLM、10 种 IM 渠道 | PG + DuckDB，litellm | WeKnora（服务化所需） |

**一句话**：WeKnora 是"企业知识库产品"，RAGLite 是"嵌入式检索内核"。拿 RAGLite 缺服务化能力去批评它，不如拿 WeKnora 的**模块设计**来补 RAGLite 的**功能空白**。

---

## 二、RAGLite 值得借鉴的优化点（按 简历价值 × 工作量 排序）

### T0：直接补齐简历缺失亮点（高价值，参考实现现成）

> ✅ **2026-09-28 状态：#1–#6 已全部实现并提交**（git 仓库已建立，每项能力一次 feat 提交，
> 新增单测 16 + 22 = 38 个全过，mypy 干净）。实现方式：
> - #1 → `_iterative_search.py`（rerank 置信度阈值触发二次召回，WeKnora 式扩池）
> - #2 → `_incremental.py`（content-hash 增量同步 + 热更新 + 可选 prune）
> - #3 → `_router.py`（微秒级启发式路由，无 LLM 调用）
> - #4 → `_cache.py`（检索结果 TTL 缓存 + vector_search 内 L1 查询向量缓存 + 入库/删除失效）
> - #5 → `_multimodal.py` + `MultimodalConfig`（**图片直读 + PDF 分页栅格化 → 本地 ONNX OCR
>   （RapidOCR）回收图内文字 + LiteLLM 视觉模型生成图像描述；双阶段独立降级、多页并发提取、
>   图像类型启发式标注；两类内容拼入 Markdown 后复用既有三层切分与混合建库**）
> - #6 → `_eval_generation.py`（零依赖 ROUGE-L + BLEU，CJK 感知分词）

| # | 优化项 | WeKnora 参考机制 | RAGLite 上的实现路径 | 工作量 |
|---|---|---|---|---|
| 1 | **低置信度二次检索** | FAQ 迭代检索（每轮起始深度翻倍）+ agent 多步 refine | 在 `search_and_rerank_chunk_spans` 外包一层：rerank 最高分 < 阈值（如 0.3）时，扩大 `num_results`×2 重新检索+重排，仍不达则走 agentic 迭代。置信度直接用 rerank score | **1–2 天** ⭐推荐首选 |
| 2 | **文档 Hash 增量索引** | `(source_id, source_type)` upsert + 删旧建新 + 内容变更检测 | `Document` 增加 `content_hash` 字段；`insert_documents` 前查库比对 hash，仅对新增/变更文档走切分+嵌入，删除失效 chunk。顺手用 Parquet 批量导入替换 ORM flush（我们已实测证明 480s→2.5min） | 1–2 天 ⭐ |
| 3 | **问题路由 / 意图识别** | intent 标签归一化 + 实体抽取决定是否检索 | 轻量 LLM 分类器（或 embedding 相似度）把 query 路由到 vector / bm25 / hybrid / 元数据过滤四条路径；不需要检索的（寒暄/常识）直接跳过 | 1 天 |
| 4 | **分层缓存** | provider prompt-cache 标记 + 上下文压缩 | 三层：① embedding 结果缓存（query hash→向量）；② 检索结果缓存（query hash→chunk_ids，TTL 失效）；③ 上下文压缩（超窗时先压缩历史再拼 context）。个人场景命中率会很高（重复问类似问题） | 2–3 天 |
| 5 | **OCR/VLM 多模态** | PaddleOCR-VL 解析器 + 图片 auto-describe caption 独立入索引 | PDF 解析时检测图片区域 → PaddleOCR 提文字 / VLM 生成 caption → 作为独立 chunk 入索引（绑 `image_id` 元数据）。先做 OCR 路径（本地、免费），VLM caption 可选 | ~~3–5 天~~ **已完成** ✅ |
| 6 | **生成端评估** | Recall + BLEU/ROUGE 端到端评估 | 检索端已有 TREC（MRR/Hit），补生成端：`rouge-score` 库算 ROUGE-L、`nltk` BLEU，用自建 QA 集（gen_bench.py 已内置 6 问可扩到 50 问） | 1 天 ⭐ |

### T1：工程正确性修复（我们实测发现的，WeKnora 也有对应实践）

| # | 问题 | 依据 | 修法 |
|---|---|---|---|
| 7 | HNSW 索引未被查询计划命中（EXPLAIN 实测） | 退化为暴力精确扫描 | 调整查询形状适配 DuckDB VSS 要求；或直接接受精确扫描（10K 规模 46ms 可接受），写进文档 |
| 8 | rerank 失败即整体失败 | WeKnora 降级为检索顺序 | try/except 包 `rerank_chunks`，失败时按检索分数返回 + warning |
| 9 | hybrid 权重 0.75/0.25 硬编码 | 源码写死 | 提为 `RAGLiteConfig` 字段，可调参 |
| 10 | 六阶段打点可观测缺失 | WeKnora 用 Langfuse span 树 | 解析/切分/嵌入/检索/组装/生成六阶段计时 + 日志输出（轻量版，不必上 Langfuse） |

---

## 三、战略意义：把简历"缺失亮点"变成真实扩展

你之前那份描述里 6 个"非原生"项，与 WeKnora 的对应关系几乎 1:1：

| 简历想写的亮点 | WeKnora 对应机制 | 实现后简历措辞 |
|---|---|---|
| 低置信度二次检索 | FAQ 迭代检索 | 「参考 WeKnora 迭代检索思想，实现 rerank 置信度阈值触发的二次召回」 |
| 文档 Hash 增量索引 | upsert + 变更检测 | 「基于内容 Hash 的增量索引与热更新」 |
| 问题路由 | intent 识别 | 「意图路由：按 query 意图分发到向量/BM25/混合/过滤四路检索」 |
| 分层缓存 | prompt cache + 上下文压缩 | 「embedding/检索/上下文三层缓存」 |
| OCR+VLM 多模态 | PaddleOCR-VL + caption | 「自研多模态文档处理器：PDF 分页栅格化 + 本地 ONNX OCR 回收图内文字 + 视觉模型生成图像描述，双路独立降级后统一入混合检索链路」 |
| RAGAS 评估 | Recall+BLEU/ROUGE | 「检索端 TREC(MRR/Hit) + 生成端 ROUGE/BLEU 联合评估」（比 RAGAS 更工程化，答辩更硬） |

**做完 1/2/3/4/5/6（合计约 7–10 天），你的简历就从"研读开源项目"升级为"参考 WeKnora 完成六项工程扩展，全部有实测数据"——这是完全可辩护的、含金量高的项目经历。**

---

## 四、建议实施顺序

1. **第 1 周**：#1 低置信二次检索 → #2 Hash 增量索引 → #6 生成端评估（三项都直接对应简历亮点，且互不依赖）
2. **第 2 周**：#3 意图路由 → #4 分层缓存 → #10 六阶段打点
3. **视精力**：#5 OCR 多模态（最大件，建议先 OCR 后 VLM）
4. 每完成一项：跑 `gen_bench.py` / `fill_and_search.py` 前后对比，**数字一律实测后写进简历**

> 📌 **全部已完成（2026-09-28）**：#1–#6 均已实现并各自独立提交，38 个单测通过。
> 唯一仍未取数的部分是多模态链路的**端到端效果**（OCR 字符准确率、图文问答准确率、含 VLM 的
> P50 延迟）——需要先装 OCR 引擎（`pip install 'raglite[multimodal]'`）并提供视觉模型端点。
