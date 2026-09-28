"""A/B evaluation of the graph-enhanced retrieval path (`raglite._graph`).

Question
--------
Does adding the graph ranking as a third RRF input improve relational questions
without hurting single-lookup questions?

Design
------
Two classes of questions over one fixed corpus:

* **single-lookup** — the question names a distinctive term and the answer lives
  in exactly one chunk that contains that term. A lexical or vector retriever
  should already get this right, so the graph path must not hurt it.
* **relational** — the question names entity X, and the answer chunk mentions
  *X's neighbour* Y but neither X nor any term of the question. The two chunks
  are connected in the graph by the relation X–Y, so one-hop expansion over the
  graph should surface the answer that no surface-form retriever can reach.

Three arms share an identical vector arm and keyword arm; only the graph is
toggled, and RRF is *not* renormalised when the third ranking is added (RRF does
not require the weights to sum to one), so the vector:keyword balance is held
constant at 0.75:0.25 in every arm.

Environment caveat (read this before trusting the delta)
--------------------------------------------------------
This machine has no LLM endpoint and no way to download a model: the default
sentence splitter (`wtpsplit` SaT) and the default embedder both need network
access and both fail here with `ProxyError 502`. As in the repository's other
benchmarks (`bench_resume.py`), the two model calls are therefore replaced with
deterministic offline stand-ins:

* `offline_split_sentences` — a regex sentence splitter.
* `offline_embed` — a hashed bag-of-words embedding at 256 dims (DuckDB's FLOAT[N] parameter
  binding costs ~3 ms per element, so 1536 dims would make the corpus insert ~4.5 s per embedding
  and dominate the run; the dimension does not affect the ablation because the vector arm is
  identical in every arm).

The hashed embedding has **no synonymy and no semantics**, so the vector arm is
weaker than a real embedder would be. That makes the relational-class baseline
*harder* than it would be in production, and so the measured graph gain is an
**upper bound**, not a production estimate. Entity extraction is likewise run
twice: once with a glossary (oracle) extractor to isolate the fusion machinery,
and once with the real offline `CooccurrenceExtractor` to show what the shipped
offline extractor actually recovers. The LLM extractor ships in `_graph.py` too,
but cannot be measured without an endpoint.

Run
---
    .venv/Scripts/python.exe bench_graph_ab.py

Writes `bench_graph_ab.json` next to this file and prints a per-arm summary.
"""

import json
import os
import re
import time
import warnings
import zlib
from dataclasses import replace
from pathlib import Path
from typing import Any

# Stop LiteLLM from fetching its remote price table (three ~7 s proxy timeouts per process).
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import numpy as np
from sqlmodel import Session, col, select

from raglite import Document, RAGLiteConfig, insert_documents
from raglite._database import Chunk, create_database_engine
from raglite._graph import (
    CooccurrenceExtractor,
    ExtractedEntity,
    ExtractedRelation,
    GraphExtraction,
    build_graph,
    graph_search,
    graph_stats,
)

EMBED_DIM = 256  # See `install_offline_models`: DuckDB's FLOAT[N] parameter binding costs ~3 ms
# per element, so a 1536-dim insert costs ~4.5 s and would dominate the whole run. The dimension
# is irrelevant to this ablation because the vector arm is a hash embedding in every arm.
K_VALUES = (1, 3, 5)  # Recall cut-offs.
OUT_JSON = Path(__file__).parent / "bench_graph_ab.json"

# ---------------------------------------------------------------------------
# Offline stand-ins for the two model calls (see the module docstring).
# ---------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?。！？])\s*")


def offline_split_sentences(content: str, max_len: int = 2048) -> list[str]:
    """Split a document into sentences with a regex, without a network model."""
    sentences: list[str] = []
    for paragraph in content.split("\n"):
        if paragraph.strip():
            sentences.extend(s.strip() for s in _SENTENCE_END.split(paragraph) if s.strip())
    # Keep every sentence under the limit so `split_chunklets` never has to hard-wrap.
    return [sentence[:max_len] for sentence in sentences]


def _tokens(text: str) -> list[str]:
    """Tokenize into lowercased Latin words and individual CJK characters."""
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())


def _stable_hash(token: str) -> int:
    """A process-independent hash (Python's `hash` is salted per process)."""
    return zlib.crc32(token.encode("utf-8"))


def offline_embed(strings: list[str], **_kwargs: Any) -> np.ndarray:
    """Deterministic hashed bag-of-words embeddings, L2-normalised.

    Deterministic across processes, so the A/B is reproducible. Has no semantics:
    two synonyms that share no characters land in unrelated dimensions.
    """
    embeddings = np.zeros((len(strings), EMBED_DIM), dtype=np.float32)
    for row, text in enumerate(strings):
        for token in _tokens(text):
            embeddings[row, _stable_hash(token) % EMBED_DIM] += 1.0
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


# ---------------------------------------------------------------------------
# Corpus: 24 short documents (one chunk each), in two domain clusters.
# ---------------------------------------------------------------------------

CORPUS: list[tuple[str, str]] = [
    # Cluster 1: aerosol metrology.
    ("d01", "SMPS 由 DMA 与 CPC 串联组成。SMPS 的时间分辨率主要由扫描时间决定。"),
    ("d02", "DMA 的鞘气流量与样品流量之比限定了仪器能覆盖的尺寸跨度。该比值漂移会带来系统偏差。"),
    ("d03", "CPC 通过丁醇过饱和使颗粒长大到光学可测尺寸。CPC 的计数效率在十纳米以下迅速下降。"),
    ("d04", "FIMS 使用法拉第杯测量气溶胶的电迁移率。FIMS 与 SMPS 共用同一条采样管路。"),
    ("d05", "扩散干燥管通过选择性扩散去除水蒸气而不显著损失颗粒。扩散干燥管常与雾化器配套使用。"),
    ("d06", "雾化器把溶液雾化成多分散液滴。雾化器产生的液滴随后进入扩散干燥管。"),
    ("d07", "管式炉在气溶胶发生链路中用于热降解或烧结。管式炉的温度直接影响颗粒的形貌。"),
    ("d08", "标定台账按三千到六千伏的实验编号归档。标定台账记录日期与文件编号。"),
    ("d17", "DMA 的传递函数描述其电场对带电颗粒的透过概率。传递函数是粒径反演的核心输入。"),
    ("d18", "CPC 的计数效率需要用小粒径标准颗粒标定。计数效率与过饱和度相关。"),
    ("d19", "采样管路的传递效率受扩散与沉降影响。传递效率损失会使计数偏低。"),
    ("d20", "电迁移率反映颗粒在电场中的迁移速度。电迁移率与颗粒的带电数成正比。"),
    # Cluster 2: retrieval internals.
    ("d09", "HNSW 是一种分层可导航小世界图索引。HNSW 用近似最近邻换取数量级的查询加速。"),
    ("d10", "BM25 基于词频与逆文档频率为文档打分。BM25 对精确术语匹配尤其有效。"),
    ("d11", "RRF 把多路排序按倒数排名融合。RRF 无需分数归一化即可融合异构检索器。"),
    ("d12", "FlashRank 是交叉编码器重排器。FlashRank 逐对编码查询与候选文档以提升精度。"),
    ("d13", "DuckDB 是进程内分析型数据库。DuckDB 同时提供向量检索与全文检索扩展。"),
    ("d14", "chunklet 是句子级的最小嵌入单元。chunklet 的嵌入在切分之前计算。"),
    ("d15", "迟绑定在切分成块之后对句子嵌入做池化以保留上下文。迟绑定依赖 token 级嵌入。"),
    ("d16", "BIP 用嵌入相似度切分最优分块边界。BIP 在块内一致性与块间差异之间取平衡。"),
    ("d21", "查询意图路由根据问题类型选择检索方法。查询意图路由不调用任何模型。"),
    ("d22", "分层缓存为查询向量与检索结果分别设置存活时间。分层缓存能降低重复查询延迟。"),
    ("d23", "低置信度二次检索在首轮重排分低于阈值时扩大候选池。二次检索以延迟换召回。"),
    ("d24", "内容哈希增量索引让未修改文档跳过重建。内容哈希增量索引与图层的跳过逻辑同源。"),
]

# The entity glossary, used by the oracle extractor and by the relation oracle.
GLOSSARY: list[str] = [
    "SMPS", "DMA", "CPC", "FIMS", "扩散干燥管", "雾化器", "管式炉", "标定台账",
    "传递函数", "计数效率", "传递效率", "电迁移率", "法拉第杯", "鞘气流量",
    "HNSW", "BM25", "RRF", "FlashRank", "DuckDB", "chunklet", "迟绑定", "BIP",
    "查询意图路由", "分层缓存", "二次检索", "内容哈希增量索引",
]

# ---------------------------------------------------------------------------
# Question set. `gold` is the set of documents that answer the question.
# ---------------------------------------------------------------------------

QUESTIONS: list[dict[str, Any]] = [
    # -- single-lookup: the answer chunk contains the question's distinctive term.
    {"id": "S1", "cls": "single", "q": "HNSW 用什么换取查询加速？", "gold": ["d09"]},
    {"id": "S2", "cls": "single", "q": "BM25 的打分依据是什么？", "gold": ["d10"]},
    {"id": "S3", "cls": "single", "q": "RRF 为什么不需要分数归一化？", "gold": ["d11"]},
    {"id": "S4", "cls": "single", "q": "FlashRank 的重排方式是什么？", "gold": ["d12"]},
    {"id": "S5", "cls": "single", "q": "DuckDB 提供哪些检索扩展？", "gold": ["d13"]},
    {"id": "S6", "cls": "single", "q": "chunklet 的嵌入在哪一步计算？", "gold": ["d14"]},
    {"id": "S7", "cls": "single", "q": "雾化器把溶液变成什么？", "gold": ["d06"]},
    {"id": "S8", "cls": "single", "q": "管式炉的温度影响颗粒的什么性质？", "gold": ["d07"]},
    # -- relational: the answer chunk mentions X's neighbour, not X.
    #    X=SMPS, neighbour=DMA -> d02. The question never says "鞘气流量" or "尺寸跨度".
    {"id": "R1", "cls": "relational", "q": "SMPS 能覆盖多宽的粒径范围，这由什么决定？", "gold": ["d02"]},
    #    X=SMPS, neighbour=CPC -> d03 / d18.
    {"id": "R2", "cls": "relational", "q": "SMPS 在小粒径端的响应为什么变差？", "gold": ["d03"]},
    {"id": "R3", "cls": "relational", "q": "SMPS 里那个传感器在小尺寸上为什么要校准？", "gold": ["d18"]},
    #    X=CPC, neighbour=管式炉/发生链路 -> d07 (no surface overlap with the question).
    {"id": "R4", "cls": "relational", "q": "颗粒进入 CPC 之前那道加热环节会影响什么？", "gold": ["d07"]},
    #    X=FIMS, neighbour=电迁移率 -> d20.
    {"id": "R5", "cls": "relational", "q": "FIMS 测量的那个物理量与带电数是什么关系？", "gold": ["d20"]},
    #    X=SMPS, neighbour=传递函数 -> d17.
    {"id": "R6", "cls": "relational", "q": "SMPS 反演粒径时依赖的那个核心输入是什么？", "gold": ["d17"]},
    #    X=chunklet/BIP chain, neighbour=迟绑定 -> d15.
    {"id": "R7", "cls": "relational", "q": "BIP 切分与哪种嵌入方式配合才能保留上下文？", "gold": ["d15"]},
    #    X=二次检索, neighbour=RRF -> d11.
    {"id": "R8", "cls": "relational", "q": "二次检索扩大候选池之后，多路结果是怎么合并的？", "gold": ["d11"]},
]


# ---------------------------------------------------------------------------
# Extractors.
# ---------------------------------------------------------------------------


class GlossaryExtractor:
    """Oracle extractor: recognises every glossary entity present in the text.

    Isolates the fusion machinery from extraction quality. Relations are all
    glossary pairs that co-occur in one sentence, which is what a competent LLM
    extraction would produce for these short technical paragraphs.
    """

    def __init__(self, glossary: list[str]) -> None:
        # Longest first, so "内容哈希增量索引" wins over a shorter overlap.
        self.glossary = sorted(glossary, key=len, reverse=True)

    def extract(self, text: str) -> GraphExtraction:
        present = [term for term in self.glossary if term.lower() in text.lower()]
        entities = [ExtractedEntity(title=term, type="concept") for term in present]
        relations: list[ExtractedRelation] = []
        for sentence in _SENTENCE_END.split(text):
            in_sentence = [term for term in present if term.lower() in sentence.lower()]
            for i, source in enumerate(in_sentence):
                for target in in_sentence[i + 1 :]:
                    relations.append(
                        ExtractedRelation(source=source, target=target, description="co-occurs")
                    )
        return GraphExtraction(entities=entities, relations=relations)


def install_offline_models() -> None:
    """Patch the model calls and the file lock that cannot run here."""
    import raglite._database as database_module
    import raglite._embed as embed_module
    import raglite._insert as insert_module
    import raglite._search as search_module

    for module in (embed_module, insert_module, search_module):
        module.embed_strings = offline_embed
    insert_module.embed_strings_without_late_chunking = offline_embed
    embed_module.embed_strings_without_late_chunking = offline_embed
    insert_module.split_sentences = offline_split_sentences
    # Pin the vector column's dimension to the offline embedding's, instead of asking LiteLLM.
    database_module.get_embedding_dim = lambda config=None: EMBED_DIM
    # `insert_documents` locks the database through a `.lock` file and deletes it on release; this
    # sandbox denies that deletion, and the run is single-process anyway, so use a no-op lock.
    insert_module.FileLock = _NullLock


class _NullLock:
    """A `filelock.FileLock` stand-in that never creates or deletes a file."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def __enter__(self) -> "_NullLock":
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False


def install_extension_loader() -> None:
    """Load DuckDB's fts/vss extensions on *every* connection.

    `create_database_engine` issues `INSTALL vss; LOAD vss;` on whichever connection its own
    session happens to check out. A later statement that touches the HNSW index on a *different*
    pooled connection then makes DuckDB try to autoload the extension itself, which on this box
    goes to the network and stalls behind a broken proxy. This hook makes `LOAD` unconditional so
    the whole run stays offline and deterministic.
    """
    from sqlalchemy import event

    import raglite._delete as delete_module
    import raglite._database as database_module
    import raglite._eval as eval_module
    import raglite._graph as graph_module
    import raglite._incremental as incremental_module
    import raglite._insert as insert_module2
    import raglite._iterative_search as iterative_module
    import raglite._search as search_module2

    original = database_module.create_database_engine

    def hooked_create_database_engine(config: Any = None) -> Any:
        engine = original(config)
        if not getattr(engine, "_graph_ab_extensions_loaded", False):
            engine._graph_ab_extensions_loaded = True  # noqa: SLF001

            @event.listens_for(engine, "connect")
            def _load_extensions(dbapi_connection: Any, _record: Any) -> None:
                cursor = dbapi_connection.cursor()
                # `LOAD` only: the extensions are already installed by `create_database_engine`,
                # and re-issuing `INSTALL` from every worker thread is what stalls.
                cursor.execute("LOAD fts;")
                cursor.execute("LOAD vss;")
                cursor.execute("SET hnsw_enable_experimental_persistence = true;")
                cursor.close()

        return engine

    for module in (
        database_module,
        graph_module,
        search_module2,
        insert_module2,
        delete_module,
        incremental_module,
        eval_module,
        iterative_module,
    ):
        if hasattr(module, "create_database_engine"):
            module.create_database_engine = hooked_create_database_engine


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------


def recall_at_k(ranked_docs: list[str], gold: set[str], k: int) -> float:
    """Fraction of gold documents found in the top k."""
    if not gold:
        return float("nan")
    return len(set(ranked_docs[:k]) & gold) / len(gold)


def reciprocal_rank(ranked_docs: list[str], gold: set[str], k: int) -> float:
    """Reciprocal rank of the first gold document in the top k, else 0."""
    for rank, doc_id in enumerate(ranked_docs[:k], start=1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0


def build_database(config: RAGLiteConfig) -> dict[str, int]:
    """Insert the corpus and build the graph, returning a build report."""
    documents = [
        Document(id=doc_id, filename=f"{doc_id}.md", content=content)
        for doc_id, content in CORPUS
    ]
    insert_documents(documents, config=config)
    with warnings.catch_warnings():
        # The co-occurrence extractor legitimately finds nothing in some chunks.
        warnings.simplefilter("ignore")
        report = build_graph(config=config)
    return {key: int(value) for key, value in report.items() if isinstance(value, int)}


def run_arm(
    name: str, *, graph_enabled: bool, graph_weight: float, probe: RAGLiteConfig, num_results: int = 5
) -> dict[str, Any]:
    """Query every question under one arm and return per-question + aggregate metrics.

    `probe` is a graph-enabled config used only to report what the graph ranking alone
    retrieves, so that the mechanism (not the fusion) can be inspected for every arm.
    """
    config = probe
    config = replace(
        config,
        hybrid_graph_weight=graph_weight,
        graph=replace(config.graph, enabled=graph_enabled),
    )
    probe = replace(probe, graph=replace(probe.graph, enabled=True))
    from raglite._search import hybrid_search

    rows: list[dict[str, Any]] = []
    for item in QUESTIONS:
        gold = set(item["gold"])
        chunk_ids, scores = hybrid_search(item["q"], num_results=num_results, config=config)
        ranked_docs = chunk_to_docs(chunk_ids, config)
        # The graph ranking on its own, to show whether the mechanism fires at all.
        graph_docs = chunk_to_docs(
            graph_search(item["q"], num_results=num_results, config=probe)[0], probe
        )
        rows.append(
            {
                "id": item["id"],
                "cls": item["cls"],
                "question": item["q"],
                "gold": sorted(gold),
                "ranked_docs": ranked_docs,
                "gold_ranks": {
                    doc_id: (ranked_docs.index(doc_id) + 1 if doc_id in ranked_docs else None)
                    for doc_id in sorted(gold)
                },
                "recall": {str(k): recall_at_k(ranked_docs, gold, k) for k in K_VALUES},
                "rr": reciprocal_rank(ranked_docs, gold, num_results),
                "graph_only_docs": graph_docs,
                "graph_only_hit": bool(set(graph_docs) & gold),
                "top_score": round(scores[0], 4) if scores else None,
            }
        )
    aggregate: dict[str, Any] = {}
    for cls in ("single", "relational"):
        subset = [row for row in rows if row["cls"] == cls]
        if not subset:
            continue
        aggregate[cls] = {
            "n": len(subset),
            **{
                f"recall@{k}": round(
                    float(np.mean([row["recall"][str(k)] for row in subset])), 4
                )
                for k in K_VALUES
            },
            "mrr@5": round(float(np.mean([row["rr"] for row in subset])), 4),
            "graph_only_hit_rate": round(
                float(np.mean([1.0 if row["graph_only_hit"] else 0.0 for row in subset])), 4
            ),
        }
    return {"arm": name, "aggregate": aggregate, "per_question": rows}


def chunk_to_docs(chunk_ids: list[str], config: RAGLiteConfig) -> list[str]:
    """Map a ranked chunk list to a de-duplicated, order-preserving document list."""
    if not chunk_ids:
        return []
    with Session(create_database_engine(config)) as session:
        rows = session.exec(
            select(Chunk.document_id).where(col(Chunk.id).in_(chunk_ids))
        ).all()
    by_chunk = dict(zip(chunk_ids, rows, strict=False))
    ordered: list[str] = []
    for chunk_id in chunk_ids:
        doc_id = by_chunk.get(chunk_id)
        if doc_id is not None and doc_id not in ordered:
            ordered.append(doc_id)
    return ordered


def main() -> None:
    os.chdir(Path(__file__).parent)
    # A fresh database per extractor, named uniquely per run: deleting an existing one is not
    # allowed here, and a stale database would silently mix corpora.
    run_id = time.strftime("%Y%m%d-%H%M%S")
    install_offline_models()
    install_extension_loader()

    arms: list[dict[str, Any]] = []
    for extractor_name, extractor in (
        ("glossary_oracle", GlossaryExtractor(GLOSSARY)),
        ("cooccurrence_offline", CooccurrenceExtractor()),
    ):
        db_file = f"bench_graph_ab_{extractor_name}_{run_id}.db"
        base = RAGLiteConfig(
            db_url=f"duckdb:///{db_file}",
            reranker=None,
            **{"llm": "gpt-4o-mini", "embedder": "text-embedding-3-small"},
        )
        build_config = replace(
            base,
            graph=replace(base.graph, enabled=True, extractor=extractor),
        )
        print(f"[build] extractor={extractor_name}", flush=True)
        report = build_database(build_config)
        stats = graph_stats(config=build_config)
        print(f"[build] done {report} {stats}", flush=True)
        for arm_name, enabled, weight in (
            ("baseline_2path", False, 0.0),
            ("graph_w0.1", True, 0.1),
            ("graph_w0.3", True, 0.3),
        ):
            print(f"[arm] {extractor_name}/{arm_name}", flush=True)
            result = run_arm(
                arm_name, graph_enabled=enabled, graph_weight=weight, probe=build_config
            )
            result["extractor"] = extractor_name
            result["graph_stats"] = stats
            result["build"] = report
            arms.append(result)

    output = {"arms": arms}
    OUT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    for result in arms:
        print(f"\n=== extractor={result['extractor']}  arm={result['arm']} ===")
        print(f"    graph stats: {result['graph_stats']}   build: {result['build']}")
        for cls, agg in result["aggregate"].items():
            print(f"    {cls:11s} {agg}")
        misses = [
            f"{row['id']}(gold_rank={list(row['gold_ranks'].values())})"
            for row in result["per_question"]
            if row["cls"] == "relational" and row["recall"]["5"] < 1.0
        ]
        if misses:
            print(f"    relational misses@5: {', '.join(misses)}")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
