"""Standard-dataset retrieval evaluation: BEIR via `ir_datasets` and `ir_measures`.

Why this script exists
----------------------
`bench_graph_ab.py` measures the graph ablation on a hand-built corpus of 24 documents and 16
questions. That is enough to decide whether a mechanism fires, but far too small to support any
effect claim, and nothing in it can be compared against published numbers. This script instead
runs the repository's DuckDB FTS (BM25) retrieval on an official BEIR dataset with official
qrels, so the resulting nDCG@10 / Recall@100 can be placed next to published baselines.

It also exercises the `IREvaluator` interface in `weknow._bench`, which until now had never been
run: the evaluator writes an ir_measures-compatible TREC run to the raglite data directory.

No model is required
--------------------
BM25 needs no embedder, no LLM and no network access beyond the one-off dataset download, so this
runs on a machine where the dense and generation paths cannot (no model endpoint, no GPU).

Run
---
    .venv/Scripts/python.exe bench_beir.py                    # beir/nfcorpus/test
    .venv/Scripts/python.exe bench_beir.py beir/scifact/test

Writes `bench_beir_<dataset>.json` next to this file and prints the aggregate metrics, followed
by the published BM25 baselines for the same dataset for context.
"""

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import duckdb
import ir_datasets
from ir_measures import MRR, Recall, ScoredDoc, calc_aggregate, nDCG
from tqdm.auto import tqdm

from weknow import keyword_search
from weknow._bench import IREvaluator
from weknow._config import RAGLiteConfig
from weknow._database import create_database_engine

DATASET_ID = sys.argv[1] if len(sys.argv) > 1 else "beir/nfcorpus/test"
NUM_RESULTS = 1000

# Published BM25 nDCG@10 for the datasets we care about, so the run can be sanity-checked against
# numbers this repository had no part in producing. Sources: the BEIR paper (Thakur et al., 2021)
# and Anserini's BEIR v1.0.0 regression tables.
PUBLISHED_BM25_NDCG10 = {
    "beir/nfcorpus/test": {"BEIR paper": 0.325, "Anserini (flat)": 0.3218, "Vespa BM25": 0.313},
    "beir/scifact/test": {"BEIR paper": 0.665, "Anserini (flat)": 0.673},
    "beir/trec-covid": {"BEIR paper": 0.656, "Vespa BM25": 0.690},
    "beir/fiqa/test": {"BEIR paper": 0.236, "Vespa BM25": 0.244},
}

# An LLM and embedder string must be set explicitly: their default factories resolve to
# llama-cpp-python, whose lazy import fails on machines without the extra installed. The reranker
# must be explicitly disabled: its default factory eagerly downloads a FlashRank model from
# HuggingFace, which this benchmark never uses (BM25 only) and which may be unreachable.
BASE_KWARGS: dict[str, Any] = {
    "llm": "gpt-4o-mini",
    "embedder": "text-embedding-3-small",
    "reranker": None,
    "self_query": False,
}


class BM25Evaluator(IREvaluator):
    """RAGLite's DuckDB FTS (BM25) retrieval, evaluated on a standard IR dataset."""

    def __init__(
        self,
        dataset: Any,
        *,
        num_results: int = NUM_RESULTS,
        config: RAGLiteConfig | None = None,
    ) -> None:
        super().__init__(dataset, num_results=num_results)
        self.db_filepath = self.cwd / f"{self.insert_id}.db"
        self.config = replace(
            config or RAGLiteConfig(**BASE_KWARGS),
            db_url=f"duckdb:///{self.db_filepath.as_posix()}",
        )

    def insert_documents(self, max_workers: int | None = None) -> None:
        """Index every dataset document as a single document-level chunk.

        BEIR is a document retrieval benchmark, so the corpus is indexed flat - one chunk per
        document holding its title and text - which is also what Anserini's "flat" regressions do.

        Rows are bulk-loaded over a native DuckDB connection instead of the ORM, for the same
        reason as in `fill_and_search.py`: the ORM flush path is not what this benchmark measures,
        and it is orders of magnitude slower than `executemany` here.
        """
        create_database_engine(self.config).dispose()  # Create tables and indexes, release pool.
        connection = duckdb.connect(self.db_filepath.as_posix())
        connection.execute("LOAD vss")
        connection.execute("LOAD fts")
        document_rows: list[tuple[str, str, None, str]] = []
        chunk_rows: list[tuple[str, str, int, str, str, str]] = []
        for dataset_document in tqdm(
            self.dataset.docs_iter(),
            total=self.dataset.docs_count(),
            desc="Indexing documents",
            unit="doc",
            dynamic_ncols=True,
        ):
            title = getattr(dataset_document, "title", "") or ""
            body = f"{title}\n\n{dataset_document.text}".strip()
            document_rows.append((dataset_document.doc_id, dataset_document.doc_id, None, "{}"))
            chunk_rows.append((dataset_document.doc_id, dataset_document.doc_id, 0, title, body, "{}"))
        connection.executemany("INSERT INTO document VALUES (?, ?, ?, ?)", document_rows)
        connection.executemany("INSERT INTO chunk VALUES (?, ?, ?, ?, ?, ?)", chunk_rows)
        connection.execute("PRAGMA create_fts_index('chunk', 'id', 'body', overwrite = 1);")
        # Flush the write-ahead log so the next connection (opened by `keyword_search`) sees the
        # rows, mirroring what `insert_documents` does with CHECKPOINT for DuckDB.
        connection.execute("CHECKPOINT;")
        connection.close()

    def search(self, query_id: str, query: str, *, num_results: int = NUM_RESULTS) -> list[ScoredDoc]:
        chunk_ids, scores = keyword_search(query, num_results=num_results, config=self.config)
        return [
            ScoredDoc(query_id=query_id, doc_id=chunk_id, score=float(score))
            for chunk_id, score in zip(chunk_ids, scores, strict=False)
        ]


def main() -> None:
    dataset = ir_datasets.load(DATASET_ID)
    evaluator = BM25Evaluator(dataset)
    print(f"dataset           : {DATASET_ID}")
    print(f"documents         : {dataset.docs_count()}")
    print(f"queries           : {dataset.queries_count()}")
    print(f"qrels             : {dataset.qrels_count()}")
    print(f"trec run          : {evaluator.trec_run_filepath}")
    print()
    # `score()` reads a cached TREC run if one exists, otherwise indexes and runs every query.
    run = list(
        tqdm(
            evaluator.score(),
            total=dataset.queries_count(),
            desc="Running queries",
            unit="query",
            dynamic_ncols=True,
        )
    )
    qrels = list(dataset.qrels_iter())
    metrics = [nDCG @ 10, Recall @ 100, nDCG @ 100, MRR @ 10]
    aggregate = calc_aggregate(metrics, qrels, run)
    scores = {str(metric): round(float(value), 4) for metric, value in aggregate.items()}
    print()
    for name, value in scores.items():
        print(f"{name:>12s} : {value}")
    print()
    baselines = PUBLISHED_BM25_NDCG10.get(DATASET_ID)
    if baselines:
        print("published BM25 nDCG@10 for reference:")
        for source, value in baselines.items():
            print(f"  {source:>16s} : {value}")
        ours = scores.get("nDCG@10")
        if ours is not None:
            published = sorted(baselines.values())
            median = published[len(published) // 2]
            gap = round(100 * (ours - median), 2)
            verdict = "comparable" if abs(gap) <= 5 else "check the pipeline, this is off"
            print(f"  -> ours {ours} vs median {median} ({gap:+.2f} points) => {verdict}")
    output_path = Path(__file__).parent / f"bench_beir_{DATASET_ID.replace('/', '_')}.json"
    output_path.write_text(
        json.dumps(
            {
                "dataset": DATASET_ID,
                "documents": dataset.docs_count(),
                "queries": dataset.queries_count(),
                "qrels": dataset.qrels_count(),
                "retriever": "raglite DuckDB FTS (match_bm25), document-level flat index",
                "num_results": NUM_RESULTS,
                "scores": scores,
                "published_bm25_ndcg10": baselines,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print(f"wrote {output_path.name}")


if __name__ == "__main__":
    main()
