"""Graph-retrieval ablation on a standard IR dataset (BEIR via `ir_datasets` / `ir_measures`).

What this measures
------------------
`bench_graph_ab.py` already ran this ablation, but on a hand-built corpus of 24 documents and 16
questions. Its conclusion ("two arms tie; the only difference is one question flipping") has no
statistical weight at that sample size. This script repeats the ablation on an official BEIR
dataset with official qrels, so the answer is a number with hundreds of queries behind it rather
than a coin flip.

Honest scope: two paths, not three
----------------------------------
`hybrid_search` fuses *vector* + *keyword* + *graph*. This machine has no reachable embedding
endpoint, so the vector path cannot run and the fusion here is **keyword (BM25) + graph**, with
the graph weight swept from 0. A weight of 0 is therefore not a different arm - it *is* the
baseline, which is a useful sanity check that the harness is wired up correctly.

Graph build
-----------
The graph is built with `CooccurrenceExtractor`, which is deterministic and needs no model: it
keeps repeated terms and relates those that share a sentence. It is not a semantic extractor, so
a weak graph here does **not** refute the mechanism - it only bounds what a purely lexical graph
buys on top of BM25. The `LLMEntityExtractor` arm remains blocked on a model endpoint.

Arms
----
    bm25            : keyword_search alone (equivalently, graph weight 0)
    graph_only      : graph_search alone (graph weight 1, keyword weight 0)
    rrf_w<W>        : reciprocal_rank_fusion(keyword, graph) with keyword weight 1 and graph W
    rrf_w<W>_dedup  : same, with `graph.deduplicate_seen_chunks` on (WeKnora's filterSeenChunk)

Run
---
    .venv/Scripts/python.exe bench_beir_graph.py                       # beir/nfcorpus/test
    .venv/Scripts/python.exe bench_beir_graph.py beir/scifact/test

Writes `bench_beir_graph_<dataset>.json` next to this file.
"""

import json
import random
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import ir_datasets
from bench_beir import BASE_KWARGS, BM25Evaluator
from ir_measures import MRR, Recall, ScoredDoc, calc_aggregate, iter_calc, nDCG
from tqdm.auto import tqdm

from raglite import keyword_search
from raglite._bench import IREvaluator
from raglite._config import RAGLiteConfig
from raglite._graph import CooccurrenceExtractor, GraphConfig, build_graph, graph_search

# `reciprocal_rank_fusion` is not re-exported from the package root; `hybrid_search` is the public
# entry point, but it insists on the vector path, which has no endpoint on this machine.
from raglite._search import reciprocal_rank_fusion

DATASET_ID = sys.argv[1] if len(sys.argv) > 1 else "beir/nfcorpus/test"
NUM_RESULTS = 100
OVERSAMPLE = 2

# The graph weights to sweep. 0.0 is the baseline (graph contributes nothing to RRF), 1.0 puts
# the graph on equal footing with BM25. `None` means "graph disabled entirely".
GRAPH_WEIGHTS: list[float | None] = [None, 0.1, 0.3, 1.0]
DEDUPLICATE_ARMS: list[float] = [0.3]

# How many chunks an entity may contribute to the chunk-chunk projection. The default
# (`GraphConfig.max_chunks_per_entity = 64`) is far too permissive for a corpus this size: it
# projects 3.0M edges over 3,633 chunks, the heaviest chunk reaching 2,852 of a possible 3,632,
# so the graph stops discriminating and every query expands across most of the corpus. Measured on
# NFCorpus, the projected edge count is 187k/484k/1.06M/1.95M/3.05M for a cap of 4/8/16/32/64, and
# query latency is flat across all of them (the expansion is bounded at query time by
# `chunk_neighbor_limit`), so the cap is set low to cut build time and storage.
CHUNK_PROJECTION_CAP = 8

METRICS = [nDCG @ 10, MRR @ 10, nDCG @ 100, Recall @ 100]

# A paired bootstrap over queries, for the deltas. The graph's contribution turns out to be small
# (hundredths of an nDCG point), and a bare aggregate cannot say whether a delta that size is real
# or sampling noise, so every delta is reported as an interval.
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260929
BOOTSTRAP_LOWER = 0.025
BOOTSTRAP_UPPER = 0.975


class GraphEvaluator(IREvaluator):
    """BM25 + graph RRF, evaluated on a standard IR dataset.

    The database is built by `BM25Evaluator.insert_documents` (one flat document-level chunk per
    BEIR document, bulk-loaded over a native DuckDB connection), then the graph is built on top of
    those chunks. `insert_id` is derived from this class name, so this arm does not share a
    database with `bench_beir.py` and can be built and inspected independently.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        graph_weight: float | None,
        deduplicate: bool = False,
        keyword_weight: float = 1.0,
        num_results: int = NUM_RESULTS,
        config: RAGLiteConfig | None = None,
    ) -> None:
        # `graph_weight=None` means the graph is off; otherwise the graph is on and this arm is a
        # distinct search variant, so each arm caches its own TREC run. The variant must carry every
        # field that changes the ranking: `rrf_w1` (keyword 1.0 + graph 1.0) and `graph_only`
        # (graph 1.0 only) share a graph weight, and naming them both "gw1" made them share a TREC
        # file, so whichever ran second silently reused the first one's run (`main` asserts against
        # this now).
        variant = "graphoff" if graph_weight is None else f"gw{graph_weight:g}"
        if keyword_weight != 1.0:
            variant += f"_kw{keyword_weight:g}"
        if deduplicate:
            variant += "_dedup"
        super().__init__(dataset, num_results=num_results, search_variant=variant)
        self.graph_weight = graph_weight
        self.deduplicate = deduplicate
        self.keyword_weight = keyword_weight
        self.graph_hits = 0
        self.db_filepath = self.cwd / f"{self.insert_id}.db"
        self.config = replace(
            config or RAGLiteConfig(**BASE_KWARGS),
            db_url=f"duckdb:///{self.db_filepath.as_posix()}",
            graph=GraphConfig(
                enabled=True,
                extractor=CooccurrenceExtractor(),
                build_chunk_graph=True,
                max_chunks_per_entity=CHUNK_PROJECTION_CAP,
                deduplicate_seen_chunks=deduplicate,
            ),
        )

    def insert_documents(self, max_workers: int | None = None) -> None:
        BM25Evaluator.insert_documents(self, max_workers=max_workers)

    def build(self) -> dict[str, Any]:
        """Build the entity graph, score its edges, and project them onto the chunks.

        `build_graph` already runs `build_chunk_relations` when `GraphConfig.build_chunk_graph` is
        set, and it finishes with `graph_stats`, so its report carries both the extraction counts
        and the resulting graph size.
        """
        return build_graph(config=self.config)

    def search(self, query_id: str, query: str, *, num_results: int = NUM_RESULTS) -> list[ScoredDoc]:
        # A `graph_weight` of None means the graph is off, which is exactly the baseline arm. The
        # graph query is then skipped rather than appended with weight 0: a zero-weight list cannot
        # change the RRF outcome, and running it anyway would make the baseline pay the graph's
        # per-query cost for nothing.
        pool = OVERSAMPLE * num_results
        keyword_ids, _ = keyword_search(query, num_results=pool, config=self.config)
        rankings = [keyword_ids]
        weights = [self.keyword_weight]
        if self.graph_weight is not None:
            seen = list(dict.fromkeys(keyword_ids)) if self.deduplicate else None
            graph_ids, _ = graph_search(
                query,
                num_results=pool,
                exclude_chunk_ids=seen,
                config=self.config,
            )
            self.graph_hits += 1 if graph_ids else 0
            rankings.append(graph_ids)
            weights.append(self.graph_weight)
        # At least one path has to carry weight, or RRF has nothing to rank.
        assert any(weight != 0.0 for weight in weights), weights
        chunk_ids, scores = reciprocal_rank_fusion(rankings, weights=weights)
        return [
            ScoredDoc(query_id=query_id, doc_id=chunk_id, score=float(score))
            for chunk_id, score in zip(chunk_ids[:num_results], scores[:num_results], strict=False)
        ]


def _trusted_trec_run(evaluator: GraphEvaluator) -> bool:
    """Whether the cached TREC run is known to be complete, setting it aside if it is not.

    `IREvaluator.score` reuses `trec_run_filepath` the moment it exists, with no completeness check,
    so a run killed mid-write gets silently scored on its first N queries and still looks like a
    number. Completeness is therefore tracked by a marker written only after the run generator has
    been fully consumed. Counting query ids cannot work: a query that retrieves nothing legitimately
    contributes no lines, so a *complete* NFCorpus run covers 308 of the 323 query ids, and an id
    count compared against `queries_count()` marks every healthy run as truncated.

    An untrusted run is renamed rather than deleted: it is evidence of an interrupted run, and an
    in-process delete can be blocked by a sandbox guard, which would abort the whole experiment.
    """
    path = evaluator.trec_run_filepath
    if path.with_suffix(path.suffix + ".done").exists():
        return True
    if path.exists():
        stale_path = path.with_suffix(path.suffix + ".incomplete")
        path.replace(stale_path)
        print(f"  set aside untrusted TREC run {path.name} -> {stale_path.name}")
    return False


def evaluate(dataset: Any, evaluator: GraphEvaluator) -> tuple[dict[str, float], list[ScoredDoc]]:
    """Score one arm, reusing the cached TREC run when it is marked complete.

    Returns the aggregate scores and the run itself, because the significance test needs the run
    once more, per query.
    """
    trusted = _trusted_trec_run(evaluator)
    evaluator.graph_hits = 0
    run = list(evaluator.score())
    if not trusted:
        run_path = evaluator.trec_run_filepath
        run_path.with_suffix(run_path.suffix + ".done").write_text(
            f"complete: {dataset.queries_count()} queries, {len(run)} scored docs\n", encoding="utf-8"
        )
    qrels = list(dataset.qrels_iter())
    aggregate = calc_aggregate(METRICS, qrels, run)
    scores = {str(metric): round(float(value), 4) for metric, value in aggregate.items()}
    return scores, run


def per_query_scores(dataset: Any, run: list[ScoredDoc]) -> dict[str, dict[str, float]]:
    """Score every query separately, as `{query_id: {metric: value}}`.

    `iter_calc` yields one `(query_id, measure, value)` triple per query and metric; its measure
    names are spelled exactly as `calc_aggregate`'s keys, so the two stay in step. Queries with no
    qrels are absent, which is the same for every arm, so the comparison stays paired.
    """
    qrels = list(dataset.qrels_iter())
    per_query: dict[str, dict[str, float]] = {}
    for query_id, metric, value in iter_calc(METRICS, qrels, run):
        per_query.setdefault(query_id, {})[str(metric)] = float(value)
    return per_query


def paired_bootstrap_delta(
    arm: dict[str, dict[str, float]], baseline: dict[str, dict[str, float]], metric: str
) -> tuple[float, float, float]:
    """Bootstrap the paired per-query delta against the baseline.

    Returns `(observed, lower, upper)` for the mean delta and its 95% interval. The interval covers
    zero exactly when the arm's difference from the baseline is within sampling noise, which is the
    question a 0.001 nDCG delta actually raises.
    """
    shared = sorted(set(arm) & set(baseline))
    differences = [arm[query_id][metric] - baseline[query_id][metric] for query_id in shared]
    observed = statistics.fmean(differences)
    count = len(differences)
    generator = random.Random(BOOTSTRAP_SEED)  # noqa: S311  (a reproducibility seed, not a secret)
    resampled = sorted(
        statistics.fmean([differences[generator.randrange(count)] for _ in range(count)])
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    return (
        observed,
        resampled[int(BOOTSTRAP_LOWER * BOOTSTRAP_SAMPLES)],
        resampled[min(BOOTSTRAP_SAMPLES - 1, int(BOOTSTRAP_UPPER * BOOTSTRAP_SAMPLES))],
    )


def compute_significance(
    dataset: Any, runs: dict[str, list[ScoredDoc]], metrics: list[str]
) -> dict[str, dict[str, list[float]]]:
    """Size every arm's delta against `bm25`, with a paired bootstrap over queries.

    A bare aggregate cannot say whether a delta of a hundredth of an nDCG point is real or noise,
    which is exactly the size the graph produces, so each delta is reported as an interval.
    """
    baseline_per_query = per_query_scores(dataset, runs["bm25"])
    significance: dict[str, dict[str, list[float]]] = {}
    for name, run in runs.items():
        if name == "bm25":
            continue
        arm_per_query = per_query_scores(dataset, run)
        significance[name] = {
            metric: list(paired_bootstrap_delta(arm_per_query, baseline_per_query, metric))
            for metric in metrics
        }
    return significance


def write_results(
    dataset: Any,
    report: dict[str, Any],
    results: dict[str, dict[str, float]],
    significance: dict[str, dict[str, list[float]]],
    graph_hits: dict[str, int],
    metrics: list[str],
) -> Path:
    """Persist the run. Called before anything is printed, so a formatting bug cannot lose it.

    It did once: the delta table keyed on `MRR@10` while `ir_measures` spells it `RR@10`, and the
    exception fired before the JSON was written.
    """
    output_path = Path(__file__).parent / f"bench_beir_graph_{DATASET_ID.replace('/', '_')}.json"
    output_path.write_text(
        json.dumps(
            {
                "dataset": DATASET_ID,
                "documents": dataset.docs_count(),
                "queries": dataset.queries_count(),
                "qrels": dataset.qrels_count(),
                "num_results": NUM_RESULTS,
                "oversample": OVERSAMPLE,
                "metrics": metrics,
                "paths": "keyword (DuckDB FTS match_bm25) + graph (CooccurrenceExtractor)",
                "caveat": "no embedding endpoint available, so the vector path is absent",
                "bootstrap": {
                    "samples": BOOTSTRAP_SAMPLES,
                    "seed": BOOTSTRAP_SEED,
                    "interval": [BOOTSTRAP_LOWER, BOOTSTRAP_UPPER],
                    "paired": True,
                    "unit": "query",
                    "layout": "arm -> metric -> [observed delta, lower, upper]",
                },
                "graph_report": {k: v for k, v in report.items() if not isinstance(v, list)},
                "scores": results,
                "delta_vs_bm25": significance,
                "graph_hits": graph_hits,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path


def print_report(
    results: dict[str, dict[str, float]],
    significance: dict[str, dict[str, list[float]]],
    graph_hits: dict[str, int],
    metrics: list[str],
    query_count: int,
) -> None:
    """Print the score table, then each arm's deltas with their confidence intervals."""
    header = f"{'arm':>16s} | " + " | ".join(f"{metric:>9s}" for metric in metrics) + " | graph-hit"
    print()
    print(header)
    print("-" * len(header))
    for name, scores in results.items():
        cells = " | ".join(f"{scores[metric]:9.4f}" for metric in metrics)
        print(f"{name:>16s} | {cells} | {graph_hits[name]:5d}/{query_count}")

    print()
    print(f"delta vs bm25, paired bootstrap over queries ({BOOTSTRAP_SAMPLES} samples):")
    print("  a delta whose interval covers 0 is not distinguishable from noise")
    for name, per_metric in significance.items():
        print(f"  {name}:")
        for metric, (observed, lower, upper) in per_metric.items():
            mark = " " if lower > 0 or upper < 0 else "~"
            print(f"    {mark} {metric:>9s} : {observed:+.4f}  95% CI [{lower:+.4f}, {upper:+.4f}]")


def build_arms(dataset: Any) -> dict[str, GraphEvaluator]:
    """Construct one evaluator per arm.

    They share a database file, because `insert_id` deliberately does not depend on the search
    variant: the corpus is indexed and the graph built exactly once, for all arms.
    """
    arms: dict[str, GraphEvaluator] = {"bm25": GraphEvaluator(dataset, graph_weight=None)}
    for weight in GRAPH_WEIGHTS:
        if weight is None:
            continue
        arms[f"rrf_w{weight:g}"] = GraphEvaluator(dataset, graph_weight=weight)
    for weight in DEDUPLICATE_ARMS:
        arms[f"rrf_w{weight:g}_dedup"] = GraphEvaluator(dataset, graph_weight=weight, deduplicate=True)
    arms["graph_only"] = GraphEvaluator(dataset, graph_weight=1.0, keyword_weight=0.0)

    # Distinct arms must have distinct TREC files. If two arms collide, the second one finds the
    # first one's cached run, trusts it, and reports the first one's scores under its own name while
    # doing no search at all (`graph_hits` of 0 is the tell). That happened with `rrf_w1` and
    # `graph_only`, which both used to be named "gw1", and the resulting table looked plausible.
    search_ids = [evaluator.search_id for evaluator in arms.values()]
    duplicates = {search_id for search_id in search_ids if search_ids.count(search_id) > 1}
    assert not duplicates, f"arms share a TREC run: {sorted(duplicates)}"
    return arms


def main() -> None:
    dataset = ir_datasets.load(DATASET_ID)
    arms = build_arms(dataset)

    reference = next(iter(arms.values()))
    print(f"dataset           : {DATASET_ID}")
    print(f"documents         : {dataset.docs_count()}")
    print(f"queries           : {dataset.queries_count()}")
    print(f"qrels             : {dataset.qrels_count()}")
    print(f"database          : {reference.db_filepath}")
    print(f"num_results       : {NUM_RESULTS} (oversample {OVERSAMPLE}x per path)")
    print()
    print("building graph with CooccurrenceExtractor ...")
    if not reference.db_filepath.exists():
        reference.insert_documents()
    report = reference.build()
    print(
        f"  graph: entities={report.get('entities')} relations={report.get('relations')} "
        f"chunks_extracted={report.get('chunks_extracted')} failed={report.get('chunks_failed')} "
        f"chunk_relations={report.get('chunk_relations')}"
    )
    # Propagation reminder: the graph lives in the same database file as the corpus, because the
    # arms share `insert_id`. Re-building is idempotent (only the delta is extracted).
    print()

    results: dict[str, Any] = {}
    runs: dict[str, list[ScoredDoc]] = {}
    graph_hits: dict[str, int] = {}
    for name, evaluator in tqdm(arms.items(), desc="Arms", unit="arm", dynamic_ncols=True):
        results[name], runs[name] = evaluate(dataset, evaluator)
        graph_hits[name] = evaluator.graph_hits

    # Metric names come from `ir_measures`, which spells MRR "RR" and Recall "R". Derive them from
    # the scored result instead of hardcoding a spelling, which differs between versions.
    metrics = list(results["bm25"])
    significance = compute_significance(dataset, runs, metrics)
    output_path = write_results(dataset, report, results, significance, graph_hits, metrics)
    print_report(results, significance, graph_hits, metrics, dataset.queries_count())
    print()
    print(f"wrote {output_path.name}")


if __name__ == "__main__":
    main()
