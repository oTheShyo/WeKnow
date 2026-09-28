"""Tests for the graph-enhanced retrieval path.

Covers title normalisation and matching, the extractor resolution and injection, graph
construction and aggregation, content-hash incremental extraction, pruning, the disabled and
no-match degradations, neighbour expansion, metadata filtering, and the RRF fusion of the graph
ranking with the vector and keyword rankings.

Every test runs offline: the scripted extractor controls the graph structure exactly, so neither
an embedding model nor an LLM endpoint is needed.
"""

from dataclasses import dataclass, replace
from typing import Any

import pytest
from sqlmodel import Session, select

from raglite import (
    CooccurrenceExtractor,
    Document,
    ExtractedEntity,
    ExtractedRelation,
    GraphConfig,
    GraphExtraction,
    LLMEntityExtractor,
    RAGLiteConfig,
    build_graph,
    clear_graph,
    entity_id_for_title,
    graph_search,
    graph_stats,
    normalize_entity_title,
    prune_graph,
    resolve_entity_extractor,
)
from raglite._database import Chunk, GraphChunk, create_database_engine
from raglite._graph import title_in_query

# Text markers that route a chunk to a scripted extraction.
MARKER_A = "MARKER-A"
MARKER_B = "MARKER-B"
MARKER_C = "MARKER-C"

BODY_A = f"""{MARKER_A} Einstein published the special theory of relativity in 1905.
The paper introduces the constancy of the speed of light and the relativity of simultaneity.
"""
BODY_B = f"""{MARKER_B} The Einstein field equations describe gravitation as the curvature of spacetime.
Einstein corresponded with Michele Besso while developing the theory.
"""
BODY_C = f"""{MARKER_C} Michele Besso was an engineer and a lifelong friend of Einstein.
"""


@dataclass
class ScriptedExtractor:
    """An extractor that returns a fixed extraction for the marker found in the text."""

    extractions: dict[str, GraphExtraction]

    def extract(self, text: str) -> GraphExtraction:
        """Return the extraction scripted for the marker this text contains."""
        for marker, extraction in self.extractions.items():
            if marker in text:
                return extraction
        return GraphExtraction()


@dataclass
class FailingExtractor:
    """An extractor that always raises, to exercise the degradation path."""

    message: str = "simulated extractor failure"

    def extract(self, text: str) -> GraphExtraction:
        """Always fail."""
        raise RuntimeError(self.message)


def _entity(title: str) -> ExtractedEntity:
    """A minimal extracted entity."""
    return ExtractedEntity(title=title, type="person", description=f"about {title}")


def _relation(source: str, target: str, strength: float = 1.0) -> ExtractedRelation:
    """A minimal extracted relation, with the strength the extractor stated for it."""
    return ExtractedRelation(
        source=source, target=target, description=f"{source} -> {target}", strength=strength
    )


# ---------------------------------------------------------------------------
# Title normalisation, identity, and matching.
# ---------------------------------------------------------------------------


def test_normalize_entity_title() -> None:
    """Surrounding punctuation, quotes, and whitespace are stripped and collapsed."""
    assert normalize_entity_title("  「量子纠错」  ") == "量子纠错"
    assert normalize_entity_title('"Albert Einstein".') == "Albert Einstein"
    assert normalize_entity_title("Albert   Einstein") == "Albert Einstein"
    assert normalize_entity_title("   ") == ""


def test_entity_id_for_title_is_case_insensitive_and_stable() -> None:
    """The same title in different cases maps to one id, and the id is stable across calls."""
    assert entity_id_for_title("Albert Einstein") == entity_id_for_title("albert einstein")
    assert entity_id_for_title("量子纠错") != entity_id_for_title("量子比特")
    assert entity_id_for_title("量子纠错") == entity_id_for_title("  「量子纠错」 ")


def test_title_in_query_uses_word_boundaries_for_latin() -> None:
    """Latin titles must not match inside a longer word; CJK titles match as substrings."""
    assert title_in_query("Einstein", "What did Einstein publish?")
    assert title_in_query("Einstein", "einstein's 1905 paper")  # Case-insensitive.
    assert not title_in_query("art", "The particle chart is partial")
    assert title_in_query("量子纠错", "量子纠错为什么重要")
    assert not title_in_query("量子比特", "量子纠错为什么重要")
    assert not title_in_query("", "anything")


# ---------------------------------------------------------------------------
# Extractor resolution and the LLM extractor.
# ---------------------------------------------------------------------------


def test_resolve_entity_extractor_variants(test_config: RAGLiteConfig) -> None:
    """None builds an LLM extractor, a string names its model, and an instance passes through."""
    default = resolve_entity_extractor(GraphConfig(), test_config)
    assert isinstance(default, LLMEntityExtractor)
    assert default.config is test_config
    assert default.temperature == GraphConfig().extraction_temperature

    named = resolve_entity_extractor(GraphConfig(extractor="qwen-plus"), test_config)
    assert isinstance(named, LLMEntityExtractor)
    assert named.config.llm == "qwen-plus"

    scripted = ScriptedExtractor({})
    assert resolve_entity_extractor(GraphConfig(extractor=scripted), test_config) is scripted


def test_llm_entity_extractor_forwards_config_and_prompt(
    test_config: RAGLiteConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The configured prompt, model, temperature, and token cap reach the extraction call."""
    import raglite._extract as extract_module

    calls: list[dict[str, Any]] = []

    def fake_extract_with_llm(
        return_type: Any, user_prompt: Any, strict: bool = False, **kwargs: Any
    ) -> Any:
        calls.append({"prompt": return_type.system_prompt, "user": user_prompt, **kwargs})
        return GraphExtraction(entities=[_entity("Einstein")])

    monkeypatch.setattr(extract_module, "extract_with_llm", fake_extract_with_llm)
    extractor = LLMEntityExtractor(
        replace(test_config, llm="qwen-plus"),
        prompt="CUSTOM PROMPT",
        temperature=0.2,
        max_tokens=256,
    )
    extraction = extractor.extract("some text")
    assert [entity.title for entity in extraction.entities] == ["Einstein"]
    assert calls[0]["prompt"] == "CUSTOM PROMPT"
    assert calls[0]["user"] == "some text"
    assert calls[0]["temperature"] == 0.2
    assert calls[0]["max_tokens"] == 256
    assert calls[0]["config"].llm == "qwen-plus"


def test_cooccurrence_extractor_keeps_repeats_and_drops_one_offs() -> None:
    """Repeated Latin terms and closed CJK n-grams are kept; one-off terms are dropped."""
    text = (
        "Einstein published the special theory. Einstein corresponded with Besso.\n"
        "量子纠错保护量子态。量子纠错需要量子比特。Nachmeldung steht nur einmal hier.\n"
    )
    extraction = CooccurrenceExtractor(min_occurrences=2).extract(text)
    titles = {entity.title for entity in extraction.entities}
    assert "Einstein" in titles
    assert "Nachmeldung" not in titles  # A one-off capitalised token.
    assert any("量子" in title for title in titles)
    assert extraction.relations, "co-occurrence should relate terms sharing a sentence"
    assert all(
        relation.source in titles and relation.target in titles for relation in extraction.relations
    )


# ---------------------------------------------------------------------------
# Graph construction.
# ---------------------------------------------------------------------------


@pytest.fixture
def test_config(tmp_path: Any) -> RAGLiteConfig:
    """A file-backed config with no model dependencies."""
    return RAGLiteConfig(
        db_url=f"duckdb:///{(tmp_path / 'graph.db').as_posix()}",
        llm="gpt-4o-mini",
        embedder="text-embedding-3-small",
        reranker=None,
    )


@pytest.fixture
def graph_config(test_config: RAGLiteConfig) -> RAGLiteConfig:
    """The test config with the graph path enabled and a scripted extractor.

    The graph spans three chunks: chunk A mentions Einstein and the field equations, chunk B
    mentions the field equations and Besso, and chunk C mentions only Besso. So a query about
    Einstein reaches chunk B after one hop and chunk C after two.
    """
    extractor = ScriptedExtractor(
        {
            MARKER_A: GraphExtraction(
                entities=[_entity("Einstein"), _entity("Field equations")],
                relations=[_relation("Einstein", "Field equations")],
            ),
            MARKER_B: GraphExtraction(
                entities=[_entity("Field equations"), _entity("Michele Besso")],
                relations=[_relation("Field equations", "Michele Besso")],
            ),
            MARKER_C: GraphExtraction(entities=[_entity("Michele Besso")]),
        }
    )
    return replace(test_config, graph=GraphConfig(enabled=True, extractor=extractor))


def _seed(config: RAGLiteConfig, bodies: dict[str, str]) -> dict[str, str]:
    """Insert one document and one chunk per entry, returning the chunk ids by key."""
    engine = create_database_engine(config)
    chunk_ids: dict[str, str] = {}
    with Session(engine) as session:
        for key, body in bodies.items():
            document = Document.from_text(body, filename=f"{key}.md")
            session.add(document)
            chunk = Chunk.from_body(document, 0, body, topic="physics")
            session.add(chunk)
            chunk_ids[key] = chunk.id
        session.commit()
    return chunk_ids


@pytest.fixture
def seeded(graph_config: RAGLiteConfig) -> dict[str, str]:
    """The graph config with chunks A, B, and C in the database."""
    return _seed(graph_config, {"a": BODY_A, "b": BODY_B, "c": BODY_C})


def test_build_graph_aggregates_entities_and_relations(
    graph_config: RAGLiteConfig, seeded: dict[str, str]
) -> None:
    """A build extracts every pending chunk and aggregates titles onto shared nodes."""
    report = build_graph(config=graph_config)
    assert report["chunks_extracted"] == 3
    assert report["chunks_skipped"] == 0
    assert report["chunks_failed"] == 0
    # Einstein, Field equations, and Michele Besso.
    assert report["entities"] == 3
    # Einstein->Field equations and Field equations->Besso.
    assert report["relations"] == 2
    assert graph_stats(config=graph_config)["entity_chunk_links"] == 5  # A:2, B:2, C:1


def test_build_graph_is_incremental(graph_config: RAGLiteConfig, seeded: dict[str, str]) -> None:
    """A second build skips every chunk, because chunk ids are content hashes."""
    build_graph(config=graph_config)
    report = build_graph(config=graph_config)
    assert report["chunks_extracted"] == 0
    assert report["chunks_skipped"] == 3
    # The totals still describe the graph, not the delta.
    assert report["entities"] == 3


def test_build_graph_restricted_to_documents(
    graph_config: RAGLiteConfig, seeded: dict[str, str]
) -> None:
    """A scoped build only extracts the requested documents."""
    engine = create_database_engine(graph_config)
    with Session(engine) as session:
        document_id = session.exec(select(Chunk.document_id).where(Chunk.id == seeded["a"])).one()
    report = build_graph(document_ids=[document_id], config=graph_config)
    assert report["chunks_extracted"] == 1
    assert report["chunks_skipped"] == 0
    assert report["entities"] == 2


@pytest.mark.filterwarnings("ignore:Graph extraction failed:UserWarning")
def test_build_graph_retries_failed_chunks(
    test_config: RAGLiteConfig,
) -> None:
    """A failed extraction is reported and left pending, so the next build retries it."""
    config = replace(test_config, graph=GraphConfig(enabled=True, extractor=FailingExtractor()))
    _seed(config, {"a": BODY_A})
    report = build_graph(config=config)
    assert report["chunks_failed"] == 1
    assert report["chunks_extracted"] == 0
    # The chunk is not recorded as extracted, so it is pending again.
    assert report["chunks_skipped"] == 0
    assert graph_stats(config=config)["chunks_indexed"] == 0


def test_extract_one_degrades_and_warns() -> None:
    """A failing extractor yields None and a warning instead of propagating."""
    import raglite._graph as graph_module

    with pytest.warns(UserWarning, match="Graph extraction failed"):
        assert graph_module._extract_one(FailingExtractor(), "text") is None


def test_build_graph_records_chunks_that_yield_nothing(test_config: RAGLiteConfig) -> None:
    """A chunk with no entities is still recorded, so it is not extracted on every build."""
    config = replace(test_config, graph=GraphConfig(enabled=True, extractor=ScriptedExtractor({})))
    _seed(config, {"a": BODY_A})
    first = build_graph(config=config)
    assert first["chunks_extracted"] == 1
    assert first["entities"] == 0
    second = build_graph(config=config)
    assert second["chunks_skipped"] == 1
    assert second["chunks_extracted"] == 0


def test_build_graph_drops_overlong_and_empty_titles(test_config: RAGLiteConfig) -> None:
    """Entities with an empty or absurdly long title are discarded."""
    extractor = ScriptedExtractor(
        {
            MARKER_A: GraphExtraction(
                entities=[
                    _entity("   "),
                    _entity("x" * 500),
                    _entity("Einstein"),
                    _entity(""),
                ]
            )
        }
    )
    config = replace(
        test_config,
        graph=GraphConfig(enabled=True, extractor=extractor, max_entity_title_length=50),
    )
    _seed(config, {"a": BODY_A})
    report = build_graph(config=config)
    assert report["entities"] == 1


def test_graph_relations_are_chunk_local(test_config: RAGLiteConfig) -> None:
    """A relation whose endpoints are not both in the chunk is dropped rather than dangling."""
    extractor = ScriptedExtractor(
        {
            MARKER_A: GraphExtraction(
                entities=[_entity("Einstein")],
                relations=[_relation("Einstein", "Nobody"), _relation("Nobody", "Einstein")],
            )
        }
    )
    config = replace(test_config, graph=GraphConfig(enabled=True, extractor=extractor))
    _seed(config, {"a": BODY_A})
    report = build_graph(config=config)
    assert report["entities"] == 1
    assert report["relations"] == 0


# ---------------------------------------------------------------------------
# Pruning and clearing.
# ---------------------------------------------------------------------------


def test_prune_graph_removes_rows_without_chunks(
    graph_config: RAGLiteConfig, seeded: dict[str, str]
) -> None:
    """Deleting a chunk prunes its links, its bookkeeping, and the entities it orphaned."""
    build_graph(config=graph_config)
    engine = create_database_engine(graph_config)
    with Session(engine) as session:
        chunk = session.get(Chunk, seeded["a"])
        assert chunk is not None
        session.delete(chunk)
        session.commit()
    removed = prune_graph(config=graph_config)
    assert removed["chunk_links"] == 2  # Einstein and Field equations lose their only chunk.
    assert removed["chunks"] == 1
    assert removed["entities"] == 1  # Einstein, orphaned by the deletion.
    assert removed["relations"] == 1  # Einstein -> Field equations.
    stats = graph_stats(config=graph_config)
    assert stats["entities"] == 2  # Field equations and Michele Besso survive.
    assert stats["relations"] == 1


def test_prune_graph_is_idempotent(graph_config: RAGLiteConfig, seeded: dict[str, str]) -> None:
    """A second prune finds nothing to remove."""
    build_graph(config=graph_config)
    assert prune_graph(config=graph_config) == {
        "chunk_links": 0,
        "chunk_relations": 0,
        "chunks": 0,
        "entities": 0,
        "relations": 0,
    }


def test_clear_graph(graph_config: RAGLiteConfig, seeded: dict[str, str]) -> None:
    """Clearing the graph leaves it empty and makes every chunk pending again."""
    build_graph(config=graph_config)
    clear_graph(config=graph_config)
    assert graph_stats(config=graph_config) == {
        "entities": 0,
        "relations": 0,
        "entity_chunk_links": 0,
        "chunk_relations": 0,
        "chunks_indexed": 0,
    }
    engine = create_database_engine(graph_config)
    with Session(engine) as session:
        assert session.exec(select(GraphChunk)).all() == []


# ---------------------------------------------------------------------------
# Graph retrieval.
# ---------------------------------------------------------------------------


def test_graph_search_disabled_returns_nothing(
    test_config: RAGLiteConfig, seeded: dict[str, str]
) -> None:
    """The disabled graph path is a no-op and does not touch the graph."""
    assert graph_search("Einstein", config=test_config) == ([], [])


def test_graph_search_matches_seed_entity(
    graph_config: RAGLiteConfig, seeded: dict[str, str]
) -> None:
    """A query mentioning an entity returns the chunks that entity was extracted from."""
    build_graph(config=graph_config)
    chunk_ids, scores = graph_search(
        "What did Einstein publish?", num_results=5, config=graph_config
    )
    assert seeded["a"] in chunk_ids
    assert all(score > 0 for score in scores)


def test_graph_search_expands_neighbours_by_hop(
    graph_config: RAGLiteConfig, seeded: dict[str, str]
) -> None:
    """One hop reaches a chunk that never mentions the query entity; two hops reach the next one."""
    build_graph(config=graph_config)
    # The chunk graph is off here so the assertion isolates the entity hops.
    one_hop = replace(
        graph_config, graph=replace(graph_config.graph, neighbor_hops=1, chunk_neighbor_hops=0)
    )
    chunk_ids, _ = graph_search("Einstein", num_results=5, config=one_hop)
    assert seeded["a"] in chunk_ids  # Directly matched.
    assert seeded["b"] in chunk_ids  # Reached through the field equations.
    assert seeded["c"] not in chunk_ids  # Two hops away.

    two_hops = replace(
        graph_config, graph=replace(graph_config.graph, neighbor_hops=2, chunk_neighbor_hops=0)
    )
    chunk_ids, _ = graph_search("Einstein", num_results=5, config=two_hops)
    assert seeded["c"] in chunk_ids


def test_chunk_graph_reaches_a_chunk_no_entity_points_at(
    graph_config: RAGLiteConfig, seeded: dict[str, str]
) -> None:
    """WeKnora's chunk projection: a chunk sharing an entity with a scored chunk is reachable."""
    build_graph(config=graph_config)
    # Chunk C only mentions Besso, which is two entity hops from Einstein.
    without = replace(
        graph_config, graph=replace(graph_config.graph, neighbor_hops=1, chunk_neighbor_hops=0)
    )
    assert seeded["c"] not in graph_search("Einstein", num_results=5, config=without)[0]

    with_projection = replace(
        graph_config, graph=replace(graph_config.graph, neighbor_hops=1, chunk_neighbor_hops=1)
    )
    chunk_ids, scores = graph_search("Einstein", num_results=5, config=with_projection)
    assert seeded["c"] in chunk_ids
    # The directly matched chunk still outranks the one reached through the projection.
    assert chunk_ids.index(seeded["a"]) < chunk_ids.index(seeded["c"])
    assert scores[chunk_ids.index(seeded["a"])] > scores[chunk_ids.index(seeded["c"])]
    assert graph_stats(config=graph_config)["chunk_relations"] > 0


def test_graph_search_prefers_the_more_specific_entity(
    graph_config: RAGLiteConfig, seeded: dict[str, str]
) -> None:
    """A longer matched title is a better seed than a shorter one contained in the query."""
    build_graph(config=graph_config)
    # "Field equations" is longer than "Einstein" and is the more specific match here.
    chunk_ids, _ = graph_search(
        "Who worked on the field equations?", num_results=5, config=graph_config
    )
    assert seeded["a"] in chunk_ids
    assert seeded["b"] in chunk_ids


def test_graph_search_returns_nothing_for_unrelated_query(
    graph_config: RAGLiteConfig, seeded: dict[str, str]
) -> None:
    """A query that mentions no known entity yields no ranking at all."""
    build_graph(config=graph_config)
    assert graph_search("How do I bake sourdough bread?", config=graph_config) == ([], [])


def test_graph_search_empty_graph(graph_config: RAGLiteConfig, seeded: dict[str, str]) -> None:
    """With nothing extracted there is nothing to rank."""
    assert graph_search("Einstein", config=graph_config) == ([], [])


def test_graph_search_respects_metadata_filter(
    graph_config: RAGLiteConfig, seeded: dict[str, str]
) -> None:
    """Chunks excluded by the metadata filter are dropped, and the graph order is preserved."""
    build_graph(config=graph_config)
    engine = create_database_engine(graph_config)
    with Session(engine) as session:
        chunk_b = session.get(Chunk, seeded["b"])
        assert chunk_b is not None
        chunk_b.metadata_ = {**chunk_b.metadata_, "topic": ["biology"]}
        session.add(chunk_b)
        session.commit()
    unfiltered, _ = graph_search("Who worked on the field equations?", config=graph_config)
    assert seeded["b"] in unfiltered
    filtered, _ = graph_search(
        "Who worked on the field equations?",
        metadata_filter={"topic": "physics"},
        config=graph_config,
    )
    assert seeded["b"] not in filtered
    assert seeded["a"] in filtered


def test_graph_search_respects_num_results(
    graph_config: RAGLiteConfig, seeded: dict[str, str]
) -> None:
    """The ranking is truncated to the requested number of results."""
    build_graph(config=graph_config)
    chunk_ids, scores = graph_search("Einstein", num_results=1, config=graph_config)
    assert len(chunk_ids) == 1
    assert len(scores) == 1


def test_graph_search_orders_by_accumulated_weight(
    graph_config: RAGLiteConfig, seeded: dict[str, str]
) -> None:
    """A chunk linked to the seed entity outranks a chunk reached only through a neighbour."""
    build_graph(config=graph_config)
    one_hop = replace(graph_config, graph=replace(graph_config.graph, neighbor_hops=1))
    chunk_ids, scores = graph_search("Einstein", num_results=5, config=one_hop)
    assert chunk_ids.index(seeded["a"]) < chunk_ids.index(seeded["b"])
    assert scores[chunk_ids.index(seeded["a"])] > scores[chunk_ids.index(seeded["b"])]


def test_graph_search_ignores_dangling_chunk_links(
    graph_config: RAGLiteConfig, seeded: dict[str, str]
) -> None:
    """A graph link left behind by a deletion is never returned as a result."""
    build_graph(config=graph_config)
    chunk_a, chunk_b = seeded["a"], seeded["b"]
    engine = create_database_engine(graph_config)
    # Deleting the chunk must succeed: the graph tables declare no foreign keys, precisely so the
    # derived graph can never block primary data deletion.
    with Session(engine) as session:
        chunk = session.get(Chunk, chunk_a)
        assert chunk is not None
        session.delete(chunk)
        session.commit()
    # No prune yet: the graph still links Einstein to the deleted chunk.
    assert graph_stats(config=graph_config)["entities"] == 3
    chunk_ids, _ = graph_search("Einstein", num_results=5, config=graph_config)
    assert chunk_a not in chunk_ids
    assert chunk_b in chunk_ids


def test_pending_graph_chunks(graph_config: RAGLiteConfig, seeded: dict[str, str]) -> None:
    """Every chunk is pending until it has been extracted, and none afterwards."""
    from raglite._graph import pending_graph_chunks

    assert set(pending_graph_chunks(config=graph_config)) == set(seeded.values())
    build_graph(config=graph_config)
    assert pending_graph_chunks(config=graph_config) == []


# ---------------------------------------------------------------------------
# Fusion with the vector and keyword rankings.
# ---------------------------------------------------------------------------


def test_hybrid_search_degrades_to_baseline_when_graph_is_empty(
    graph_config: RAGLiteConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty graph ranking leaves the two-path result unchanged."""
    import raglite._search as search_module

    monkeypatch.setattr(
        search_module,
        "vector_search",
        lambda query, **kwargs: (["v1", "v2", "v3"], [1.0, 0.9, 0.8]),
    )
    monkeypatch.setattr(
        search_module,
        "keyword_search",
        lambda query, **kwargs: (["k1", "k2"], [1.0, 0.9]),
    )
    monkeypatch.setattr(search_module, "graph_search", lambda query, **kwargs: ([], []))

    baseline_config = replace(graph_config, graph=GraphConfig())
    baseline, baseline_scores = search_module.hybrid_search(
        "q", num_results=4, config=baseline_config
    )
    fused, fused_scores = search_module.hybrid_search("q", num_results=4, config=graph_config)
    assert fused == baseline
    assert fused_scores == baseline_scores


def test_hybrid_search_fuses_the_graph_ranking(
    graph_config: RAGLiteConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The graph ranking participates in the fusion, and its weight is configurable."""
    import raglite._search as search_module

    monkeypatch.setattr(
        search_module, "vector_search", lambda query, **kwargs: (["v1", "v2"], [1.0, 0.9])
    )
    monkeypatch.setattr(
        search_module, "keyword_search", lambda query, **kwargs: (["k1", "k2"], [1.0, 0.9])
    )
    monkeypatch.setattr(
        search_module, "graph_search", lambda query, **kwargs: (["g1", "g2"], [2.0, 1.0])
    )
    # A strong graph weight puts the graph-only chunks on top...
    chunk_ids, _ = search_module.hybrid_search(
        "q", num_results=6, graph_search_weight=1.0, config=graph_config
    )
    assert chunk_ids[:2] == ["g1", "g2"]
    # ...and the default (small) graph weight leaves them at the bottom.
    chunk_ids, _ = search_module.hybrid_search("q", num_results=6, config=graph_config)
    assert chunk_ids[-2:] == ["g1", "g2"]


def test_hybrid_search_does_not_consult_the_graph_when_disabled(
    graph_config: RAGLiteConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A disabled graph never runs a graph search."""
    import raglite._search as search_module

    monkeypatch.setattr(search_module, "vector_search", lambda query, **kwargs: (["v1"], [1.0]))
    monkeypatch.setattr(search_module, "keyword_search", lambda query, **kwargs: (["k1"], [1.0]))

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("graph_search must not run when the graph is disabled")

    monkeypatch.setattr(search_module, "graph_search", unexpected)
    disabled = replace(graph_config, graph=GraphConfig())
    assert search_module.hybrid_search("q", config=disabled)[0]


# ---------------------------------------------------------------------------
# Incremental sync wiring.
# ---------------------------------------------------------------------------


def test_sync_documents_rebuilds_the_graph_for_the_delta(
    graph_config: RAGLiteConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An enabled graph is rebuilt for the inserted documents only, and pruned first."""
    import raglite._graph as graph_module
    import raglite._incremental as incremental_module

    calls: list[dict[str, Any]] = []

    def record_prune(**kwargs: Any) -> None:
        calls.append({"pruned": True})

    def record_build(**kwargs: Any) -> dict[str, int]:
        calls.append(kwargs)
        return {"chunks_extracted": 0}

    monkeypatch.setattr("raglite._insert.insert_documents", lambda documents, **kwargs: None)
    monkeypatch.setattr(
        incremental_module, "_existing_documents_by_key", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(graph_module, "prune_graph", record_prune)
    monkeypatch.setattr(graph_module, "build_graph", record_build)
    report = incremental_module.sync_documents(
        [Document(id="doc-new", filename="new.md", content="content")], config=graph_config
    )
    assert report["inserted"] == ["doc-new"]
    assert calls[0] == {"pruned": True}
    assert calls[1]["document_ids"] == ["doc-new"]
    assert calls[1]["prune_missing"] is False  # Already pruned above.
    assert report["graph"] == {"chunks_extracted": 0}


def test_sync_documents_skips_the_graph_when_disabled(
    test_config: RAGLiteConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A disabled graph is never built, so `sync_documents` stays cheap by default."""
    import raglite._graph as graph_module
    import raglite._incremental as incremental_module

    def unexpected(**kwargs: Any) -> Any:
        raise AssertionError("the graph must not be built when it is disabled")

    monkeypatch.setattr("raglite._insert.insert_documents", lambda documents, **kwargs: None)
    monkeypatch.setattr(
        incremental_module, "_existing_documents_by_key", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(graph_module, "build_graph", unexpected)
    monkeypatch.setattr(graph_module, "prune_graph", unexpected)
    report = incremental_module.sync_documents(
        [Document(id="doc-new", filename="new.md", content="content")], config=test_config
    )
    assert report["inserted"] == ["doc-new"]
    assert "graph" not in report


# ---------------------------------------------------------------------------
# WeKnora alignment: edge weighting, the chunk projection, and the query side.
# ---------------------------------------------------------------------------


def test_pointwise_mutual_information_matches_the_definition() -> None:
    """PMI is `log2(P(x, y) / (P(x) * P(y)))`, clamped at zero."""
    from raglite._graph import pointwise_mutual_information

    # Together a quarter of the time, each a quarter of the time: four times more than chance,
    # which is two bits of mutual information.
    assert pointwise_mutual_information(0.25, 0.25, 0.25) == pytest.approx(2.0)
    # Exactly chance: no mutual information.
    assert pointwise_mutual_information(0.0625, 0.25, 0.25) == pytest.approx(0.0)
    # Less than chance: clamped, never negative.
    assert pointwise_mutual_information(0.01, 0.25, 0.25) == 0.0
    # An entity that never occurs carries no information.
    assert pointwise_mutual_information(0.0, 0.25, 0.0) == 0.0


def test_recompute_relation_weights_scales_pmi_and_strength(test_config: RAGLiteConfig) -> None:
    """The edge weight combines normalised PMI and the extractor's strength into 1-10."""
    from raglite._database import GraphRelation
    from raglite._graph import recompute_relation_weights

    extractor = ScriptedExtractor(
        {
            MARKER_A: GraphExtraction(
                entities=[_entity("Einstein"), _entity("Field equations")],
                relations=[_relation("Einstein", "Field equations", strength=10.0)],
            ),
            MARKER_B: GraphExtraction(
                entities=[_entity("Field equations"), _entity("Michele Besso")],
                relations=[_relation("Field equations", "Michele Besso", strength=1.0)],
            ),
            MARKER_C: GraphExtraction(entities=[_entity("Michele Besso")]),
        }
    )
    config = replace(test_config, graph=GraphConfig(enabled=True, extractor=extractor))
    _seed(config, {"a": BODY_A, "b": BODY_B, "c": BODY_C})
    report = build_graph(config=config)
    assert report["relations_scored"] == 2

    with Session(create_database_engine(config)) as session:
        relations = {
            (relation.source_id, relation.target_id): relation
            for relation in session.exec(select(GraphRelation)).all()
        }
    einstein = entity_id_for_title("Einstein")
    field_equations = entity_id_for_title("Field equations")
    besso = entity_id_for_title("Michele Besso")
    strong = relations[(einstein, field_equations)]
    weak = relations[(field_equations, besso)]
    # Both land in the 1-10 range...
    assert 1.0 <= strong.weight <= 10.0
    assert 1.0 <= weak.weight <= 10.0
    # ...and the relation with the higher strength and equal PMI wins.
    assert strong.weight > weak.weight
    assert strong.strength == pytest.approx(1.0)
    assert weak.strength == pytest.approx(0.1)
    # Re-running is idempotent: the weights are recomputed from the graph, not accumulated.
    weights_before = {key: relation.weight for key, relation in relations.items()}
    recompute_relation_weights(config=config)
    with Session(create_database_engine(config)) as session:
        after = {
            (relation.source_id, relation.target_id): relation.weight
            for relation in session.exec(select(GraphRelation)).all()
        }
    assert after == weights_before


def test_relation_strength_is_a_mean_over_the_chunks(test_config: RAGLiteConfig) -> None:
    """An incremental build folds a new mention into the mean strength instead of replacing it."""
    from raglite._database import GraphRelation

    extractor = ScriptedExtractor(
        {
            MARKER_A: GraphExtraction(
                entities=[_entity("Einstein"), _entity("Relativity")],
                relations=[_relation("Einstein", "Relativity", strength=10.0)],
            ),
            MARKER_B: GraphExtraction(
                entities=[_entity("Relativity"), _entity("Einstein")],
                relations=[_relation("Einstein", "Relativity", strength=1.0)],
            ),
        }
    )
    config = replace(test_config, graph=GraphConfig(enabled=True, extractor=extractor))
    _seed(config, {"a": BODY_A, "b": BODY_B})
    build_graph(config=config)
    with Session(create_database_engine(config)) as session:
        relation = session.exec(select(GraphRelation)).one()
    assert relation.observations == 2
    assert relation.strength == pytest.approx(0.55)  # (1.0 + 0.1) / 2


def test_extraction_prompt_carries_the_configured_types_and_examples() -> None:
    """WeKnora's `ExtractConfig` constraints are appended to the prompt when they are set."""
    from raglite._graph import build_extraction_prompt

    plain = build_extraction_prompt(GraphConfig())
    assert "knowledge graph extractor" in plain
    constrained = build_extraction_prompt(
        GraphConfig(
            entity_types=("person", "theory"),
            relation_types=("authored-by",),
            extraction_examples="Example: Einstein -> authored-by -> relativity.",
        )
    )
    assert "person, theory" in constrained
    assert "authored-by" in constrained
    assert constrained.endswith("Example: Einstein -> authored-by -> relativity.")
    assert plain == GraphConfig().extraction_prompt


def test_query_extraction_seeds_entities_the_lexical_matcher_misses(
    test_config: RAGLiteConfig,
) -> None:
    """Query-time extraction finds an entity the question never spells out."""

    @dataclass
    class QueryAwareExtractor:
        """An extractor that reads the question's entities when it is given the question."""

        def extract(self, text: str) -> GraphExtraction:
            if "annus mirabilis" in text:
                return GraphExtraction(entities=[_entity("Michele Besso")])
            if MARKER_A in text:
                return GraphExtraction(entities=[_entity("Einstein")])
            return GraphExtraction(entities=[_entity("Michele Besso")])

    config = replace(
        test_config,
        graph=GraphConfig(enabled=True, extractor=QueryAwareExtractor()),
    )
    chunk_ids = _seed(config, {"a": BODY_A, "b": BODY_B})
    build_graph(config=config)
    # "Who wrote the annus mirabilis papers?" mentions no entity title, so lexical matching finds
    # nothing until query extraction is turned on.
    assert graph_search("Who wrote the annus mirabilis papers?", config=config)[0] == []

    with_extraction = replace(config, graph=replace(config.graph, query_extraction=True))
    assert graph_search("Who wrote the annus mirabilis papers?", config=with_extraction)[0] == [
        chunk_ids["b"]
    ]


def test_graph_search_excludes_the_chunks_the_other_paths_found(
    graph_config: RAGLiteConfig, seeded: dict[str, str]
) -> None:
    """`exclude_chunk_ids` is WeKnora's `filterSeenChunk`: only new chunks are contributed."""
    build_graph(config=graph_config)
    full, _ = graph_search("Einstein", num_results=5, config=graph_config)
    assert seeded["a"] in full
    filtered, _ = graph_search(
        "Einstein", num_results=5, exclude_chunk_ids=[seeded["a"]], config=graph_config
    )
    assert seeded["a"] not in filtered
    assert seeded["b"] in filtered


def test_hybrid_search_deduplicates_when_configured(
    graph_config: RAGLiteConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`deduplicate_seen_chunks` forwards what the other paths found to the graph search."""
    import raglite._search as search_module

    seen: list[Any] = []

    def record_graph_search(query: str, **kwargs: Any) -> tuple[list[str], list[float]]:
        seen.append(kwargs.get("exclude_chunk_ids"))
        return (["g1"], [1.0])

    monkeypatch.setattr(search_module, "vector_search", lambda query, **kwargs: (["v1"], [1.0]))
    monkeypatch.setattr(search_module, "keyword_search", lambda query, **kwargs: (["k1"], [1.0]))
    monkeypatch.setattr(search_module, "graph_search", record_graph_search)
    search_module.hybrid_search("q", config=graph_config)
    assert seen == [None]  # Off by default: RRF rewards agreement between paths.

    deduplicating = replace(
        graph_config, graph=replace(graph_config.graph, deduplicate_seen_chunks=True)
    )
    search_module.hybrid_search("q", config=deduplicating)
    assert seen[-1] == ["v1", "k1"]


def test_query_knowledge_graph_returns_the_neighbourhood(
    graph_config: RAGLiteConfig, seeded: dict[str, str]
) -> None:
    """The agent-facing tool returns the entities, their relations, and their chunks."""
    from raglite._graph import query_knowledge_graph

    build_graph(config=graph_config)
    result = query_knowledge_graph("What did Einstein publish?", config=graph_config)
    titles = {entity.title for entity in result.entities}
    assert "Einstein" in titles
    assert "Field equations" in titles  # One hop away.
    assert "Michele Besso" not in titles  # Two hops away.
    assert ("Einstein", "Field equations") in {
        (relation.source, relation.target) for relation in result.relations
    }
    assert seeded["a"] in result.chunk_ids
    assert result.metadata["seeds"] >= 1
    # An unrelated query yields an empty subgraph rather than an error.
    assert query_knowledge_graph("How do I bake bread?", config=graph_config).entities == ()


def test_query_knowledge_graph_is_a_two_hop_neighbourhood_on_request(
    graph_config: RAGLiteConfig, seeded: dict[str, str]
) -> None:
    """`neighbor_hops` widens the returned subgraph."""
    from raglite._graph import query_knowledge_graph

    build_graph(config=graph_config)
    result = query_knowledge_graph("Einstein", neighbor_hops=2, config=graph_config)
    assert {entity.title for entity in result.entities} == {
        "Einstein",
        "Field equations",
        "Michele Besso",
    }
