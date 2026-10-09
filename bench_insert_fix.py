"""Verify the batched-embedding-insert fix end to end.

Times every stage of a real `insert_documents` call on a real document, so the
per-stage cost is visible next to the total. Run with the project's venv python.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import numpy as np  # noqa: E402
from sqlmodel import Session, select  # noqa: E402

from weknow._config import RAGLiteConfig  # noqa: E402, F401
from weknow._database import (  # noqa: E402
    Chunk,
    ChunkEmbedding,
    Document,
    create_database_engine,
    hash_bytes,
)
from weknow._dotenv import config_from_environ  # noqa: E402
from weknow._insert import _insert_chunk_embeddings_fast, insert_documents  # noqa: E402

DOC = Path("README.md")
DB_URL = "duckdb:///E:/weknow/_fixverify.db"

# A fresh database every run, so nothing is skipped as a duplicate.
for suffix in (".db", ".db.wal", ".lock"):
    p = Path("E:/weknow/_fixverify" + suffix)
    if p.exists():
        p.unlink()


def timed(label: str, fn):  # noqa: ANN001, ANN201
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    print(f"{label:<34} {elapsed * 1000:>10.1f} ms")
    return result, elapsed


def main() -> None:
    config = config_from_environ(
        disable_default_reranker=True, autoprobe=False, db_url=DB_URL
    )
    print(f"embedder = {config.embedder}")
    print(f"dimension = {config.embedder_dimension}")
    print(f"batch size = {config.embedder_batch_size}")
    print(f"vector_search_multivector = {config.vector_search_multivector}")
    print("-" * 50)

    content = DOC.read_text(encoding="utf-8")
    print(f"document = {DOC} ({len(content)} chars)")
    document = Document(
        id=hash_bytes(content.encode("utf-8"), max_len=16),
        content=content,
        filename=str(DOC),
        metadata_={"filename": str(DOC), "size": len(content)},
    )

    print("-" * 50)
    _, total = timed("insert_documents (whole)", lambda: insert_documents([document], config=config))
    print("-" * 50)

    # Inspect what landed.
    engine = create_database_engine(config)
    with Session(engine) as session:
        num_chunks = len(session.exec(select(Chunk)).all())
        num_embeddings = len(session.exec(select(ChunkEmbedding.chunk_id)).all())
    print(f"chunks inserted      = {num_chunks}")
    print(f"embeddings inserted  = {num_embeddings}")
    print(f"total insert time    = {total:.2f} s")

    # Isolate the embedding write, which is what the fix targets: reuse a real chunk id so the
    # foreign key holds, and time the same 7-row batch at dim=1024 that took 28.75 s before.
    print("-" * 50)
    with Session(engine) as session:
        real_chunk_id = str(session.exec(select(Chunk.id)).first())
        dim = int(config.embedder_dimension or 1024)
        records = [
            ChunkEmbedding(chunk_id=real_chunk_id, embedding=np.zeros(dim)) for _ in range(7)
        ]
        _, emb_time = timed(
            f"7 rows @ dim={dim}, multi-row VALUES",
            lambda: _insert_chunk_embeddings_fast(session, [[r] for r in records], dim=dim),
        )
        session.rollback()
    print(f"\nisolated embedding write = {emb_time * 1000:.0f} ms")
    print("before the fix           = ~28750 ms (ORM flush, 7 rows @ dim=1024)")
    if emb_time > 0:
        print(f"speedup                  = {28.75 / max(emb_time, 1e-9):.0f}x")


if __name__ == "__main__":
    main()
