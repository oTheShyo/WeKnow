"""WeKnow web server — a WeKnora-style frontend for the RAG toolkit.

Serves a single-page application (WeKnora-like: knowledge-base management, chat with
citations, retrieval testing) plus a small JSON/SSE API on top of the WeKnow core.

Everything is configured through the project `.env` via `config_from_environ`; the server
adds no configuration of its own. Documents are inserted through the same
`insert_documents` path as the CLI, so content-hash deduplication and the parser registry
apply here too.
"""

from __future__ import annotations

import contextlib
import json
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from weknow._dotenv import config_from_environ, ensure_dotenv_loaded

_WEB_DIR = Path(__file__).parent / "_web"
_UPLOAD_DIR = Path(tempfile.gettempdir()) / "weknow_uploads"

app = FastAPI(title="WeKnow", docs_url=None, redoc_url=None)

# Upload tasks are tracked in memory: {task_id: {name, status, detail, chunks}}. Status is
# best-effort; a restart clears it, and the document table remains the source of truth.
_TASKS: dict[str, dict[str, Any]] = {}
_TASKS_LOCK = threading.Lock()


def _get_config() -> Any:
    """Build the config from the environment. Cheap after the first call (cached downstream)."""
    return config_from_environ(disable_default_reranker=True)


def _task_update(task_id: str, **fields: Any) -> None:
    with _TASKS_LOCK:
        if task_id in _TASKS:
            _TASKS[task_id].update(fields)


def _sse(payload: dict[str, Any]) -> str:
    """Serialize a payload for an SSE data line (single line, UTF-8)."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@contextlib.contextmanager
def _db_lock() -> Iterator[None]:
    """Serialise one unit of database work behind the process-wide lock.

    `threading.Lock` is itself a context manager, but wrapping it keeps the type simple for callers
    typed against `AbstractContextManager[None]`.
    """
    with _DB_LOCK:
        yield


# DuckDB allows a single writer at a time, and the SQLAlchemy connection pool hands out a new
# connection per request. A background insert thread holding a write transaction while a request
# thread opens another connection deadlocks the whole process (observed: the server stopped
# answering within seconds of an upload). Serialise every database operation through one lock, and
# share one engine so the pool cannot multiply connections.
_DB_LOCK = threading.Lock()
_ENGINES: dict[int, Any] = {}


def _get_engine(config: Any) -> Any:
    """Return the process-wide engine for this config's database URL."""
    from weknow._database import create_database_engine

    key = id(config.db_url) if not isinstance(config.db_url, str) else hash(config.db_url)
    with _DB_LOCK:
        engine = _ENGINES.get(key)
        if engine is None:
            engine = create_database_engine(config)
            _ENGINES[key] = engine
        return engine


class _DbSession:
    """A Session under the process-wide database lock.

    Use as a context manager: `with _db_session(config) as session: ...`. The lock is what keeps
    DuckDB happy with concurrent requests; the shared engine keeps the pool to one connection.
    """

    def __init__(self, config: Any) -> None:
        from sqlmodel import Session

        self._session = Session(_get_engine(config))

    def __enter__(self) -> Any:
        _DB_LOCK.acquire()
        return self._session

    def __exit__(self, *exc: object) -> None:
        with contextlib.suppress(Exception):
            self._session.close()
        _DB_LOCK.release()


def _insert_in_background(task_id: str, tmp_path: Path, display_name: str | None = None) -> None:
    """Insert an uploaded document off the request path, updating the task as it goes.

    `display_name` is the name the user's file had; it is kept as the document's filename because
    the temp path the bytes landed in would otherwise become the user-visible name.
    """
    from sqlmodel import select

    from weknow import Document, insert_documents
    from weknow._database import Document as DocumentRow

    try:
        _task_update(task_id, status="parsing")
        config = _get_config()
        document = Document.from_path(tmp_path, config=config)
        if display_name:
            document.filename = display_name
            metadata = dict(document.metadata_ or {})
            metadata["filename"] = display_name
            document.metadata_ = metadata
        document_id = str(document.id)
        # `insert_documents` silently skips documents whose content hash is already present,
        # so report that case explicitly instead of claiming success.
        with _DbSession(config) as session:
            exists = (
                session.exec(
                    select(DocumentRow.id).where(DocumentRow.id == document_id)  # type: ignore[arg-type]
                ).first()
                is not None
            )
        if exists:
            _task_update(
                task_id, status="duplicate", detail="内容已存在（content-hash 相同），已跳过"
            )
            return
        _task_update(task_id, status="embedding")
        # `document_id` was read before the insert: `insert_documents` releases its ORM instances on
        # the way out, so touching `document.id` afterwards raises `DetachedInstanceError`.
        insert_documents([document], config=config)
        _task_update(task_id, status="ready", detail="", document_id=document_id)
    except Exception as exc:  # noqa: BLE001
        _task_update(task_id, status="failed", detail=str(exc)[:300])
    finally:
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)


class ChatRequest(BaseModel):
    """A chat request: the full OpenAI-style message history minus the pending answer."""

    messages: list[dict[str, str]]


class SearchRequest(BaseModel):
    """A retrieval-test request."""

    query: str
    top_k: int = 5


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the single-page frontend."""
    return HTMLResponse((_WEB_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/tasks")
def list_tasks() -> dict[str, Any]:
    """Return the in-memory upload tasks only.

    The frontend polls this while an upload is in flight. It deliberately does not touch the
    database: the background insert thread holds the process-wide database lock for the duration
    of the insert, so a poll that read the document table would block behind it (and, before the
    lock existed, contend for the single DuckDB writer). Task state lives in memory, so this
    endpoint answers immediately and never waits on the insert.
    """
    with _TASKS_LOCK:
        tasks = [dict(task, task_id=task_id) for task_id, task in _TASKS.items()]
    return {"tasks": tasks}


@app.get("/api/documents")
def list_documents() -> dict[str, Any]:
    """List every document in the knowledge base, with its chunk count."""
    from sqlmodel import select

    from weknow._database import Chunk, Document

    config = _get_config()
    with _DbSession(config) as session:
        documents = session.exec(select(Document)).all()
        chunk_counts: dict[str, int] = {}
        for document_id in session.exec(select(Chunk.document_id)).all():
            chunk_counts[str(document_id)] = chunk_counts.get(str(document_id), 0) + 1
    items = [
        {
            "id": str(doc.id),
            "filename": doc.filename,
            "size": (doc.metadata_ or {}).get("size"),
            "chunks": chunk_counts.get(str(doc.id), 0),
        }
        for doc in documents
    ]
    with _TASKS_LOCK:
        tasks = [dict(task, task_id=task_id) for task_id, task in _TASKS.items()]
    return {"documents": items, "tasks": tasks}


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)) -> dict[str, str]:  # type: ignore[no-untyped-def]
    """Accept an uploaded document and insert it in the background."""
    task_id = uuid.uuid4().hex[:12]
    display_name = file.filename or "document"
    suffix = Path(display_name).suffix or ".txt"
    _UPLOAD_DIR.mkdir(exist_ok=True)
    tmp_path = _UPLOAD_DIR / f"{task_id}{suffix}"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="空文件")
    tmp_path.write_bytes(content)
    with _TASKS_LOCK:
        _TASKS[task_id] = {
            "task_id": task_id,
            "filename": display_name,
            "status": "uploading",
            "detail": "",
            "started": time.time(),
        }
    threading.Thread(
        target=_insert_in_background,
        args=(task_id, tmp_path, display_name),
        daemon=True,
    ).start()
    return {"task_id": task_id}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str) -> dict[str, str]:
    """Delete a document and all of its chunks and embeddings."""
    from weknow import delete_documents

    # `delete_documents` opens its own sessions internally, so hold the process-wide lock across
    # the whole call rather than wrapping a session here.
    with _DB_LOCK:
        delete_documents([doc_id], config=_get_config())
    return {"deleted": doc_id}


@app.post("/api/search")
def search(request: SearchRequest) -> dict[str, Any]:
    """Run a hybrid retrieval and return the ranked chunks with their scores."""
    from weknow import hybrid_search, retrieve_chunks

    config = _get_config()
    # Both calls open sessions internally (the search functions take a Session per query), so hold
    # the process-wide lock across both.
    with _DB_LOCK:
        chunk_ids, scores = hybrid_search(request.query, config=config)
        chunks = retrieve_chunks(chunk_ids[: request.top_k], config=config)
    results = [
        {
            "document_id": str(chunk.document_id),
            "filename": chunk.document.filename,
            "headings": chunk.headings,
            "body": chunk.body,
            "score": round(scores[i], 4) if i < len(scores) else None,
        }
        for i, chunk in enumerate(chunks)
    ]
    return {"results": results}


def _serialize_sources(spans: list[Any]) -> list[dict[str, Any]]:
    """Render retrieved chunk spans as the JSON payload the frontend cites."""
    return [
        {
            "index": i + 1,
            "filename": span.document.filename,
            "document_id": str(span.document.id),
            "text": str(span)[:800],
        }
        for i, span in enumerate(spans)
    ]


@dataclass
class _AnswerStream:
    """Cross-thread plumbing for one streaming answer.

    The producer thread owns `queue` and `failure`; the response generator (a different thread,
    since Starlette iterates a sync generator in a threadpool) drains the queue and reads `failure`
    once the producer signals completion by putting `None`.
    """

    queue: SimpleQueue[str | None]
    failure: list[str]


def _produce_answer(
    messages: list[dict[str, Any]],
    *,
    config: Any,
    on_retrieval: Callable[[list[Any]], None],
    stream: _AnswerStream,
) -> None:
    """Stream the answer, serialising only the database work.

    `rag` interleaves two kinds of work: agentic retrieval rounds that touch the database, and LLM
    calls (comprehension + answer generation) that do not. Holding the process-wide database lock
    for the *whole* answer is what froze the UI: an answer takes tens of seconds here, and every
    other request queued behind it for that entire time (measured: `/api/documents` went from ~6 ms
    idle to ~5-12 s while an answer streamed), so the page could not repaint.

    Instead the lock is handed to `rag` as a `db_scope`, which enters it only around the actual
    knowledge-base searches. The slow model round-trips run with the database free.
    """
    from weknow import rag

    queue, failure = stream.queue, stream.failure
    try:
        for token in rag(
            messages,
            on_retrieval=on_retrieval,
            config=config,
            db_scope=lambda: _db_lock(),
        ):
            queue.put(token)
    except Exception as exc:  # noqa: BLE001
        failure.append(str(exc)[:300])
    finally:
        queue.put(None)


@app.post("/api/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    """Stream a RAG answer as server-sent events, then the retrieved sources."""
    config = _get_config()
    messages = [m for m in request.messages if m.get("content")]
    # Written by the producer thread below and read after it finishes, so the answer's citations
    # survive outside `stream()`.
    retrieved: list[Any] = []

    def stream() -> Any:
        stream_state = _AnswerStream(queue=SimpleQueue(), failure=[])

        def on_retrieval(spans: list[Any]) -> None:
            if spans:
                retrieved.extend(spans)

        threading.Thread(
            target=_produce_answer,
            args=(messages,),
            kwargs={
                "config": config,
                "on_retrieval": on_retrieval,
                "stream": stream_state,
            },
            daemon=True,
        ).start()

        announced = False
        while True:
            item = stream_state.queue.get()
            if item is None:
                break
            if not announced:
                announced = True
                yield _sse({"status": f"已检索到 {len(retrieved)} 段上下文，正在生成回答…"})
            yield _sse({"token": item})
        if stream_state.failure:
            yield _sse({"error": stream_state.failure[0]})
            return
        yield _sse({"sources": _serialize_sources(retrieved)})
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Proxies (including the local one on this machine) buffer responses unless told not
            # to, which would deliver the whole answer in one chunk and destroy the streaming feel.
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def main(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the WeKnow web server."""
    import uvicorn

    ensure_dotenv_loaded()
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
