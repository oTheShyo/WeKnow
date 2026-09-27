"""Content-hash based incremental indexing and hot update.

`Document.from_text` already derives the document id from a hash of the content by default, so a
changed document automatically gets a new id. What is missing for incremental indexing is the
orchestration: detect new, unchanged, and changed documents against the database, delete the stale
versions, and insert only the new or changed ones. This module provides that orchestration.
"""

from collections.abc import Sequence
from typing import Literal

from sqlmodel import Session, col, select

from raglite._cache import get_retrieval_cache
from raglite._config import RAGLiteConfig
from raglite._database import Document, create_database_engine
from raglite._typing import DocumentId

SyncKey = Literal["id", "filename", "uri", "url"]


def _document_key_value(document: Document, key: SyncKey) -> str:
    """Get the sync key value of a document."""
    if key == "id":
        return document.id
    value = getattr(document, key, None)
    if value is None:
        error_message = f"Document {document.id!r} has no {key!r} to sync by."
        raise ValueError(error_message)
    return str(value)


def _record_group_key(record: Document, key: SyncKey) -> str:
    """Get the sync key value of a stored document record (None attributes fall back to id)."""
    if key == "id":
        return record.id
    value = getattr(record, key, None)
    return str(value) if value is not None else record.id


def _existing_documents_by_key(
    session: Session, key: SyncKey, key_values: list[str]
) -> dict[str, list[Document]]:
    """Look up existing documents grouped by the sync key value."""
    if not key_values:
        return {}
    column = col(Document.id) if key == "id" else getattr(Document, key)
    existing_by_key: dict[str, list[Document]] = {}
    batch_size = 128
    for i in range(0, len(key_values), batch_size):
        batch = key_values[i : i + batch_size]
        for record in session.exec(select(Document).where(column.in_(batch))).all():
            existing_by_key.setdefault(_record_group_key(record, key), []).append(record)
    return existing_by_key


def sync_documents(  # noqa: PLR0913
    documents: Sequence[Document],
    *,
    key: SyncKey = "id",
    prune: bool = False,
    max_workers: int | None = None,
    config: RAGLiteConfig | None = None,
    invalidate_query_adapter: bool = False,
) -> dict[str, list[DocumentId]]:
    """Incrementally synchronize documents with the database by content hash.

    Documents whose content is unchanged are skipped (same content hash implies the same document
    id). Changed documents get a new id, so their stale version is deleted before reinsertion.
    Documents that are new are inserted. This makes document updates cheap: only the delta goes
    through parsing, splitting, and embedding.

    Parameters
    ----------
    documents
        The authoritative set of documents to synchronize.
    key
        The document attribute that identifies "the same document" across versions: `"id"` (the
        content hash itself), `"filename"`, `"uri"`, or `"url"`. With `"id"`, a changed document
        is indistinguishable from a new one unless `prune` removes the stale version. With
        `"filename"` (recommended for hot updates), a changed document is detected as the same
        document with a new content hash, and its stale version is replaced.
    prune
        If True, delete database documents whose sync key is not present in `documents`. Use with
        care: with `key="id"` this deletes every database document that is not in `documents`.
    max_workers
        The maximum number of worker threads used for insertion.
    config
        The RAGLite config to use.
    invalidate_query_adapter
        If True, invalidate the query adapter after deletion.

    Returns
    -------
    dict
        A report with the inserted, unchanged, and deleted document ids.
    """
    from raglite._delete import delete_documents
    from raglite._insert import insert_documents

    config = config or RAGLiteConfig()
    # Deduplicate the input documents by id, keeping insertion order.
    documents = list({doc.id: doc for doc in documents}.values())
    documents = [doc for doc in documents if isinstance(doc.content, str) and doc.content.strip()]
    document_keys = {doc.id: _document_key_value(doc, key) for doc in documents}
    report: dict[str, list[DocumentId]] = {"inserted": [], "unchanged": [], "deleted": []}
    # Look up the existing documents that match the sync keys of the input documents.
    with Session(create_database_engine(config)) as session:
        existing_by_key = _existing_documents_by_key(session, key, list(set(document_keys.values())))
    # Decide per document: unchanged, or delete stale versions and reinsert.
    stale_document_ids: set[DocumentId] = set()
    documents_to_insert: list[Document] = []
    for document in documents:
        existing = existing_by_key.get(document_keys[document.id], [])
        if any(record.id == document.id for record in existing):
            report["unchanged"].append(document.id)
            continue
        stale_document_ids.update(record.id for record in existing)
        report["deleted"].extend(record.id for record in existing)
        documents_to_insert.append(document)
        report["inserted"].append(document.id)
    # Optionally prune database documents whose sync key is not in the authoritative set.
    if prune:
        with Session(create_database_engine(config)) as session:
            if key == "id":
                current_keys = set(document_keys.values())
                prune_records = [
                    record
                    for record in session.exec(select(Document)).all()
                    if record.id not in current_keys
                ]
            else:
                column = getattr(Document, key)
                existing_keys = list(set(document_keys.values()))
                matching: list[Document] = []
                batch_size = 128
                for i in range(0, len(existing_keys), batch_size):
                    batch = existing_keys[i : i + batch_size]
                    matching.extend(session.exec(select(Document).where(column.in_(batch))).all())
                current_keys = set(document_keys.values())
                prune_records = [
                    record for record in matching if _document_key_value(record, key) not in current_keys
                ]
            for record in prune_records:
                stale_document_ids.add(record.id)
                report["deleted"].append(record.id)
    # Delete the stale versions, then insert only the new or changed documents.
    if stale_document_ids:
        delete_documents(
            list(stale_document_ids), config=config, invalidate_query_adapter=invalidate_query_adapter
        )
    if documents_to_insert:
        insert_documents(documents_to_insert, max_workers=max_workers, config=config)
    # Any index mutation invalidates the retrieval result cache.
    if report["inserted"] or report["deleted"]:
        get_retrieval_cache().clear()
    return report


def get_document_content_hashes(
    document_ids: list[DocumentId] | None = None, *, config: RAGLiteConfig | None = None
) -> dict[DocumentId, str]:
    """Get the stored content hash (the document id) of documents.

    Diffs documents against the database without loading their content.
    """
    config = config or RAGLiteConfig()
    with Session(create_database_engine(config)) as session:
        statement = select(Document.id, Document.filename)
        if document_ids:
            statement = statement.where(col(Document.id).in_(document_ids))
        return {doc_id: (filename or "") for doc_id, filename in session.exec(statement).all()}


def diff_documents(
    documents: Sequence[Document], *, config: RAGLiteConfig | None = None
) -> dict[str, list[DocumentId]]:
    """Report which documents are new, unchanged, or changed, without touching the database index.

    A document is unchanged when its content hash (the document id) is already in the database.
    A document is changed when a document with the same filename exists but has a different hash.
    Everything else is new.
    """
    config = config or RAGLiteConfig()
    document_ids = [doc.id for doc in documents]
    stored_hashes = get_document_content_hashes(document_ids, config=config)
    stored_filenames = {
        filename: doc_id for doc_id, filename in get_document_content_hashes(config=config).items()
    }
    report: dict[str, list[DocumentId]] = {"new": [], "unchanged": [], "changed": []}
    for document in documents:
        if document.id in stored_hashes:
            report["unchanged"].append(document.id)
        elif document.filename and document.filename in stored_filenames:
            report["changed"].append(document.id)
        else:
            report["new"].append(document.id)
    return report
