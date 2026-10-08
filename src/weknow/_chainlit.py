"""Chainlit frontend for WeKnow."""

import os
from pathlib import Path
from typing import TYPE_CHECKING

import chainlit as cl
from chainlit.input_widget import Switch, TextInput

from weknow import (
    Document,
    RAGLiteConfig,
    async_rag,
    hybrid_search,
    insert_documents,
    rerank_chunks,
)
from weknow._dotenv import (
    config_from_environ,
    default_database_url,
    default_embedder,
    default_llm,
    ensure_dotenv_loaded,
)
from weknow._markdown import document_to_markdown

if TYPE_CHECKING:
    from weknow._database import ChunkSpan

async_insert_documents = cl.make_async(insert_documents)
async_hybrid_search = cl.make_async(hybrid_search)
async_rerank_chunks = cl.make_async(rerank_chunks)


@cl.on_chat_start
async def start_chat() -> None:
    """Initialize the chat."""
    # Set tokenizers parallelism to avoid a deadlock warning.
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    # Pick up the project `.env` so the frontend, the CLI, and the benchmark scripts agree. Chainlit
    # does not read `.env` on its own, and without this the frontend silently falls back to the
    # llama.cpp defaults and tries to download weights.
    ensure_dotenv_loaded()
    # Build the initial config from the environment. `config_from_environ` resolves LLM, embedder,
    # and reranker without instantiating `RAGLiteConfig()` bare: those default factories import
    # `llama_cpp` and download FlashRank weights, both of which fail on an offline machine and would
    # make the frontend unusable. `disable_default_reranker` stops a missing `.env` from turning
    # into a multi-hundred-megabyte download at startup; opt back in with `RAG_RERANKER=<model>`.
    config = config_from_environ(disable_default_reranker=True)
    settings = await cl.ChatSettings(  # type: ignore[no-untyped-call]
        [
            TextInput(id="db_url", label="Database URL", initial=str(config.db_url)),
            TextInput(id="llm", label="LLM", initial=config.llm),
            TextInput(id="embedder", label="Embedder", initial=config.embedder),
            Switch(id="vector_search_query_adapter", label="Query adapter", initial=True),
        ]
    ).send()
    await update_config(settings)


@cl.on_settings_update  # type: ignore[arg-type]
async def update_config(settings: cl.ChatSettings) -> None:
    """Update the RAGLite config."""
    # The settings dialog only exposes the four fields above; everything else (reranker, parser
    # engine, graph config, chunking, ...) is taken from the environment so that the frontend
    # behaves exactly like the CLI. The dialog values win where they overlap.
    config = config_from_environ(
        db_url=settings["db_url"],  # type: ignore[index]
        llm=settings["llm"],  # type: ignore[index]
        embedder=settings["embedder"],  # type: ignore[index]
        vector_search_query_adapter=settings["vector_search_query_adapter"],  # type: ignore[index]
        disable_default_reranker=True,
    )
    cl.user_session.set("config", config)  # type: ignore[no-untyped-call]
    # Run a search to prime the pipeline if it's a local pipeline.
    if config.embedder.startswith("llama-cpp-python"):
        query = "Hello world"
        chunk_ids, _ = await async_hybrid_search(query=query, config=config)
        _ = await async_rerank_chunks(query=query, chunk_ids=chunk_ids, config=config)


@cl.on_message
async def handle_message(user_message: cl.Message) -> None:
    """Respond to a user message."""
    # Get the config and message history from the user session.
    config: RAGLiteConfig = cl.user_session.get("config")  # type: ignore[no-untyped-call]
    # Determine what to do with the attachments.
    inline_attachments = []
    for file in user_message.elements:
        if file.path:
            doc_md = document_to_markdown(Path(file.path), config=config)
            if len(doc_md) // 3 <= 5 * (config.chunk_max_size // 3):
                # Document is small enough to attach to the context.
                inline_attachments.append(f"{Path(file.path).name}:\n\n{doc_md}")
            else:
                # Document is too large and must be inserted into the database.
                async with cl.Step(name="insert", type="run") as step:
                    step.input = Path(file.path).name
                    document = Document.from_path(Path(file.path), config=config)
                    await async_insert_documents([document], config=config)
    # Append any inline attachments to the user prompt.
    user_prompt = (
        "\n\n".join(
            f'<attachment index="{i}">\n{attachment.strip()}\n</attachment>'
            for i, attachment in enumerate(inline_attachments)
        )
        + f"\n\n{user_message.content}"
    ).strip()
    # Stream the LLM response.
    assistant_message = cl.Message(content="")
    chunk_spans: list[ChunkSpan] = []
    messages: list[dict[str, str]] = cl.chat_context.to_openai()[:-1]  # type: ignore[no-untyped-call]
    messages.append({"role": "user", "content": user_prompt})
    async for token in async_rag(messages, on_retrieval=chunk_spans.extend, config=config):
        await assistant_message.stream_token(token)
    # Append RAG sources, if any.
    if chunk_spans:
        rag_sources: dict[str, list[str]] = {}
        for chunk_span in chunk_spans:
            rag_sources.setdefault(chunk_span.document.id, [])
            rag_sources[chunk_span.document.id].append(str(chunk_span))
        assistant_message.content += "\n\nSources: " + ", ".join(  # Rendered as hyperlinks.
            f"[{i + 1}]" for i in range(len(rag_sources))
        )
        assistant_message.elements = [  # Markdown content is rendered in sidebar.
            cl.Text(name=f"[{i + 1}]", content="\n\n---\n\n".join(content), display="side")  # type: ignore[misc]
            for i, (_, content) in enumerate(rag_sources.items())
        ]
    await assistant_message.update()  # type: ignore[no-untyped-call]
