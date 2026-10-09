"""Profile the chat path to find where the answer stalls.

Times the LLM round-trips and the retrieval rounds separately, because "it freezes on every
question" is compatible with several very different root causes (slow model, too many agentic
rounds, reranker download, lock contention, context blow-up).

    .venv/Scripts/python.exe bench_chat_profile.py
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
# The reranker downloads from HuggingFace on first use on this machine, which hangs for minutes
# behind the proxy. The server disables it too (`disable_default_reranker=True`).
os.environ.setdefault("RAG_RERANKER", "")

from weknow._dotenv import config_from_environ  # noqa: E402

QUESTION = "WeKnow 的检索内核用了哪些检索路径？"


def main() -> None:
    config = config_from_environ(disable_default_reranker=True, autoprobe=False)
    print(f"llm                 = {config.llm}")
    print(f"agentic_iterations  = {config.agentic_iterations}")
    print(f"llm_context_size    = {config.llm_context_size}")
    print(f"reranker            = {config.reranker!r}")
    print(f"chunk_max_size      = {config.chunk_max_size}")
    print(f"vector_multivector  = {config.vector_search_multivector}")
    print("-" * 62)

    # 1. Bare LLM round-trip with no retrieval, to separate model latency from retrieval cost.
    from litellm import completion

    messages = [{"role": "user", "content": QUESTION}]
    t0 = time.perf_counter()
    response = completion(model=config.llm, messages=messages)
    llm_ms = (time.perf_counter() - t0) * 1000
    print(f"bare LLM round-trip (no tools)   {llm_ms:>9.0f} ms")

    # 2. Retrieval on its own.
    from weknow import hybrid_search

    t1 = time.perf_counter()
    chunk_ids, scores = hybrid_search(QUESTION, config=config)
    search_ms = (time.perf_counter() - t1) * 1000
    print(f"hybrid_search                    {search_ms:>9.0f} ms  ({len(chunk_ids)} chunks)")

    # 3. The real thing: rag() with the agentic retrieval tool, counted round by round.
    from weknow import rag

    spans: list = []
    t2 = time.perf_counter()
    first_token_ms = None
    tokens = 0
    for token in rag(messages, on_retrieval=spans.extend, config=config):
        if first_token_ms is None:
            first_token_ms = (time.perf_counter() - t2) * 1000
        tokens += len(token)
    rag_ms = (time.perf_counter() - t2) * 1000
    print("-" * 62)
    print(f"rag() end to end                 {rag_ms:>9.0f} ms")
    print(f"  first token                    {first_token_ms or 0:>9.0f} ms")
    print(f"  tokens                         {tokens:>9}")
    print(f"  retrieval spans                {len(spans):>9}")
    print("-" * 62)
    # LLM cost implied by the gap between retrieval-only and the full rag() call.
    print(f"implied LLM work in rag()        {max(0, rag_ms - search_ms):>9.0f} ms")
    print(f"bare LLM vs rag() ratio          {rag_ms / max(llm_ms, 1e-9):>9.1f}x")


if __name__ == "__main__":
    main()
