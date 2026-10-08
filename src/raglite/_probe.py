"""Probe a model endpoint for the capabilities that LiteLLM cannot tell us, and cache the result.

LiteLLM resolves the context size, function calling support, and embedding dimension from its
bundled cost map. Any third-party or private endpoint therefore reports nothing, and RAGLite falls
back to a 2048-token context window and an extra network round trip to learn the embedding
dimension. Both are avoidable: the endpoints themselves can tell us, and the answer is worth
persisting because it does not change between runs.

`probe_and_cache` writes its findings into the project `.env` as the `RAG_LLM_CONTEXT_SIZE` and
`RAG_EMBEDDER_DIMENSION` keys, which `config_from_environ` then reads. Probing is best effort and
never raises: a failure just means the caller keeps the conservative fallback.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import re
import warnings
from dataclasses import dataclass

# Keys managed by this module, rewritten in place on every successful probe.
LLM_CONTEXT_SIZE_KEY = "RAG_LLM_CONTEXT_SIZE"
EMBEDDER_DIMENSION_KEY = "RAG_EMBEDDER_DIMENSION"
_MANAGED_KEYS = (LLM_CONTEXT_SIZE_KEY, EMBEDDER_DIMENSION_KEY)

# A pragmatic upper bound. Probing for an unknown model means sending a request that is large enough
# to observe the real limit, so we stop at a sane ceiling instead of trying to exhaust the endpoint.
MAX_PROBE_CONTEXT_SIZE = 131072

# `context length is only N tokens` / `maximum context length is N tokens` and similar phrasings.
_CONTEXT_PATTERN = re.compile(
    r"(?:context length|maximum context length|max_tokens|context window|input length)"
    r"[^0-9]{0,40}?(\d{3,7})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of probing an endpoint: which values were learned, and whether they changed."""

    context_size: int | None = None
    embedding_dimension: int | None = None
    changed: bool = False
    detail: str = ""

    @property
    def found_anything(self) -> bool:
        return self.context_size is not None or self.embedding_dimension is not None


def probe_embedding_dimension(embedder: str) -> int | None:
    """Return the embedding dimension reported by `embedder`, or `None` on failure.

    Calls the embedding API directly rather than going through `embed_strings`, so the probe costs
    exactly one request and does not depend on late chunking being supported by the endpoint.
    """
    with contextlib.suppress(Exception):
        import litellm

        from raglite._litellm import register_model_metadata

        # Forget any previously registered (possibly wrong) dimension, so the probe observes the
        # endpoint instead of our own declaration.
        _forget_registered_model(embedder)
        response = litellm.embedding(model=embedder, input=["Hello world"], timeout=30)
        dimension = len(response.data[0]["embedding"])  # type: ignore[attr-defined]
        if dimension > 0:
            # Record it for this process too, so the current run does not probe twice.
            register_model_metadata(embedder, output_vector_size=dimension, mode="embedding")
            return dimension
    return None


def probe_context_size(llm: str, *, ceiling: int = MAX_PROBE_CONTEXT_SIZE) -> int | None:
    """Return the context size of `llm`, or `None` if it cannot be determined cheaply.

    Only asked of the endpoint when it volunteers the number in an error message: we send a single
    over-long request and read the limit out of the rejection. Models that accept the request are
    probed by bisection, which costs a handful of calls but converges on the real value. This is
    deliberately conservative — a wrong context size silently truncates prompts, so returning `None`
    (and keeping the documented fallback) is better than guessing.
    """
    import litellm

    def _ask(prompt: str) -> tuple[bool, int | None]:
        """Send one request; return (accepted, limit_if_reported)."""
        try:
            litellm.completion(
                model=llm,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
                timeout=30,
            )
            return True, None
        except Exception as exc:  # noqa: BLE001
            match = _CONTEXT_PATTERN.search(str(exc))
            if match:
                return False, int(match.group(1))
            return False, None

    # Start high, and let the endpoint tell us the real limit in its rejection.
    accepted, reported = _ask("x" * (ceiling * 3))
    if reported is not None:
        return reported
    if accepted:
        # The endpoint happily accepted far more than `ceiling`, so `ceiling` is a safe lower bound.
        return ceiling

    # The rejection did not name a limit: bisect between 1024 and the ceiling.
    low, high = 1024, ceiling
    best: int | None = None
    while low <= high:
        mid = (low + high) // 2
        ok, limit = _ask("x" * (mid * 3))
        if limit is not None:
            return limit
        if ok:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best


def probe_and_cache(
    *,
    llm: str | None = None,
    embedder: str | None = None,
    dotenv_path: str | os.PathLike[str] = ".env",
    probe_llm: bool = True,
    probe_embedder: bool = True,
) -> ProbeResult:
    """Probe the configured endpoints and persist what we learn into `.env`.

    Values already present in `.env` (or in the environment) are respected: the probe only fills in
    what is missing, and never overwrites an explicit user choice. Pass `probe_llm=False` or
    `probe_embedder=False` to skip one side.
    """
    from raglite._dotenv import ensure_dotenv_loaded

    ensure_dotenv_loaded(dotenv_path)
    llm = llm or os.environ.get("RAG_LLM_MODEL")
    embedder = embedder or os.environ.get("RAG_EMBED_MODEL")

    updates: dict[str, int] = {}
    detail_parts: list[str] = []

    if probe_embedder and embedder and not os.environ.get(EMBEDDER_DIMENSION_KEY):
        dimension = probe_embedding_dimension(embedder)
        if dimension is not None:
            updates[EMBEDDER_DIMENSION_KEY] = dimension
            detail_parts.append(f"embedding_dimension={dimension}")
        else:
            detail_parts.append("embedding_dimension=unavailable")

    if probe_llm and llm and not os.environ.get(LLM_CONTEXT_SIZE_KEY):
        # Only the embedding dimension is cheap to probe. A context size probe needs several large
        # requests, so it is opt-in via `probe_llm=True` and still bounded by the rejection message.
        context_size = probe_context_size(llm)
        if context_size is not None:
            updates[LLM_CONTEXT_SIZE_KEY] = context_size
            detail_parts.append(f"llm_context_size={context_size}")
        else:
            detail_parts.append("llm_context_size=unavailable")

    changed = False
    if updates:
        changed = _update_dotenv(dotenv_path, updates)

    return ProbeResult(
        context_size=updates.get(LLM_CONTEXT_SIZE_KEY),
        embedding_dimension=updates.get(EMBEDDER_DIMENSION_KEY),
        changed=changed,
        detail="; ".join(detail_parts),
    )


def _forget_registered_model(model: str) -> None:
    """Remove `model` from LiteLLM's model map so the next lookup hits the endpoint again."""
    import litellm

    litellm.model_cost.pop(model, None)
    with contextlib.suppress(Exception):
        from litellm import get_model_info

        get_model_info.cache_clear()  # type: ignore[attr-defined]


def _update_dotenv(path: str | os.PathLike[str], updates: dict[str, int]) -> bool:
    """Write `updates` into the `.env` file at `path`, preserving the file's existing layout.

    Keys already present are rewritten in place, so comments and ordering survive. Missing keys are
    appended under a labelled section. Returns whether the file content changed.
    """
    dotenv_path = pathlib.Path(path)
    section_header = "# Managed automatically by `python -m raglite._probe` — safe to edit."
    if not dotenv_path.is_file():
        dotenv_path.write_text(
            "\n".join([section_header, *(f"{k}={v}" for k, v in updates.items()), ""]),
            encoding="utf-8",
        )
        return True

    try:
        lines = dotenv_path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        warnings.warn(f"Could not read {dotenv_path}: {exc}", stacklevel=2)
        return False

    remaining = dict(updates)
    rewritten: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.partition("=")[0].strip()
            if key in remaining:
                rewritten.append(f"{key}={remaining.pop(key)}")
                continue
        rewritten.append(line)

    if remaining:
        if rewritten and rewritten[-1].strip():
            rewritten.append("")
        rewritten.append(section_header)
        rewritten.extend(f"{k}={v}" for k, v in remaining.items())

    old_content = ""
    with contextlib.suppress(OSError):
        old_content = dotenv_path.read_text(encoding="utf-8-sig")
    new_content = "\n".join(rewritten).rstrip("\n") + "\n"
    if new_content == old_content:
        return False
    try:
        dotenv_path.write_text(new_content, encoding="utf-8")
    except OSError as exc:
        warnings.warn(f"Could not write {dotenv_path}: {exc}", stacklevel=2)
        return False
    return True


def main() -> int:
    """Probe the configured endpoints and report what was learned."""
    from raglite._dotenv import ensure_dotenv_loaded

    ensure_dotenv_loaded()
    result = probe_and_cache()
    if not result.found_anything:
        if result.detail:
            print(f"Nothing to probe: {result.detail}.")
            print("Already-cached values are in effect.")
        else:
            print("Nothing to probe: RAG_LLM_MODEL and RAG_EMBED_MODEL are both unset.")
        return 0
    print(f"Probe result: {result.detail}")
    if result.changed:
        print("Updated .env with the discovered capabilities.")
    else:
        print(".env already up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
