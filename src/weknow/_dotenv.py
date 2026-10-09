"""Load a project `.env` file and resolve safe model defaults.

The CLI, the Chainlit frontend, and the benchmark scripts all need the same two things:

1. **Read the project `.env`.** Only the CLI goes through `pydantic-settings` with
   `env_file=".env"`; the Chainlit frontend and the standalone scripts read `os.environ` directly
   and therefore silently ignore `.env`. Loading it here makes the three entry points agree.

2. **Avoid the `llama_cpp` import when it is not installed.** `RAGLiteConfig()` evaluates its `llm`
   and `embedder` default factories, which lazily import `llama_cpp` and raise
   `ModuleNotFoundError` on any machine without that extra. Entry points that only need a database
   URL — or that intend to override the models from the environment — must not instantiate
   `RAGLiteConfig()` blindly.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import pathlib

from weknow._config import RAGLiteConfig

# Fallbacks used when `llama-cpp-python` is unavailable. These are OpenAI-compatible model names,
# so they are only meaningful together with a reachable OpenAI-compatible endpoint.
FALLBACK_LLM = "openai/gpt-4o-mini"
FALLBACK_EMBEDDER = "openai/text-embedding-3-small"

_DOTENV_LOADED = False


def llama_cpp_is_available() -> bool:
    """Return whether `llama-cpp-python` is importable."""
    return importlib.util.find_spec("llama_cpp") is not None


def default_database_url() -> str:
    """Return the default database URL without resolving the model defaults.

    Reading the dataclass field avoids evaluating the `llm` / `embedder` default factories, which
    would import `llama_cpp`.
    """
    return str(RAGLiteConfig.__dataclass_fields__["db_url"].default)  # type: ignore[attr-defined]


def load_dotenv(path: str | os.PathLike[str] = ".env", *, override: bool = False) -> bool:
    """Load a `.env` file into `os.environ`. Returns whether a file was loaded.

    A deliberately minimal parser: blank lines and `#` comments are skipped, each line is split on
    the first `=`, and surrounding quotes are stripped. Reading with `utf-8-sig` swallows a BOM.
    Existing environment variables win unless `override` is set, so an explicit shell export always
    takes precedence over the file.
    """
    global _DOTENV_LOADED
    dotenv_path = pathlib.Path(path)
    if not dotenv_path.is_file():
        return False
    try:
        text = dotenv_path.read_text(encoding="utf-8-sig")
    except OSError:
        return False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and (override or key not in os.environ):
            os.environ[key] = value
    _DOTENV_LOADED = True
    return True


def ensure_dotenv_loaded(path: str | os.PathLike[str] = ".env") -> None:
    """Load `.env` once per process. Safe to call from every entry point."""
    if not _DOTENV_LOADED:
        load_dotenv(path)


def default_llm() -> str:
    """Return the default LLM, falling back to an OpenAI-compatible name without `llama_cpp`."""
    return RAGLiteConfig().llm if llama_cpp_is_available() else FALLBACK_LLM


def default_embedder() -> str:
    """Return the default embedder, falling back to an OpenAI-compatible name without `llama_cpp`."""
    return RAGLiteConfig().embedder if llama_cpp_is_available() else FALLBACK_EMBEDDER


def config_from_environ(
    *,
    prefix: str = "RAG_",
    db_url_var: str | None = None,
    llm_var: str | None = None,
    embedder_var: str | None = None,
    disable_default_reranker: bool = False,
    autoprobe: bool = True,
    dotenv_path: str | os.PathLike[str] = ".env",
    **overrides: object,
) -> RAGLiteConfig:
    """Build a `RAGLiteConfig` from the environment, after loading `.env`.

    Only the keyword arguments that are actually present in the environment are passed through, so
    the `RAGLiteConfig` defaults still apply to everything else. Empty environment entries are
    ignored rather than becoming empty strings in the config.

    `RAG_RERANKER` receives special treatment, because the default reranker downloads model weights
    from Hugging Face on first use:

    * unset → the `RAGLiteConfig` default reranker is used, unless `disable_default_reranker` is set
    * set but empty (`RAG_RERANKER=`) → the reranker is disabled (`None`), which is what offline
      setups want
    * set to a value → passed through as-is

    Pass `disable_default_reranker=True` from interactive entry points (the Chainlit frontend, the
    CLI) so that a missing `.env` cannot turn into a multi-hundred-megabyte download at startup.

    With `autoprobe` (the default), any capability the user has not declared is asked of the endpoint
    once and cached back into `.env`, so a fresh checkout only pays for it on the first run. Set it
    to `False` for callers that must not make network requests at config time.
    """
    ensure_dotenv_loaded(dotenv_path)
    resolved: dict[str, object] = {}
    for var, key in (
        (db_url_var or f"{prefix}DB_URL", "db_url"),
        (llm_var or f"{prefix}LLM_MODEL", "llm"),
        (embedder_var or f"{prefix}EMBED_MODEL", "embedder"),
    ):
        value = os.environ.get(var)
        if value:
            resolved[key] = value
    # Declared capability metadata, used for models missing from LiteLLM's bundled cost map. When
    # the user has not declared them, LiteLLM falls back to a 2048-token context window and has to
    # probe the embedding dimension over the network. `autoprobe` fills the gap by asking the
    # endpoints once and caching the answer in `.env`.
    for var, key in (
        (f"{prefix}LLM_CONTEXT_SIZE", "llm_context_size"),
        (f"{prefix}EMBEDDER_DIMENSION", "embedder_dimension"),
        (f"{prefix}EMBEDDER_BATCH_SIZE", "embedder_batch_size"),
        (f"{prefix}AGENTIC_ITERATIONS", "agentic_iterations"),
    ):
        raw = os.environ.get(var)
        if raw:
            try:
                resolved[key] = int(raw)
            except ValueError:
                pass
    if autoprobe:
        _autoprobe_missing_capabilities(
            resolved,
            llm=resolved.get("llm"),
            embedder=resolved.get("embedder"),
            dotenv_path=dotenv_path,
        )
    reranker_var = f"{prefix}RERANKER"
    if reranker_var in os.environ:
        reranker_spec = os.environ.get(reranker_var, "").strip()
        resolved["reranker"] = reranker_spec or None
    elif disable_default_reranker:
        # No explicit choice, but we must not let the default factory pull weights over the network.
        resolved["reranker"] = None
    resolved.update({k: v for k, v in overrides.items() if v is not None})
    return RAGLiteConfig(**resolved)  # type: ignore[arg-type]


def _autoprobe_missing_capabilities(
    resolved: dict[str, object],
    *,
    llm: object,
    embedder: object,
    dotenv_path: str | os.PathLike[str],
) -> None:
    """Probe whichever capability is still unknown, and record it in `resolved` and in `.env`.

    Entirely best effort: any failure leaves the conservative fallbacks in place rather than
    interrupting the caller. The probe imports LiteLLM, so it is done lazily and only when there is
    actually a gap to fill.
    """
    context_key = "llm_context_size"
    dimension_key = "embedder_dimension"
    want_context = context_key not in resolved and isinstance(llm, str) and bool(llm)
    want_dimension = dimension_key not in resolved and isinstance(embedder, str) and bool(embedder)
    if not (want_context or want_dimension):
        return
    with contextlib.suppress(Exception):
        from weknow import _probe

        result = _probe.probe_and_cache(
            llm=llm if isinstance(llm, str) else None,
            embedder=embedder if isinstance(embedder, str) else None,
            dotenv_path=dotenv_path,
            probe_llm=want_context,
            probe_embedder=want_dimension,
        )
        if result.context_size is not None:
            resolved[context_key] = result.context_size
        if result.embedding_dimension is not None:
            resolved[dimension_key] = result.embedding_dimension
        # The probe already registered both models with LiteLLM, so the learned values take effect
        # even though the environment variables are still absent.
