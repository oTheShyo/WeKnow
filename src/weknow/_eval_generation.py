"""Generation-side evaluation: ROUGE-L and BLEU over question-answer pairs.

These metrics cover the generation side of the pipeline. The retrieval side is measured separately
by the ablation in `bench_graph_ab.py` (recall@k and MRR over a hand-built question set), while
`_bench.py` offers an ir_measures-compatible TREC run interface for standard datasets. The
implementations here are dependency-free (token-level, CJK-aware tokenization), so they run
anywhere the rest of RAGLite runs.
"""

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

_TOKEN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


def _tokenize(text: str) -> list[str]:
    """Lowercase tokens: alnum words and individual CJK characters."""
    return _TOKEN.findall(text.lower())


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence of two token sequences.

    O(len(a) * len(b)) time, O(min(len(a), len(b))) space.
    """
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return 0
    previous = [0] * (len(b) + 1)
    for item_a in a:
        current = [0]
        for j, item_b in enumerate(b, start=1):
            if item_a == item_b:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(current[j - 1], previous[j]))
        previous = current
    return previous[-1]


def rouge_l_f1(reference: str, hypothesis: str) -> float:
    """ROUGE-L F1 score between a reference and a hypothesis (0.0 to 1.0)."""
    reference_tokens, hypothesis_tokens = _tokenize(reference), _tokenize(hypothesis)
    if not reference_tokens or not hypothesis_tokens:
        return 0.0
    lcs = _lcs_length(reference_tokens, hypothesis_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(hypothesis_tokens)
    recall = lcs / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu_n(reference: str, hypothesis: str, *, max_n: int = 4) -> float:
    """BLEU score with add-one smoothing between a reference and a hypothesis (0.0 to 1.0)."""
    reference_tokens, hypothesis_tokens = _tokenize(reference), _tokenize(hypothesis)
    if not reference_tokens or not hypothesis_tokens:
        return 0.0
    # Clipped n-gram precisions with add-one smoothing.
    log_precisions = []
    for n in range(1, max_n + 1):
        hypothesis_ngrams = Counter(
            tuple(hypothesis_tokens[i : i + n]) for i in range(len(hypothesis_tokens) - n + 1)
        )
        reference_ngrams = Counter(
            tuple(reference_tokens[i : i + n]) for i in range(len(reference_tokens) - n + 1)
        )
        matches = sum(min(count, reference_ngrams[gram]) for gram, count in hypothesis_ngrams.items())
        total = max(sum(hypothesis_ngrams.values()), 1)
        log_precisions.append(math.log((matches + 1) / (total + 1)))
    # Brevity penalty.
    bp = min(0.0, 1 - len(reference_tokens) / len(hypothesis_tokens))
    return math.exp(bp + sum(log_precisions) / max_n)


def evaluate_generation(
    qa_pairs: Iterable[tuple[str, str]],
    *,
    answers: Sequence[str] | None = None,
    num_chunks: int = 5,
    config: Any = None,
) -> dict[str, Any]:
    """Evaluate generated answers against reference answers with ROUGE-L and BLEU.

    Parameters
    ----------
    qa_pairs
        Pairs of (question, reference answer).
    answers
        Pre-generated answers. When omitted, answers are generated with `rag` (requires an LLM in
        the config).
    num_chunks
        The number of chunks to retrieve per question when generating answers.
    config
        The RAGLite config used for generation.

    Returns
    -------
    dict
        Mean ROUGE-L F1, mean BLEU, and per-question details.
    """
    from weknow._rag import add_context, rag, retrieve_context

    qa_pairs = list(qa_pairs)
    if answers is None:
        if config is None:
            error_message = "Either pre-generated `answers` or a `config` with an LLM is required."
            raise ValueError(error_message)
        generated_answers = []
        for question, _ in qa_pairs:
            chunk_spans = retrieve_context(question, num_chunks=num_chunks, config=config)
            messages = [{"role": "user", "content": question}]
            if chunk_spans:
                messages = [add_context(question, context=chunk_spans, config=config)]
            generated_answers.append("".join(rag(messages, config=config)))
        answers = generated_answers
    if len(answers) != len(qa_pairs):
        error_message = f"Got {len(answers)} answers for {len(qa_pairs)} question-answer pairs."
        raise ValueError(error_message)
    details: list[dict[str, Any]] = [
        {
            "question": question,
            "reference": reference,
            "answer": answer,
            "rouge_l": round(rouge_l_f1(reference, answer), 4),
            "bleu": round(bleu_n(reference, answer), 4),
        }
        for (question, reference), answer in zip(qa_pairs, answers, strict=True)
    ]
    return {
        "n": len(details),
        "mean_rouge_l": round(sum(item["rouge_l"] for item in details) / max(len(details), 1), 4),
        "mean_bleu": round(sum(item["bleu"] for item in details) / max(len(details), 1), 4),
        "details": details,
    }
