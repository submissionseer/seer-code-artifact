from __future__ import annotations

import asyncio
from typing import Any

from seer.openrouter import OpenRouterConfig, run_completions, build_cache_key
from seer.util_text import normalize_text


def _f1_score(pred: str, gold: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def em_f1(pred: str, gold: str | list[str]) -> tuple[float, float]:
    """
    Compute Exact Match and F1 scores.

    Args:
        pred: Predicted answer string
        gold: Gold answer string or list of acceptable answers

    Returns:
        Tuple of (EM score, F1 score) - both in [0, 1]
    """
    if not pred:
        return 0.0, 0.0
    golds = [gold] if isinstance(gold, str) else gold
    golds = [g for g in golds if g]  # Filter empty strings
    if not golds:
        return 0.0, 0.0
    em = max(float(normalize_text(pred) == normalize_text(g)) for g in golds)
    f1 = max(_f1_score(pred, g) for g in golds)
    return em, f1


READER_PROMPT_VERSION = "reader_short_answer_v2"
READER_MAX_TOKENS = 12

READER_SYSTEM_PROMPT = """You are a question answering system. Answer the question using only the provided context.

Rules:
- Output ONLY the final answer text (a short answer span or phrase)
- Do NOT include explanations, citations, prefixes, or full sentences
- If the answer is directly stated in the context, copy it exactly
- If the answer can be inferred from the context, provide the shortest correct answer
- If the context does not contain enough information, respond exactly with UNANSWERABLE
- Do not make up information not in the context"""

READER_USER_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""


def build_reader_prompt(question: str, context: str) -> list[dict[str, str]]:
    """Build messages for the reader prompt."""
    return [
        {"role": "system", "content": READER_SYSTEM_PROMPT},
        {"role": "user", "content": READER_USER_TEMPLATE.format(question=question, context=context)},
    ]


def _clean_reader_answer(text: str) -> str:
    """Normalize common formatting issues for answer-only EM/F1 eval."""
    if not text:
        return ""
    ans = text.strip()
    # Keep the first non-empty line only (models sometimes append explanations on later lines).
    lines = [ln.strip() for ln in ans.splitlines() if ln.strip()]
    if lines:
        ans = lines[0]
    # Strip trivial answer prefixes.
    for prefix in ("Answer:", "Final answer:", "The answer is"):
        if ans.lower().startswith(prefix.lower()):
            ans = ans[len(prefix):].strip(" :")
    # Remove surrounding quotes.
    if len(ans) >= 2 and ans[0] == ans[-1] and ans[0] in {'"', "'"}:
        ans = ans[1:-1].strip()
    return ans


def format_context_for_reader(passages: list[dict]) -> str:
    """Format passages into a context string for the reader."""
    lines = []
    for p in passages:
        text = p.get("text", "")
        title = p.get("title", "")
        if title:
            lines.append(f"[{title}] {text}")
        else:
            lines.append(text)
    return "\n\n".join(lines)


async def run_frozen_reader(
    config: OpenRouterConfig,
    cache_dir: str,
    model: str,
    question: str,
    passages: list[dict],
) -> dict[str, Any]:
    """
    Run the frozen reader to generate an answer from context.

    Args:
        config: OpenRouter configuration
        cache_dir: Directory for caching
        model: Model slug (should be fixed for the experiment)
        question: The query
        passages: List of passage dicts with 'text' key

    Returns:
        Dict with 'answer' and metadata
    """
    context = format_context_for_reader(passages)
    messages = build_reader_prompt(question, context)

    cache_key = build_cache_key(
        model,
        f"{READER_SYSTEM_PROMPT}\n[{READER_PROMPT_VERSION}]",
        READER_USER_TEMPLATE,
        question,
        context,
    )

    requests = [{
        "model": model,
        "messages": messages,
        "cache_key": cache_key,
        "max_tokens": READER_MAX_TOKENS,
        "meta": {"type": "frozen_reader"},
    }]

    results = await run_completions(config, cache_dir, requests)

    if results:
        response = results[0].get("response", {})
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {
            "answer": _clean_reader_answer(content),
            "model": model,
            "cached": "cache_key" in results[0],
        }

    return {"answer": "", "model": model, "error": "No response"}


def run_frozen_reader_sync(
    config: OpenRouterConfig,
    cache_dir: str,
    model: str,
    question: str,
    passages: list[dict],
) -> dict[str, Any]:
    """Synchronous wrapper for run_frozen_reader."""
    return asyncio.run(run_frozen_reader(config, cache_dir, model, question, passages))


async def evaluate_reader_batch(
    config: OpenRouterConfig,
    cache_dir: str,
    model: str,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Evaluate the frozen reader on a batch of records.

    Args:
        config: OpenRouter configuration
        cache_dir: Cache directory
        model: Model slug
        records: List of dicts with 'question', 'context' (list of passages), 'gold' (with 'answer')

    Returns:
        List of dicts with EM, F1, and predicted answer
    """
    requests = []
    for record in records:
        question = record["question"]
        passages = record.get("context", [])
        context = format_context_for_reader(passages)
        messages = build_reader_prompt(question, context)
        cache_key = build_cache_key(
            model,
            f"{READER_SYSTEM_PROMPT}\n[{READER_PROMPT_VERSION}]",
            READER_USER_TEMPLATE,
            question,
            context,
        )
        requests.append({
            "model": model,
            "messages": messages,
            "cache_key": cache_key,
            "max_tokens": READER_MAX_TOKENS,
            "meta": {"type": "frozen_reader"},
        })

    responses = await run_completions(config, cache_dir, requests)
    results = []
    for record, response in zip(records, responses):
        gold = record.get("gold", {})
        gold_answer = gold.get("answer", gold.get("answers", ""))
        payload = response.get("response", response)
        content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
        pred_answer = _clean_reader_answer(content)
        em, f1 = em_f1(pred_answer, gold_answer)
        results.append({
            "qid": record.get("qid", ""),
            "predicted_answer": pred_answer,
            "gold_answer": gold_answer,
            "em": em,
            "f1": f1,
            "is_unanswerable": pred_answer.upper() == "UNANSWERABLE",
            "gold_is_answerable": gold.get("is_answerable", True),
        })

    return results


def run_reader_batch_sync(
    config: OpenRouterConfig,
    cache_dir: str,
    model: str,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Synchronous wrapper for evaluate_reader_batch."""
    return asyncio.run(evaluate_reader_batch(config, cache_dir, model, records))
