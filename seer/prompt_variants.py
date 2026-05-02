"""Helpers for DSPy prompt variant extraction."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import dspy
from dspy.adapters.chat_adapter import ChatAdapter


def extract_prompt_from_history(lm: Any) -> str | None:
    if not hasattr(lm, "history") or not lm.history:
        return None
    last = lm.history[-1]
    if isinstance(last, list):
        if not last:
            return None
        last = last[-1]
    if isinstance(last, str):
        return last
    if isinstance(last, dict):
        if "prompt" in last and last["prompt"]:
            prompt = last["prompt"]
            if isinstance(prompt, (list, dict)):
                return json.dumps(prompt)
            return prompt
        if "messages" in last and last["messages"]:
            messages = last["messages"]
            if isinstance(messages, str):
                return messages
            if isinstance(messages, list):
                for msg in reversed(messages):
                    if isinstance(msg, str):
                        return msg
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        return msg.get("content")
    return None


def _format_messages(messages: list[dict[str, Any]]) -> str:
    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, (dict, list)):
            content = json.dumps(content)
        lines.append(f"[{role}]\n{content}")
    return "\n\n".join(lines).strip()


def run_program_for_prompt(prog: Any, lm: Any, logger: Any) -> str | None:
    predictor = getattr(prog, "generate_query", None)
    demos = getattr(predictor, "demos", None) if predictor is not None else None
    signature = getattr(predictor, "signature", None) if predictor is not None else None
    if demos and signature is not None:
        try:
            adapter = ChatAdapter()
            messages = adapter.format(
                signature=signature,
                demos=demos,
                inputs={"context": "{context}", "question": "{question}"},
            )
            prompt = _format_messages(messages)
            if prompt:
                return prompt
        except Exception as exc:
            logger.warning("Failed to format prompt from demos: %s", exc, exc_info=True)
    try:
        try:
            _ = prog("{PROMPT}")
        except AssertionError:
            pass
    except Exception:
        try:
            _ = prog(question="{question}", context=[])
        except Exception as exc:
            logger.warning("Failed to run program for prompt capture: %s", exc, exc_info=True)
            return None
    try:
        prompt = extract_prompt_from_history(lm)
    except Exception as exc:
        logger.warning("Failed to extract prompt: %s", exc, exc_info=True)
        return None
    if not prompt:
        logger.warning("Prompt capture returned empty prompt.")
        return None
    return prompt


def _extract_demos_from_state(state: dict) -> list[dict]:
    if not isinstance(state, dict):
        return []
    preferred_keys = (
        "generate_query",
        "generate_query.predict",
        "generate_query.predictor",
        "generate_query.pred",
    )
    for key in preferred_keys:
        container = state.get(key)
        if isinstance(container, dict):
            demos = container.get("demos")
            if isinstance(demos, list) and demos:
                return demos
    for value in state.values():
        if isinstance(value, dict):
            demos = value.get("demos")
            if isinstance(demos, list) and demos:
                return demos
    return []


def _format_prompt_from_demos(demos: list[dict], signature: Any | None = None) -> str | None:
    if not demos:
        return None
    if signature is None:
        predictor = dspy.Predict("context, question -> search_query")
        signature = predictor.signature
    adapter = ChatAdapter()
    messages = adapter.format(
        signature=signature,
        demos=demos,
        inputs={"context": "{context}", "question": "{question}"},
    )
    return _format_messages(messages)


def load_dspy_states(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and isinstance(data.get("states"), list):
        states = data["states"]
    elif isinstance(data, list):
        states = data
    elif isinstance(data, dict):
        states = [data]
    else:
        raise ValueError("DSPy state JSON must be a dict or list.")
    return [state for state in states if isinstance(state, dict)]


def prompts_from_dspy_state(path: Path, limit: int | None = None) -> list[str]:
    states = load_dspy_states(path)
    prompts = []
    signature = None
    try:
        signature = dspy.Predict("context, question -> search_query").signature
    except Exception:
        signature = None
    for state in states:
        demos = _extract_demos_from_state(state)
        if not demos:
            continue
        prompt = _format_prompt_from_demos(demos, signature)
        if prompt:
            prompts.append(prompt)
        if limit and len(prompts) >= limit:
            break
    return dedupe_prompts(prompts, limit or len(prompts))


def collect_candidate_programs(compiled: Any, optimizer: Any, logger: Any) -> list[Any]:
    sources = [
        ("compiled.candidate_programs", getattr(compiled, "candidate_programs", None)),
        ("optimizer.candidate_programs", getattr(optimizer, "candidate_programs", None)),
        ("compiled.teleprompter.candidate_programs", getattr(getattr(compiled, "teleprompter", None), "candidate_programs", None)),
        ("compiled.programs", getattr(compiled, "programs", None)),
    ]
    for name, value in sources:
        if not value:
            logger.info("%s empty", name)
            continue
        try:
            logger.info("%s length=%d", name, len(value))
            logger.info("%s sample_type=%s", name, type(value[0]))
        except Exception:
            logger.info("%s length/type unavailable", name)
        programs = []
        for item in value:
            prog = item[-1] if isinstance(item, (list, tuple)) else item
            if isinstance(prog, dict):
                try:
                    logger.info("%s dict keys=%s", name, sorted(prog.keys()))
                except Exception:
                    logger.info("%s dict keys unavailable", name)
                for key in ("program", "candidate_program", "module", "prog"):
                    candidate = prog.get(key)
                    if callable(candidate):
                        programs.append(candidate)
                        break
                continue
            if callable(prog):
                programs.append(prog)
                continue
            logger.info("%s non-callable item type=%s", name, type(prog))
        if programs:
            logger.info("Using %s with %d programs", name, len(programs))
            return programs
        logger.info("%s had no callable programs", name)
    logger.info("No candidate programs found in compiled/optimizer.")
    return []


def dedupe_prompts(prompts: list[str], limit: int) -> list[str]:
    seen = set()
    deduped = []
    for prompt in prompts:
        if prompt in seen:
            continue
        seen.add(prompt)
        deduped.append(prompt)
        if len(deduped) >= limit:
            break
    return deduped


def collect_prompts_from_programs(
    programs: list[Any],
    lm: Any,
    logger: Any,
    limit: int,
) -> list[str]:
    prompts = []
    for prog in programs:
        if len(prompts) >= limit:
            break
        prompt = run_program_for_prompt(prog, lm, logger)
        if prompt:
            prompts.append(prompt)
    return dedupe_prompts(prompts, limit)
