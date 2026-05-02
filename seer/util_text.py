from __future__ import annotations

import re


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_NON_WORD = re.compile(r"[^\w\s]")


def normalize_text(text: str) -> str:
    text = text.lower()
    text = _NON_WORD.sub(" ", text)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def safe_format_prompt(template: str, **kwargs: str) -> str:
    if "PROMPT" not in kwargs and "question" in kwargs:
        kwargs["PROMPT"] = kwargs["question"]
    try:
        return template.format_map(_SafeFormatDict(**kwargs))
    except Exception:
        return template
