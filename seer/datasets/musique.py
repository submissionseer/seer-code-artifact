from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .base import DatasetAdapter, dedupe_preserve_order
from .schema import CandidatePassage, NormalizedQuestion


_HOPS_RE = re.compile(r"([234])hop", re.IGNORECASE)


class MuSiQueAdapter(DatasetAdapter):
    name = "musique"
    aliases = ("musique_qa",)

    def normalize_raw(self, item: Mapping[str, Any], *, split: str | None = None) -> NormalizedQuestion:
        paragraphs = item.get("paragraphs") or item.get("paragraphs_info") or []
        candidates = []
        for idx, para in enumerate(paragraphs):
            if not isinstance(para, Mapping):
                continue
            candidates.append(
                CandidatePassage(
                    cand_id=f"c{idx}",
                    title=str(para.get("title", "")),
                    text=str(para.get("paragraph_text", para.get("text", ""))),
                    is_supporting=bool(para.get("is_supporting", False)),
                    supporting_fact_ids=[],
                )
            )

        qid = str(item.get("id", item.get("qid", "")))
        metadata: dict[str, Any] = {
            "gold_titles": self.extract_gold_titles(item),
            "num_hops": self.infer_num_hops(qid, item),
        }
        question_decomposition = item.get("question_decomposition")
        if isinstance(question_decomposition, list):
            metadata["question_decomposition"] = question_decomposition
        decomposition = item.get("decomposition")
        if isinstance(decomposition, list):
            metadata["decomposition"] = decomposition

        return NormalizedQuestion(
            dataset="musique",
            split=str(split or item.get("split", "unknown")),
            qid=qid,
            question=str(item.get("question", "")),
            answer=str(item.get("answer", "")),
            is_answerable=bool(item.get("answerable", True)),
            candidates=candidates,
            metadata=metadata,
        )

    def extract_gold_titles(self, item: Mapping[str, Any]) -> list[str]:
        if "gold_titles" in item and isinstance(item.get("gold_titles"), list):
            return dedupe_preserve_order([str(t) for t in item.get("gold_titles", [])])

        candidates = item.get("candidates")
        if isinstance(candidates, list):
            titles = [str(c.get("title", "")) for c in candidates if c.get("is_supporting", False)]
            if titles:
                return dedupe_preserve_order(titles)

        paragraphs = item.get("paragraphs") or item.get("paragraphs_info") or []
        titles = []
        for para in paragraphs:
            if isinstance(para, Mapping) and para.get("is_supporting", False):
                titles.append(str(para.get("title", "")))
        return dedupe_preserve_order(titles)

    def infer_num_hops(self, qid: str, item: Mapping[str, Any] | None = None) -> int:
        if item and item.get("num_hops") is not None:
            try:
                return int(item["num_hops"])
            except (TypeError, ValueError):
                pass

        if item:
            for key in ("question_decomposition", "decomposition"):
                value = item.get(key)
                if isinstance(value, list) and value:
                    return len(value)

        match = _HOPS_RE.search(str(qid or ""))
        if match:
            return int(match.group(1))
        return 2
