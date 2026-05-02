from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any


@dataclass(slots=True)
class CandidatePassage:
    cand_id: str
    title: str
    text: str
    is_supporting: bool = False
    supporting_fact_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "CandidatePassage":
        return cls(
            cand_id=str(record.get("cand_id", "")),
            title=str(record.get("title", "")),
            text=str(record.get("text", "")),
            is_supporting=bool(record.get("is_supporting", False)),
            supporting_fact_ids=[str(x) for x in (record.get("supporting_fact_ids") or [])],
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "cand_id": self.cand_id,
            "title": self.title,
            "text": self.text,
            "is_supporting": bool(self.is_supporting),
            "supporting_fact_ids": list(self.supporting_fact_ids),
        }


@dataclass(slots=True)
class NormalizedQuestion:
    dataset: str
    split: str
    qid: str
    question: str
    answer: str
    is_answerable: bool
    candidates: list[CandidatePassage] = field(default_factory=list)
    relevant_doc_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "NormalizedQuestion":
        known_keys = {
            "dataset",
            "split",
            "qid",
            "question",
            "answer",
            "is_answerable",
            "candidates",
            "relevant_doc_ids",
        }
        metadata = {
            str(k): v
            for k, v in record.items()
            if k not in known_keys
        }
        return cls(
            dataset=str(record.get("dataset", "")),
            split=str(record.get("split", "unknown")),
            qid=str(record.get("qid", "")),
            question=str(record.get("question", "")),
            answer=str(record.get("answer", "")),
            is_answerable=bool(record.get("is_answerable", True)),
            candidates=[
                CandidatePassage.from_record(c)
                for c in (record.get("candidates") or [])
            ],
            relevant_doc_ids=[str(x) for x in (record.get("relevant_doc_ids") or [])],
            metadata=metadata,
        )

    def gold_titles(self) -> list[str]:
        if "gold_titles" in self.metadata and isinstance(self.metadata["gold_titles"], list):
            seen: set[str] = set()
            out: list[str] = []
            for title in self.metadata["gold_titles"]:
                t = str(title or "").strip()
                if t and t not in seen:
                    seen.add(t)
                    out.append(t)
            return out

        seen = set()
        out = []
        for cand in self.candidates:
            if not cand.is_supporting:
                continue
            title = cand.title.strip()
            if title and title not in seen:
                seen.add(title)
                out.append(title)
        return out

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "dataset": self.dataset,
            "split": self.split,
            "qid": self.qid,
            "question": self.question,
            "answer": self.answer,
            "is_answerable": bool(self.is_answerable),
            "candidates": [cand.to_record() for cand in self.candidates],
        }
        if self.relevant_doc_ids:
            record["relevant_doc_ids"] = list(self.relevant_doc_ids)
        record.update(self.metadata)
        return record
