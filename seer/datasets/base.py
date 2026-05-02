from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from .schema import NormalizedQuestion


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


class DatasetAdapter(ABC):
    name: str
    aliases: tuple[str, ...] = ()

    @abstractmethod
    def normalize_raw(self, item: Mapping[str, Any], *, split: str | None = None) -> NormalizedQuestion:
        """Normalize one raw dataset record into the shared schema."""

    @abstractmethod
    def extract_gold_titles(self, item: Mapping[str, Any]) -> list[str]:
        """Extract gold titles from either a raw or normalized-style record."""

    def infer_num_hops(self, qid: str, item: Mapping[str, Any] | None = None) -> int:
        """Dataset-specific hop count. Default to 2-hop."""
        return 2

    def normalize_to_record(self, item: Mapping[str, Any], *, split: str | None = None) -> dict[str, Any]:
        return self.normalize_raw(item, split=split).to_record()
