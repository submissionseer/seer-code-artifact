from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from .base import DatasetAdapter, dedupe_preserve_order
from .schema import CandidatePassage, NormalizedQuestion


class HotpotAdapter(DatasetAdapter):
    name = "hotpot"
    aliases = ("hotpot_fullwiki", "hotpotqa")

    def normalize_raw(self, item: Mapping[str, Any], *, split: str | None = None) -> NormalizedQuestion:
        supporting_fact_ids_by_title = self._supporting_fact_ids_by_title(item)
        supporting_titles = set(supporting_fact_ids_by_title.keys())

        candidates = []
        context = item.get("context", {})
        if isinstance(context, Mapping):
            titles = context.get("title", []) or []
            sentences_list = context.get("sentences", []) or []
            for idx, (title, sentences) in enumerate(zip(titles, sentences_list)):
                text = " ".join(sentences) if isinstance(sentences, list) else str(sentences or "")
                title_s = str(title or "")
                candidates.append(
                    CandidatePassage(
                        cand_id=f"c{idx}",
                        title=title_s,
                        text=text,
                        is_supporting=title_s in supporting_titles,
                        supporting_fact_ids=supporting_fact_ids_by_title.get(title_s, []),
                    )
                )
        else:
            for idx, pair in enumerate(context or []):
                if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                    continue
                title = str(pair[0] or "")
                sentences = pair[1]
                text = " ".join(sentences) if isinstance(sentences, list) else str(sentences or "")
                candidates.append(
                    CandidatePassage(
                        cand_id=f"c{idx}",
                        title=title,
                        text=text,
                        is_supporting=title in supporting_titles,
                        supporting_fact_ids=supporting_fact_ids_by_title.get(title, []),
                    )
                )

        qid = str(item.get("_id", item.get("id", item.get("qid", ""))))
        return NormalizedQuestion(
            dataset="hotpot",
            split=str(split or item.get("split", "unknown")),
            qid=qid,
            question=str(item.get("question", "")),
            answer=str(item.get("answer", "")),
            is_answerable=True,
            candidates=candidates,
            metadata={"gold_titles": self.extract_gold_titles(item)},
        )

    def extract_gold_titles(self, item: Mapping[str, Any]) -> list[str]:
        if "gold_titles" in item and isinstance(item.get("gold_titles"), list):
            return dedupe_preserve_order([str(t) for t in item.get("gold_titles", [])])

        candidates = item.get("candidates")
        if isinstance(candidates, list):
            titles = [str(c.get("title", "")) for c in candidates if c.get("is_supporting", False)]
            if titles:
                return dedupe_preserve_order(titles)

        supporting_facts = item.get("supporting_facts", [])
        if isinstance(supporting_facts, Mapping):
            titles = [str(t) for t in (supporting_facts.get("title", []) or [])]
            return dedupe_preserve_order(titles)

        titles = []
        for sf in supporting_facts or []:
            if isinstance(sf, (list, tuple)):
                if sf:
                    titles.append(str(sf[0]))
            elif isinstance(sf, Mapping):
                titles.append(str(sf.get("title", "")))
        return dedupe_preserve_order(titles)

    def infer_num_hops(self, qid: str, item: Mapping[str, Any] | None = None) -> int:
        return 2

    def _supporting_fact_ids_by_title(self, item: Mapping[str, Any]) -> dict[str, list[str]]:
        supporting_facts = item.get("supporting_facts", [])
        by_title: dict[str, list[str]] = defaultdict(list)

        if isinstance(supporting_facts, Mapping):
            titles = supporting_facts.get("title", []) or []
            sent_ids = supporting_facts.get("sent_id", []) or supporting_facts.get("sentence_id", []) or []
            for idx, title in enumerate(titles):
                t = str(title or "")
                if not t:
                    continue
                sent_val = sent_ids[idx] if idx < len(sent_ids) else None
                if sent_val is not None:
                    by_title[t].append(str(sent_val))
                elif t not in by_title:
                    by_title[t] = []
            return dict(by_title)

        for sf in supporting_facts or []:
            if isinstance(sf, (list, tuple)):
                if not sf:
                    continue
                title = str(sf[0] or "")
                if not title:
                    continue
                if len(sf) > 1 and sf[1] is not None:
                    by_title[title].append(str(sf[1]))
                elif title not in by_title:
                    by_title[title] = []
            elif isinstance(sf, Mapping):
                title = str(sf.get("title", "") or "")
                if not title:
                    continue
                sent_id = sf.get("sent_id", sf.get("sentence_id"))
                if sent_id is not None:
                    by_title[title].append(str(sent_id))
                elif title not in by_title:
                    by_title[title] = []
        return dict(by_title)
