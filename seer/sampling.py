from __future__ import annotations

import random
from typing import Sequence


def stratified_sample(items: Sequence[dict], key: str, counts: dict[str, int]) -> list[dict]:
    buckets: dict[str, list[dict]] = {}
    for item in items:
        buckets.setdefault(item.get(key, "unknown"), []).append(item)
    sampled = []
    for bucket, count in counts.items():
        sampled.extend(random.sample(buckets.get(bucket, []), k=min(count, len(buckets.get(bucket, [])))))
    return sampled
