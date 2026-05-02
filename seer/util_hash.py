from __future__ import annotations

import hashlib


def sha256_text(*parts: str) -> str:
    joined = "||".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()
