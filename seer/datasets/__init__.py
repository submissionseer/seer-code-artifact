from .base import DatasetAdapter
from .hotpot import HotpotAdapter
from .musique import MuSiQueAdapter
from .schema import CandidatePassage, NormalizedQuestion

_ADAPTERS = (
    HotpotAdapter(),
    MuSiQueAdapter(),
)


def get_dataset_adapter(name: str) -> DatasetAdapter:
    normalized = name.strip().lower().replace("-", "_")
    for adapter in _ADAPTERS:
        if normalized == adapter.name or normalized in adapter.aliases:
            return adapter
    raise KeyError(f"Unknown dataset adapter: {name}")


def list_dataset_adapters() -> list[str]:
    return [adapter.name for adapter in _ADAPTERS]


__all__ = [
    "CandidatePassage",
    "DatasetAdapter",
    "HotpotAdapter",
    "MuSiQueAdapter",
    "NormalizedQuestion",
    "get_dataset_adapter",
    "list_dataset_adapters",
]
