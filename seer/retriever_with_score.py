from __future__ import annotations

from typing import Iterable

import dspy
from dspy.predict.parameter import Parameter
from dspy.primitives.prediction import Prediction


def _retrieve(query: str, k: int, **kwargs) -> list:
    if not dspy.settings.rm:
        raise AssertionError("No RM is loaded.")
    passages = dspy.settings.rm(query, k=k, **kwargs)
    if isinstance(passages, Iterable):
        return list(passages)
    return [passages]


class RetrieveWithScore(Parameter):
    name = "Search"
    input_variable = "query"
    desc = "takes a search query and returns one or more passages"

    def __init__(self, k: int = 3):
        self.k = k

    def reset(self) -> None:
        pass

    def dump_state(self, json_mode: bool | None = None):
        _ = json_mode
        return {"k": self.k}

    def load_state(self, state):
        for name, value in state.items():
            setattr(self, name, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, query: str, k: int | None = None, **kwargs) -> Prediction:
        k = k if k is not None else self.k
        passages = _retrieve(query, k, **kwargs)
        return Prediction(passages=passages)
