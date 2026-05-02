from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def plot_bar(values: dict[str, float], output_path: str | Path, title: str) -> None:
    labels = list(values.keys())
    scores = list(values.values())
    plt.figure(figsize=(8, 4))
    plt.bar(labels, scores)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
