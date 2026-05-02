from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "dpo" / "validate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("dpo_validate", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_metric_score_decomp_binary_uses_legacy_field_when_primary_none():
    mod = _load_module()
    candidate = {
        "decomp_binary_ap": None,
        "decomposed_binary_ap": 0.62,
    }
    assert mod._metric_score(candidate, "decomp_binary", "decomp_binary_ap") == 0.62


def test_metric_score_non_decomp_uses_declared_key():
    mod = _load_module()
    candidate = {"mmr_ap": 0.51}
    assert mod._metric_score(candidate, "mmr", "mmr_ap") == 0.51
