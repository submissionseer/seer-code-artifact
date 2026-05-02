from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "sft" / "prepare_data.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("sft_prepare_data", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _rollout(base_qid: str, variant_id: int, hop_cov: tuple[float, float], q1: str, q2: str, seer_xml: str = ""):
    return {
        "base_qid": base_qid,
        "prompt_variant_id": variant_id,
        "question": "Who did X?",
        "retrieved": [
            {"hop": 1, "title": "T1", "text": "A"},
            {"hop": 2, "title": "T2", "text": "B"},
        ],
        "seer_label": seer_xml,
        "hop_details": [
            {"hop": 1, "query": q1, "coverage_after_hop": hop_cov[0]},
            {"hop": 2, "query": q2, "coverage_after_hop": hop_cov[1]},
        ],
    }


def test_build_examples_gold_selects_best_variant_per_hop():
    mod = _load_module()
    grouped = {
        "q1": [
            _rollout("q1", 0, (0.8, 0.2), "q1v0h1", "q1v0h2"),
            _rollout("q1", 1, (0.1, 0.9), "q1v1h1", "q1v1h2"),
        ]
    }
    examples, stats = mod.build_examples(
        grouped,
        selector="gold",
        min_score=0.0,
        instruction="inst",
        selector_scores=None,
    )
    assert len(examples) == 2
    assert "q1v0h1" in examples[0]["output"]
    assert "q1v1h2" in examples[1]["output"]
    assert stats["hops_selected"] == 2


def test_build_examples_mmr_selects_by_scores():
    mod = _load_module()
    grouped = {
        "q2": [
            _rollout("q2", 0, (0.1, 0.1), "q2v0h1", "q2v0h2"),
            _rollout("q2", 1, (0.1, 0.1), "q2v1h1", "q2v1h2"),
        ]
    }
    mmr_scores = {
        ("q2", 0, 1): 0.3,
        ("q2", 1, 1): 0.7,
        ("q2", 0, 2): 0.9,
        ("q2", 1, 2): 0.1,
    }
    examples, _ = mod.build_examples(
        grouped,
        selector="mmr",
        min_score=-999.0,
        instruction="inst",
        selector_scores=mmr_scores,
    )
    assert len(examples) == 2
    assert "q2v1h1" in examples[0]["output"]
    assert "q2v0h2" in examples[1]["output"]


def test_build_examples_seer_selects_by_scores():
    mod = _load_module()
    grouped = {
        "qseer": [
            _rollout("qseer", 0, (0.1, 0.1), "qseerv0h1", "qseerv0h2"),
            _rollout("qseer", 1, (0.1, 0.1), "qseerv1h1", "qseerv1h2"),
        ]
    }
    seer_scores = {
        ("qseer", 0, 1): 0.2,
        ("qseer", 1, 1): 0.9,
        ("qseer", 0, 2): 0.8,
        ("qseer", 1, 2): 0.1,
    }
    examples, _ = mod.build_examples(
        grouped,
        selector="seer",
        min_score=0.0,
        instruction="inst",
        selector_scores=seer_scores,
    )
    assert len(examples) == 2
    assert "qseerv1h1" in examples[0]["output"]
    assert "qseerv0h2" in examples[1]["output"]


def test_load_selector_scores_supports_seer(tmp_path):
    mod = _load_module()
    scores_path = tmp_path / "seer_scores.jsonl"
    rows = [
        {"qid": "q1", "variant_id": 0, "hop": 1, "seer_ap": 0.25},
        {"qid": "q1", "variant_id": 1, "hop": 1, "seer_ap": 0.75},
    ]
    with scores_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    loaded = mod.load_selector_scores(scores_path, selector="seer")
    assert loaded[("q1", 0, 1)] == 0.25
    assert loaded[("q1", 1, 1)] == 0.75


def test_load_selector_scores_supports_decomp_fallback(tmp_path):
    mod = _load_module()
    scores_path = tmp_path / "decomp_scores.jsonl"
    rows = [
        {
            "qid": "q1",
            "variant_id": 0,
            "hop": 1,
            "decomp_binary_ap": 0.2,
            "decomposed_binary_ap": 0.9,
        },
        {
            "qid": "q1",
            "variant_id": 1,
            "hop": 1,
            "decomp_binary_ap": None,
            "decomposed_binary_ap": 0.8,
        },
    ]
    with scores_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    loaded = mod.load_selector_scores(scores_path, selector="decomp_binary")
    assert loaded[("q1", 0, 1)] == 0.2
    assert loaded[("q1", 1, 1)] == 0.8


def test_load_selector_scores_supports_raw_jina_variants(tmp_path):
    mod = _load_module()
    scores_path = tmp_path / "raw_scores.jsonl"
    rows = [
        {
            "qid": "q1",
            "variant_id": 0,
            "hop": 1,
            "raw_jina_maxmean_top3": 0.9,
            "raw_jina_max": 0.95,
        },
        {
            "qid": "q1",
            "variant_id": 1,
            "hop": 1,
            "raw_jina_maxmean": 0.7,
            "raw_jina_max": 0.8,
        },
    ]
    with scores_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    loaded_mean = mod.load_selector_scores(scores_path, selector="raw_jina_maxmean")
    loaded_max = mod.load_selector_scores(scores_path, selector="raw_jina_max")
    assert loaded_mean[("q1", 0, 1)] == 0.9
    assert loaded_mean[("q1", 1, 1)] == 0.7
    assert loaded_max[("q1", 0, 1)] == 0.95
    assert loaded_max[("q1", 1, 1)] == 0.8


def test_build_examples_decomp_binary_selects_by_hop_ap():
    mod = _load_module()
    grouped = {
        "q2": [
            {
                **_rollout("q2", 0, (0.1, 0.1), "q2v0h1", "q2v0h2"),
                "decomp_binary_ap_hop1": 0.2,
                "decomp_binary_ap_hop2": 0.9,
            },
            {
                **_rollout("q2", 1, (0.1, 0.1), "q2v1h1", "q2v1h2"),
                "decomp_binary_ap_hop1": 0.8,
                "decomp_binary_ap_hop2": 0.1,
            },
        ]
    }
    examples, _ = mod.build_examples(
        grouped,
        selector="decomp_binary",
        min_score=-999.0,
        instruction="inst",
        selector_scores=None,
    )
    assert len(examples) == 2
    assert "q2v1h1" in examples[0]["output"]
    assert "q2v0h2" in examples[1]["output"]


def test_build_examples_raw_jina_selects_by_scores():
    mod = _load_module()
    grouped = {
        "q2": [
            _rollout("q2", 0, (0.1, 0.1), "q2v0h1", "q2v0h2"),
            _rollout("q2", 1, (0.1, 0.1), "q2v1h1", "q2v1h2"),
        ]
    }
    raw_scores = {
        ("q2", 0, 1): 0.15,
        ("q2", 1, 1): 0.22,
        ("q2", 0, 2): 0.9,
        ("q2", 1, 2): 0.8,
    }
    examples, _ = mod.build_examples(
        grouped,
        selector="raw_jina_max",
        min_score=-999.0,
        instruction="inst",
        selector_scores=raw_scores,
    )
    assert len(examples) == 2
    assert "q2v1h1" in examples[0]["output"]
    assert "q2v0h2" in examples[1]["output"]


def test_seer_ap_for_hop_uses_cumulative_hops():
    mod = _load_module()
    seer_xml = """
<specific_information_required>
f1) req one
f2) req two
</specific_information_required>
<present_information>
f1->p1
f2->p2
</present_information>
<missing_information>
NONE
</missing_information>
""".strip()
    rollout = {
        "seer_label": seer_xml,
        "retrieved": [{"hop": 1}, {"hop": 2}],
    }
    hop1 = mod.seer_ap_for_hop(rollout, 0)
    hop2 = mod.seer_ap_for_hop(rollout, 1)
    assert hop1 == 0.5
    assert hop2 == 1.0


def test_build_examples_gold_skips_invalid_top_variant_and_uses_next_best():
    mod = _load_module()
    grouped = {
        "q3": [
            _rollout(
                "q3",
                0,
                (0.9, 0.9),
                "q3v0h1",
                "There is no information in the provided context about this entity.",
            ),
            _rollout("q3", 1, (0.8, 0.8), "q3v1h1", "q3v1h2"),
        ]
    }
    examples, stats = mod.build_examples(
        grouped,
        selector="gold",
        min_score=0.0,
        instruction="inst",
        selector_scores=None,
    )
    assert len(examples) == 2
    assert "q3v0h1" in examples[0]["output"]
    assert "q3v1h2" in examples[1]["output"]
    assert stats["variants_invalid_query"] >= 1


def test_normalize_rollout_query_rejects_missing_completed_marker():
    mod = _load_module()
    query, reason = mod.normalize_rollout_query(
        '[[ ## search_query ## ]]\n"hertfordshire east of england region creation year"\n'
    )
    assert query is None
    assert reason == "missing_completed_marker"


def test_normalize_rollout_query_rejects_short_no_such_context_phrase():
    mod = _load_module()
    query, reason = mod.normalize_rollout_query("No such date mentioned in the context")
    assert query is None
    assert reason == "narrative_like"


def test_build_jina_tasks_uses_dynamic_hop_count():
    mod = _load_module()
    grouped = {
        "qdyn": [
            {
                "base_qid": "qdyn",
                "prompt_variant_id": 0,
                "question": "Q dyn",
                "retrieved": [
                    {"hop": 1, "title": "A1", "text": "a"},
                    {"hop": 2, "title": "A2", "text": "b"},
                    {"hop": 3, "title": "A3", "text": "c"},
                ],
                "hop_details": [
                    {"hop": 1, "query": "h1", "coverage_after_hop": 0.0},
                    {"hop": 2, "query": "h2", "coverage_after_hop": 0.0},
                    {"hop": 3, "query": "h3", "coverage_after_hop": 0.0},
                ],
            },
            {
                "base_qid": "qdyn",
                "prompt_variant_id": 1,
                "question": "Q dyn",
                "retrieved": [
                    {"hop": 1, "title": "B1", "text": "x"},
                    {"hop": 2, "title": "B2", "text": "y"},
                    {"hop": 3, "title": "B3", "text": "z"},
                ],
                "hop_details": [
                    {"hop": 1, "query": "h1b", "coverage_after_hop": 0.0},
                    {"hop": 2, "query": "h2b", "coverage_after_hop": 0.0},
                    {"hop": 3, "query": "h3b", "coverage_after_hop": 0.0},
                ],
            },
        ]
    }
    requirements_by_question = {"Q dyn": ["req1"]}

    tasks, _passage_maps, variant_passage_map = mod._build_jina_tasks(grouped, requirements_by_question)
    hop_set = {t["hop"] for t in tasks}
    assert hop_set == {1, 2, 3}
    assert len(tasks) == 3
    assert ("qdyn", 0, 3) in variant_passage_map
    assert ("qdyn", 1, 3) in variant_passage_map
