from __future__ import annotations

from seer.dpo_utils import iter_hop_records, selected_context_docs_upto_hop


def test_iter_hop_records_supports_canonical_hops_list():
    rollout = {
        "hops": [
            {"hop": 1, "candidates": []},
            {"hop": 2, "candidates": []},
            {"hop": 3, "candidates": []},
        ]
    }
    hops = iter_hop_records(rollout)
    assert [h for h, _ in hops] == [1, 2, 3]


def test_iter_hop_records_supports_legacy_hop_keys():
    rollout = {
        "hop1": {"candidates": []},
        "hop2": {"candidates": []},
    }
    hops = iter_hop_records(rollout)
    assert [h for h, _ in hops] == [1, 2]


def test_selected_context_docs_upto_hop_accumulates_chain():
    doc1 = {"doc_id": 1, "title": "T1", "text": "A"}
    doc2 = {"doc_id": 2, "title": "T2", "text": "B"}
    doc3 = {"doc_id": 3, "title": "T3", "text": "C"}
    rollout = {
        "hops": [
            {"hop": 1, "candidates": [{"retrieved": [doc1]}, {"retrieved": [doc2]}]},
            {
                "hop": 2,
                "context_source": {"selected_candidate_idx": 1},  # select hop1 cand1 => doc2
                "candidates": [{"retrieved": [doc3]}, {"retrieved": []}],
            },
            {
                "hop": 3,
                "context_source": {"selected_candidate_idx": 0},  # select hop2 cand0 => doc3
                "candidates": [{"retrieved": []}],
            },
        ]
    }
    hop2_ctx = selected_context_docs_upto_hop(rollout, 2)
    hop3_ctx = selected_context_docs_upto_hop(rollout, 3)
    assert [d["doc_id"] for d in hop2_ctx] == [2]
    assert [d["doc_id"] for d in hop3_ctx] == [2, 3]

