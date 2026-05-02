"""DSPy optimization metrics for Seer prompt optimization.

Metrics that compare Seer's predicted AP to Gold AP, used with
DSPy MIPROv2 and GEPA optimizers.
"""

from __future__ import annotations

import dspy

from seer.parse_seer_xml import parse_seer_xml
from seer.seer_metrics import compute_seer_metrics


def seer_gold_correlation_metric(example, prediction, trace=None):
    """Metric: how close is Seer's predicted score to Gold AP?

    Returns: 1.0 - |seer_ap_predicted - gold_ap|
    Higher is better. Penalizes false positives and false negatives equally.

    For bootstrapping (trace is not None): returns True if error < 0.15.
    """
    gold_ap = float(example.gold_ap)
    seer_ap = _extract_seer_ap(prediction, example)

    error = abs(seer_ap - gold_ap)

    if trace is not None:
        return error < 0.15

    return 1.0 - error


def seer_gold_feedback_metric(gold, pred, trace=None):
    """GEPA-compatible metric with natural language feedback.

    Returns a dspy.Prediction with score and feedback fields for GEPA
    to use when mutating instructions.
    """
    gold_ap = float(gold.gold_ap)
    seer_ap = _extract_seer_ap(pred, gold)
    score = 1.0 - abs(seer_ap - gold_ap)

    if seer_ap > gold_ap + 0.3:
        feedback = (
            f"False positive: Seer scored {seer_ap:.2f} but Gold is {gold_ap:.2f}. "
            f"The model likely marked a requirement as satisfied when the passage only "
            f"mentions the entity tangentially without providing the specific fact needed."
        )
    elif seer_ap < gold_ap - 0.3:
        feedback = (
            f"False negative: Seer scored {seer_ap:.2f} but Gold is {gold_ap:.2f}. "
            f"The model missed that the passage does contain the required information."
        )
    elif abs(seer_ap - gold_ap) > 0.15:
        direction = "overestimated" if seer_ap > gold_ap else "underestimated"
        feedback = (
            f"Moderate error: Seer={seer_ap:.2f}, Gold={gold_ap:.2f}. "
            f"The model {direction} retrieval quality."
        )
    else:
        feedback = (
            f"Close: Seer={seer_ap:.2f}, Gold={gold_ap:.2f}. Good calibration."
        )

    return dspy.Prediction(score=score, feedback=feedback)


def _extract_seer_ap(prediction, example) -> float:
    """Extract seer_ap from a prediction, handling both XML and structured formats."""
    # If the prediction already has seer_ap computed (from SeerXmlModule)
    seer_ap = getattr(prediction, "seer_ap", None)
    if seer_ap is not None:
        return float(seer_ap)

    # Try to get it from raw_xml (SeerXmlModule output)
    raw_xml = getattr(prediction, "raw_xml", None)
    if raw_xml:
        return _parse_ap_from_xml(raw_xml, example)

    # Try xml_response field (from SeerXmlPrompt signature)
    xml_response = getattr(prediction, "xml_response", None)
    if xml_response:
        return _parse_ap_from_xml(xml_response, example)

    # Try recall field (structured module)
    recall = getattr(prediction, "recall", None)
    if recall is not None:
        return float(recall)

    return 0.0


def _parse_ap_from_xml(xml_text: str, example) -> float:
    """Parse XML text and compute seer_ap."""
    try:
        parsed = parse_seer_xml(xml_text)
        n_passages = int(getattr(example, "n_passages", 3))
        metrics = compute_seer_metrics(parsed, num_passages=n_passages)
        return metrics.get("seer_ap", 0.0)
    except Exception:
        return 0.0
