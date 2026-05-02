from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score


def bootstrap_ci(
    values: Iterable[float],
    num_samples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Compute bootstrap confidence interval for the mean.

    Args:
        values: Iterable of values
        num_samples: Number of bootstrap samples
        alpha: Significance level (default 0.05 for 95% CI)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (lower, upper) bounds
    """
    values = np.array(list(values))
    if values.size == 0:
        return 0.0, 0.0
    means = []
    rng = np.random.default_rng(seed)
    for _ in range(num_samples):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, 100 * alpha / 2)
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    num_samples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for AUROC.

    Args:
        y_true: Binary ground truth labels
        y_score: Predicted scores/probabilities
        num_samples: Number of bootstrap samples
        alpha: Significance level
        seed: Random seed

    Returns:
        Tuple of (auroc, lower_ci, upper_ci)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5, 0.5, 0.5

    auroc = roc_auc_score(y_true, y_score)

    rng = np.random.default_rng(seed)
    aurocs = []
    for _ in range(num_samples):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        y_true_boot = y_true[idx]
        y_score_boot = y_score[idx]
        # Skip if bootstrap sample has only one class
        if len(np.unique(y_true_boot)) < 2:
            continue
        aurocs.append(roc_auc_score(y_true_boot, y_score_boot))

    if not aurocs:
        return auroc, auroc, auroc

    lower = np.percentile(aurocs, 100 * alpha / 2)
    upper = np.percentile(aurocs, 100 * (1 - alpha / 2))
    return float(auroc), float(lower), float(upper)


def delong_test(
    y_true: np.ndarray,
    y_score1: np.ndarray,
    y_score2: np.ndarray,
) -> tuple[float, float, float]:
    """
    DeLong test for comparing two AUROC values.

    Implementation based on:
    DeLong et al. (1988) "Comparing the Areas under Two or More Correlated
    Receiver Operating Characteristic Curves: A Nonparametric Approach"

    Args:
        y_true: Binary ground truth labels (0/1)
        y_score1: Predicted scores from model 1
        y_score2: Predicted scores from model 2

    Returns:
        Tuple of (z_statistic, p_value, auroc_diff)
    """
    y_true = np.asarray(y_true).astype(int)
    y_score1 = np.asarray(y_score1)
    y_score2 = np.asarray(y_score2)

    # Separate positive and negative samples
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    m = len(pos_idx)  # Number of positives
    n = len(neg_idx)  # Number of negatives

    if m == 0 or n == 0:
        return 0.0, 1.0, 0.0

    # Compute AUROCs
    auc1 = roc_auc_score(y_true, y_score1)
    auc2 = roc_auc_score(y_true, y_score2)

    # Compute placement values (structural components)
    # V10: for each positive, fraction of negatives with lower score
    # V01: for each negative, fraction of positives with higher score

    def compute_placements(y_score):
        pos_scores = y_score[pos_idx]
        neg_scores = y_score[neg_idx]

        # V10[i] = (1/n) * sum_j I(X_j < Y_i) for positive i
        v10 = np.zeros(m)
        for i, ps in enumerate(pos_scores):
            v10[i] = np.mean(neg_scores < ps) + 0.5 * np.mean(neg_scores == ps)

        # V01[j] = (1/m) * sum_i I(Y_i < X_j) for negative j
        v01 = np.zeros(n)
        for j, ns in enumerate(neg_scores):
            v01[j] = np.mean(pos_scores > ns) + 0.5 * np.mean(pos_scores == ns)

        return v10, v01

    v10_1, v01_1 = compute_placements(y_score1)
    v10_2, v01_2 = compute_placements(y_score2)

    # Compute covariance matrix
    # S10 = Cov(V10_1, V10_2) for positives
    # S01 = Cov(V01_1, V01_2) for negatives

    s10 = np.cov(v10_1, v10_2)
    s01 = np.cov(v01_1, v01_2)

    # Variance of the difference
    # Var(AUC1 - AUC2) = (1/m)*(s10[0,0] + s10[1,1] - 2*s10[0,1]) +
    #                   (1/n)*(s01[0,0] + s01[1,1] - 2*s01[0,1])
    var_diff = (
        (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m +
        (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    )

    if var_diff <= 0:
        return 0.0, 1.0, auc1 - auc2

    # Z statistic
    z = (auc1 - auc2) / np.sqrt(var_diff)

    # Two-tailed p-value
    p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z)))

    return float(z), float(p_value), float(auc1 - auc2)


def holm_bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """
    Apply Holm-Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of p-values from multiple tests
        alpha: Family-wise error rate

    Returns:
        List of booleans indicating which hypotheses to reject
    """
    n = len(p_values)
    if n == 0:
        return []

    # Sort p-values and keep track of original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    # Apply Holm correction
    reject = [False] * n
    for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
        threshold = alpha / (n - i)
        if p <= threshold:
            reject[idx] = True
        else:
            # Once we fail to reject, stop
            break

    return reject


def compute_all_auroc_comparisons(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    primary_metric: str | None = None,
) -> dict[str, dict]:
    """
    Compute AUROC with CIs for all metrics and pairwise DeLong tests.

    Args:
        y_true: Binary ground truth
        predictions: Dict mapping metric names to predicted scores
        primary_metric: If set, this is the pre-registered primary comparison

    Returns:
        Dict with 'aurocs' (per-metric results) and 'comparisons' (pairwise tests)
    """
    results = {"aurocs": {}, "comparisons": [], "primary_comparison": None}

    # Compute AUROC with CI for each metric
    for name, scores in predictions.items():
        auroc, lower, upper = bootstrap_auroc_ci(y_true, scores)
        results["aurocs"][name] = {
            "auroc": auroc,
            "ci_lower": lower,
            "ci_upper": upper,
        }

    # Pairwise DeLong tests
    names = list(predictions.keys())
    p_values = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            z, p, diff = delong_test(y_true, predictions[names[i]], predictions[names[j]])
            comparison = {
                "metric1": names[i],
                "metric2": names[j],
                "z_statistic": z,
                "p_value": p,
                "auroc_diff": diff,
            }
            results["comparisons"].append(comparison)
            p_values.append(p)

            # Check if this is the primary comparison
            if primary_metric and (names[i] == primary_metric or names[j] == primary_metric):
                if results["primary_comparison"] is None:
                    results["primary_comparison"] = comparison

    # Apply Holm correction to secondary comparisons
    if p_values:
        reject = holm_bonferroni_correction(p_values)
        for i, comp in enumerate(results["comparisons"]):
            comp["reject_h0_holm"] = reject[i]

    return results
