from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t


def _two_sample_t_pvalue(y: np.ndarray, z: np.ndarray) -> float:
    treated = y[z == 1]
    control = y[z == 0]
    n1 = treated.shape[0]
    n0 = control.shape[0]
    s1 = float(np.var(treated, ddof=1))
    s0 = float(np.var(control, ddof=1))
    se = float(np.sqrt(s1 / n1 + s0 / n0))
    diff = float(np.mean(treated) - np.mean(control))
    if se == 0.0:
        return 1.0
    t_stat = diff / se
    df_num = (s1 / n1 + s0 / n0) ** 2
    df_den = ((s1 / n1) ** 2) / (n1 - 1) + ((s0 / n0) ** 2) / (n0 - 1)
    if df_den == 0.0:
        return 1.0
    df = df_num / df_den
    return float(2.0 * t.sf(np.abs(t_stat), df=df))


def simulate_null_pvalues(config: dict[str, Any]) -> pd.DataFrame:
    """
    Generate p-values under the complete null for L simulations.
    Return columns: sim_id, hypothesis_id, p_value.
    """
    rng = np.random.default_rng(int(config["seed_null"]))
    n = int(config["N"])
    m = int(config["M"])
    l = int(config["L"])
    p_treat = float(config["p_treat"])

    rows: list[dict[str, float | int]] = []
    for sim_id in range(l):
        z = (rng.random(n) < p_treat).astype(int)
        for hypothesis_id in range(m):
            y = rng.normal(loc=0.0, scale=1.0, size=n)
            p_value = _two_sample_t_pvalue(y=y, z=z)
            rows.append(
                {
                    "sim_id": sim_id,
                    "hypothesis_id": hypothesis_id,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def simulate_mixed_pvalues(config: dict[str, Any]) -> pd.DataFrame:
    """
    Generate p-values under mixed true and false null hypotheses for L simulations.
    Return columns: sim_id, hypothesis_id, p_value, is_true_null.
    """
    rng = np.random.default_rng(int(config["seed_mixed"]))
    n = int(config["N"])
    m = int(config["M"])
    m0 = int(config["M0"])
    l = int(config["L"])
    p_treat = float(config["p_treat"])
    tau_alt = float(config["tau_alternative"])

    rows: list[dict[str, float | int | bool]] = []
    for sim_id in range(l):
        z = (rng.random(n) < p_treat).astype(int)
        for hypothesis_id in range(m):
            is_true_null = hypothesis_id >= (m - m0)
            effect = 0.0 if is_true_null else tau_alt
            y = rng.normal(loc=0.0, scale=1.0, size=n) + effect * z
            p_value = _two_sample_t_pvalue(y=y, z=z)
            rows.append(
                {
                    "sim_id": sim_id,
                    "hypothesis_id": hypothesis_id,
                    "p_value": p_value,
                    "is_true_null": is_true_null,
                }
            )
    return pd.DataFrame(rows)


def bonferroni_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Bonferroni correction.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = p_values.shape[0]
    if m == 0:
        return np.array([], dtype=bool)
    threshold = alpha / m
    return p_values <= threshold


def holm_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Holm step-down correction.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = p_values.shape[0]
    if m == 0:
        return np.array([], dtype=bool)
    order = np.argsort(p_values, kind="mergesort")
    sorted_p = p_values[order]
    rejected_sorted_idx: list[int] = []
    for j in range(m):
        # Rank (j+1) compared to alpha / (M - j)
        if sorted_p[j] <= alpha / (m - j):
            rejected_sorted_idx.append(j)
        else:
            break
    out = np.zeros(m, dtype=bool)
    if rejected_sorted_idx:
        out[order[np.array(rejected_sorted_idx, dtype=int)]] = True
    return out


def benjamini_hochberg_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Benjamini-Hochberg correction.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = p_values.shape[0]
    if m == 0:
        return np.array([], dtype=bool)
    order = np.argsort(p_values, kind="mergesort")
    sorted_p = p_values[order]
    k_star = 0
    for k in range(m, 0, -1):
        if sorted_p[k - 1] <= (k / m) * alpha:
            k_star = k
            break
    out = np.zeros(m, dtype=bool)
    if k_star > 0:
        out[order[:k_star]] = True
    return out


def benjamini_yekutieli_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Benjamini-Yekutieli correction.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = p_values.shape[0]
    if m == 0:
        return np.array([], dtype=bool)
    harmonic_sum = float(np.sum(1.0 / np.arange(1, m + 1, dtype=float)))
    order = np.argsort(p_values, kind="mergesort")
    sorted_p = p_values[order]
    k_star = 0
    for k in range(m, 0, -1):
        threshold = (k / m) * alpha / harmonic_sum
        if sorted_p[k - 1] <= threshold:
            k_star = k
            break
    out = np.zeros(m, dtype=bool)
    if k_star > 0:
        out[order[:k_star]] = True
    return out


def compute_fwer(rejections_null: np.ndarray) -> float:
    """
    Return family-wise error rate from a [L, M] rejection matrix under the complete null.
    """
    rejections_null = np.asarray(rejections_null, dtype=bool)
    if rejections_null.size == 0:
        return 0.0
    return float(np.mean(np.any(rejections_null, axis=1)))


def compute_fdr(rejections: np.ndarray, is_true_null: np.ndarray) -> float:
    """
    Return FDR for one simulation: false discoveries among all discoveries.
    Use 0.0 when there are no rejections.
    """
    rejections = np.asarray(rejections, dtype=bool)
    is_true_null = np.asarray(is_true_null, dtype=bool)
    n_discoveries = int(np.sum(rejections))
    if n_discoveries == 0:
        return 0.0
    false_discoveries = int(np.sum(rejections & is_true_null))
    return float(false_discoveries / n_discoveries)


def compute_power(rejections: np.ndarray, is_true_null: np.ndarray) -> float:
    """
    Return power for one simulation: true rejections among false null hypotheses.
    """
    rejections = np.asarray(rejections, dtype=bool)
    is_true_null = np.asarray(is_true_null, dtype=bool)
    false_null = ~is_true_null
    n_false_null = int(np.sum(false_null))
    if n_false_null == 0:
        return 0.0
    true_rejections = int(np.sum(rejections & false_null))
    return float(true_rejections / n_false_null)


def summarize_multiple_testing(
    null_pvalues: pd.DataFrame,
    mixed_pvalues: pd.DataFrame,
    alpha: float,
) -> dict[str, float]:
    """
    Return summary metrics:
      fwer_uncorrected, fwer_bonferroni, fwer_holm,
      fdr_uncorrected, fdr_bh, fdr_by,
      power_uncorrected, power_bh, power_by.
    """
    null_sims = sorted(null_pvalues["sim_id"].unique())
    unc_rows: list[np.ndarray] = []
    bonf_rows: list[np.ndarray] = []
    holm_rows: list[np.ndarray] = []
    for sid in null_sims:
        g = null_pvalues[null_pvalues["sim_id"] == sid].sort_values("hypothesis_id")
        p = g["p_value"].to_numpy(dtype=float)
        unc_rows.append(p <= alpha)
        bonf_rows.append(bonferroni_rejections(p, alpha))
        holm_rows.append(holm_rejections(p, alpha))
    rej_unc = np.stack(unc_rows, axis=0)
    rej_bonf = np.stack(bonf_rows, axis=0)
    rej_holm = np.stack(holm_rows, axis=0)

    mixed_sims = sorted(mixed_pvalues["sim_id"].unique())
    fdr_u: list[float] = []
    fdr_bh_l: list[float] = []
    fdr_by_l: list[float] = []
    pow_u: list[float] = []
    pow_bh_l: list[float] = []
    pow_by_l: list[float] = []
    for sid in mixed_sims:
        g = mixed_pvalues[mixed_pvalues["sim_id"] == sid].sort_values("hypothesis_id")
        p = g["p_value"].to_numpy(dtype=float)
        is_null = g["is_true_null"].astype(bool).to_numpy()
        fdr_u.append(compute_fdr(p <= alpha, is_null))
        bh = benjamini_hochberg_rejections(p, alpha)
        by = benjamini_yekutieli_rejections(p, alpha)
        fdr_bh_l.append(compute_fdr(bh, is_null))
        fdr_by_l.append(compute_fdr(by, is_null))
        pow_u.append(compute_power(p <= alpha, is_null))
        pow_bh_l.append(compute_power(bh, is_null))
        pow_by_l.append(compute_power(by, is_null))

    n_mixed = len(mixed_sims)
    if n_mixed == 0:
        avg_fdr_u = avg_fdr_bh = avg_fdr_by = 0.0
        avg_pow_u = avg_pow_bh = avg_pow_by = 0.0
    else:
        avg_fdr_u = float(np.mean(fdr_u))
        avg_fdr_bh = float(np.mean(fdr_bh_l))
        avg_fdr_by = float(np.mean(fdr_by_l))
        avg_pow_u = float(np.mean(pow_u))
        avg_pow_bh = float(np.mean(pow_bh_l))
        avg_pow_by = float(np.mean(pow_by_l))

    return {
        "fwer_uncorrected": compute_fwer(rej_unc),
        "fwer_bonferroni": compute_fwer(rej_bonf),
        "fwer_holm": compute_fwer(rej_holm),
        "fdr_uncorrected": avg_fdr_u,
        "fdr_bh": avg_fdr_bh,
        "fdr_by": avg_fdr_by,
        "power_uncorrected": avg_pow_u,
        "power_bh": avg_pow_bh,
        "power_by": avg_pow_by,
    }
