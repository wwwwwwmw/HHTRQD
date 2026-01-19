from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import BENEFIT_CRITERIA, COST_CRITERIA


def _safe_divide(numer: pd.Series | float, denom: pd.Series | float) -> pd.Series | float:
    return numer / (denom + 1e-9)


def parse_mpg(value) -> float:
    if pd.isna(value):
        return 0.0
    try:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "unknown"}:
            return 0.0
        if "-" in text:
            low, high = text.split("-", 1)
            return (float(low) + float(high)) / 2.0
        return float(text)
    except Exception:
        return 0.0


def default_ahp_matrix(criteria: list[str]) -> np.ndarray:
    A = np.array(
        [
            [1, 3, 5, 5, 7, 3, 3, 5, 3],
            [1 / 3, 1, 3, 3, 5, 3, 3, 3, 3],
            [1 / 5, 1 / 3, 1, 3, 3, 3, 3, 3, 3],
            [1 / 5, 1 / 3, 1 / 3, 1, 3, 3, 3, 3, 3],
            [1 / 7, 1 / 5, 1 / 3, 1 / 3, 1, 1, 1, 3, 1],
            [1 / 3, 1 / 3, 1 / 3, 1 / 3, 1, 1, 3, 3, 3],
            [1 / 3, 1 / 3, 1 / 3, 1 / 3, 1, 1 / 3, 1, 3, 3],
            [1 / 5, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1, 1],
            [1 / 3, 1 / 3, 1 / 3, 1 / 3, 1, 1 / 3, 1 / 3, 1, 1],
        ],
        dtype=float,
    )
    if A.shape != (len(criteria), len(criteria)):
        raise ValueError("Ma trận AHP không khớp số tiêu chí")
    return A


def compute_ahp_weights(A: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    col_sum = A.sum(axis=0)
    A_norm = A / col_sum
    weights = A_norm.mean(axis=1)
    weights = weights / weights.sum()

    n = A.shape[0]
    Aw = A.dot(weights)
    lambda_max = float(np.mean(Aw / weights))
    CI = float((lambda_max - n) / (n - 1))

    ri_map = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    RI = float(ri_map.get(n, 1.49))
    CR = float(CI / RI) if RI > 0 else 0.0

    return weights, lambda_max, CI, CR


def compute_ahp_score(
    df: pd.DataFrame,
    criteria: list[str],
    weights: dict[str, float],
    benefit_criteria: list[str] | None = None,
    cost_criteria: list[str] | None = None,
) -> pd.Series:
    df_work = df.copy()
    if "mpg" in df_work.columns:
        df_work["mpg"] = df_work["mpg"].apply(parse_mpg).astype(float)

    df_ahp = pd.DataFrame(index=df_work.index)
    for col in criteria:
        if col in df_work.columns:
            df_ahp[col] = df_work[col]
        else:
            df_ahp[col] = 0.0

    benefit = benefit_criteria if benefit_criteria is not None else BENEFIT_CRITERIA
    cost = cost_criteria if cost_criteria is not None else COST_CRITERIA

    for col in benefit:
        if col not in df_ahp.columns:
            df_ahp[col] = 0.0
        max_val = float(df_ahp[col].max()) if len(df_ahp) else 0.0
        df_ahp[col] = _safe_divide(df_ahp[col], max_val if max_val > 0 else 1.0)

    for col in cost:
        if col not in df_ahp.columns:
            df_ahp[col] = 0.0
        min_val = float(df_ahp[col].min()) if len(df_ahp) else 0.0
        df_ahp[col] = _safe_divide(min_val, df_ahp[col] + 1e-6) if min_val > 0 else 0.0

    score = 0.0
    for col in criteria:
        score = score + (df_ahp[col].astype(float) * float(weights.get(col, 0.0)))
    return pd.Series(score, index=df.index, name="ahp_score")
