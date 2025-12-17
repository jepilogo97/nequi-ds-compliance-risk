from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.risk_config import RiskMatrixConfig, ensure_config


def _score_fixed(values: pd.Series, bins: tuple[float, float, float, float]) -> pd.Series:
    """
    bins define thresholds for 1..5:
    <= b0 -> 1
    <= b1 -> 2
    <= b2 -> 3
    <= b3 -> 4
    >  b3 -> 5
    """
    b0, b1, b2, b3 = bins
    return pd.Series(
        np.select(
            [values <= b0, values <= b1, values <= b2, values <= b3, values > b3],
            [1, 2, 3, 4, 5],
        ),
        index=values.index,
        dtype=int,
    )


def _score_quantile(values: pd.Series) -> pd.Series:
    """
    Score 1..5 usando cuantiles.
    Robusto ante empates / baja cardinalidad.
    """
    try:
        return pd.qcut(values.rank(method="average"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    except ValueError:
        if values.nunique(dropna=True) <= 1:
            return pd.Series(3, index=values.index, dtype=int)
        r = values.rank(method="average")
        return pd.cut(r, bins=5, labels=[1, 2, 3, 4, 5], include_lowest=True).astype(int)


def assign_scores(df_metrics: pd.DataFrame, cfg: RiskMatrixConfig | None = None) -> pd.DataFrame:
    """
    - probabilidad 1–5 desde freq_annual_adj
    - impacto 1–5 desde severity_avg
    - risk_score = prob * impact
    """
    cfg = ensure_config(cfg)
    out = df_metrics.copy()

    freq = out["freq_annual_adj"]
    sev = out["severity_avg"]

    if cfg.probability_method == "fixed":
        out["probability_1_5"] = _score_fixed(freq, cfg.freq_bins)
    else:
        out["probability_1_5"] = _score_quantile(freq)

    if cfg.impact_method == "fixed":
        out["impact_1_5"] = _score_fixed(sev, cfg.sev_bins)
    else:
        out["impact_1_5"] = _score_quantile(sev)

    out["risk_score"] = out["probability_1_5"] * out["impact_1_5"]

    # bandera crítica: por input o por score alto
    out["is_critical"] = out["critical_flag"].astype(bool) | (out["risk_score"] >= 16)

    return out
