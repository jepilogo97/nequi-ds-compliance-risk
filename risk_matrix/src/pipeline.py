from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from src.config.risk_config import RiskMatrixConfig, ensure_config
from src.metrics.calculations import compute_metrics
from src.scoring.scoring import assign_scores
from src.matrix.risk_matrix import build_risk_matrix
from src.aggregation.aggregations import aggregate_by_process_and_type


def run_pipeline(df: pd.DataFrame, cfg: RiskMatrixConfig | None = None) -> dict[str, Any]:
    """
    Pipeline completo:
    1) métricas
    2) scoring 1–5 (prob/impact) + score
    3) matriz 5x5
    4) agregaciones por proceso y tipo
    """
    cfg = ensure_config(cfg)

    metrics = compute_metrics(df, cfg)
    scored = assign_scores(metrics, cfg)
    matrix_long, matrix_counts, matrix_el = build_risk_matrix(scored)
    agg = aggregate_by_process_and_type(scored)

    return {
        "config": asdict(cfg),
        "metrics": metrics,
        "scored": scored,
        "matrix_long": matrix_long,
        "matrix_counts": matrix_counts,
        "matrix_el": matrix_el,
        "agg_process_type": agg,
    }
