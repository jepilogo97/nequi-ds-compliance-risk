from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.risk_config import RiskMatrixConfig, ensure_config
from src.validation.validators import validate_input


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    denom = denom.replace(0, np.nan)
    out = numer / denom
    return out.fillna(0.0)


def compute_metrics(df: pd.DataFrame, cfg: RiskMatrixConfig | None = None) -> pd.DataFrame:
    """
    Parte A – Métricas base:
    - frecuencia anual (base y ajustada por controles)
    - severidad promedio
    - pérdida esperada anual (base y ajustada)
    """
    cfg = ensure_config(cfg)
    validate_input(df)

    out = df.copy()

    # Frecuencia anual base (eventos/año)
    out["freq_annual"] = out["events_last_3y"] / cfg.years_window

    # Severidad promedio (loss/evento) — si no hay eventos, severidad 0
    # Round total losses to whole units before computing severity average
    out["severity_avg"] = _safe_div(out["total_loss_last_3y"].round(0), out["events_last_3y"])

    # Pérdida esperada anual base (Expected Loss): freq * severity
    out["el_annual"] = out["freq_annual"] * out["severity_avg"]

    cl = out["controls_level"]

    # Si viene texto, mapear a escala 1-5
    if cl.dtype == "object":
        mapped = cl.map(cfg.controls_level_mapping)
        if mapped.isna().any():
            bad = sorted(cl[mapped.isna()].unique().tolist())
            raise ValueError(f"controls_level contains unknown labels: {bad}")
        out["controls_level_num"] = mapped
        cl_num = out["controls_level_num"]
    else:
        cl_num = cl

    # Ajuste por controles (riesgo residual): ajusta la frecuencia
    mult = cl_num.map(cfg.controls_freq_multiplier)
    if mult.isna().any():
        bad = sorted(out.loc[mult.isna(), "controls_level"].unique().tolist())
        raise ValueError(f"controls_level contains unmapped values: {bad}")

    out["freq_annual_adj"] = out["freq_annual"] * mult
    out["el_annual_adj"] = out["freq_annual_adj"] * out["severity_avg"]

    return out
