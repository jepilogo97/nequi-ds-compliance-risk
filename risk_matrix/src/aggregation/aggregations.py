from __future__ import annotations

import pandas as pd


def aggregate_by_process_and_type(df_scored: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega por process y risk_type:
    - # riesgos
    - EL anual ajustada total
    - score promedio
    - # críticos
    - promedios de frecuencia/severidad (para diagnóstico)
    """
    agg = (
        df_scored.groupby(["process", "risk_type"], as_index=False)
        .agg(
            risks=("risk_id", "nunique"),
            el_annual_adj=("el_annual_adj", "sum"),
            avg_score=("risk_score", "mean"),
            critical_risks=("is_critical", "sum"),
            avg_freq=("freq_annual_adj", "mean"),
            avg_severity=("severity_avg", "mean"),
        )
        .sort_values(["el_annual_adj", "avg_score"], ascending=False)
    )
    return agg
