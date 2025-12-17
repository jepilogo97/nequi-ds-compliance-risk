from __future__ import annotations

import pandas as pd


def build_risk_matrix(df_scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Construye matriz 5x5 en:
    - formato largo (por celda)
    - pivot de conteos
    - pivot de pérdida esperada anual ajustada
    """
    out = df_scored.copy()

    matrix = (
        out.groupby(["probability_1_5", "impact_1_5"], as_index=False)
        .agg(
            risks=("risk_id", "nunique"),
            el_annual_adj=("el_annual_adj", "sum"),
            critical_risks=("is_critical", "sum"),
        )
    )

    matrix["matrix_cell"] = matrix["probability_1_5"].astype(str) + "x" + matrix["impact_1_5"].astype(str)

    pivot_counts = (
        matrix.pivot(index="probability_1_5", columns="impact_1_5", values="risks")
        .reindex(index=[1, 2, 3, 4, 5], columns=[1, 2, 3, 4, 5])
        .fillna(0)
        .astype(int)
    )

    pivot_el = (
        matrix.pivot(index="probability_1_5", columns="impact_1_5", values="el_annual_adj")
        .reindex(index=[1, 2, 3, 4, 5], columns=[1, 2, 3, 4, 5])
        .fillna(0.0)
    )

    return matrix, pivot_counts, pivot_el
