from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


@dataclass(frozen=True)
class FeatureSpec:
    target: str = "critical_flag"
    id_cols: Tuple[str, ...] = ("risk_id",)

    # Features base del dataset
    numeric_cols: Tuple[str, ...] = ("events_last_3y", "total_loss_last_3y")
    categorical_cols: Tuple[str, ...] = ("process", "risk_type", "controls_level")
    optional_numeric_cols: Tuple[str, ...] = ("freq_annual_adj", "severity_avg", "el_annual_adj")


def build_preprocessor(df: pd.DataFrame, spec: FeatureSpec) -> tuple[ColumnTransformer, List[str], List[str]]:
    """
    Construye un preprocesador robusto:
    - Imputa numéricos (median)
    - OHE para categóricos (ignore unknown)
    Además, agrega features opcionales si existen.
    """
    # - `numeric` incluye las columnas numéricas básicas y opcionales si están
    #   presentes en el dataframe (p. ej. `freq_annual_adj`, `severity_avg`).
    # - Para numéricos usamos `SimpleImputer(strategy='median')` para resiliencia
    #   frente a outliers y valores faltantes.
    # - Para categóricos imputamos la moda y aplicamos OneHotEncoder con
    #   `handle_unknown='ignore'` para que el pipeline sea tolerant a nuevas
    #   categorías en datos de producción.
    numeric = list(spec.numeric_cols)
    for c in spec.optional_numeric_cols:
        if c in df.columns:
            numeric.append(c)

    categorical = [c for c in spec.categorical_cols if c in df.columns]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric),
            ("cat", cat_pipe, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor, numeric, categorical


def split_xy(df: pd.DataFrame, spec: FeatureSpec) -> tuple[pd.DataFrame, pd.Series]:
    if spec.target not in df.columns:
        raise ValueError(f"Target column '{spec.target}' not found in dataframe")

    X = df.drop(columns=[spec.target], errors="ignore").copy()
    y = df[spec.target].astype(int).copy()
    return X, y
