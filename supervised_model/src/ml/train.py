from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.inspection import permutation_importance

from src.ml.features import FeatureSpec, build_preprocessor, split_xy
from src.ml.model import ModelConfig, build_model


def _safe_roc_auc(y_true, y_score) -> float:
    # roc_auc requiere ambos labels presentes
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def _safe_pr_auc(y_true, y_score) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return average_precision_score(y_true, y_score)


def _get_feature_names(pipe: Pipeline) -> list[str]:
    # pipe: preprocessor -> model
    pre = pipe.named_steps["preprocessor"]
    try:
        names = pre.get_feature_names_out().tolist()
        return names
    except Exception:
        return []


def _explain_model(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, top_k: int = 15) -> dict[str, Any]:
    """
    Explicabilidad ligera (sin SHAP):
    - Si es LogisticRegression: coeficientes (signo + magnitud)
    - Para cualquier modelo: permutation importance (sobre X_test)
    """
    model = pipe.named_steps["model"]
    feature_names = _get_feature_names(pipe)

    explanation: dict[str, Any] = {"feature_names_available": bool(feature_names)}

    # 1) Coefs (solo logreg)
    if hasattr(model, "coef_") and feature_names:
        coefs = model.coef_.ravel()
        coef_df = (
            pd.DataFrame({"feature": feature_names, "coef": coefs})
            .assign(abs_coef=lambda d: d["coef"].abs())
            .sort_values("abs_coef", ascending=False)
            .head(top_k)
        )
        explanation["top_coefficients"] = coef_df

    # 2) Permutation importance (modelo agnóstico)
    try:
        perm = permutation_importance(
            pipe,
            X_test,
            y_test,
            n_repeats=10,
            random_state=42,
            scoring="f1",
        )
        if feature_names and len(feature_names) == len(perm.importances_mean):
            perm_df = (
                pd.DataFrame({"feature": feature_names, "importance_mean": perm.importances_mean})
                .sort_values("importance_mean", ascending=False)
                .head(top_k)
            )
            explanation["top_permutation_importance"] = perm_df
        else:
            explanation["top_permutation_importance"] = None
    except Exception:
        explanation["top_permutation_importance"] = None

    return explanation


def train_supervised_model(
    df: pd.DataFrame,
    feature_spec: Optional[FeatureSpec] = None,
    model_cfg: Optional[ModelConfig] = None,
    test_size: float = 0.25,
    random_state: int = 42,
    use_smote: bool = False,
) -> dict[str, Any]:
    """
    Parte B – Modelo Supervisado:
    - Entrenar modelo simple para critical_flag
    - Métricas
    - Manejo de desbalanceo (class_weight y opcional SMOTE)
    - Explicabilidad (coef/permutation importance)
    """
    feature_spec = feature_spec or FeatureSpec()
    model_cfg = model_cfg or ModelConfig()

    X, y = split_xy(df, feature_spec)

    # Split estratificado (si se puede)
    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(df, feature_spec)
    model = None
    if model_cfg.model_type == "xgb":
        try:
            from xgboost import XGBClassifier
        except Exception as e:
            raise ImportError("To use model_type='xgb' install xgboost: pip install xgboost") from e

        # calcular ratio negativos/positivos en train para scale_pos_weight
        pos = int(y_train.sum())
        neg = int(len(y_train) - pos)
        scale_pos_weight = 1.0
        if model_cfg.class_weight == "balanced":
            if pos == 0:
                scale_pos_weight = 1.0
            else:
                scale_pos_weight = neg / pos if pos > 0 else 1.0
        
        print(f"⚡ XGBoost: scale_pos_weight set to {scale_pos_weight:.4f} (Pos: {pos}, Neg: {neg})")

        model = XGBClassifier(
            n_estimators=model_cfg.n_estimators,
            max_depth=model_cfg.max_depth,
            learning_rate=model_cfg.learning_rate,
            use_label_encoder=getattr(model_cfg, "use_label_encoder", False),
            random_state=model_cfg.random_state,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
        )
    else:
        model = build_model(model_cfg)

    # Pipeline base
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # Opción SMOTE
    if use_smote:
        try:
            from imblearn.pipeline import Pipeline as ImbPipeline
            from imblearn.over_sampling import SMOTE

            pipe = ImbPipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("smote", SMOTE(random_state=random_state)),
                    ("model", model),
                ]
            )
        except Exception as e:
            raise ImportError(
                "use_smote=True requiere instalar 'imbalanced-learn'. "
                "Instala: pip install imbalanced-learn"
            ) from e

    pipe.fit(X_train, y_train)

    # Predicciones
    y_pred = pipe.predict(X_test)

    # Scores para AUCs
    y_score = None
    if hasattr(pipe, "predict_proba"):
        y_score = pipe.predict_proba(X_test)[:, 1]
    elif hasattr(pipe, "decision_function"):
        y_score = pipe.decision_function(X_test)

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)

    roc_auc = _safe_roc_auc(y_test, y_score) if y_score is not None else float("nan")
    pr_auc = _safe_pr_auc(y_test, y_score) if y_score is not None else float("nan")

    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, zero_division=0)

    # Explicabilidad
    explanation = _explain_model(pipe, X_test, y_test, top_k=15)

    # Devolver también test labels y scores para generación de artefactos (curvas/matriz)
    y_test_list = y_test.tolist() if hasattr(y_test, "tolist") else list(y_test)
    y_pred_list = y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred)
    y_score_list = None
    if y_score is not None:
        y_score_list = y_score.tolist() if hasattr(y_score, "tolist") else list(y_score)

    return {
        "model_cfg": asdict(model_cfg),
        "feature_spec": asdict(feature_spec),
        "used_features": {"numeric": numeric_cols, "categorical": categorical_cols},
        "pipeline": pipe,
        "metrics": {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "confusion_matrix": cm,
            "classification_report": report,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "positive_rate_train": float(y_train.mean()) if len(y_train) else 0.0,
            "positive_rate_test": float(y_test.mean()) if len(y_test) else 0.0,
        },
        "explainability": explanation,
        "y_test": y_test_list,
        "y_pred": y_pred_list,
        "y_score": y_score_list,
    }
