from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

ModelType = Literal["logreg", "rf", "xgb"]


@dataclass(frozen=True)
class ModelConfig:
    model_type: ModelType = "logreg"

    # Manejo de desbalanceo: opción simple y muy efectiva
    class_weight: Optional[str] = "balanced"  # None o "balanced"

    # Logistic Regression params
    C: float = 1.0
    max_iter: int = 2000

    # RandomForest params
    n_estimators: int = 300
    max_depth: Optional[int] = None
    random_state: int = 42
    # XGBoost params
    learning_rate: float = 0.1
    use_label_encoder: bool = False


def build_model(cfg: ModelConfig):
    if cfg.model_type == "logreg":
        return LogisticRegression(
            C=cfg.C,
            max_iter=cfg.max_iter,
            class_weight=cfg.class_weight,
            solver="liblinear",
        )
    if cfg.model_type == "rf":
        return RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            class_weight=cfg.class_weight,
            random_state=cfg.random_state,
            n_jobs=-1,
        )
    if cfg.model_type == "xgb":
        return XGBClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            use_label_encoder=cfg.use_label_encoder,
            random_state=cfg.random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model_type: {cfg.model_type}")

# - `ModelConfig.class_weight` por defecto está en 'balanced', lo que hace que
#   los clasificadores penalicen más los errores sobre la clase minoritaria.
#   Es una solución rápida y efectiva para desbalance moderado.
# - `logreg` (Regresión Logística) es una buena línea base: interpretable,
#   rápida y permite obtener coeficientes para explicar la relación entre
#   features y la probabilidad de `critical_flag`.
# - `RandomForest` es una opción más robusta frente a relaciones no lineales
#   y outliers; también acepta `class_weight` y proporciona medidas de
#   importancia de features (aunque menos interpretables que coeficientes).
# - Para desbalance severo se recomienda combinar `class_weight` con técnicas
#   de sobremuestreo (SMOTE) o submuestreo dependiente del caso de uso.
