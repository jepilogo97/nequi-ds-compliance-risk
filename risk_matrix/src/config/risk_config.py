from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .data_driven_config import (
    DEFAULT_CONTROLS_FREQ_MULTIPLIER,
    DEFAULT_CONTROLS_LEVEL_MAPPING,
    DEFAULT_FREQ_BINS,
    DEFAULT_SEV_BINS,
)

ScoreMethod = Literal["quantile", "fixed"]


@dataclass(frozen=True)
class RiskMatrixConfig:
    """
    Configuración de la Matriz de Riesgo.

    - years_window: ventana histórica en años (por defecto 3)
    - probability_method / impact_method:
        - "quantile": escala 1–5 usando cuantiles
        - "fixed": escala 1–5 usando bins definidos
    - freq_bins y sev_bins se usan si method="fixed"
    - controls_freq_multiplier ajusta el riesgo residual vía frecuencia
    """
    years_window: float = 3.0

    probability_method: ScoreMethod = "quantile"
    impact_method: ScoreMethod = "quantile"

    # Frecuencia anual (eventos/año) -> probabilidad 1..5
    freq_bins: tuple[float, float, float, float] = DEFAULT_FREQ_BINS

    # Severidad promedio (loss/evento) -> impacto 1..5
    sev_bins: tuple[float, float, float, float] = DEFAULT_SEV_BINS

    # controls_level esperado: 1=alto control ... 5=bajo control
    # Factor multiplicativo sobre la frecuencia anual:
    controls_freq_multiplier: dict[int, float] = None  # se completa en runtime

    controls_level_mapping: dict[str, int] = None  # se completa en runtime

def ensure_config(cfg: RiskMatrixConfig | None) -> RiskMatrixConfig:
    if cfg is None:
        cfg = RiskMatrixConfig()
    if cfg.controls_freq_multiplier is None:
        return RiskMatrixConfig(
            years_window=cfg.years_window,
            probability_method=cfg.probability_method,
            impact_method=cfg.impact_method,
            freq_bins=cfg.freq_bins,
            sev_bins=cfg.sev_bins,
            controls_freq_multiplier=DEFAULT_CONTROLS_FREQ_MULTIPLIER,
            controls_level_mapping=DEFAULT_CONTROLS_LEVEL_MAPPING,
        )
    return cfg
