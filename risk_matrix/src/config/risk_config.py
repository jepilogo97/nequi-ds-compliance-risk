from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
    freq_bins: tuple[float, float, float, float] = (0.2, 0.5, 1.0, 2.0)

    # Severidad promedio (loss/evento) -> impacto 1..5
    sev_bins: tuple[float, float, float, float] = (1_000, 10_000, 50_000, 200_000)

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
            controls_freq_multiplier={1: 0.70, 2: 0.85, 3: 1.00, 4: 1.15, 5: 1.30},
            controls_level_mapping={
                                    "Alto": 1,
                                    "Medio": 3,
                                    "Bajo": 5,
                                }
        )
    return cfg
