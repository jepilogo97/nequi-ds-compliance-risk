import pandas as pd

from src.config.risk_config import RiskMatrixConfig
from src.metrics.calculations import compute_metrics
from src.scoring.scoring import assign_scores
from src.matrix.risk_matrix import build_risk_matrix


def test_build_risk_matrix_shapes():
    df = pd.DataFrame(
        {
            "risk_id": [1, 2, 3, 4],
            "process": ["Pagos", "Reclamaciones", "Onboarding", "Credito"],
            "risk_type": ["Cumplimiento", "Operativo", "Cumplimiento", "Reputacional"],
            "events_last_3y": [0, 3, 6, 10],
            "total_loss_last_3y": [0.0, 30000.19, 60000.20, 240000.0],
            "controls_level": ["Bajo", "Medio", "Medio", "Alto"],
            "critical_flag": [0, 0, 1, 0],
        }
    )

    cfg = RiskMatrixConfig(probability_method="quantile", impact_method="quantile")
    m = compute_metrics(df, cfg)
    s = assign_scores(m, cfg)

    long, counts, el = build_risk_matrix(s)
    assert counts.shape == (5, 5)
    assert el.shape == (5, 5)
    assert {"probability_1_5", "impact_1_5", "risks", "el_annual_adj"}.issubset(long.columns)
