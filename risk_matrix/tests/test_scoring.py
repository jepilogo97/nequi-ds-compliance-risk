import pandas as pd

from src.config.risk_config import RiskMatrixConfig
from src.metrics.calculations import compute_metrics
from src.scoring.scoring import assign_scores


def test_assign_scores_ranges_quantile():
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

    assert s["probability_1_5"].between(1, 5).all()
    assert s["impact_1_5"].between(1, 5).all()
    assert (s["risk_score"] == s["probability_1_5"] * s["impact_1_5"]).all()
