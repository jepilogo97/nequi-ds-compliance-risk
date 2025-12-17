import pandas as pd
import pytest

from src.config.risk_config import RiskMatrixConfig
from src.metrics.calculations import compute_metrics


def test_compute_metrics_basic():
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

    m = compute_metrics(df, RiskMatrixConfig(probability_method="fixed", impact_method="fixed"))

    # risk_id=2: 3 eventos / 3 años => 1 evento/año
    assert m.loc[m["risk_id"] == 2, "freq_annual"].iloc[0] == pytest.approx(1.0)
    # severidad promedio = 30000/3 = 10000
    assert m.loc[m["risk_id"] == 2, "severity_avg"].iloc[0] == pytest.approx(10000.0)
    # events=0 => severidad 0
    assert m.loc[m["risk_id"] == 1, "severity_avg"].iloc[0] == pytest.approx(0.0)
