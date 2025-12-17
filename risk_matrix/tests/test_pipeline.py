import pandas as pd

from src.pipeline import run_pipeline
from src.config.risk_config import RiskMatrixConfig


def test_run_pipeline_returns_all():
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
    
    res = run_pipeline(df, RiskMatrixConfig(probability_method="quantile", impact_method="quantile"))

    for k in ["metrics", "scored", "matrix_long", "matrix_counts", "matrix_el", "agg_process_type"]:
        assert k in res
