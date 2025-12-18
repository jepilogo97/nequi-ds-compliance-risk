import pandas as pd

from src.ml.train import train_supervised_model
from src.ml.model import ModelConfig


def test_train_supervised_model_runs():
    df = pd.DataFrame(
        {
            "risk_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "process": ["Pagos", "Pagos", "Credito", "Credito", "Onboarding", "Onboarding", "Reclamaciones", "Reclamaciones"],
            "risk_type": ["Operativo", "Cumplimiento", "Operativo", "Reputacional", "Operativo", "Cumplimiento", "Operativo", "Cumplimiento"],
            "events_last_3y": [0, 3, 1, 10, 2, 6, 0, 4],
            "total_loss_last_3y": [0.0, 30000.0, 2000.0, 240000.0, 5000.0, 60000.0, 0.0, 12000.0],
            "controls_level": ["Alto", "Medio", "Alto", "Bajo", "Medio", "Bajo", "Alto", "Medio"],
            "critical_flag": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    res = train_supervised_model(df, model_cfg=ModelConfig(model_type="logreg"))
    assert "metrics" in res
    assert res["metrics"]["f1"] >= 0.0
