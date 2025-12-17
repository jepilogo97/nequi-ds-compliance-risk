import pandas as pd
import pytest

from src.validation.validators import validate_input


def test_validate_missing_columns():
    df = pd.DataFrame({"risk_id": [1]})
    with pytest.raises(ValueError):
        validate_input(df)


def test_validate_negative_values():
    df = pd.DataFrame(
        {
            "risk_id": [1],
            "process": ["Onboarding"],
            "risk_type": ["Operativo"],
            "events_last_3y": [-1],
            "total_loss_last_3y": [100.0],
            "controls_level": ["Bajo"],
            "critical_flag": [0],
        }
    )
    with pytest.raises(ValueError):
        validate_input(df)
