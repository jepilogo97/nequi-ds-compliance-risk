from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {
    "risk_id",
    "process",
    "risk_type",
    "events_last_3y",
    "total_loss_last_3y",
    "controls_level",
    "critical_flag",
}


def validate_input(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df["events_last_3y"].isna().any():
        raise ValueError("events_last_3y has nulls")
    if df["total_loss_last_3y"].isna().any():
        raise ValueError("total_loss_last_3y has nulls")
    if df["controls_level"].isna().any():
        raise ValueError("controls_level has nulls")

    if (df["events_last_3y"] < 0).any():
        raise ValueError("events_last_3y must be non-negative")
    if (df["total_loss_last_3y"] < 0).any():
        raise ValueError("total_loss_last_3y must be non-negative")
