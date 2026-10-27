from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_features(frame: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Pure, reusable transformations for both training and inference."""
    result = frame.copy()
    limit = result["Credit_Limit"].replace(0, np.nan)
    result["Credit_Utilization"] = (result["Current_Balance"] / limit).clip(0, 5).fillna(0)
    result["High_Utilization_Flag"] = (
        result["Credit_Utilization"] >= settings["utilization_high_cutoff"]
    ).astype(int)
    result["Payment_Delay_Band"] = pd.cut(
        result["Payment_Delay_Days"].fillna(0),
        bins=[-1, 0, settings["late_payment_cutoff_days"], float("inf")],
        labels=["on_time", "short_delay", "long_delay"],
    )
    # Domain feature: repeated spending above the approved limit signals stress.
    result["Monthly_Spend_to_Limit"] = (result["Monthly_Spend"] / limit).clip(0, 5).fillna(0)
    return result
