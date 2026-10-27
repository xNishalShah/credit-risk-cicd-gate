import pandas as pd

from credit_risk.features import engineer_features


SETTINGS = {"utilization_high_cutoff": 0.70, "late_payment_cutoff_days": 5}


def test_engineered_features_have_expected_values():
    source = pd.DataFrame({"Credit_Limit": [1000.0], "Current_Balance": [800.0], "Monthly_Spend": [250.0], "Payment_Delay_Days": [8]})
    out = engineer_features(source, SETTINGS)
    assert out.loc[0, "Credit_Utilization"] == 0.8
    assert out.loc[0, "High_Utilization_Flag"] == 1
    assert out.loc[0, "Payment_Delay_Band"] == "long_delay"
    assert out.loc[0, "Monthly_Spend_to_Limit"] == 0.25


def test_zero_credit_limit_is_safe_and_does_not_create_infinity():
    source = pd.DataFrame({"Credit_Limit": [0.0], "Current_Balance": [100.0], "Monthly_Spend": [50.0], "Payment_Delay_Days": [None]})
    out = engineer_features(source, SETTINGS)
    assert out.loc[0, "Credit_Utilization"] == 0
    assert out.loc[0, "Monthly_Spend_to_Limit"] == 0
    assert out.loc[0, "Payment_Delay_Band"] == "on_time"
