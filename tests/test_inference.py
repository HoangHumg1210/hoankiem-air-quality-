import pandas as pd

from inference import build_history_feature_frame, prepare_raw_frame


DATA_PATH = "data/processed/data2225_done.csv"


def test_prepare_raw_frame_resamples_to_three_hours():
    raw_df = pd.read_csv(DATA_PATH)
    frame_3h = prepare_raw_frame(raw_df, step_hours=3)

    assert frame_3h.index.is_monotonic_increasing
    assert len(frame_3h) < len(raw_df)
    assert "PM25" in frame_3h.columns
    assert "IsHoliday" in frame_3h.columns


def test_build_history_feature_frame_contains_expected_columns():
    raw_df = pd.read_csv(DATA_PATH)
    frame_3h = prepare_raw_frame(raw_df, step_hours=3)
    feature_frame = build_history_feature_frame(frame_3h)

    expected_cols = {
        "PM25_lag_1",
        "PM25_lag_8",
        "PM25_lag_24",
        "PM25_lag_56",
        "PM25_roll_mean_8",
        "PM25_roll_std_24",
        "PM25_same_hour_mean_3d",
        "PM25_same_hour_max_7d",
        "hour_sin",
        "dow_cos",
        "month_cos",
    }
    assert expected_cols.issubset(feature_frame.columns)
