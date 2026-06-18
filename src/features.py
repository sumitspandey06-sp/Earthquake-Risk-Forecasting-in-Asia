from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "latitude_cell",
    "longitude_cell",
    "event_count",
    "max_magnitude",
    "mean_magnitude",
    "mean_depth",
    "days_since_previous_event",
]


def add_bins(df: pd.DataFrame, grid_size: float, window_days: int) -> pd.DataFrame:
    binned = df.copy()
    binned["latitude_cell"] = np.floor(binned["latitude"] / grid_size) * grid_size
    binned["longitude_cell"] = np.floor(binned["longitude"] / grid_size) * grid_size
    first_time = binned["time"].min()
    binned["window_index"] = ((binned["time"] - first_time).dt.days // window_days).astype(int)
    return binned


def build_feature_table(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    feature_config = config["features"]
    grid_size = float(feature_config["grid_size_degrees"])
    window_days = int(feature_config["window_days"])
    target_magnitude = float(feature_config["target_magnitude"])

    binned = add_bins(df, grid_size, window_days)
    grouped = (
        binned.groupby(["latitude_cell", "longitude_cell", "window_index"], as_index=False)
        .agg(
            event_count=("mag", "size"),
            max_magnitude=("mag", "max"),
            mean_magnitude=("mag", "mean"),
            mean_depth=("depth", "mean"),
            window_start=("time", "min"),
        )
        .sort_values(["latitude_cell", "longitude_cell", "window_index"])
    )

    grouped["previous_event_time"] = grouped.groupby(["latitude_cell", "longitude_cell"])[
        "window_start"
    ].shift(1)
    grouped["days_since_previous_event"] = (
        grouped["window_start"] - grouped["previous_event_time"]
    ).dt.days
    grouped["days_since_previous_event"] = grouped["days_since_previous_event"].fillna(
        window_days * 10
    )

    grouped["future_max_magnitude"] = grouped.groupby(["latitude_cell", "longitude_cell"])[
        "max_magnitude"
    ].shift(-1)
    grouped["target"] = (grouped["future_max_magnitude"] >= target_magnitude).astype(int)

    model_df = grouped.dropna(subset=["future_max_magnitude"]).copy()
    model_df = model_df[FEATURE_COLUMNS + ["window_index", "window_start", "target"]]
    return model_df.reset_index(drop=True)
