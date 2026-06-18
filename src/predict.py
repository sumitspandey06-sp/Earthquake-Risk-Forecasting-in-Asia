from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.features import FEATURE_COLUMNS


def risk_label(probability: float) -> str:
    if probability >= 0.60:
        return "High"
    if probability >= 0.30:
        return "Medium"
    return "Low"


def build_prediction_row(latitude: float, longitude: float, features_path: Path) -> pd.DataFrame:
    features = pd.read_csv(features_path)
    ranked = features.assign(
        distance=(features["latitude_cell"] - latitude).abs()
        + (features["longitude_cell"] - longitude).abs()
    ).sort_values(["distance", "window_index"], ascending=[True, False])

    if ranked.empty:
        raise ValueError("No feature rows found. Run build-features and train first.")

    latest_nearby = ranked.iloc[0].copy()
    return pd.DataFrame([latest_nearby[FEATURE_COLUMNS]])


def predict_risk(latitude: float, longitude: float, model_path: Path, features_path: Path) -> dict:
    artifact = joblib.load(model_path)
    model = artifact["model"]
    row = build_prediction_row(latitude, longitude, features_path)
    probability = float(model.predict_proba(row)[:, 1][0])
    return {
        "latitude": latitude,
        "longitude": longitude,
        "probability": probability,
        "risk": risk_label(probability),
    }
