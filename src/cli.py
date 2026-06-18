from __future__ import annotations

import argparse
import json

from src.config import load_config, project_path
from src.data import download_earthquake_data, load_earthquake_data
from src.eda import run_eda
from src.features import build_feature_table
from src.model import train_model
from src.predict import predict_risk


def cmd_download(config: dict) -> None:
    output_path = project_path(config["data"]["raw_path"])
    saved_path = download_earthquake_data(config, output_path)
    print(f"Downloaded earthquake data to {saved_path}")


def cmd_build_features(config: dict) -> None:
    raw_path = project_path(config["data"]["raw_path"])
    processed_path = project_path(config["data"]["processed_path"])
    df = load_earthquake_data(raw_path)
    features = build_feature_table(df, config)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(processed_path, index=False)
    print(f"Saved {len(features)} feature rows to {processed_path}")


def cmd_analyze(config: dict) -> None:
    raw_path = project_path(config["data"]["raw_path"])
    analysis_config = config["analysis"]
    output_dir = project_path(analysis_config["output_dir"])
    summary_path = project_path(analysis_config["summary_path"])
    df = load_earthquake_data(raw_path)
    outputs = run_eda(df, output_dir, summary_path)
    for name, path in outputs.items():
        print(f"{name}: {path}")


def cmd_train(config: dict) -> None:
    processed_path = project_path(config["data"]["processed_path"])
    model_path = project_path(config["model"]["model_path"])
    metrics_path = project_path(config["model"]["metrics_path"])
    features = load_feature_data(processed_path)
    metrics = train_model(features, config, model_path, metrics_path)
    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")


def load_feature_data(path):
    import pandas as pd

    return pd.read_csv(path, parse_dates=["window_start"])


def cmd_predict(config: dict, latitude: float, longitude: float) -> None:
    model_path = project_path(config["model"]["model_path"])
    features_path = project_path(config["data"]["processed_path"])
    result = predict_risk(latitude, longitude, model_path, features_path)
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Earthquake risk forecasting project")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("download", help="Download earthquake data from USGS")
    subparsers.add_parser("analyze", help="Create EDA summary tables and figures")
    subparsers.add_parser("build-features", help="Create ML-ready feature table")
    subparsers.add_parser("train", help="Train and evaluate the ML model")

    predict_parser = subparsers.add_parser("predict", help="Predict risk for a location")
    predict_parser.add_argument("--latitude", type=float, required=True)
    predict_parser.add_argument("--longitude", type=float, required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "download":
        cmd_download(config)
    elif args.command == "analyze":
        cmd_analyze(config)
    elif args.command == "build-features":
        cmd_build_features(config)
    elif args.command == "train":
        cmd_train(config)
    elif args.command == "predict":
        cmd_predict(config, args.latitude, args.longitude)


if __name__ == "__main__":
    main()
