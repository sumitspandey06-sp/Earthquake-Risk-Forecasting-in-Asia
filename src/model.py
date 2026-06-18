from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLUMNS


def chronological_split(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_df = df.sort_values("window_index").reset_index(drop=True)
    split_index = int(len(sorted_df) * (1 - test_fraction))
    split_index = max(1, min(split_index, len(sorted_df) - 1))
    return sorted_df.iloc[:split_index], sorted_df.iloc[split_index:]


def train_model(features: pd.DataFrame, config: dict, model_path: Path, metrics_path: Path) -> dict:
    model_config = config["model"]
    train_df, test_df = chronological_split(features, float(model_config["test_fraction"]))

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["target"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["target"]

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=int(model_config["random_state"]),
            ),
        ),
        "Histogram Gradient Boosting": HistGradientBoostingClassifier(
            random_state=int(model_config["random_state"]),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=int(model_config["random_state"]),
            min_samples_leaf=2,
            n_jobs=-1,
        ),
    }

    comparison_rows = []
    fitted_models = {}
    for name, candidate in models.items():
        candidate.fit(X_train, y_train)
        candidate_predictions = candidate.predict(X_test)
        candidate_probabilities = candidate.predict_proba(X_test)[:, 1]
        candidate_roc_auc = (
            roc_auc_score(y_test, candidate_probabilities) if y_test.nunique() > 1 else float("nan")
        )
        comparison_rows.append(
            {
                "model": name,
                "accuracy": float(accuracy_score(y_test, candidate_predictions)),
                "roc_auc": float(candidate_roc_auc),
            }
        )
        fitted_models[name] = candidate

    model = fitted_models["Random Forest"]
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, probabilities) if y_test.nunique() > 1 else float("nan")

    comparison_path = Path(model_config["comparison_path"])
    feature_importance_path = Path(model_config["feature_importance_path"])
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    feature_importance_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    write_feature_importance(model, feature_importance_path)

    metrics = {
        "rows": len(features),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "positive_rate": float(features["target"].mean()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc),
        "comparison_path": str(comparison_path),
        "feature_importance_path": str(feature_importance_path),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, zero_division=0),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS, "config": config}, model_path)
    write_metrics(metrics, metrics_path)
    return metrics


def write_feature_importance(model: RandomForestClassifier, output_path: Path) -> None:
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values()
    plt.figure(figsize=(8, 5))
    importance.plot(kind="barh", color="#2f6f73")
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_metrics(metrics: dict, metrics_path: Path) -> None:
    lines = [
        "Earthquake Risk Model Metrics",
        "=============================",
        f"Rows: {metrics['rows']}",
        f"Train rows: {metrics['train_rows']}",
        f"Test rows: {metrics['test_rows']}",
        f"Positive target rate: {metrics['positive_rate']:.4f}",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"ROC-AUC: {metrics['roc_auc']:.4f}",
        f"Confusion matrix: {metrics['confusion_matrix']}",
        f"Model comparison file: {metrics['comparison_path']}",
        f"Feature importance figure: {metrics['feature_importance_path']}",
        "",
        metrics["classification_report"],
    ]
    metrics_path.write_text("\n".join(lines), encoding="utf-8")
