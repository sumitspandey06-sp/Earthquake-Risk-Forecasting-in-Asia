from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def run_eda(raw_df: pd.DataFrame, output_dir: Path, summary_path: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    df = raw_df.copy()
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.tz_convert(None).dt.to_period("M").dt.to_timestamp()
    df["magnitude_class"] = pd.cut(
        df["mag"],
        bins=[0, 4.5, 5.5, 6.5, 10],
        labels=["4.0-4.4", "4.5-5.4", "5.5-6.4", "6.5+"],
        right=False,
    )

    summary = pd.DataFrame(
        {
            "metric": [
                "total_events",
                "start_date",
                "end_date",
                "minimum_magnitude",
                "maximum_magnitude",
                "mean_magnitude",
                "mean_depth_km",
                "deepest_event_km",
            ],
            "value": [
                len(df),
                df["time"].min().date().isoformat(),
                df["time"].max().date().isoformat(),
                round(float(df["mag"].min()), 2),
                round(float(df["mag"].max()), 2),
                round(float(df["mag"].mean()), 2),
                round(float(df["depth"].mean()), 2),
                round(float(df["depth"].max()), 2),
            ],
        }
    )
    summary.to_csv(summary_path, index=False)

    figure_paths = {
        "magnitude_distribution": output_dir / "magnitude_distribution.png",
        "monthly_event_trend": output_dir / "monthly_event_trend.png",
        "depth_vs_magnitude": output_dir / "depth_vs_magnitude.png",
        "magnitude_class_counts": output_dir / "magnitude_class_counts.png",
        "asia_event_map": output_dir / "asia_event_map.png",
    }

    plt.figure(figsize=(8, 5))
    df["mag"].hist(bins=30, color="#2f6f73", edgecolor="white")
    plt.title("Magnitude Distribution of Earthquakes in Asia")
    plt.xlabel("Magnitude")
    plt.ylabel("Number of Events")
    plt.tight_layout()
    plt.savefig(figure_paths["magnitude_distribution"], dpi=160)
    plt.close()

    monthly_counts = df.groupby("month").size()
    plt.figure(figsize=(10, 5))
    monthly_counts.plot(color="#7a3f2a", linewidth=1.8)
    plt.title("Monthly Earthquake Event Trend in Asia")
    plt.xlabel("Month")
    plt.ylabel("Number of Events")
    plt.tight_layout()
    plt.savefig(figure_paths["monthly_event_trend"], dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(df["depth"], df["mag"], s=10, alpha=0.35, color="#44546a")
    plt.title("Depth vs Magnitude")
    plt.xlabel("Depth (km)")
    plt.ylabel("Magnitude")
    plt.tight_layout()
    plt.savefig(figure_paths["depth_vs_magnitude"], dpi=160)
    plt.close()

    class_counts = df["magnitude_class"].value_counts().sort_index()
    plt.figure(figsize=(7, 5))
    class_counts.plot(kind="bar", color="#596f2f", edgecolor="white")
    plt.title("Earthquake Count by Magnitude Class")
    plt.xlabel("Magnitude Class")
    plt.ylabel("Number of Events")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figure_paths["magnitude_class_counts"], dpi=160)
    plt.close()

    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        df["longitude"],
        df["latitude"],
        c=df["mag"],
        s=df["mag"] ** 2,
        cmap="viridis",
        alpha=0.55,
    )
    plt.title("Spatial Distribution of Earthquakes in Asia")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.colorbar(scatter, label="Magnitude")
    plt.tight_layout()
    plt.savefig(figure_paths["asia_event_map"], dpi=160)
    plt.close()

    return figure_paths | {"summary": summary_path}
