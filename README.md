# Earthquake Risk Forecasting in Asia

MCA AI & ML major project for real-life earthquake data analysis and short-term earthquake risk forecasting in Asia.

Student: PANDEY SUMIT SHIVPUJAN

## Important Note

This project does not claim to predict the exact time, place, or magnitude of a future earthquake. Reliable exact earthquake prediction is not currently possible with standard public catalog data. Instead, this project builds a machine learning model that estimates the probability of an earthquake above a chosen magnitude threshold occurring in a geographic grid cell during the next time window.

That framing is realistic, scientifically safer, and still suitable as an AI/ML major project.

## Problem Statement

Given historical earthquake records containing time, latitude, longitude, depth, and magnitude, build a machine learning system that forecasts whether a selected geographic area is likely to experience a significant earthquake in the near future.

## Objectives

- Collect historical earthquake data from the USGS earthquake catalog.
- Clean and transform earthquake records into time-window and grid-based features.
- Train a supervised ML classification model.
- Compare multiple ML models.
- Evaluate the model using classification metrics and ROC-AUC.
- Generate feature importance for interpretability.
- Provide a prediction script for estimating risk at a latitude and longitude.

## Dataset

The project uses the United States Geological Survey earthquake catalog API:

- Official API documentation: https://earthquake.usgs.gov/fdsnws/event/1/
- CSV format reference: https://earthquake.usgs.gov/earthquakes/feed/v1.0/csv.php

The scripts can download catalog data directly in CSV format.

## Project Structure

```text
earthquake-prediction-model/
|-- config.yaml
|-- requirements.txt
|-- README.md
|-- AMITY_PROJECT_REPORT.md
|-- PROJECT_REPORT.md
|-- src/
|   |-- cli.py
|   |-- data.py
|   |-- eda.py
|   |-- features.py
|   |-- model.py
|   `-- predict.py
|-- data/
|   |-- raw/
|   `-- processed/
|-- models/
`-- reports/
    |-- SAMPLE_OUTPUT.md
    `-- figures/
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick Start

Download earthquake data:

```powershell
python -m src.cli download
```

Build ML-ready features:

```powershell
python -m src.cli build-features
```

Create real-life EDA charts and summary tables:

```powershell
python -m src.cli analyze
```

Train and evaluate the model:

```powershell
python -m src.cli train
```

Training generates:

```text
reports/metrics.txt
reports/model_comparison.csv
reports/figures/feature_importance.png
models/earthquake_risk_model.joblib
```

Predict earthquake risk for a location:

```powershell
python -m src.cli predict --latitude 35.68 --longitude 139.69
```

## Model Design

The earth surface is divided into latitude-longitude grid cells. Historical data is grouped into fixed time windows. For each cell and time window, the system calculates seismic activity features such as:

- Number of recent earthquakes
- Maximum recent magnitude
- Mean recent magnitude
- Mean earthquake depth
- Days since previous earthquake in the cell

The prediction target is:

```text
1 = at least one earthquake with magnitude >= target threshold occurs in the next forecast window
0 = no such earthquake occurs in the next forecast window
```

The default model is `RandomForestClassifier`, which is a strong baseline for tabular ML projects.

The project also compares Logistic Regression, Histogram Gradient Boosting, and Random Forest.

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

For imbalanced earthquake data, ROC-AUC, recall, and precision are more meaningful than accuracy alone.

## Limitations

- Public earthquake catalog data alone cannot guarantee exact earthquake prediction.
- The model learns statistical patterns, not physical tectonic causality.
- Results depend heavily on region, magnitude threshold, grid size, and time window.
- The model should not be used for emergency decisions.

## Future Enhancements

- Add tectonic plate boundary distance features.
- Add seismic energy release features.
- Compare Random Forest, XGBoost, LSTM, and Transformer models.
- Build a Streamlit dashboard for visualization.
- Use region-specific training for India, Japan, California, or global seismic zones.

## Submission Files

- `README.md`: setup and usage guide
- `AMITY_PROJECT_REPORT.md`: Amity-format project report draft
- `PROJECT_REPORT.md`: report content for MCA submission
- `reports/SAMPLE_OUTPUT.md`: sample metric and prediction output format
