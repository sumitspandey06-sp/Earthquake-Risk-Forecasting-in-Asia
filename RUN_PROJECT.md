# Project Run Guide

## Project Title

Earthquake Risk Forecasting in Asia

## Software Requirements

- Python 3.10 or above
- pip package manager
- Internet connection for downloading earthquake data from USGS

## Step 1: Open Project Folder

Open PowerShell or Command Prompt in the project folder.

```powershell
cd Major_Project_PANDEY_SUMIT_SHIVPUJAN
```

## Step 2: Create Virtual Environment

```powershell
python -m venv .venv
```

## Step 3: Activate Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Step 4: Install Dependencies

```powershell
pip install -r requirements.txt
```

## Step 5: Download Real Earthquake Data

```powershell
python -m src.cli download
```

This downloads Asia-region earthquake records from the USGS earthquake catalog.

## Step 6: Generate Data Analysis Charts

```powershell
python -m src.cli analyze
```

The generated figures are saved in:

```text
reports/figures/
```

## Step 7: Build Machine Learning Features

```powershell
python -m src.cli build-features
```

The processed feature dataset is saved in:

```text
data/processed/features.csv
```

## Step 8: Train the Model

```powershell
python -m src.cli train
```

The trained model is saved in:

```text
models/earthquake_risk_model.joblib
```

The evaluation report is saved in:

```text
reports/metrics.txt
```

## Step 9: Run a Sample Prediction

Example for Delhi, India:

```powershell
python -m src.cli predict --latitude 28.61 --longitude 77.20
```

Example for Tokyo, Japan:

```powershell
python -m src.cli predict --latitude 35.68 --longitude 139.69
```

## Important Output Files

```text
data/raw/earthquakes.csv
data/processed/features.csv
reports/asia_earthquake_summary.csv
reports/metrics.txt
reports/figures/
models/earthquake_risk_model.joblib
AMITY_PROJECT_REPORT.md
```

## Suggested Submission File

The main project report content is available in:

```text
AMITY_PROJECT_REPORT.md
```

Before final submission, replace all placeholders for student name, enrollment number, guide name, designation, course, semester, and academic year.
