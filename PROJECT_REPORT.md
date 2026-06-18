# Earthquake Prediction Model - MCA AI & ML Major Project Report

## 1. Title

Earthquake Prediction Model using Machine Learning

## 2. Abstract

Earthquakes are among the most destructive natural disasters, and estimating future seismic risk is an important research problem. This project develops a machine learning based earthquake risk forecasting system using historical earthquake catalog data. The model does not attempt exact deterministic earthquake prediction. Instead, it predicts whether a geographical grid cell is likely to experience an earthquake above a selected magnitude threshold in the next time window.

The system collects data from the USGS earthquake catalog, preprocesses the records, creates spatio-temporal features, trains a Random Forest classifier, and evaluates performance using standard classification metrics. The final model can estimate earthquake risk for a given latitude and longitude.

## 3. Introduction

Earthquake prediction is a difficult scientific problem because earthquakes are controlled by complex geological and tectonic processes. Traditional exact prediction requires identifying the exact location, time, and magnitude of a future event, which is not reliably possible with public historical earthquake data alone.

Therefore, this project follows a practical AI/ML approach: short-term risk forecasting. Historical seismic activity is converted into numerical features, and a supervised learning model is trained to estimate whether a significant event may occur in a future time window.

## 4. Problem Definition

Given historical earthquake data with timestamp, latitude, longitude, depth, and magnitude, predict whether a grid cell will experience an earthquake with magnitude greater than or equal to a configured threshold during the next forecast period.

## 5. Scope

The project includes:

- Data collection from USGS
- Data preprocessing
- Feature engineering
- Model training
- Model evaluation
- Risk prediction for a user-provided location

The project excludes:

- Real-time warning systems
- Official disaster management recommendations
- Exact earthquake time and magnitude prediction

## 6. Dataset Description

The dataset is obtained from the USGS earthquake catalog API in CSV format. Important fields include:

- `time`: event timestamp
- `latitude`: event latitude
- `longitude`: event longitude
- `depth`: event depth in kilometers
- `mag`: earthquake magnitude
- `place`: textual location description
- `type`: event type

Only earthquake events with valid latitude, longitude, depth, magnitude, and time are used.

## 7. Methodology

### 7.1 Data Collection

The system downloads earthquake records using configurable date range, magnitude threshold, and bounding box values.

### 7.2 Data Cleaning

Rows with missing essential values are removed. The `time` column is converted to a timezone-aware datetime format.

### 7.3 Spatial and Temporal Binning

The earth is divided into latitude-longitude grid cells. Time is divided into fixed windows, such as seven-day intervals. Each earthquake is assigned to a grid cell and a time window.

### 7.4 Feature Engineering

For each grid cell and time window, the system calculates:

- Recent earthquake count
- Recent maximum magnitude
- Recent mean magnitude
- Recent mean depth
- Days since the previous earthquake

### 7.5 Target Creation

The model predicts whether at least one earthquake with magnitude above the target threshold occurs in the next time window for the same grid cell.

### 7.6 Model Training

A Random Forest classifier is used because it handles non-linear relationships, works well with tabular data, and provides strong baseline performance.

### 7.7 Evaluation

The dataset is split chronologically so the model is trained on earlier data and tested on later data. This avoids using future information during training.

## 8. Algorithms Used

### Random Forest Classifier

Random Forest is an ensemble machine learning algorithm that builds multiple decision trees and combines their predictions. It reduces overfitting compared to a single decision tree and performs well on structured data.

## 9. Expected Output

The trained model outputs a probability score:

```text
Low risk: probability < 0.30
Medium risk: probability between 0.30 and 0.60
High risk: probability >= 0.60
```

## 10. Limitations

- Earthquake prediction remains scientifically uncertain.
- Historical catalog patterns may not capture hidden tectonic stress.
- Class imbalance can reduce recall for rare major events.
- Results should be interpreted as educational risk estimates only.

## 11. Future Work

- Use deep learning models for sequence forecasting.
- Include geological features such as fault-line distance.
- Create a Streamlit web dashboard.
- Add map-based visualization.
- Train separate regional models.

## 12. Conclusion

This project demonstrates how artificial intelligence and machine learning can be applied to earthquake risk forecasting. While exact earthquake prediction is not reliable, the implemented system provides a practical educational framework for learning data collection, spatio-temporal feature engineering, supervised classification, and model evaluation.
