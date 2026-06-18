# AMITY UNIVERSITY ONLINE, NOIDA, UTTAR PRADESH

## Project Report

In partial fulfilment of the requirement for the award of degree of Master of Computer Applications, Discipline - IT

## Title

Earthquake Risk Forecasting in Asia

Guide Details:

Name: ____________________  
Designation: ____________________

Submitted By:

Name of the Student: PANDEY SUMIT SHIVPUJAN  
Enrollment No.: ____________________

Font for final formatting: Times New Roman, 12 pt body text, double spacing, 1-inch margin.

## Abstract

Earthquakes are high-impact natural hazards that can cause severe damage to human life, infrastructure, economic activity, and public safety. Asia is one of the most seismically active regions in the world because it includes multiple tectonic plate boundaries and earthquake-prone zones such as the Himalayan belt, Japan, Indonesia, the Philippines, Iran, Afghanistan, Turkey, and parts of Central Asia. The uncertainty associated with earthquake occurrence makes it an important topic for scientific research and data-driven risk analysis. However, exact earthquake prediction, in the strict sense of identifying the precise time, location, and magnitude of a future earthquake, is not considered reliably achievable using public historical catalog data alone. Therefore, this project treats the problem as earthquake risk forecasting rather than deterministic prediction.

The objective of this project is to design and implement a machine learning based earthquake risk forecasting model for the Asia region using real-life earthquake catalog data obtained from the United States Geological Survey (USGS). The project analyzes earthquakes recorded between 2014-01-01 and 2026-05-25 within the selected Asia geographical boundary. The dataset contains 78,686 earthquake records with magnitude 4.0 and above. The main fields used for analysis are event time, latitude, longitude, depth, magnitude, and location description. The study applies exploratory data analysis to understand the distribution of earthquake magnitude, depth, spatial occurrence, and monthly trends. It then converts the historical earthquake records into a supervised machine learning dataset using grid-based spatial binning and seven-day temporal windows.

The predictive modeling task is framed as a binary classification problem. For each latitude-longitude grid cell and time window, engineered features such as event count, maximum magnitude, mean magnitude, mean depth, and days since the previous event are calculated. The target variable indicates whether an earthquake of magnitude 5.0 or above occurs in the next forecast window for the same grid cell. A Random Forest classifier is trained using chronological splitting, where older records are used for training and newer records are used for testing. This split method is selected to reduce future-data leakage and to simulate a more realistic forecasting scenario.

The final processed machine learning dataset contains 45,413 feature rows. The trained Random Forest model achieved an accuracy of 80.97 percent and ROC-AUC of 0.5448 on the test set. The positive target rate was 12.59 percent, indicating that significant earthquake events are relatively infrequent compared with non-event windows. The results show that historical catalog features alone provide limited predictive strength for short-term earthquake forecasting. The model performs better in identifying non-event windows than future event windows, which is an important finding for the study. This supports the scientific understanding that earthquake forecasting is a complex problem and that public historical catalog data should be supplemented with geological, geophysical, and tectonic features for stronger predictive performance.

This project contributes an educational and practical machine learning workflow for earthquake risk analysis. It demonstrates real data collection, data cleaning, exploratory data analysis, feature engineering, supervised classification, model evaluation, and interpretation of results. The project also emphasizes ethical and scientific caution by clearly stating that its output should not be used as an official disaster warning system. Instead, it should be understood as a data analysis and AI/ML learning framework that can be extended with additional domain-specific features and improved modeling techniques.

Keywords: earthquake forecasting, machine learning, Asia, USGS, Random Forest, seismic risk, data analysis, AI, ML

## Declaration

I, PANDEY SUMIT SHIVPUJAN, a student pursuing Master of Computer Applications at Amity University Online, hereby declare that the project work entitled "Earthquake Risk Forecasting in Asia" has been prepared by me during the academic year 2025-2026 under the guidance of ____________________, Department of Computer Applications, Amity University Online. I assert that this project is a piece of original bona fide work done by me. It is the outcome of my own effort and has not been submitted to any other university for the award of any degree.

Signature of Student

## Certificate

This is to certify that PANDEY SUMIT SHIVPUJAN of Amity University Online has carried out the project work presented in this project report entitled "Earthquake Risk Forecasting in Asia" for the award of Master of Computer Applications under my guidance. The project report embodies results of original work, and studies are carried out by the student himself/herself. Certified further, that to the best of my knowledge the work reported herein does not form the basis for the award of any other degree to the candidate or to anybody else from this or any other University/Institution.

Signature  
[Name of Guide]  
[Designation]

## Table of Contents

1. Title Page  
2. Abstract  
3. Declaration  
4. Certificate  
5. Table of Contents  
6. List of Tables  
7. List of Figures  
8. Chapter 1: Introduction to the Topic  
9. Chapter 2: Review of Literature  
10. Chapter 3: Research Objectives and Methodology  
11. Chapter 4: Data Analysis, Results, and Interpretation  
12. Chapter 5: Findings and Conclusion  
13. Chapter 6: Recommendations and Limitations of the Study  
14. Bibliography / References  
15. Appendix

## List of Tables

Table 1: Dataset configuration for Asia earthquake study  
Table 2: Summary statistics of real earthquake data  
Table 3: Machine learning model evaluation metrics  
Table 4: Model comparison results  
Table 5: Confusion matrix of Random Forest classifier

## List of Figures

Figure 1: Magnitude distribution of earthquakes in Asia  
Figure 2: Monthly earthquake event trend in Asia  
Figure 3: Depth vs magnitude scatter plot  
Figure 4: Earthquake count by magnitude class  
Figure 5: Spatial distribution of earthquakes in Asia  
Figure 6: Feature importance of Random Forest model

# Chapter 1: Introduction to the Topic

Earthquakes are sudden movements of the earth's crust caused by the release of accumulated stress along faults and tectonic plate boundaries. The energy released during an earthquake travels through seismic waves and can result in ground shaking, landslides, tsunamis, soil liquefaction, and damage to buildings and infrastructure. Because earthquakes can occur with limited direct warning, they represent a major concern for disaster management, urban planning, engineering, public administration, and scientific research.

Asia has been selected as the study region because it contains several of the world's most seismically active zones. The region includes the Himalayan collision zone, the Indonesian subduction zone, the Japanese island arc, the Philippine region, and earthquake-prone areas across Iran, Afghanistan, Pakistan, Nepal, China, and Turkey. These areas experience frequent moderate earthquakes and occasional high-magnitude events. The diversity of tectonic settings makes Asia a suitable region for studying earthquake patterns through real-life data analysis.

The project topic is relevant to the MCA AI and ML specialization because it applies machine learning to a real-world problem that involves large-scale data, uncertainty, classification, model evaluation, and responsible interpretation. The project uses historical earthquake records and transforms them into a supervised learning problem. The purpose is not to claim exact prediction, but to investigate whether recent seismic activity in a grid cell can help estimate the probability of a significant future event.

The topic has been selected for four main reasons. First, earthquake risk is socially important because it affects public safety and infrastructure resilience. Second, real-world earthquake data is publicly available from scientific agencies such as USGS, which makes the study feasible and reproducible. Third, the problem is suitable for AI/ML because it involves spatio-temporal data, feature engineering, classification, and model validation. Fourth, the study encourages critical thinking because earthquake forecasting has clear scientific limitations, and the model results must be interpreted carefully.

Modern data science can support earthquake research by identifying statistical patterns in historical records. For example, earthquake catalogs can be analyzed to identify high-activity zones, magnitude distributions, event frequency, and changes over time. Machine learning can then be used to learn relationships between historical activity and future event occurrence. However, seismic systems are complex, non-linear, and controlled by hidden geophysical processes. Therefore, model outputs should be understood as risk scores, not official warnings.

This project uses a practical AI/ML workflow. First, data is downloaded from the USGS earthquake catalog using configurable geographical and temporal filters. Second, the data is cleaned and converted into a structured format. Third, exploratory data analysis is performed using summary statistics and visualizations. Fourth, a feature table is created by grouping earthquakes into latitude-longitude grid cells and weekly time windows. Fifth, a Random Forest classifier is trained and evaluated. Finally, the findings are interpreted in the context of earthquake forecasting limitations.

# Chapter 2: Review of Literature

Earthquake prediction and earthquake forecasting are related but different concepts. Prediction usually refers to specifying the time, place, and magnitude of an earthquake within narrow limits. Forecasting refers to estimating the probability of earthquakes in a region and time period. The USGS states that scientists cannot predict exact earthquakes and can only calculate the probability of significant earthquakes in a particular area over a period of time. This distinction is important for the present project because the model is designed for probabilistic risk forecasting rather than exact prediction.

Geller, Jackson, Kagan, and Mulargia (1997) argued that reliable deterministic earthquake prediction is not presently achievable. Their work is frequently cited in discussions about the scientific difficulty of exact earthquake prediction. The reason is that earthquake processes involve complex stress accumulation and fracture dynamics, and small changes in conditions may influence whether a rupture grows into a larger event. This literature supports the cautious framing of the present study.

Machine learning has increasingly been applied to earthquake-related problems, including magnitude estimation, aftershock forecasting, event detection, and seismic signal classification. Mignan and Broccardo (2019) reviewed neural network applications in earthquake prediction and highlighted both opportunities and limitations. Their work indicates that AI models can be useful, but they must be evaluated carefully because earthquake datasets are often imbalanced, noisy, and region-dependent.

Random Forest is a widely used machine learning algorithm for tabular classification problems. Breiman (2001) introduced Random Forests as an ensemble of decision trees that improves predictive performance by combining multiple tree-based learners. In this project, Random Forest is selected because it can capture non-linear relationships, handle numerical features, and work effectively as a baseline model for structured earthquake catalog data.

The literature also suggests that earthquake forecasting should be evaluated with realistic validation techniques. Random data splitting can create leakage when future events influence training. Therefore, chronological splitting is used in this project. The model is trained on earlier time periods and tested on later time periods. This is more consistent with the real forecasting problem, where future data is not available during training.

# Chapter 3: Research Objectives and Methodology

## Research Objectives

- To collect and analyze real-life earthquake catalog data for the Asia region from the USGS earthquake database.
- To perform exploratory data analysis on earthquake magnitude, depth, frequency, and spatial distribution.
- To develop a supervised machine learning model for short-term earthquake risk forecasting using engineered spatio-temporal features.
- To evaluate model performance using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix.
- To identify the limitations of using historical earthquake catalog data for forecasting significant earthquake events.

## Research Problem

The research problem is to determine whether historical earthquake activity in a geographic grid cell can be used to forecast the occurrence of a significant earthquake in the next time window. The prediction target is whether an earthquake of magnitude 5.0 or above occurs in the next seven-day period for the same grid cell.

## Study Hypotheses

Null hypothesis: Historical earthquake catalog features do not provide meaningful predictive information for forecasting significant earthquake occurrence in the next time window.

Alternative hypothesis: Historical earthquake catalog features provide meaningful predictive information for forecasting significant earthquake occurrence in the next time window.

## Research Design

The research design is descriptive and empirical. It is descriptive because the study summarizes real earthquake data using statistics and visualizations. It is empirical because a machine learning model is trained and evaluated on real historical data.

## Type of Data Used

The project uses secondary quantitative data. The data consists of earthquake records collected by the USGS earthquake catalog.

## Data Collection Method

Data is collected using the USGS FDSN Event Web Service API. The project script requests earthquake events in CSV format for the Asia study boundary.

## Data Collection Instrument

The data collection instrument is a Python-based API downloader implemented in the project. It uses the `requests` library to fetch data and `pandas` to store and process it.

## Sample Size

The real-life dataset contains 78,686 earthquake records for Asia from 2014-01-01 to 2026-05-25 with magnitude 4.0 and above. After feature engineering, the machine learning dataset contains 45,413 rows.

## Sampling Technique

The study uses purposive regional sampling. The geographic boundary is selected to represent Asia and nearby seismically active zones.

## Data Analysis Tool

The analysis tools are Python, pandas, matplotlib, scikit-learn, joblib, and YAML configuration files.

# Chapter 4: Data Analysis, Results, and Interpretation

## Dataset Configuration

Table 1: Dataset configuration for Asia earthquake study

| Field | Value |
|---|---|
| Region | Asia |
| Start date | 2014-01-01 |
| End date | 2026-05-26 query, latest event in dataset 2026-05-25 |
| Minimum magnitude | 4.0 |
| Latitude boundary | -10 to 60 |
| Longitude boundary | 25 to 150 |
| Forecast window | 7 days |
| Target earthquake threshold | Magnitude 5.0 or above |
| Grid size | 2.0 degrees |

## Summary Statistics

Table 2: Summary statistics of real earthquake data

| Metric | Value |
|---|---:|
| Total events | 78,686 |
| Start date | 2014-01-01 |
| End date | 2026-05-25 |
| Minimum magnitude | 4.0 |
| Maximum magnitude | 7.8 |
| Mean magnitude | 4.49 |
| Mean depth | 68.97 km |
| Deepest event | 686.39 km |

The dataset shows a large number of moderate earthquakes across Asia. The mean magnitude of 4.49 indicates that most events are moderate rather than extreme. The maximum magnitude of 7.8 confirms that the dataset includes major earthquake events. The mean depth of 68.97 km suggests that the region includes both shallow and deeper seismic activity.

## Exploratory Data Analysis

Figure files generated by the project are stored in `reports/figures`.

Figure 1, `magnitude_distribution.png`, shows the distribution of earthquake magnitudes. The distribution is expected to be right-skewed because smaller events occur more frequently than larger events. This is consistent with known earthquake frequency-magnitude behavior.

Figure 2, `monthly_event_trend.png`, shows how the number of recorded events changes over time. This helps identify months or periods with increased seismic activity. However, short-term increases should not be interpreted as exact precursors without additional geophysical evidence.

Figure 3, `depth_vs_magnitude.png`, compares earthquake depth and magnitude. This helps examine whether deeper earthquakes in the dataset are associated with different magnitude patterns. The scatter plot supports exploratory interpretation but does not alone prove causality.

Figure 4, `magnitude_class_counts.png`, groups events by magnitude class. It shows that moderate earthquakes are much more common than high-magnitude earthquakes, which explains the class imbalance in the machine learning target.

Figure 5, `asia_event_map.png`, plots earthquake latitude and longitude across the selected Asia boundary. Dense event clusters are expected around tectonically active belts such as Indonesia, Japan, the Himalayas, and neighboring seismic zones.

Figure 6, `feature_importance.png`, shows the relative feature importance values learned by the Random Forest classifier. This helps identify which engineered variables contributed more strongly to the model's decision-making process.

## Machine Learning Results

Table 3: Machine learning model evaluation metrics

| Metric | Value |
|---|---:|
| Feature rows | 45,413 |
| Training rows | 34,059 |
| Testing rows | 11,354 |
| Positive target rate | 0.1259 |
| Accuracy | 0.8097 |
| ROC-AUC | 0.5448 |

Table 4: Model comparison results

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.5524 | 0.5803 |
| Histogram Gradient Boosting | 0.8763 | 0.5802 |
| Random Forest | 0.8097 | 0.5448 |

The model comparison shows that Histogram Gradient Boosting achieved the highest accuracy and ROC-AUC among the tested models. Logistic Regression achieved lower accuracy but a similar ROC-AUC, which indicates that threshold selection and class imbalance strongly affect the final classification output. Random Forest was retained as the main saved model because it provides feature importance values and is easier to explain for academic evaluation.

Table 5: Confusion matrix of Random Forest classifier

| Actual / Predicted | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 9,007 | 936 |
| Actual 1 | 1,225 | 186 |

The Random Forest model achieved 80.97 percent accuracy, but accuracy alone is not sufficient because the dataset is imbalanced. Only 12.59 percent of rows belong to the positive class, meaning that significant future-event windows are much less frequent than non-event windows. The ROC-AUC score of 0.5448 is only slightly above random classification. This indicates that the selected historical catalog features provide weak but limited predictive signal.

The confusion matrix shows that the model correctly classified many non-event cases but missed many positive cases. The recall for the positive class is low. This is an important result because it shows that earthquake risk forecasting cannot be considered solved using only simple historical catalog features. The model may improve if additional features are added, such as tectonic plate boundary distance, fault-line information, seismic energy release, regional clustering, and geological variables.

# Chapter 5: Findings and Conclusion

The first major finding is that Asia has a high volume of earthquake activity. The collected USGS dataset contains 78,686 earthquake events of magnitude 4.0 and above between 2014 and 2026. This confirms that Asia is a suitable region for seismic data analysis because it contains diverse and active earthquake zones.

The second finding is that most earthquakes in the dataset are moderate magnitude events. The mean magnitude is 4.49, while the maximum recorded magnitude in the selected dataset is 7.8. This creates a natural class imbalance because high-magnitude or significant future events are much less common than non-event windows.

The third finding is that historical earthquake catalog features alone have limited ability to forecast significant future earthquakes. The Random Forest model achieved 80.97 percent accuracy but only 0.5448 ROC-AUC and low recall for the positive class. This means that the model is much better at identifying non-event cases than identifying future significant-event cases.

The study concludes that machine learning can be used to build a structured earthquake risk analysis workflow, but exact earthquake prediction is not supported by this dataset or model. The project is useful as an AI/ML application because it demonstrates real data collection, exploratory analysis, feature engineering, classification, and interpretation. At the same time, the results support the scientific view that earthquake forecasting requires more than public historical catalog data.

# Chapter 6: Recommendations and Limitations of the Study

## Recommendations

1. Add tectonic plate boundary distance as a feature.
2. Add known fault-line proximity features.
3. Include seismic energy release calculations.
4. Train separate models for sub-regions such as Japan, Indonesia, Himalayas, and Turkey.
5. Compare Random Forest with XGBoost, Logistic Regression, SVM, and neural network models.
6. Use precision-recall curves because earthquake forecasting data is imbalanced.
7. Apply threshold tuning to improve positive-class recall.
8. Use longer historical data if consistent catalog quality is available.
9. Add map-based visualizations for better interpretation.
10. Build a Streamlit dashboard for interactive analysis.
11. Include feature importance analysis to identify the most useful predictors.
12. Use cross-validation based on time periods rather than random splitting.
13. Add earthquake depth categories as separate features.
14. Use regional geological datasets along with earthquake catalog data.
15. Clearly communicate that model outputs are educational risk scores, not official warnings.

## Limitations of the Study

1. The model does not predict the exact date, location, or magnitude of future earthquakes.
2. The dataset includes only public USGS catalog features and does not include geological stress measurements.
3. The target class is imbalanced because significant earthquakes are relatively rare.
4. The selected grid size and time window affect model performance.
5. The Asia boundary is broad and includes different tectonic environments.
6. The model may perform differently in specific sub-regions.
7. Historical earthquake patterns may not fully represent future seismic behavior.
8. The model has low recall for significant future-event cases.
9. The study is intended for academic analysis and not disaster warning.
10. Additional domain expertise is required for operational seismic hazard assessment.

# Bibliography / References

Breiman, L. (2001). Random forests. Machine Learning, 45, 5-32.

Geller, R. J., Jackson, D. D., Kagan, Y. Y., & Mulargia, F. (1997). Earthquakes cannot be predicted. Science, 275(5306), 1616-1617.

Mignan, A., & Broccardo, M. (2019). Neural network applications in earthquake prediction (1994-2019): Meta-analytic insight on their limitations. arXiv. https://arxiv.org/abs/1910.01178

United States Geological Survey. (n.d.). Can you predict earthquakes? https://www.usgs.gov/faqs/can-you-predict-earthquakes

United States Geological Survey. (n.d.). FDSN Event Web Service. https://earthquake.usgs.gov/fdsnws/event/1/

United States Geological Survey. (n.d.). Earthquake CSV feeds and formats. https://earthquake.usgs.gov/earthquakes/feed/v1.0/csv.php

# Appendix

## Appendix A: Important Project Commands

```powershell
python -m src.cli download
python -m src.cli analyze
python -m src.cli build-features
python -m src.cli train
python -m src.cli predict --latitude 35.68 --longitude 139.69
```

## Appendix B: Generated Output Files

- `data/raw/earthquakes.csv`
- `data/processed/features.csv`
- `reports/asia_earthquake_summary.csv`
- `reports/metrics.txt`
- `reports/figures/magnitude_distribution.png`
- `reports/figures/monthly_event_trend.png`
- `reports/figures/depth_vs_magnitude.png`
- `reports/figures/magnitude_class_counts.png`
- `reports/figures/asia_event_map.png`
- `models/earthquake_risk_model.joblib`
