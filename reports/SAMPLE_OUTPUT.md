# Sample Output

After training, `reports/metrics.txt` will contain output similar to:

```text
Earthquake Risk Model Metrics
=============================
Rows: 12500
Train rows: 9375
Test rows: 3125
Positive target rate: 0.0824
Accuracy: 0.8912
ROC-AUC: 0.7435
Confusion matrix: [[2701, 168], [172, 84]]
```

Prediction command:

```powershell
python -m src.cli predict --latitude 35.68 --longitude 139.69
```

Example prediction output:

```json
{
  "latitude": 35.68,
  "longitude": 139.69,
  "probability": 0.42,
  "risk": "Medium"
}
```

The probability is an educational risk score for the configured forecast window. It is not an official earthquake warning.
