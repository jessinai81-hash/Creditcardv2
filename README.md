# PySpark Credit Card Approval Model

This workspace contains a PySpark training script that builds a Random Forest classifier to predict `approval_status` from `Credit_Card_Approval_10000_70_30.csv`.

Quick start (PowerShell, from workspace root):

```powershell
# Activate venv (if not active)
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -r requirements.txt

# Train
python train_pyspark.py "Credit_Card_Approval_10000_70_30.csv" models/pyspark_rf_model
```

Model artifacts will be saved under `models/pyspark_rf_model/pipeline`.
