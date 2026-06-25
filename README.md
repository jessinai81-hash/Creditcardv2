# Credit Card Approval App

This project now includes:

- `app.py`: Streamlit web app for interactive credit-card approval prediction.
- `train_pyspark.py`: PySpark training script for local Spark experimentation.
- `Credit_Card_Approval_10000_70_30.csv`: dataset used by both scripts.

## Run Streamlit app locally

```powershell
# Activate venv (if not active)
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -r requirements.txt

# Start app
streamlit run app.py
```

## Streamlit Cloud setup

- Repository: `jessinai81-hash/Creditcardv2`
- Main file path: `app.py`
- Python version: default is fine
- `requirements.txt` is used automatically

## Optional: Run PySpark trainer

```powershell
python -m pip install -r requirements-pyspark.txt
python train_pyspark.py "Credit_Card_Approval_10000_70_30.csv" models/pyspark_rf_model
```

If running PySpark on Windows, Java and Hadoop/winutils setup may be required.
