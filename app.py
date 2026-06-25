import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = "Credit_Card_Approval_10000_70_30.csv"
TARGET = "approval_status"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def train_model(df: pd.DataFrame):
    work_df = df.copy()
    work_df = work_df.drop(columns=["application_id", "application_date"], errors="ignore")
    work_df[TARGET] = work_df[TARGET].map({"Rejected": 0, "Approved": 1})
    work_df = work_df.dropna(subset=[TARGET])

    X = work_df.drop(columns=[TARGET])
    y = work_df[TARGET]

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(n_estimators=250, max_depth=None, random_state=42),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "auc": roc_auc_score(y_test, prob),
        "feature_columns": X.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "categorical_choices": {
            col: sorted(df[col].dropna().astype(str).unique().tolist()) for col in categorical_cols
        },
    }
    return model, metrics


def main():
    st.set_page_config(page_title="Credit Card Approval", page_icon="💳", layout="wide")

    st.title("Credit Card Approval Predictor")
    st.caption("Interactive model app for your Streamlit deployment")

    try:
        df = load_data(DATA_PATH)
    except FileNotFoundError:
        st.error("Dataset file not found. Ensure `Credit_Card_Approval_10000_70_30.csv` is in the repo root.")
        return

    model, metrics = train_model(df)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Validation Accuracy", f"{metrics['accuracy']:.3f}")
    with col2:
        st.metric("Validation ROC AUC", f"{metrics['auc']:.3f}")

    st.subheader("Enter applicant details")

    input_payload = {}
    form_cols = st.columns(3)

    num_i = 0
    cat_i = 0

    for col in metrics["feature_columns"]:
        if col in metrics["numeric_columns"]:
            median_val = float(df[col].median()) if col in df.columns else 0.0
            min_val = float(df[col].min()) if col in df.columns and pd.notna(df[col].min()) else 0.0
            max_val = float(df[col].max()) if col in df.columns and pd.notna(df[col].max()) else 100.0
            with form_cols[num_i % 3]:
                input_payload[col] = st.number_input(
                    col.replace("_", " ").title(),
                    value=median_val,
                    min_value=min_val,
                    max_value=max_val,
                )
            num_i += 1
        else:
            choices = metrics["categorical_choices"].get(col, ["Unknown"])
            with form_cols[cat_i % 3]:
                input_payload[col] = st.selectbox(
                    col.replace("_", " ").title(),
                    options=choices,
                )
            cat_i += 1

    if st.button("Predict Approval", type="primary"):
        pred_df = pd.DataFrame([input_payload])
        pred_class = int(model.predict(pred_df)[0])
        pred_prob = float(model.predict_proba(pred_df)[0][1])

        if pred_class == 1:
            st.success(f"Prediction: Approved (probability: {pred_prob:.2%})")
        else:
            st.error(f"Prediction: Rejected (probability of approval: {pred_prob:.2%})")

        st.write("Input used for prediction")
        st.dataframe(pred_df, use_container_width=True)

    with st.expander("View sample dataset"):
        st.dataframe(df.head(20), use_container_width=True)


if __name__ == "__main__":
    main()
