import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score, 
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
import seaborn as sns

st.set_page_config(page_title="💳 Credit Card Churn Prediction", layout="wide")

st.title("💳 Credit Card Churn Prediction App")

# -----------------------------
# Constants
# -----------------------------
TARGET_COL = "Exited"

NUM_COLS = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "EstimatedSalary",
    "HasCrCard", "IsActiveMember"
]

CAT_COLS = ["Geography", "Gender"]

# -----------------------------
# Model Loader
# -----------------------------
@st.cache_resource
def load_model(model_name):
    return joblib.load(f"models/{model_name}.pkl")

# -----------------------------
# Sidebar - Model Selection
# -----------------------------
st.sidebar.header("⚙️ Settings")

model_name = st.sidebar.selectbox(
    "Select Model",
    ["logistic_regression", "random_forest", "XGBoost"]
)

model = load_model(model_name)

# -----------------------------
# Dataset Upload
# -----------------------------
st.subheader("📂 Upload Test Dataset (CSV)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("### 🔍 Preview of Uploaded Data")
    st.dataframe(df.head())

    # -----------------------------
    # Validation
    # -----------------------------
    missing_cols = set(NUM_COLS + CAT_COLS + [TARGET_COL]) - set(df.columns)
    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
        st.stop()

    X_test = df[NUM_COLS + CAT_COLS]
    y_test = df[TARGET_COL]

    # Force correct dtypes (CRITICAL)
    X_test[NUM_COLS] = X_test[NUM_COLS].astype(float)
    X_test[["HasCrCard", "IsActiveMember"]] = X_test[
        ["HasCrCard", "IsActiveMember"]
    ].astype(int)

    # -----------------------------
    # Prediction
    # -----------------------------
    y_pred = model.predict(X_test)

    # -----------------------------
    # Evaluation Metrics
    # -----------------------------
    st.subheader("📊 Evaluation Metrics")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
    col4.metric("F1 Score", f"{f1_score(y_test, y_pred):.3f}")
    col5.metric("ROC AUC", f"{roc_auc_score(y_test, y_pred):.3f}")
    col6.metric("MCC AUC", f"{matthews_corrcoef(y_test, y_pred):.3f}")


    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    st.subheader("🧩 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # -----------------------------
    # Classification Report
    # -----------------------------
    st.subheader("📄 Classification Report")
    st.text(classification_report(y_test, y_pred))
