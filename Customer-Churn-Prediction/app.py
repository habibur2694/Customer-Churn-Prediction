"""
app.py
======
Streamlit dashboard for the Customer Churn Prediction System.

Provides:
    - A customer input form with live prediction (Stay / Churn)
    - Confidence score and churn probability gauge
    - Dataset preview and summary statistics
    - Model performance dashboard (accuracy, precision, recall, F1, ROC-AUC)
    - Training accuracy / loss curves
    - Confusion matrix and ROC curve visualizations

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_dataset
from src.evaluator import evaluate_model
from src.predictor import load_trained_model, predict_customer
from src.preprocessing import prepare_dataset
from src.trainer import train_model
from src.utils import (
    DATA_PATH,
    HISTORY_PATH,
    METRICS_PATH,
    MODEL_PATH,
    FeatureConfig,
    ensure_directories,
    get_logger,
)
from src.visualization import (
    plot_confusion_matrix,
    plot_prediction_probability,
    plot_roc_curve,
    plot_training_history,
)

logger = get_logger(__name__)
CONFIG = FeatureConfig()

# --------------------------------------------------------------------------- #
# Page configuration & styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main { background-color: #0e1117; }
    .stMetric {
        background-color: #1b1e2b;
        border-radius: 12px;
        padding: 12px;
        border: 1px solid #2c2f3e;
    }
    div[data-testid="stMetricValue"] { color: #f5f5f5; }
    .result-card {
        padding: 1.5rem;
        border-radius: 14px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 1rem;
    }
    .stay { background: linear-gradient(135deg, #1e5631, #2ecc71); color: white; }
    .churn { background: linear-gradient(135deg, #7a1f1f, #e74c3c); color: white; }
    section[data-testid="stSidebar"] { background-color: #12141c; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Cached data / resource loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    """Load (or generate) the raw dataset, cached across reruns."""
    ensure_directories()
    return load_dataset(path=DATA_PATH, n_records=10_000)


@st.cache_resource(show_spinner=False)
def get_model():
    """Load the trained model, cached as a resource across reruns."""
    return load_trained_model()


def model_artifacts_exist() -> bool:
    """Check whether a trained model and its metrics already exist on disk."""
    return os.path.exists(MODEL_PATH) and os.path.exists(METRICS_PATH) and os.path.exists(HISTORY_PATH)


def run_full_training_pipeline(progress_callback=None) -> None:
    """Run the full data prep -> train -> evaluate pipeline from within the app."""
    df = get_dataset()
    if progress_callback:
        progress_callback(0.15, "Cleaning and encoding data ...")
    X_train, X_test, y_train, y_test, _ = prepare_dataset(df, CONFIG)

    if progress_callback:
        progress_callback(0.35, "Training neural network (this may take a minute) ...")
    model, history = train_model(X_train, y_train, epochs=60, batch_size=32, patience=8)

    if progress_callback:
        progress_callback(0.85, "Evaluating model and generating charts ...")
    metrics = evaluate_model(model, X_test, y_test)
    plot_training_history(history.history)
    plot_confusion_matrix(metrics["confusion_matrix"])
    plot_roc_curve(metrics["fpr"], metrics["tpr"], metrics["roc_auc"])

    if progress_callback:
        progress_callback(1.0, "Done!")

    st.cache_resource.clear()


def load_metrics() -> dict:
    with open(METRICS_PATH, "rb") as f:
        return pickle.load(f)


def load_history() -> dict:
    with open(HISTORY_PATH, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- #
# Sidebar navigation
# --------------------------------------------------------------------------- #
st.sidebar.title("📊 Churn Predictor")
st.sidebar.caption("AI-powered Customer Retention Insights")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "🔮 Predict Churn", "📈 Model Performance", "📂 Dataset Explorer", "ℹ️ About"],
)

st.sidebar.divider()
st.sidebar.subheader("Model Status")

if model_artifacts_exist():
    st.sidebar.success("Trained model found ✅")
else:
    st.sidebar.warning("No trained model found yet.")

if st.sidebar.button("🚀 Train / Retrain Model", use_container_width=True):
    progress_bar = st.sidebar.progress(0.0, text="Starting training pipeline ...")

    def _update(value: float, text: str) -> None:
        progress_bar.progress(value, text=text)

    with st.spinner("Training in progress ..."):
        run_full_training_pipeline(progress_callback=_update)
    st.sidebar.success("Training complete! Reloading model ...")
    st.rerun()

st.sidebar.divider()
st.sidebar.caption("Built with TensorFlow, Keras & Streamlit")


# --------------------------------------------------------------------------- #
# Page: Overview
# --------------------------------------------------------------------------- #
if page == "🏠 Overview":
    st.title("📊 Customer Churn Prediction System")
    st.markdown(
        "A production-ready deep learning application that predicts whether a "
        "customer is likely to **stay** or **churn**, built with TensorFlow/Keras "
        "and served through an interactive Streamlit dashboard."
    )

    df = get_dataset()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(df):,}")
    churn_rate = (df["Churn"] == "Yes").mean() * 100
    col2.metric("Churn Rate", f"{churn_rate:.1f}%")
    col3.metric("Avg. Tenure (months)", f"{df['tenure'].mean():.1f}")
    col4.metric("Avg. Monthly Charges", f"${df['MonthlyCharges'].mean():.2f}")

    st.subheader("What this app does")
    st.markdown(
        """
        - **Predict** whether an individual customer will stay or churn, with a confidence score.
        - **Explore** the underlying dataset and its key statistics.
        - **Monitor** model performance through accuracy, precision, recall, F1, and ROC-AUC.
        - **Retrain** the model at any time directly from the sidebar.
        """
    )

    if not model_artifacts_exist():
        st.info("👈 No trained model yet. Click **Train / Retrain Model** in the sidebar to get started.")


# --------------------------------------------------------------------------- #
# Page: Predict Churn
# --------------------------------------------------------------------------- #
elif page == "🔮 Predict Churn":
    st.title("🔮 Predict Customer Churn")

    if not model_artifacts_exist():
        st.warning("No trained model found. Please train the model first using the sidebar button.")
        st.stop()

    st.markdown("Enter the customer's details below to generate a live churn prediction.")

    with st.form("customer_form"):
        st.subheader("Customer Profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            partner = st.selectbox("Has Partner", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        with c2:
            age = st.slider("Age", 18, 90, 35)
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=780.0, step=10.0)
        with c3:
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

        st.subheader("Services")
        s1, s2, s3 = st.columns(3)
        with s1:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        with s2:
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        with s3:
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])

        submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

    if submitted:
        num_services = sum(
            [
                phone_service == "Yes",
                internet_service != "No",
                online_security == "Yes",
                tech_support == "Yes",
                streaming_tv == "Yes",
            ]
        )

        record = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "Age": age,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "NumServices": num_services,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        try:
            label, confidence, churn_probability = predict_customer(record, CONFIG)

            result_col, chart_col = st.columns([1, 1.4])
            with result_col:
                if label == "Stay":
                    st.markdown(
                        f'<div class="result-card stay">✅ Prediction: Stay<br>'
                        f'Confidence: {confidence * 100:.1f}%</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="result-card churn">❌ Prediction: Churn<br>'
                        f'Confidence: {confidence * 100:.1f}%</div>',
                        unsafe_allow_html=True,
                    )
                st.metric("Churn Probability", f"{churn_probability * 100:.2f}%")
                st.metric("Retention Probability", f"{(1 - churn_probability) * 100:.2f}%")

            with chart_col:
                fig = plot_prediction_probability(churn_probability)
                st.pyplot(fig, use_container_width=True)

        except Exception as exc:  # noqa: BLE001
            logger.exception("Prediction failed")
            st.error(f"Prediction failed: {exc}")


# --------------------------------------------------------------------------- #
# Page: Model Performance
# --------------------------------------------------------------------------- #
elif page == "📈 Model Performance":
    st.title("📈 Model Performance Dashboard")

    if not model_artifacts_exist():
        st.warning("No trained model found. Please train the model first using the sidebar button.")
        st.stop()

    metrics = load_metrics()
    history = load_history()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    m2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
    m3.metric("Recall", f"{metrics['recall'] * 100:.2f}%")
    m4.metric("F1 Score", f"{metrics['f1_score'] * 100:.2f}%")
    m5.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")

    st.divider()
    st.subheader("Training Curves")
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        fig_acc, _ = plot_training_history(history, save=False)
        st.pyplot(fig_acc, use_container_width=True)
    with tcol2:
        _, fig_loss = plot_training_history(history, save=False)
        st.pyplot(fig_loss, use_container_width=True)

    st.divider()
    st.subheader("Confusion Matrix & ROC Curve")
    ccol1, ccol2 = st.columns(2)
    with ccol1:
        fig_cm = plot_confusion_matrix(metrics["confusion_matrix"], save=False)
        st.pyplot(fig_cm, use_container_width=True)
    with ccol2:
        fig_roc = plot_roc_curve(metrics["fpr"], metrics["tpr"], metrics["roc_auc"], save=False)
        st.pyplot(fig_roc, use_container_width=True)

    st.divider()
    st.subheader("Classification Report")
    st.code(metrics["classification_report"], language="text")


# --------------------------------------------------------------------------- #
# Page: Dataset Explorer
# --------------------------------------------------------------------------- #
elif page == "📂 Dataset Explorer":
    st.title("📂 Dataset Explorer")

    df = get_dataset()
    st.markdown(f"Dataset shape: **{df.shape[0]:,} rows × {df.shape[1]} columns**")

    st.subheader("Preview")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

    st.subheader("Churn Distribution")
    churn_counts = df["Churn"].value_counts()
    st.bar_chart(churn_counts)

    st.subheader("Numerical Feature Distributions")
    numeric_df = df.select_dtypes(include=[np.number])
    selected_col = st.selectbox("Select a numerical feature", numeric_df.columns)
    st.bar_chart(numeric_df[selected_col].value_counts(bins=20).sort_index())


# --------------------------------------------------------------------------- #
# Page: About
# --------------------------------------------------------------------------- #
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown(
        """
        ### Customer Churn Prediction System

        A production-ready machine learning application for predicting
        customer churn, built as part of an AI/ML engineering portfolio.

        **Technology Stack**
        - Python 3.12+
        - TensorFlow 2.x / Keras
        - Scikit-learn
        - Pandas & NumPy
        - Matplotlib
        - Streamlit

        **Model Architecture**
        - Dense(128, ReLU) → BatchNorm → Dropout
        - Dense(64, ReLU) → Dropout
        - Dense(32, ReLU)
        - Dense(1, Sigmoid)

        See the project `README.md` for full documentation, installation
        instructions, and deployment guidance.
        """
    )
