# 📊 Customer Churn Prediction System

A production-ready **deep learning application** that predicts whether a customer will **stay** or **churn**, built with **TensorFlow/Keras**, **Scikit-learn**, and an interactive **Streamlit** dashboard.

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🚀 Overview

This project implements an end-to-end machine learning pipeline for customer churn prediction:

- Automatic **dataset loading** (Telco Customer Churn) with **synthetic data generation** fallback (10,000+ realistic records) so the project runs with zero manual setup.
- A modular, testable **preprocessing pipeline** (cleaning, encoding, scaling).
- A **TensorFlow/Keras neural network** with batch normalization, dropout regularization, early stopping, and model checkpointing.
- A full **evaluation suite**: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and classification report.
- A polished, **dark-mode friendly Streamlit dashboard** for live predictions and model monitoring.

---

## ✨ Features

### Data Processing
- Automatic dataset loading or synthetic data generation
- Missing value imputation (median for numerical, mode for categorical)
- Duplicate removal
- Categorical encoding (label encoding, persisted for inference)
- Numerical feature normalization (standard scaling, persisted for inference)
- Stratified train/test split

### Deep Learning Model
- Sequential neural network:
  `Input → Dense(128, ReLU) → BatchNorm → Dropout → Dense(64, ReLU) → Dropout → Dense(32, ReLU) → Dense(1, Sigmoid)`
- Adam optimizer, binary cross-entropy loss
- EarlyStopping + ModelCheckpoint + ReduceLROnPlateau callbacks
- Validation split during training

### Evaluation
- Accuracy, Precision, Recall, F1 Score, ROC-AUC
- Confusion matrix heatmap
- ROC curve
- Full classification report
- All charts auto-saved to `assets/`

### Streamlit Dashboard
- Sidebar navigation across 5 pages (Overview, Predict, Performance, Dataset Explorer, About)
- One-click **Train / Retrain Model** button
- Live customer input form with instant prediction
- Confidence score + churn probability gauge chart
- Dataset preview and summary statistics
- Model performance dashboard with training curves, confusion matrix, and ROC curve
- Dark-mode friendly custom styling

---

## 🛠️ Technologies

| Category            | Tools                                  |
|---------------------|-----------------------------------------|
| Language             | Python 3.12+                           |
| Deep Learning        | TensorFlow, Keras                      |
| Data Processing      | Pandas, NumPy, Scikit-learn            |
| Visualization        | Matplotlib                             |
| Web App              | Streamlit                              |

---

## 📁 Folder Structure

```
Customer-Churn-Prediction/
│
├── app.py                    # Streamlit dashboard entry point
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── data/
│   └── customer_churn.csv    # Auto-generated / loaded dataset
│
├── models/
│   ├── churn_model.keras     # Trained model
│   ├── scaler.pkl            # Fitted StandardScaler
│   ├── encoders.pkl          # Fitted LabelEncoders
│   ├── feature_columns.pkl   # Feature ordering for inference
│   ├── history.pkl           # Training history
│   └── metrics.pkl           # Evaluation metrics
│
├── assets/                   # Saved charts (accuracy, loss, ROC, confusion matrix)
│
├── notebooks/                # Optional exploratory notebooks
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Dataset loading / synthetic generation
│   ├── preprocessing.py      # Cleaning, encoding, scaling, splitting
│   ├── model.py               # Neural network architecture
│   ├── trainer.py            # Training loop with callbacks
│   ├── evaluator.py          # Metrics computation
│   ├── predictor.py          # Single-customer inference
│   ├── visualization.py      # Matplotlib chart generation
│   └── utils.py              # Logging, paths, configuration
│
├── scripts/
│   └── train.py              # CLI entry point for the full training pipeline
│
└── screenshots/               # App screenshots for documentation
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Option A — Train via command line, then launch the app
```bash
python scripts/train.py
streamlit run app.py
```

### Option B — Train directly from the Streamlit sidebar
```bash
streamlit run app.py
```
Then click **🚀 Train / Retrain Model** in the sidebar. The dataset will be generated automatically if it doesn't already exist, and the model will train, evaluate, and save itself — no manual steps required.

The app will be available at **http://localhost:8501**.

---

## 📊 Dataset

By default, the project uses (or generates) a dataset modeled on the well-known **Telco Customer Churn** dataset, including features such as:

- Demographics: `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `Age`
- Account info: `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`
- Services: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `TechSupport`, `StreamingTV`
- Billing: `MonthlyCharges`, `TotalCharges`
- Target: `Churn` (Yes / No)

If `data/customer_churn.csv` is not found, a realistic **synthetic dataset of 10,000+ records** is generated automatically with statistically grounded churn logic (e.g., month-to-month contracts and electronic-check payments correlate with higher churn, consistent with real-world churn research).

---

## 🖼️ Screenshots

> Add your own screenshots to the `screenshots/` folder and reference them here, e.g.:
>
> ![Dashboard Overview](screenshots/2DB.png)
> ![Prediction Page](screenshots/1home.png)

---

## 🔮 Future Improvements

- [ ] Hyperparameter tuning with Keras Tuner / Optuna
- [ ] SHAP-based model explainability (feature importance per prediction)
- [ ] Batch prediction via CSV upload
- [ ] REST API endpoint (FastAPI) alongside the Streamlit UI
- [ ] Dockerfile for containerized deployment
- [ ] CI pipeline (GitHub Actions) for linting and automated tests
- [ ] Support for real Telco Customer Churn dataset ingestion from Kaggle API

---

## ☁️ Deployment

### Local Execution
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Community Cloud
1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select this repository and branch, and set the main file path to `app.py`.
4. Click **Deploy**. Streamlit Cloud will install dependencies from `requirements.txt` automatically.
5. On first load, use the sidebar **Train / Retrain Model** button to generate the dataset and train the model (or commit a pretrained `models/` directory to skip this step).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
