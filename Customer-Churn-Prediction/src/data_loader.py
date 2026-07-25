

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.utils import DATA_PATH, RANDOM_STATE, ensure_directories, get_logger

logger = get_logger(__name__)


def _generate_synthetic_dataset(n_records: int = 10_000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Generate a realistic synthetic customer churn dataset.

    The distributions and churn logic are designed to mimic the
    well-known Telco Customer Churn dataset so that downstream
    preprocessing, modeling, and evaluation code behaves realistically.

    Args:
        n_records: Number of customer records to generate.
        seed: Random seed for reproducibility.

    Returns:
        A pandas DataFrame containing the synthetic dataset.
    """
    rng = np.random.default_rng(seed)
    logger.info("Generating synthetic dataset with %d records ...", n_records)

    gender = rng.choice(["Male", "Female"], size=n_records)
    senior_citizen = rng.choice([0, 1], size=n_records, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n_records, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n_records, p=[0.30, 0.70])
    age = rng.integers(18, 80, size=n_records)

    tenure = rng.integers(0, 73, size=n_records)
    phone_service = rng.choice(["Yes", "No"], size=n_records, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["Yes", "No"], size=n_records),
    )
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], size=n_records, p=[0.35, 0.44, 0.21]
    )
    online_security = np.where(
        internet_service == "No",
        "No internet service",
        rng.choice(["Yes", "No"], size=n_records),
    )
    tech_support = np.where(
        internet_service == "No",
        "No internet service",
        rng.choice(["Yes", "No"], size=n_records),
    )
    streaming_tv = np.where(
        internet_service == "No",
        "No internet service",
        rng.choice(["Yes", "No"], size=n_records),
    )
    num_services = (
        (phone_service == "Yes").astype(int)
        + (internet_service != "No").astype(int)
        + (online_security == "Yes").astype(int)
        + (tech_support == "Yes").astype(int)
        + (streaming_tv == "Yes").astype(int)
    )

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n_records, p=[0.55, 0.24, 0.21]
    )
    paperless_billing = rng.choice(["Yes", "No"], size=n_records, p=[0.59, 0.41])
    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        size=n_records,
    )

    base_charge = np.select(
        [internet_service == "Fiber optic", internet_service == "DSL"],
        [70.0, 45.0],
        default=20.0,
    )
    monthly_charges = np.round(
        base_charge + num_services * rng.uniform(2, 8, size=n_records) + rng.normal(0, 5, size=n_records),
        2,
    )
    monthly_charges = np.clip(monthly_charges, 18.0, 120.0)
    total_charges = np.round(monthly_charges * tenure + rng.normal(0, 20, size=n_records), 2)
    total_charges = np.clip(total_charges, 0, None)

    # Construct a churn probability that depends on realistic risk factors.
    churn_score = (
        0.35 * (contract == "Month-to-month")
        + 0.15 * (payment_method == "Electronic check")
        + 0.20 * (internet_service == "Fiber optic")
        - 0.25 * (tenure / 72)
        - 0.15 * (contract == "Two year")
        + 0.10 * (paperless_billing == "Yes")
        + 0.10 * (online_security == "No")
        - 0.10 * (partner == "Yes")
        + rng.normal(0, 0.15, size=n_records)
    )
    churn_prob = 1 / (1 + np.exp(-6 * (churn_score - churn_score.mean())))
    churn = rng.binomial(1, np.clip(churn_prob, 0.02, 0.95))
    churn_labels = np.where(churn == 1, "Yes", "No")

    df = pd.DataFrame(
        {
            "customerID": [f"CUST-{i:06d}" for i in range(n_records)],
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
            "Churn": churn_labels,
        }
    )

    # Inject a small amount of realistic missingness and a few duplicate
    # rows so the preprocessing pipeline has genuine cleaning work to do.
    missing_idx = rng.choice(n_records, size=int(n_records * 0.01), replace=False)
    df.loc[missing_idx, "TotalCharges"] = np.nan
    dup_rows = df.sample(n=int(n_records * 0.005), random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)

    logger.info("Synthetic dataset generated: %d rows, %d columns.", *df.shape)
    return df


def load_dataset(path: str = DATA_PATH, n_records: int = 10_000, force_regenerate: bool = False) -> pd.DataFrame:
    """
    Load the customer churn dataset from disk, generating a synthetic
    dataset automatically if the file does not already exist.

    Args:
        path: File path of the CSV dataset.
        n_records: Number of rows to generate if creating a synthetic dataset.
        force_regenerate: If True, always regenerate the synthetic dataset.

    Returns:
        A pandas DataFrame containing the raw dataset.
    """
    ensure_directories()

    if os.path.exists(path) and not force_regenerate:
        logger.info("Loading existing dataset from %s", path)
        df = pd.read_csv(path)
    else:
        df = _generate_synthetic_dataset(n_records=n_records)
        df.to_csv(path, index=False)
        logger.info("Synthetic dataset saved to %s", path)

    logger.info("Dataset loaded with shape %s", df.shape)
    return df


if __name__ == "__main__":
    dataset = load_dataset(force_regenerate=True)
    print(dataset.head())
    print(dataset["Churn"].value_counts(normalize=True))
