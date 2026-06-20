import pandas as pd
import os


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def merge_tables(df1: pd.DataFrame, df2: pd.DataFrame, on_columns: list, how: str = "inner") -> pd.DataFrame:
    return pd.merge(df1, df2, on=on_columns, how=how)


def save_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def build_merged_dataset(patients_path: str, visits_path: str, billing_path: str, output_path: str) -> pd.DataFrame:
    patients = load_csv(patients_path)
    visits = load_csv(visits_path)
    billing = load_csv(billing_path)

    merged = merge_tables(patients, visits, on_columns=["patient_id"])
    merged = merge_tables(merged, billing, on_columns=["visit_id"])

    save_csv(merged, output_path)
    return merged


if __name__ == "__main__":
    BASE = "/Users/narendrafuloria/Desktop/Capstone_IITM/capstone_healthcare_analytics"

    build_merged_dataset(
        patients_path=f"{BASE}/data/raw/patients.csv",
        visits_path=f"{BASE}/data/raw/visits.csv",
        billing_path=f"{BASE}/data/raw/billing.csv",
        output_path=f"{BASE}/data/processed/merged_data.csv",
    )
