"""
build_features.py
-----------------
Phase 2 Feature Engineering — Healthcare Capstone (IITM)

Reads  : dataset_no_null_values.csv
Outputs: data/processed/model_table_risk.csv   (Model A — risk_score classifier)
         data/processed/model_table_claim.csv  (Model B — claim_status classifier)

Encoding strategy : One-Hot Encoding with drop='first' to avoid the dummy
                    variable trap (required for Logistic Regression baseline).

Run from the project root:
    python build_features.py
"""

import os
from pathlib import Path
import pandas as pd

# ── 0. Setup ──────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = str(_PROJECT_ROOT / "data" / "interim" / "dataset_no_null_values.csv")
OUTPUT_DIR = str(_PROJECT_ROOT / "data" / "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  Phase 2 — Feature Engineering")
print("=" * 60)

# ── 1. Load raw data ──────────────────────────────────────────────────────────

df = pd.read_csv(INPUT_PATH)
print(f"\n[1] Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── 2. Drop leakage columns ───────────────────────────────────────────────────
# approved_amount and approval_ratio are definitionally derived from
# claim_status. Including them in any model would constitute data leakage.

LEAKAGE_COLS = ["approved_amount", "approval_ratio"]
df.drop(columns=LEAKAGE_COLS, inplace=True)
print(f"\n[2] Dropped leakage columns: {LEAKAGE_COLS}")

# ── 3. Drop date and ID columns ───────────────────────────────────────────────
# Date columns confirmed independent of both targets in EDA.
# ID columns are identifiers, not features.

DROP_COLS = [
    "visit_date", "registration_date", "billing_date",  # temporal
    "patient_id", "visit_id", "bill_id", "doctor_id"    # identifiers
]
df.drop(columns=DROP_COLS, inplace=True)
print(f"\n[3] Dropped date and ID columns: {DROP_COLS}")
print(f"    Working dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── 4. Engineered features (shared across both models) ────────────────────────

print("\n[4] Engineering features...")

# 4a. age_group — standard clinical age bands
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 18, 35, 65, 100],
    labels=["Child", "Young_Adult", "Adult", "Senior"],
    right=True
)
print("    ✓ age_group          : Child / Young_Adult / Adult / Senior")

# 4b. los_band — quartile-based LOS bands (Q1=9.9h, Q3=27.3h)
df["los_band"] = pd.cut(
    df["length_of_stay_hours"],
    bins=[0, 9.9, 27.3, float("inf")],
    labels=["Short", "Normal", "Long"],
    right=True
)
print("    ✓ los_band           : Short (≤9.9h) / Normal (9.9–27.3h) / Long (>27.3h)")

# 4c. billed_amount_band — quartile-based amount bands (Q1=11,595, Q3=28,403)
df["billed_amount_band"] = pd.cut(
    df["billed_amount"],
    bins=[0, 11595, 28403, float("inf")],
    labels=["Low", "Mid", "High"],
    right=True
)
print("    ✓ billed_amount_band : Low (≤11,595) / Mid (11,595–28,403) / High (>28,403)")

# 4d. payment_days_band — quartile-based payment speed bands (Q1=8d, Q3=17d)
#     ⚠ Caveat: Rejected claims have non-zero payment_days due to null
#     imputation during Phase 2 null handling. Values retained as-is (Option A).
#     Document as a soft signal caveat in the Phase 3 model card.
df["payment_days_band"] = pd.cut(
    df["payment_days"],
    bins=[0, 8, 17, float("inf")],
    labels=["Fast", "Normal", "Slow"],
    right=True
)
print("    ✓ payment_days_band  : Fast (≤8d) / Normal (8–17d) / Slow (>17d)")
print("      ⚠ Caveat: Rejected claims have non-zero payment_days (imputation artefact).")
print("        Values retained as-is. Document in Phase 3 model card.")

# ── 5. One-Hot Encoding ───────────────────────────────────────────────────────
# drop='first' removes one dummy per variable to avoid multicollinearity
# (the dummy variable trap). Required for Logistic Regression.
# e.g. gender: only gender_M is kept; gender_F is implied when gender_M = 0.

print("\n[5] Applying One-Hot Encoding (drop='first')...")

OHE_COLS = [
    "gender", "city", "insurance_provider",
    "department", "visit_type",
    "age_group", "los_band",
    "billed_amount_band", "payment_days_band"
]

df = pd.get_dummies(df, columns=OHE_COLS, drop_first=True, dtype=int)

new_ohe_cols = [c for c in df.columns if any(c.startswith(base + "_") for base in OHE_COLS)]
print(f"    Generated {len(new_ohe_cols)} dummy columns from {len(OHE_COLS)} categorical variables:")
for col in new_ohe_cols:
    print(f"      {col}")

# ── 6. Encode targets ─────────────────────────────────────────────────────────

print("\n[6] Encoding target variables...")

df["risk_score"]   = df["risk_score"].map({"Low": 0, "Medium": 1, "High": 2})
df["claim_status"] = df["claim_status"].map({"Paid": 0, "Pending": 1, "Rejected": 2})
print("    ✓ risk_score   : Low=0, Medium=1, High=2")
print("    ✓ claim_status : Paid=0, Pending=1, Rejected=2")

# ── 7. Build Model A dataset — risk_score classifier ─────────────────────────
# Exclude: claim_status (other model's target)
#          payment_days + payment_days_band dummies — temporal leakage:
#            risk score is assigned at visit time, before billing occurs.

print("\n[7] Building Model A dataset (target: risk_score)...")

MODEL_A_EXCLUDE = ["claim_status", "payment_days"] + \
                  [c for c in df.columns if c.startswith("payment_days_band_")]

df_risk = df.drop(columns=MODEL_A_EXCLUDE).copy()

risk_path = os.path.join(OUTPUT_DIR, "model_table_risk.csv")
df_risk.to_csv(risk_path, index=False)

feature_cols_a = [c for c in df_risk.columns if c != "risk_score"]
print(f"    Saved  → {risk_path}")
print(f"    Shape  : {df_risk.shape[0]:,} rows × {df_risk.shape[1]} columns")
print(f"    Features ({len(feature_cols_a)}): {feature_cols_a}")

# ── 8. Build Model B dataset — claim_status classifier ───────────────────────
# Exclude: risk_score (independent of claim_status, EDA p=0.319)

print("\n[8] Building Model B dataset (target: claim_status)...")

MODEL_B_EXCLUDE = ["risk_score"]

df_claim = df.drop(columns=MODEL_B_EXCLUDE).copy()

claim_path = os.path.join(OUTPUT_DIR, "model_table_claim.csv")
df_claim.to_csv(claim_path, index=False)

feature_cols_b = [c for c in df_claim.columns if c != "claim_status"]
print(f"    Saved  → {claim_path}")
print(f"    Shape  : {df_claim.shape[0]:,} rows × {df_claim.shape[1]} columns")
print(f"    Features ({len(feature_cols_b)}): {feature_cols_b}")

# ── 9. Summary ────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  OUTPUT SUMMARY")
print("=" * 60)

print("\n── Model A : risk_score classifier ──────────────────────────")
print(f"   File  : {risk_path}")
print(f"   Rows  : {df_risk.shape[0]:,}")
print(f"   Cols  : {df_risk.shape[1]} (including target)")
print("   Target distribution:")
risk_counts = df_risk["risk_score"].value_counts().sort_index()
risk_labels  = {0: "Low", 1: "Medium", 2: "High"}
for k, v in risk_counts.items():
    print(f"     {risk_labels[k]:8s} ({k}): {v:,}  ({v / len(df_risk) * 100:.1f}%)")

print("\n── Model B : claim_status classifier ────────────────────────")
print(f"   File  : {claim_path}")
print(f"   Rows  : {df_claim.shape[0]:,}")
print(f"   Cols  : {df_claim.shape[1]} (including target)")
print("   Target distribution:")
claim_counts = df_claim["claim_status"].value_counts().sort_index()
claim_labels  = {0: "Paid", 1: "Pending", 2: "Rejected"}
for k, v in claim_counts.items():
    print(f"     {claim_labels[k]:8s} ({k}): {v:,}  ({v / len(df_claim) * 100:.1f}%)")

print("\n── Feature Engineering Decisions ────────────────────────────")
print("""
   Engineered features (shared):
     age_group          : Clinical age bands (Child/Young_Adult/Adult/Senior)
     los_band           : LOS quartile bands (Short/Normal/Long)
     billed_amount_band : Amount quartile bands (Low/Mid/High)
     payment_days_band  : Payment speed bands (Fast/Normal/Slow)

   Encoding:
     One-Hot Encoding with drop='first' (avoids dummy variable trap)
     risk_score   → Low=0, Medium=1, High=2
     claim_status → Paid=0, Pending=1, Rejected=2

   Leakage dropped (both models):
     approved_amount, approval_ratio

   Temporal leakage dropped from Model A:
     payment_days, payment_days_band_*
     (billing data unavailable at visit-time risk scoring)

   Cross-target dropped:
     Model A drops claim_status
     Model B drops risk_score (EDA: p=0.319, independent)

   Caveats for Phase 3 model card:
     payment_days for Rejected claims contains non-zero values
     due to null imputation during Phase 2 null handling.
     Treat payment_days_band as a soft signal for Model B.
""")

print("=" * 60)
print("  Feature engineering complete.")
print("=" * 60)
