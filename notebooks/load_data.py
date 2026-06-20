import pandas as pd 
import mysql.connector

from dotenv import load_dotenv
import os

load_dotenv()

conn = mysql.connector.connect(
    host = os.getenv("DB_HOST"),
    user = os.getenv("DB_USER"),
    password = os.getenv("DB_PASSWORD"),
    database = os.getenv("DB_NAME")
)

cursor = conn.cursor()

#--------CREATE TABLES----------------
cursor.execute("""  
CREATE TABLE IF NOT EXISTS patients (
    patient_id INT PRIMARY KEY,
    age INT,
    gender VARCHAR(10),
    city VARCHAR(100),
    insurance_provider VARCHAR(100),
    chronic_flag TINYINT,
    registration_date DATE
    )
               
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS visits (
    visit_id              INT PRIMARY KEY,
    patient_id            INT,
    visit_date            DATE,
    department            VARCHAR(100),
    visit_type            VARCHAR(50),
    length_of_stay_hours  FLOAT,
    risk_score            VARCHAR(20),
    doctor_id             INT,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS billing (
    bill_id          INT PRIMARY KEY,
    visit_id         INT,
    billed_amount    FLOAT,
    approved_amount  FLOAT,
    claim_status     VARCHAR(50),
    payment_days     FLOAT,
    billing_date     DATE,
    FOREIGN KEY (visit_id) REFERENCES visits(visit_id)
)
""")

#--------LOAD DATA FROM CSV FILES----------------

def load_csv(filepath, table_name):
    df = pd.read_csv(filepath)
    
    # Explicitly convert all NaN variants to None
    df = df.astype(object).where(pd.notnull(df), None)
    
    cols = ", ".join(df.columns)
    placeholders = ", ".join(["%s"] * len(df.columns))
    sql = f"INSERT IGNORE INTO {table_name} ({cols}) VALUES ({placeholders})"
    
    # Convert to list of tuples using explicit None handling
    data = []
    for _, row in df.iterrows():
        clean_row = tuple(None if val != val else val for val in row)
        # val != val is True only for NaN (NaN is never equal to itself)
        data.append(clean_row)
    
    # Insert in batches of 500
    batch_size = 500
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        cursor.executemany(sql, batch)
        conn.commit()
        print(f"  Inserted rows {i} to {i + len(batch)} into {table_name}")
    
    print(f"✓ Finished loading {len(data)} rows into {table_name}")

load_csv("/Users/narendrafuloria/Desktop/Capstone_IITM/capstone_healthcare_analytics/data/raw/patients.csv", "patients")
load_csv("/Users/narendrafuloria/Desktop/Capstone_IITM/capstone_healthcare_analytics/data/raw/visits.csv", "visits")
load_csv("/Users/narendrafuloria/Desktop/Capstone_IITM/capstone_healthcare_analytics/data/raw/billing.csv", "billing")

print("Done.")

# ── CREATE INDEXES (run after data is loaded) ──────────────────────────────

print("\nCreating indexes...")

indexes = [
    "CREATE INDEX idx_visits_patient_id  ON visits(patient_id)",
    "CREATE INDEX idx_visits_department  ON visits(department)",
    "CREATE INDEX idx_visits_visit_date  ON visits(visit_date)",
    "CREATE INDEX idx_billing_visit_id     ON billing(visit_id)",
    "CREATE INDEX idx_billing_claim_status ON billing(claim_status)"
]

for idx_sql in indexes:
    try:
        cursor.execute(idx_sql)
        print(f"✓ Created: {idx_sql.split('CREATE INDEX ')[1].split(' ON')[0]}")
    except mysql.connector.errors.DatabaseError as e:
        if "Duplicate key name" in str(e):
            print(f"⚠ Already exists, skipping: {idx_sql.split('CREATE INDEX ')[1].split(' ON')[0]}")
        else:
            raise e

conn.commit()
print("\n✓ All indexes created.")

cursor.close()
conn.close()