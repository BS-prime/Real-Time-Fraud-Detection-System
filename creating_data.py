import pandas as pd
import numpy as np
import random
from faker import Faker
from sqlalchemy import create_engine
from datetime import timedelta

# -------------------------
# Global setup
# -------------------------
fake = Faker()
random.seed(42)
np.random.seed(42)

# -------------------------
# Configuration
# -------------------------
N_USERS = 500
N_TX = 10_000

CATEGORY_RULES = {
    "grocery": (5, 200),
    "food": (5, 150),
    "entertainment": (10, 500),
    "utilities": (50, 400),
    "tech": (100, 3000),
    "travel": (500, 5000)
}

AUTH_METHODS = ["PIN", "Biometric", "Password"]

# -------------------------
# User profiles
# -------------------------
def create_users(n_users):
    users = {}

    for i in range(n_users):
        user_id = f"user_{i}"

        # Home location cluster
        home_lat = float(fake.latitude())
        home_lon = float(fake.longitude())

        # Each user owns 1–3 devices
        devices = [fake.uuid4()[:8] for _ in range(random.randint(1, 3))]

        users[user_id] = {
            "devices": devices,
            "home_lat": home_lat,
            "home_lon": home_lon,
            "last_tx_time": fake.date_time_this_year()
        }

    return users

# -------------------------
# Main generator
# -------------------------
def generate_data(n_tx=10_000):
    users = create_users(N_USERS)
    data = []

    for _ in range(n_tx):
        
        is_fraud = 0
        user_id = random.choice(list(users.keys()))
        profile = users[user_id]

        # ---- Timestamp (sequential behavior)
        timestamp = profile["last_tx_time"] + timedelta(
            minutes=random.randint(5, 1440)
        )
        profile["last_tx_time"] = timestamp

        
        # ========================================
        # ---- Normal behavior defaults
        # ========================================


        category = random.choice(
            ["grocery", "food", "entertainment", "utilities", "tech"]
        )
        min_amt, max_amt = CATEGORY_RULES[category]
        
        amount = round(random.uniform(min_amt, max_amt), 2)
        
        device = random.choice(profile["devices"])
        
        auth = random.choice(["PIN", "Biometric", "Password"])
        
        lat = round(np.random.normal(profile["home_lat"], 0.05), 6)
        lon = round(np.random.normal(profile["home_lon"], 0.05), 6)

        
        # =========================================
        # ---- Fraud injection (single random draw)
        # =========================================

        r = random.random()

        # Pattern 1: High Roller
        if r < 0.01:
            category = "tech"
            amount = round(random.uniform(3000, 7000), 2)
            auth = "Password"
            is_fraud = 1

        # Pattern 2: Impossible Traveller
        elif r < 0.015:
            category = "travel"
            amount = round(random.uniform(1000, 5000), 2)
            auth = "Password"
            lat, lon = 12.9716, 77.5946  # Fraud hotspot
            timestamp += timedelta(minutes=5)
            is_fraud = 1

        data.append([
            fake.uuid4()[:12],
            timestamp,
            user_id,
            amount,
            category,
            device,
            auth,
            lat,
            lon,
            fake.ipv4(),
            is_fraud
        ])

    return pd.DataFrame(
        data,
        columns=[
            "tx_id",
            "timestamp",
            "user_id",
            "amount",
            "category",
            "device_id",
            "auth_method",
            "lat",
            "lon",
            "ip_address",
            "is_fraud"
        ]
    )

# -------------------------
# Generate data
# -------------------------
df = generate_data(N_TX)

# -------------------------
# MySQL export
# -------------------------
engine = create_engine(
    "mysql+mysqlconnector://root:password@localhost/banking_db"
)

df.to_sql(
    name="transactions",
    con=engine,
    if_exists="replace",
    index=False,
    chunksize=1000
)

print("Data successfully exported to MySQL.")
