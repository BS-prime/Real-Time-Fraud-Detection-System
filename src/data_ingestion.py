# ============================================================================
# --- Import Libraries ---
# ============================================================================

import os
import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import timedelta ,timezone, datetime
from pathlib import Path



# ============================================================================
# --- Configurations ---
# ============================================================================

CATEGORY_RULES = {
    "grocery": (5, 200),
    "food": (5, 150),
    "entertainment": (10, 500),
    "utilities": (50, 400),
    "tech": (100, 3000),
    "travel": (500, 5000)
}

AUTH_METHODS = ["PIN", "Biometric", "Password"]



# ============================================================================
# --- Data Generation Function ---
# ============================================================================

def generate_transactions_data(
    n_tx: int | None = 10_000,
    n_users: int | None = 500,
    seed: int | None = 42,
    output_path: str | None = None,
    save: bool | None = True,
):
    
    '''
    Generate synthetic transaction data for fraud detection.
    '''
    
    if seed is not None:
        
        Faker.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    

    fake = Faker()
    


    # ============================================================================
    # --- Define the user profiles ---
    # ============================================================================

    def create_users(n_users):

        '''
        Create synthetic user profiles.
        '''


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



    # ============================================================================
    # --- Genrerate Transactions ---
    # ============================================================================

    def generate_data(
            n_tx, 
            n_users
        ) -> pd.DataFrame:

        '''
        Using the user profiles, 
        generate synthetic transaction data,
        injecting fraud patterns.
        '''



        users = create_users(n_users)
        data = []

        for _ in range(n_tx):
            
            is_fraud = 0
            user_id = random.choice(list(users.keys()))
            profile = users[user_id]

            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # ---- Timestamp (sequential behavior) ---
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            timestamp = profile["last_tx_time"] + timedelta(
                minutes=random.randint(5, 1440)
            )
            profile["last_tx_time"] = timestamp

            
            
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # ---- Normal behavior defaults ---
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


            category = random.choice(
                ["grocery", "food", "entertainment", "utilities", "tech"]
            )
            min_amt, max_amt = CATEGORY_RULES[category]
            
            amount = round(random.uniform(min_amt, max_amt), 2)
            
            device = random.choice(profile["devices"])
            
            auth = random.choice(["PIN", "Biometric", "Password"])
            
            # Generally users stay inside 60 km radius from their homes
            lat = round(profile["home_lat"] + random.uniform(-0.05, 0.05), 6) 
            lon = round(profile["home_lon"] + random.uniform(-0.05, 0.05), 6) 



            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # ---- Fraud injection (single random draw)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
                lat, lon = 40.7128, -74.0060  # Fraud hotspot
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

        df = pd.DataFrame(
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

        return df



    # ===========================================================================================
    # --- Save Data ---
    # ===========================================================================================    
    
    df = generate_data(n_tx, n_users)

    if save:
        if output_path is None:

            output_dir = Path(__file__).resolve().parents[1] / "artifacts" / "data"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"simulated_transactions_v{seed}_0.csv"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)      
            
            # Save to CSV
        df.to_csv(output_path, index=False)

            # Return path info
        print(f"Data saved to {output_path}")
        
    return df 



# ============================================================================================
# --- Main Execution ---
# ============================================================================================
if __name__ == "__main__":

    print('=' * 70)
    print("Generating synthetic transaction data...")
    print('=' * 70)
    print()
    # Generate and save data
    generate_transactions_data()
    
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE!!!")
    print("=" * 70)