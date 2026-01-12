import pandas as pd
import numpy as np
from faker import Faker
import random
from sqlalchemy import create_engine

fake = Faker()
random.seed(42)
np.random.seed(42)

def generate_data(n=10000):
    data = []

    # Create a pool of 500 users and 100 devices
    user_id = [f"user_{i}" for i in range(500)]
    device_id = [fake.uuid4()[:8] for _ in range(100)]

    for i in range(n):

        
        # The legitamite data
        is_fraud = 0 # Initially we just treat all transaction legit
        user = random.choice(user_id)
        amount = round(np.random.exponential(50), 2)
        timestamp = fake.date_time_this_year
        category = random.choice(['grocery', 'entertainment', 'utilities', 'food', 'tech'])
        auth = random.choice(['PIN', 'Biometric', 'Password'])
        device = random.choice(device_id)
        lat, lon = float(fake.latitude()), float(fake.longitude())


        # Let's inject fraud patterns
        # Pattern 1: The "High Roller" (sudden huge amount)
        if random.random() < 0.01:
            amount = round(random.uniform(2000,5000),2)
            is_fraud = 1
        
        # Pattern 2: The "Impossible Traveller" (High risk location + Fast Succession)
        elif random.random() < 0.005:
            category = 'travel' 
            auth = 'password' # Fraudsters rarely have biometrics and pin
            lat , lon = 12.9716, 77.5946 # From a Fraudsters concentrated location (latitude and longitude)
            is_fraud = 1

        # Fill the empty list
        data.append([
            fake.uuid4()[:12], timestamp, user, amount, category, 
            device, auth, lat, lon, fake.ipv4(), is_fraud
        ])

    df = pd.DataFrame(data, columns=[
        'tx_id', 'timestamp', 'user_id', 'amount', 'category', 
        'device_id', 'auth_method', 'lat', 'lon', 'ip_address', 'is_fraud'
    ])
    
    return df

# 2. Create the connection engine
# Format: mysql+mysqlconnector://[user]:[password]@[host]/[database]
engine = create_engine("mysql+mysqlconnector://root:password@localhost/banking_db")

# 3. Convert DataFrame to MySQL Table
df.to_sql(
    name='transactions',      # Name of the table in MySQL
    con=engine,               # The SQLAlchemy engine
    if_exists='replace',      # options: 'fail', 'replace', 'append'
    index=False,              # Don't save the DataFrame index as a column
    chunksize=1000            # For very large dataframes, insert in batches
)

print("Data successfully exported to MySQL.")

