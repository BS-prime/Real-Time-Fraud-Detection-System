# =====================================================================================
# --- Import Libraries ---
# =====================================================================================

from pathlib import Path
import pandas as pd
import numpy as np



# =====================================================================================
# --- Feature Engineering ---
# =====================================================================================

def feature_engineering(
        input_path: str | None = 'artifacts/data/simulated_transactions_seed_42.csv',
        save: bool | None = True
    ):

    '''
    Function to perform feature engineering on the transaction data.
    It reads the raw transaction data, creates new features, 
    and saves the processed features to a new CSV file.
    It can be called from notebooks also run directly with default data.
    '''

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # --- Load csv ---
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def load_data(input_path: str) -> pd.DataFrame:

        ROOT_DIR = Path(__file__).resolve().parents[1]
        FILE_PATH = ROOT_DIR / Path(input_path)

        df = pd.read_csv(FILE_PATH)
        return df
    

    df = load_data(input_path)


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # --- Define Helper functions ---
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Haversine Distance (km) 

    def haversine(
            lat1, 
            lon1, 
            lat2, 
            lon2
        ):

        '''
        Docstring for haversine
        
        :param lat1: Lattitude of first coordinate in degrees
        :param lon1: Longitude of first coordinate in degrees
        :param lat2: Lattitude of second coordinate in degress
        :param lon2: Longitude of second coordinate in degress
        '''


        """
        Calculate distance between two coordinates on Earth.
        Inputs are in degrees. Output is in kilometers.
        """


        R = 6371.0  # Earth radius in km

        lat1 = np.radians(lat1)
        lon1 = np.radians(lon1)
        lat2 = np.radians(lat2)
        lon2 = np.radians(lon2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c
    

    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # --- Define the function ---
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        '''
        Docstring for engineer_features
        
        :param df: Load the csv into a dataframe
        :return: Model ready features
        :rtype: DataFrame
        '''

        '''
        Just feature for the model and system
        '''

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # --- 0. Initial preprocessing ---
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # Convert 'timestamp' to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Ensure correct ordering - critical for time-based features
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)



        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 1. Time-based features
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        # Extract hour of day and day of week

        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek

        
        
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 2. Transaction velocity
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        
        # Number of transactions in the past 24 hours per user  
        
        counts = (
            df
            .groupby('user_id')
            .rolling('24h', on='timestamp')['tx_id']
            .count()
            .values
        )
        
        # Then assign it back to a column
        df['tx_count_24h'] = counts

        
        
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 3. Behavioral features
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        # Calculate average spend per user up to (but not including) current transaction
        
        df['avg_spend_user'] = (
            df
            .groupby('user_id')['amount']
            .transform(lambda x: x.shift(1).expanding().mean())
        )
        
        # Calculate amount ratio to average spend

        df['amount_ratio'] = np.where(
            df['avg_spend_user'] > 0, 
            df['amount'] / df['avg_spend_user'], 
            0
        )

        
        
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 4. Geospatial features
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        # Previous transaction coordinates and timestamp

        df['prev_lat'] = df.groupby('user_id')['lat'].shift(1)
        df['prev_lon'] = df.groupby('user_id')['lon'].shift(1)
        df['prev_ts']  = df.groupby('user_id')['timestamp'].shift(1)

        # Distance from previous transaction (km) using Haversine formula

        df['dist_from_last_tx_km'] = haversine(
            df['lat'], 
            df['lon'],
            df['prev_lat'], 
            df['prev_lon']
        ).fillna(0)

        # Calculate time difference in hours
     
        time_diff_hours = (
            (df['timestamp'] - df['prev_ts'])
            .dt
            .total_seconds()
            .div(3600)
            .clip(lower=1e-3)
        )

        # Calculate travel velocity (km/h) - handle division by zero
        
        df['travel_velocity_kmph'] = np.where(
            time_diff_hours == 0,
            df['dist_from_last_tx_km'] / time_diff_hours,
            0
        )



        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 5. Categorical encoding
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        df = pd.get_dummies(
            df,
            columns=['auth_method', 'category'],
            drop_first=True
        )

        
        
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 6. Drop non-model columns
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        df = df.drop(
            columns=[
                'tx_id',
                'prev_lat',
                'prev_lon',
                'prev_ts'
            ]
        )

        
        
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 7. Final numeric safety
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        num_cols = df.select_dtypes(include='number').columns
        df[num_cols] = df[num_cols].fillna(0)

        return df
    
    # =====================================================================================
    # --- Save the features ---
    # =====================================================================================

    df_features = engineer_features(df)

    # Extract version number from file name for saving
    file_text = input_path
    curr_ver = __import__('re').search(r'_seed_(\d+)', file_text).group(1)

    if save:
        
        output_dir = Path(__file__).resolve().parents[1] / Path("artifacts/features")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"fraud_features_seed_{curr_ver}.csv"
        
        
        # Save to CSV
        df_features.to_csv(output_path, index=False)

        # Print the file Path
        print(f"File saved to {output_path}")

    return df_features


# ===========================================================================================
# --- Main Execution ---
# ===========================================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("Performing feature engineering...")
    print("=" * 70)
    print()
    
    feature_engineering()

    print("\n" + "=" * 70)
    print("Feature engineering successfully completed!!!")
    print("=" * 70)


