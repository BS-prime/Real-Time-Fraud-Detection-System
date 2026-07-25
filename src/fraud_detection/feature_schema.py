"""Canonical model feature names and categorical values."""

TARGET_COLUMN = "is_fraud"

AUTH_METHODS = ["Biometric", "PIN", "Password"]
CATEGORIES = ["entertainment", "food", "grocery", "tech", "travel", "utilities"]

FEATURE_COLUMNS = [
    "amount",
    "lat",
    "lon",
    "hour",
    "day_of_week",
    "tx_count_24h",
    "avg_spend_user",
    "amount_ratio",
    "dist_from_last_tx_km",
    "travel_velocity_kmph",
    "auth_method_PIN",
    "auth_method_Password",
    "category_food",
    "category_grocery",
    "category_tech",
    "category_travel",
    "category_utilities",
]
