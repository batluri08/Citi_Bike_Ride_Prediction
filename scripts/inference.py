from datetime import datetime
import pandas as pd
import numpy as np
import hopsworks
import os
import joblib

# --- Config ---
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = "BhumikaTaxiFareMLProject"
FG_NAME = "citibike_hourly_features"
FG_VERSION = 1
MODEL_NAME = "citibike_lightgbm_full"
MODEL_VERSION = 1
WINDOW_SIZE = 28
PRED_FG_NAME = "citibike_hourly_predictions"

# --- Login to Hopsworks ---
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
fs = project.get_feature_store()

# --- Load latest features from Hopsworks ---
fg = fs.get_feature_group(FG_NAME, version=FG_VERSION)
query = fg.select_all()
features_df = query.read()

# --- Drop target column (used for training only) ---
if "target" in features_df.columns:
    features_df = features_df.drop(columns=["target"])

# --- Select latest row per top 3 locations ---
latest_rows = features_df.sort_values("pickup_hour").drop_duplicates("pickup_location_id", keep="last")
top_locations = latest_rows["pickup_location_id"].value_counts().head(3).index.tolist()

inference_rows = []

for loc in top_locations:
    df_loc = features_df[features_df["pickup_location_id"] == loc].sort_values("pickup_hour")
    if df_loc.empty:
        print(f"⚠️ No data for location {loc}")
        continue

    latest_row = df_loc.iloc[-1]

    row = [latest_row[f"feature_{i+1}"] for i in range(WINDOW_SIZE)]
    row += [
        latest_row["hour_of_day"],
        latest_row["day_of_week"],
        loc
    ]

    inference_rows.append(row)

# --- Prepare final inference DataFrame ---
columns = [f"feature_{i+1}" for i in range(WINDOW_SIZE)] + ["hour_of_day", "day_of_week", "location_id"]
inference_df = pd.DataFrame(inference_rows, columns=columns)

# Enforce integer types
feature_cols = [col for col in columns if "feature_" in col]
inference_df[feature_cols] = inference_df[feature_cols].astype(np.int32)

# --- Load model ---
mr = project.get_model_registry()
model = mr.get_model(MODEL_NAME, version=MODEL_VERSION)
model_dir = model.download()
model_path = os.path.join(model_dir, "lightgbm_full_model.pkl")
model = joblib.load(model_path)

# --- Predict with slight noise to make output dynamic ---
preds = model.predict(inference_df)
noise = np.random.randint(-2, 3, size=preds.shape)
preds_noisy = np.clip(preds + noise, a_min=0, a_max=None)

inference_df["predicted_rides"] = preds_noisy.astype(int)

# Add prediction time
inference_df["prediction_time"] = pd.Timestamp.utcnow()

# Cast datatypes
inference_df["location_id"] = inference_df["location_id"].astype(np.int64)
inference_df["predicted_rides"] = inference_df["predicted_rides"].astype(np.int64)
inference_df["prediction_time"] = pd.to_datetime(inference_df["prediction_time"])

# --- Upload predictions to Hopsworks ---
try:
    pred_fg = fs.get_feature_group(PRED_FG_NAME, version=1)
    print("📦 Using existing prediction feature group")
except:
    pred_fg = fs.create_feature_group(
        name=PRED_FG_NAME,
        version=1,
        description="Hourly predicted rides with noise for demo",
        primary_key=["location_id", "prediction_time"],
        event_time="prediction_time"
    )
    print("🆕 Created prediction feature group")

pred_fg.insert(
    inference_df[["location_id", "predicted_rides", "prediction_time"]],
    write_options={"wait_for_job": True}
)

print("\n✅ Predictions uploaded to Hopsworks:")
print(inference_df[["location_id", "predicted_rides", "prediction_time"]])
