from datetime import datetime, timedelta
from hsfs.feature_group import FeatureGroup
from hsml.schema import Schema
import pandas as pd
import numpy as np
import requests
import hopsworks
import os
import joblib

# --- Config ---
HOPSWORKS_API_KEY = "hcd5CJN4URxAz0LC.CXXUwj6ljLaUBxrXZC500JG5azgUPdrJmSkljCG2JSE0DoRqK0Sc9nEliTPs5m82"
HOPSWORKS_PROJECT = "BhumikaTaxiFareMLProject"
FG_NAME = "citibike_hourly_features"
FG_VERSION = 1
MODEL_NAME = "citibike_lightgbm_full"
MODEL_VERSION = 1
WINDOW_SIZE = 28

# --- Login to Hopsworks ---
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
fs = project.get_feature_store()

# --- Load latest features from Hopsworks ---
fg = fs.get_feature_group(FG_NAME, version=FG_VERSION)
query = fg.select_all()
features_df = query.read()

# --- Get latest 28 hours per top location ---
top_locations = features_df.groupby("pickup_location_id")["target"].sum().sort_values(ascending=False).head(3).index.tolist()

inference_rows = []

for loc in top_locations:
    df_loc = features_df[features_df["pickup_location_id"] == loc].sort_values("pickup_hour")
    if df_loc.empty:
        print(f"⚠️ No data for location {loc}")
        continue

    # Just take the latest row (already has all 28 lag features + hour/day)
    latest_row = df_loc.iloc[-1]

    row = [latest_row[f"feature_{i+1}"] for i in range(WINDOW_SIZE)]
    row += [
        latest_row["hour_of_day"],
        latest_row["day_of_week"],
        loc  # keep pickup_location_id for mapping
    ]

    inference_rows.append(row)

# --- Build inference dataframe ---
columns = [f"feature_{i+1}" for i in range(WINDOW_SIZE)] + ["hour_of_day", "day_of_week", "pickup_location_id"]
inference_df = pd.DataFrame(inference_rows, columns=columns)
inference_df[[c for c in inference_df.columns if c.startswith("feature_")]] = inference_df[
    [c for c in inference_df.columns if c.startswith("feature_")]
].astype(np.int32)

# ✅ Rename to match training feature name
inference_df = inference_df.rename(columns={"pickup_location_id": "location_id"})

# --- Load model and predict ---
mr = project.get_model_registry()
model = mr.get_model("citibike_lightgbm_full", version=1)
model_dir = model.download()
model_path = os.path.join(model_dir, "lightgbm_full_model.pkl")
model = joblib.load(model_path)

# ✅ Predict using all 31 features
X_pred = inference_df
preds = model.predict(X_pred)
inference_df["predicted_rides"] = preds.astype(int)


# --- Output ---
print("\n📈 Inference Results:")
print(inference_df[["location_id", "predicted_rides"]])
print(features_df.sort_values("pickup_hour", ascending=False).head(5))
print(inference_df[["location_id", "hour_of_day", "day_of_week"] + [f"feature_{i+1}" for i in range(28)]])


# --- Step 9: Upload Predictions to Hopsworks Feature Store ---
from hsml.schema import Schema

# Add timestamp
inference_df["prediction_time"] = pd.Timestamp.utcnow()

# Clean datatypes
inference_df["location_id"] = inference_df["location_id"].astype(int)
inference_df["predicted_rides"] = inference_df["predicted_rides"].astype(int)
inference_df["prediction_time"] = pd.to_datetime(inference_df["prediction_time"])

# Define schema only if needed
schema = Schema(
    [
        {"name": "location_id", "type": "int"},
        {"name": "predicted_rides", "type": "int"},
        {"name": "prediction_time", "type": "timestamp"}
    ]
)

# Get FG or create
pred_fg = fs.get_feature_group("citibike_hourly_predictions", version=1)

if pred_fg is None:
    print("🆕 Creating new prediction feature group")
    pred_fg = fs.create_feature_group(
        name="citibike_hourly_predictions",
        version=1,
        description="Hourly predicted rides for top 3 locations",
        primary_key=["location_id", "prediction_time"],
        event_time="prediction_time",
    )
else:
    print("📦 Using existing prediction feature group")

# Insert prediction data
pred_fg.insert(
    inference_df[["location_id", "predicted_rides", "prediction_time"]],
    write_options={"wait_for_job": True}
)

print("✅ Predictions uploaded to Hopsworks successfully.")
