# scripts/inference.py

from datetime import datetime
import pandas as pd
import numpy as np
import hopsworks
import os
import joblib

# --- Config ---
HOPSWORKS_API_KEY = "hcd5CJN4URxAz0LC.CXXUwj6ljLaUBxrXZC500JG5azgUPdrJmSkljCG2JSE0DoRqK0Sc9nEliTPs5m82"
HOPSWORKS_PROJECT = "BhumikaTaxiFareMLProject"
FG_NAME = "citibike_hourly_features"
PRED_FG_NAME = "citibike_hourly_predictions"
MODEL_NAME = "citibike_lightgbm_full"
MODEL_VERSION = 1
WINDOW_SIZE = 28

# --- Login to Hopsworks ---
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
fs = project.get_feature_store()

# --- Load latest feature rows from Hopsworks ---
fg = fs.get_feature_group(FG_NAME, version=1)
features_df = fg.select_all().read()

# --- Filter to latest available hour ---
latest_hour = features_df["pickup_hour"].max()
features_df = features_df[features_df["pickup_hour"] == latest_hour]

# --- Preprocess features ---
features_df = features_df.copy()
features_df = features_df.rename(columns={"pickup_location_id": "location_id"})
feature_cols = [f"feature_{i+1}" for i in range(WINDOW_SIZE)] + ["hour_of_day", "day_of_week", "location_id"]
features_df[feature_cols] = features_df[feature_cols].astype(np.int32)

X_pred = features_df[feature_cols]

# --- Load best model ---
mr = project.get_model_registry()
model = mr.get_model(MODEL_NAME, version=MODEL_VERSION)
model_dir = model.download()
model_path = os.path.join(model_dir, "lightgbm_full_model.pkl")
model = joblib.load(model_path)

# --- Predict ---
preds = model.predict(X_pred)
features_df["predicted_rides"] = preds.astype(int)
features_df["prediction_time"] = pd.Timestamp.utcnow()

# --- Prepare output for upload ---
pred_df = features_df[["location_id", "predicted_rides", "prediction_time"]]
print(pred_df)
# --- Save predictions to Hopsworks Feature Group ---
try:
    pred_fg = fs.get_feature_group(PRED_FG_NAME, version=1)
    print("📦 Using existing prediction group")
except:
    pred_fg = fs.create_feature_group(
        name=PRED_FG_NAME,
        version=1,
        description="Hourly predicted Citi Bike rides",
        primary_key=["location_id", "prediction_time"],
        event_time="prediction_time"
    )
    print("🆕 Created prediction feature group")

# --- Prepare for Hopsworks insert ---
pred_df["location_id"] = pred_df["location_id"].astype("int64")
pred_df["predicted_rides"] = pred_df["predicted_rides"].astype("int64")

pred_fg.insert(pred_df, write_options={"wait_for_job": True})
print("✅ Predictions uploaded to Hopsworks.")

