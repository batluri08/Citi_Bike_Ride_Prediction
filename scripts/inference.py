from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import hopsworks
import os

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

# --- Load model and predict ---
mr = project.get_model_registry()
model = mr.get_model(MODEL_NAME, version=MODEL_VERSION)
model_dir = model.download()
model = model.load()

preds = model.predict(inference_df.drop(columns=["pickup_location_id"]))
inference_df["predicted_rides"] = preds.astype(int)

# --- Output ---
print("\n📈 Inference Results:")
print(inference_df[["pickup_location_id", "predicted_rides"]])
