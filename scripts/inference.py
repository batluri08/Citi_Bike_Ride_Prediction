from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import hopsworks
import os

# --- Config ---
HOPSWORKS_API_KEY = "your-api-key"
HOPSWORKS_PROJECT = "BhumikaTaxiFareMLProject"
FG_NAME = "citibike_hourly_features"
FG_VERSION = 1
MODEL_NAME = "citibike_predictor"
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
    if len(df_loc) < WINDOW_SIZE:
        print(f"❌ Not enough data for location {loc}")
        continue

    latest = df_loc.iloc[-WINDOW_SIZE:]
    row = latest[[f"feature_{i+1}" for i in range(WINDOW_SIZE)]].values.flatten().tolist()
    row += [
        latest.iloc[-1]["hour_of_day"],
        latest.iloc[-1]["day_of_week"],
        loc
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
