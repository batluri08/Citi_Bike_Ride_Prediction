from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import requests
from zipfile import ZipFile
from io import BytesIO
import hopsworks
from hsml.schema import Schema
import os

# --- Step 1: Get previous full month's info ---
today = datetime.today()
year = today.year
month = today.month - 1 if today.month > 1 else 12
if today.month == 1:
    year -= 1

# --- Step 2: Download monthly data ZIP ---
url = f"https://s3.amazonaws.com/tripdata/{year}{month:02}-citibike-tripdata.zip"
response = requests.get(url)

if response.status_code != 200:
    raise Exception(f"❌ Failed to download {url}")

with ZipFile(BytesIO(response.content)) as zf:
    csv_filename = [f for f in zf.namelist() if f.endswith('.csv')][0]
    with zf.open(csv_filename) as file:
        df = pd.read_csv(file, low_memory=False)

# --- Step 3: Clean + Prepare ---
df = df[df['started_at'].notnull() & df['ended_at'].notnull()]
df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
df['ended_at'] = pd.to_datetime(df['ended_at'], errors='coerce')
df['duration'] = df['ended_at'] - df['started_at']
df = df[(df['duration'] > pd.Timedelta(0)) & (df['duration'] <= pd.Timedelta(hours=5))]

df = df[df['start_station_id'].notnull()]
df['pickup_location_id'] = pd.to_numeric(df['start_station_id'], errors='coerce')
df = df[df['pickup_location_id'].notnull()]
df['pickup_location_id'] = df['pickup_location_id'].round().astype(int)
df['pickup_hour'] = df['started_at'].dt.floor("H")

# --- Step 4: Round to hourly and aggregate ---
hourly_counts = df.groupby(['pickup_hour', 'pickup_location_id']).size().reset_index(name="rides")
full_hours = pd.date_range(start=hourly_counts['pickup_hour'].min(),
                           end=hourly_counts['pickup_hour'].max(), freq='H')
all_locations = hourly_counts['pickup_location_id'].unique()
grid = pd.MultiIndex.from_product([full_hours, all_locations], names=['pickup_hour', 'pickup_location_id'])
grid_df = pd.DataFrame(index=grid).reset_index()
ts_df = pd.merge(grid_df, hourly_counts, on=["pickup_hour", "pickup_location_id"], how="left")
ts_df["rides"] = ts_df["rides"].fillna(0).astype(int)

# --- Step 5: Function for dynamic lag feature generation ---
def extract_latest_lag_features(df, location_id, window_size=28):
    data = df[df["pickup_location_id"] == location_id].sort_values("pickup_hour").reset_index(drop=True)
    current_hour = pd.Timestamp.utcnow().floor("H")
    latest_available_hour = current_hour - pd.Timedelta(hours=1)

    idx = data[data["pickup_hour"] == latest_available_hour].index
    if len(idx) == 0:
        return pd.DataFrame()
    
    idx = idx[0]
    if idx < window_size:
        return pd.DataFrame()
    
    values = data["rides"].values
    hours = data["pickup_hour"].dt.hour.values
    days = data["pickup_hour"].dt.dayofweek.values

    lags = values[idx - window_size: idx]
    hour = hours[idx]
    day = days[idx]
    target = values[idx]

    row = list(lags) + [hour, day, target]
    columns = [f"feature_{i+1}" for i in range(window_size)] + ["hour_of_day", "day_of_week", "target"]
    return pd.DataFrame([row], columns=columns)

# --- Step 6: Identify top 3 locations dynamically ---
top_locations = ts_df.groupby("pickup_location_id")["rides"].sum().sort_values(ascending=False).head(3).index.tolist()

# --- Step 7: Build latest features for top locations ---
final_features = []
for loc in top_locations:
    features_df = extract_latest_lag_features(ts_df, loc)
    if not features_df.empty:
        features_df["pickup_location_id"] = loc
        final_features.append(features_df)

final_features = pd.concat(final_features, ignore_index=True)
final_features["pickup_hour"] = pd.Timestamp.utcnow().floor("H")

# --- Step 8: Upload to Hopsworks Feature Store ---
HOPSWORKS_API_KEY = "hcd5CJN4URxAz0LC.CXXUwj6ljLaUBxrXZC500JG5azgUPdrJmSkljCG2JSE0DoRqK0Sc9nEliTPs5m82"
HOPSWORKS_PROJECT = "BhumikaTaxiFareMLProject"

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
fs = project.get_feature_store()

FG_NAME = "citibike_hourly_features"
FG_VERSION = 1

try:
    fg = fs.get_feature_group(FG_NAME, version=FG_VERSION)
    print("📦 Using existing feature group")
except:
    fg = fs.create_feature_group(
        name=FG_NAME,
        version=FG_VERSION,
        description="28-hour lag features for hourly Citi Bike predictions",
        primary_key=["pickup_location_id", "pickup_hour"],
        event_time="pickup_hour",
    )
    print("🆕 Created new feature group")

# --- Fix types and upload ---
int_cols = [col for col in final_features.columns if col.startswith("feature_")] + ["target"]
final_features[int_cols] = final_features[int_cols].astype(np.int32)

fg.insert(final_features, write_options={"wait_for_job": True})
print("✅ Features uploaded to Hopsworks successfully.")
