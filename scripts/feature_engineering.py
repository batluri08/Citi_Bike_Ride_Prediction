from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import numpy as np
import requests
from zipfile import ZipFile
from io import BytesIO
import hopsworks
from hsml.schema import Schema
import os

# --- Step 1: Set current prediction hour (UTC) ---
current_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

# --- Step 2: Fetch relevant files from May 2024 to now ---
data_frames = []
year = 2024
months = list(range(5, 13))  # May to Dec 2024
if current_hour.year > 2024:
    months += list(range(1, current_hour.month + 1))  # Jan to current month of 2025

for month in months:
    fetch_year = 2024 if month >= 5 else 2025
    url = f"https://s3.amazonaws.com/tripdata/{fetch_year}{month:02}-citibike-tripdata.zip"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with ZipFile(BytesIO(response.content)) as zf:
                csv_file = [f for f in zf.namelist() if f.endswith('.csv')][0]
                with zf.open(csv_file) as f:
                    df = pd.read_csv(f, low_memory=False)
                    data_frames.append(df)
    except:
        continue

# Combine all data
df = pd.concat(data_frames, ignore_index=True)

# --- Step 3: Clean + Filter ---
df = df[df['started_at'].notnull() & df['ended_at'].notnull()]
df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce', utc=True)
df['ended_at'] = pd.to_datetime(df['ended_at'], errors='coerce', utc=True)
df['duration'] = df['ended_at'] - df['started_at']
df = df[(df['duration'] > pd.Timedelta(0)) & (df['duration'] <= pd.Timedelta(hours=5))]
df = df[df['start_station_id'].notnull()]
df['pickup_location_id'] = pd.to_numeric(df['start_station_id'], errors='coerce')
df = df[df['pickup_location_id'].notnull()]
df['pickup_location_id'] = df['pickup_location_id'].round().astype(int)
df['pickup_hour'] = df['started_at'].dt.floor("H")

# --- Step 4: Hourly aggregation ---
hourly_counts = df.groupby(['pickup_hour', 'pickup_location_id']).size().reset_index(name="rides")
full_hours = pd.date_range(start=hourly_counts['pickup_hour'].min(),
                           end=hourly_counts['pickup_hour'].max(), freq='H', tz='UTC')
all_locations = hourly_counts['pickup_location_id'].unique()
grid = pd.MultiIndex.from_product([full_hours, all_locations], names=['pickup_hour', 'pickup_location_id'])
grid_df = pd.DataFrame(index=grid).reset_index()
ts_df = pd.merge(grid_df, hourly_counts, on=["pickup_hour", "pickup_location_id"], how="left")
ts_df["rides"] = ts_df["rides"].fillna(0).astype(int)

# --- Step 5: Remove future rows ---
ts_df = ts_df[ts_df["pickup_hour"] < current_hour]

# --- Step 6: Lag feature generation ---
def make_lag_features(df, location_id, window_size=28):
    data = df[df["pickup_location_id"] == location_id].sort_values("pickup_hour")
    values = data["rides"].values
    hours = data["pickup_hour"].dt.hour.values
    days = data["pickup_hour"].dt.dayofweek.values

    if len(values) <= window_size:
        return pd.DataFrame()

    features = []
    for i in range(len(values) - window_size):
        lags = values[i:i + window_size]
        hour = hours[i + window_size]
        day = days[i + window_size]
        row = list(lags) + [hour, day]
        features.append(row)

    columns = [f"feature_{i+1}" for i in range(window_size)] + ["hour_of_day", "day_of_week"]
    return pd.DataFrame(features, columns=columns)

# --- Step 7: Top 3 locations and latest features ---
top_locations = ts_df.groupby("pickup_location_id")["rides"].sum().sort_values(ascending=False).head(3).index.tolist()
latest_rows = []
for loc in top_locations:
    features_df = make_lag_features(ts_df, loc)
    if not features_df.empty:
        features_df["pickup_location_id"] = loc
        latest_rows.append(features_df.iloc[-1:])

final_features = pd.concat(latest_rows, ignore_index=True)
final_features["pickup_hour"] = [pd.Timestamp(current_hour)] * len(final_features)

# --- Step 8: Upload to Hopsworks ---
project = hopsworks.login(api_key_value=os.getenv("hcd5CJN4URxAz0LC.CXXUwj6ljLaUBxrXZC500JG5azgUPdrJmSkljCG2JSE0DoRqK0Sc9nEliTPs5m82"), project="BhumikaTaxiFareMLProject")
fs = project.get_feature_store()
schema = Schema(final_features)

FG_NAME = "citibike_hourly_features"
FG_VERSION = 1

try:
    fg = fs.get_feature_group(FG_NAME, version=FG_VERSION)
except:
    fg = fs.create_feature_group(
        name=FG_NAME,
        version=FG_VERSION,
        description="28-hour lag features for hourly Citi Bike predictions",
        primary_key=["pickup_location_id", "pickup_hour"],
        event_time="pickup_hour"
    )

final_features[[col for col in final_features.columns if col.startswith("feature_")]] = final_features[
    [col for col in final_features.columns if col.startswith("feature_")]
].astype(np.int32)

fg.insert(final_features, write_options={"wait_for_job": True})
print("✅ Features uploaded to Hopsworks.")

