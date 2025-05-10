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
df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
df['ended_at'] = pd.to_datetime(df['ended_at'], errors='coerce')
df = df[df['started_at'].notnull() & df['ended_at'].notnull()]
df['duration'] = df['ended_at'] - df['started_at']
df = df[(df['duration'] > pd.Timedelta(0)) & (df['duration'] <= pd.Timedelta(hours=5))]

df = df[df['start_station_id'].notnull()]
df['pickup_location_id'] = pd.to_numeric(df['start_station_id'], errors='coerce')
df = df[df['pickup_location_id'].notnull()]
df['pickup_location_id'] = df['pickup_location_id'].round().astype(int)

# --- Step 4: Round to hourly and aggregate ---
df['pickup_hour'] = df['started_at'].dt.floor("H")
hourly_counts = df.groupby(['pickup_hour', 'pickup_location_id']).size().reset_index(name="rides")

# --- Step 5: Build complete hourly grid for missing hours ---
full_hours = pd.date_range(hourly_counts['pickup_hour'].min(), hourly_counts['pickup_hour'].max(), freq='H')
all_locations = hourly_counts['pickup_location_id'].unique()
grid = pd.MultiIndex.from_product([full_hours, all_locations], names=['pickup_hour', 'pickup_location_id'])
grid_df = pd.DataFrame(index=grid).reset_index()

ts_df = pd.merge(grid_df, hourly_counts, on=["pickup_hour", "pickup_location_id"], how="left")
ts_df["rides"] = ts_df["rides"].fillna(0).astype(int)

# --- Step 6: Transform into lag features ---
def make_lag_features(df, location_id, window_size=28, step_size=1):
    data = df[df["pickup_location_id"] == location_id].sort_values("pickup_hour")
    values = data["rides"].values
    hours = data["pickup_hour"].dt.hour.values
    days = data["pickup_hour"].dt.dayofweek.values

    if len(values) <= window_size:
        return pd.DataFrame()

    rows = []
    for i in range(0, len(values) - window_size, step_size):
        lags = values[i:i + window_size]
        target = values[i + window_size]
        hour = hours[i + window_size]
        day = days[i + window_size]
        row = list(lags) + [hour, day, target]
        rows.append(row)

    columns = [f"feature_{i+1}" for i in range(window_size)] + ["hour_of_day", "day_of_week", "target"]
    return pd.DataFrame(rows, columns=columns)

# --- Step 7: Get top 3 locations and prepare features ---
top_locations = ts_df.groupby("pickup_location_id")["rides"].sum().sort_values(ascending=False).head(3).index.tolist()
combined_features = []

for loc in top_locations:
    features_df = make_lag_features(ts_df, loc)
    if not features_df.empty:
        features_df["pickup_location_id"] = loc
        combined_features.append(features_df)

latest_rows = [df.iloc[-1:] for df in combined_features if not df.empty]
final_features = pd.concat(latest_rows, ignore_index=True)

# --- Step 8: Add hourly timestamps for Hopsworks ---
start_time = ts_df["pickup_hour"].min() + pd.Timedelta(hours=28)
final_features["pickup_hour"] = pd.date_range(start=start_time, periods=len(final_features), freq="H")

# --- Step 9: Upload to Hopsworks Feature Store ---
HOPSWORKS_API_KEY = "hcd5CJN4URxAz0LC.CXXUwj6ljLaUBxrXZC500JG5azgUPdrJmSkljCG2JSE0DoRqK0Sc9nEliTPs5m82"
HOPSWORKS_PROJECT = "BhumikaTaxiFareMLProject"

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)

fs = project.get_feature_store()
schema = Schema(final_features)

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

int_columns = [col for col in final_features.columns if col.startswith("feature_") or col == "target"]
final_features[int_columns] = final_features[int_columns].astype(np.int32)

fg.insert(final_features, write_options={"wait_for_job": True})
print("✅ Features uploaded to Hopsworks successfully.")
