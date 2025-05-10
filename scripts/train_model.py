from datetime import datetime
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import hopsworks
from hsml.schema import Schema

# --- Step 1: Login to Hopsworks ---
HOPSWORKS_API_KEY = "hcd5CJN4URxAz0LC.CXXUwj6ljLaUBxrXZC500JG5azgUPdrJmSkljCG2JSE0DoRqK0Sc9nEliTPs5m82"
HOPSWORKS_PROJECT = "BhumikaTaxiFareMLProject"

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
fs = project.get_feature_store()
mr = project.get_model_registry()

# --- Step 2: Read feature data ---
FG_NAME = "citibike_hourly_features"
FG_VERSION = 1

fg = fs.get_feature_group(FG_NAME, version=FG_VERSION)
query = fg.select_all()
df = query.read()
print(f"📊 Loaded {len(df)} records from Feature Group '{FG_NAME}'")

# --- Step 3: Prepare data ---
df = df.dropna()
X = df[[f"feature_{i+1}" for i in range(28)] + ["hour_of_day", "day_of_week", "pickup_location_id"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# --- Step 4: Train LightGBM model ---
model = lgb.LGBMRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f"✅ Trained LightGBM model. MAE = {mae:.2f}")

# --- Step 5: Save model locally ---
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "lightgbm_full_model.pkl")
joblib.dump(model, model_path)
print(f"💾 Model saved at: {model_path}")

# --- Step 6: Upload to Hopsworks Model Registry ---
schema = Schema(X_train)

model_obj = mr.python.create_model(
    name="citibike_lightgbm_full",
    version=1,
    description="LightGBM model trained on hourly lag features",
    metrics={"mae": mae},
    input_example=X_train.iloc[:5],
    model_schema=schema
)

model_obj.save(model_path)
print("🚀 Model uploaded to Hopsworks Model Registry")
