import streamlit as st
import pandas as pd
import hopsworks
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import requests
import lightgbm as lgb
import joblib
import os
import seaborn as sns

# --- Page Config ---
st.set_page_config(page_title="CitiBike Predictions", layout="wide")

# --- Lottie Animation ---
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    return None

lottie_cycling = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_touohxv0.json")
st_lottie(lottie_cycling, height=150)

# --- Login to Hopsworks ---
HOPSWORKS_API_KEY = st.secrets["HOPSWORKS_API_KEY"]
HOPSWORKS_PROJECT = st.secrets["HOPSWORKS_PROJECT_NAME"]

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
fs = project.get_feature_store()
mr = project.get_model_registry()

# --- Load Predictions ---
fg = fs.get_feature_group("citibike_hourly_predictions", version=1)
query = fg.select_all()
df = query.read()
df["prediction_time"] = pd.to_datetime(df["prediction_time"])

# --- Header ---
st.title("CitiBike Ride Predictions")
st.markdown("Real-time predictions for top NYC CitiBike pickup locations")

# --- Latest Predictions ---
st.markdown("## \U0001F504 Latest Predictions")
latest_time = df["prediction_time"].max()
df_latest = df[df["prediction_time"] == latest_time].sort_values("predicted_rides", ascending=False)
st.markdown(f"#### As of `{latest_time.strftime('%Y-%m-%d %H:%M:%S')} UTC`")

# Redesigned card layout with icons
prediction_cards = "<div style='display: flex; flex-wrap: wrap; gap: 1.5rem; justify-content: start;'>"
for row in df_latest.itertuples():
    prediction_cards += (
        "<div style='background-color: #1e1e1e; padding: 1.5rem; border-radius: 12px; text-align: center; "
        "width: 220px; box-shadow: 0 4px 8px rgba(0,0,0,0.3);'>"
        f"<div style='font-size: 1.1rem; font-weight: 600; color: #9ca3af;'>📍 Location {row.location_id}</div>"
        f"<div style='font-size: 2rem; font-weight: bold; color: #10b981; margin-top: 0.5rem;'>🚗 {row.predicted_rides} rides</div>"
        "</div>"
    )
prediction_cards += "</div>"
st.markdown(prediction_cards, unsafe_allow_html=True)

# --- Layout Split ---
st.markdown("### Compare Top 3 Locations")
top_locs = df["location_id"].value_counts().head(3).index.tolist()
df_top3 = df[df["location_id"].isin(top_locs)].sort_values("prediction_time")
df_pivot = df_top3.pivot(index="prediction_time", columns="location_id", values="predicted_rides")
st.line_chart(df_pivot, height=300)

# --- Prediction Trends ---
st.markdown("### Prediction Trend Over Time")
col1, col2 = st.columns([3, 1])

with col1:
    location = st.selectbox("Trend: Select location:", df["location_id"].unique(), key="trend_loc")
    df_loc = df[df["location_id"] == location].sort_values("prediction_time")

with col2:
    hours = st.slider("Past hours:", min_value=6, max_value=168, value=24, step=6)
    df_loc = df_loc.tail(hours)

st.line_chart(df_loc.set_index("prediction_time")["predicted_rides"], height=300)

# --- Model Version and Metrics ---
st.markdown("### Model Info & Feature Importance")
model_versions = sorted([m.version for m in mr.get_models("citibike_lightgbm_full")])
version = st.selectbox("Choose model version:", model_versions)
model = mr.get_model("citibike_lightgbm_full", version=version)

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.markdown(f"- **Model Name:** `{model.name}`")
    st.markdown(f"- **Version:** `{model.version}`")
    st.markdown(f"- **Description:** {model.description}")

# --- Feature Importance ---
with col_m2:
    model_dir = model.download()
    model_path = os.path.join(model_dir, "lightgbm_full_model.pkl")
    model_local = joblib.load(model_path)
    booster = model_local.booster_
    importance_df = pd.DataFrame({
        "Feature": booster.feature_name(),
        "Importance": booster.feature_importance()
    }).sort_values("Importance", ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis", ax=ax)
    ax.set_title("Top 10 Feature Importances")
    st.pyplot(fig)
