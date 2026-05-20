"""
CompostIQ — IoT Data Pipeline & ML Prediction Engine
=====================================================
Author: Ali Ibtisam
Project: AI + IoT Smart Composting Research — AIU
Description: Simulates ESP32 sensor data ingestion, runs feature engineering,
             and trains a Random Forest model to predict compost readiness.

Stack: Python, Pandas, Scikit-Learn, NumPy, Matplotlib
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. SIMULATE ESP32 IoT SENSOR DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("  CompostIQ — ML Pipeline v1.0")
print("  AI + IoT Smart Composting Research · AIU")
print("=" * 60)

np.random.seed(42)
n_samples = 500
days = np.arange(n_samples)

# Simulate realistic composting sensor readings
temperature = 40 + 15 * np.sin(days / 30) + np.random.normal(0, 2, n_samples)
moisture    = 55 + 10 * np.sin(days / 20 + 1) + np.random.normal(0, 3, n_samples)
ph_level    = 6.5 + 0.5 * np.sin(days / 15) + np.random.normal(0, 0.15, n_samples)
co2_level   = 400 + 100 * np.abs(np.sin(days / 10)) + np.random.normal(0, 20, n_samples)
nitrogen    = 2.5 + 0.8 * np.random.random(n_samples)
carbon      = 25 + 5 * np.random.random(n_samples)

# Clip to realistic ranges
temperature = np.clip(temperature, 25, 75)
moisture    = np.clip(moisture, 30, 90)
ph_level    = np.clip(ph_level, 5.5, 8.5)
co2_level   = np.clip(co2_level, 300, 800)

# Target: decomposition rate (% per day) — function of all factors
decomposition_rate = (
    0.3 * (1 - abs(temperature - 55) / 30)
  + 0.25 * (1 - abs(moisture - 55) / 45)
  + 0.2 * (1 - abs(ph_level - 7.0) / 2.5)
  + 0.15 * (1 - co2_level / 1000)
  + 0.1 * nitrogen / 3.5
  + np.random.normal(0, 0.02, n_samples)
)
decomposition_rate = np.clip(decomposition_rate * 100, 0, 100)

df = pd.DataFrame({
    "day": days,
    "temperature_c": temperature,
    "moisture_pct": moisture,
    "ph_level": ph_level,
    "co2_ppm": co2_level,
    "nitrogen_pct": nitrogen,
    "carbon_pct": carbon,
    "decomposition_rate": decomposition_rate
})

print(f"\n[DATA] Loaded {len(df)} sensor readings from ESP32 network")
print(df.describe().round(2).to_string())

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[FEATURES] Engineering time-series features...")

# Lag features (previous readings matter)
for lag in [1, 3, 7]:
    df[f"temp_lag_{lag}d"]  = df["temperature_c"].shift(lag)
    df[f"moist_lag_{lag}d"] = df["moisture_pct"].shift(lag)

# Rolling statistics
df["temp_roll_7d_mean"]  = df["temperature_c"].rolling(7).mean()
df["temp_roll_7d_std"]   = df["temperature_c"].rolling(7).std()
df["moist_roll_3d_mean"] = df["moisture_pct"].rolling(3).mean()

# Domain-specific features
df["cn_ratio"]        = df["carbon_pct"] / df["nitrogen_pct"]
df["temp_optimal"]    = ((df["temperature_c"] >= 50) & (df["temperature_c"] <= 60)).astype(int)
df["moist_optimal"]   = ((df["moisture_pct"] >= 50) & (df["moisture_pct"] <= 65)).astype(int)
df["ph_optimal"]      = ((df["ph_level"] >= 6.5) & (df["ph_level"] <= 7.5)).astype(int)
df["health_score"]    = df["temp_optimal"] + df["moist_optimal"] + df["ph_optimal"]
df["co2_risk_flag"]   = (df["co2_ppm"] > 600).astype(int)
df["compost_maturity"]= (df["temperature_c"] < 45) & (df["ph_level"].between(6.8, 7.2))

df = df.dropna()
print(f"  → {df.shape[1]} features engineered, {len(df)} samples after lag cleanup")

# ─────────────────────────────────────────────
# 3. TRAIN/TEST SPLIT + MODEL
# ─────────────────────────────────────────────
feature_cols = [c for c in df.columns if c not in ["day", "decomposition_rate"]]
X = df[feature_cols]
y = df["decomposition_rate"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

print("\n[MODEL] Training Random Forest Regressor...")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ─────────────────────────────────────────────
# 4. EVALUATION
# ─────────────────────────────────────────────
r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse= mean_squared_error(y_test, y_pred) ** 0.5

print("\n[RESULTS] Model Performance:")
print(f"  R² Score : {r2:.4f}")
print(f"  MAE      : {mae:.2f}%/day")
print(f"  RMSE     : {rmse:.2f}%/day")

# Feature importance
importances = pd.Series(model.feature_importances_, index=feature_cols)
top_features = importances.sort_values(ascending=False).head(8)

print("\n[FEATURES] Top 8 Most Important Features:")
for feat, imp in top_features.items():
    bar = "█" * int(imp * 100)
    print(f"  {feat:<28} {imp:.3f}  {bar}")

# ─────────────────────────────────────────────
# 5. COMPOST READINESS PREDICTION
# ─────────────────────────────────────────────
print("\n[PREDICT] Running readiness predictions for active bins...")

bins = {
    "Bin #1 (Baling)": {
        "temperature_c": 54, "moisture_pct": 61, "ph_level": 6.8,
        "co2_ppm": 420, "nitrogen_pct": 2.8, "carbon_pct": 28,
        "temp_lag_1d": 53, "moist_lag_1d": 62, "temp_lag_3d": 51,
        "moist_lag_3d": 63, "temp_lag_7d": 49, "moist_lag_7d": 64,
        "temp_roll_7d_mean": 52, "temp_roll_7d_std": 2.1,
        "moist_roll_3d_mean": 62, "cn_ratio": 10,
        "temp_optimal": 1, "moist_optimal": 1, "ph_optimal": 1,
        "health_score": 3, "co2_risk_flag": 0, "compost_maturity": False
    },
    "Bin #2 (Sik)": {
        "temperature_c": 68, "moisture_pct": 58, "ph_level": 7.1,
        "co2_ppm": 680, "nitrogen_pct": 3.2, "carbon_pct": 22,
        "temp_lag_1d": 67, "moist_lag_1d": 59, "temp_lag_3d": 65,
        "moist_lag_3d": 60, "temp_lag_7d": 60, "moist_lag_7d": 61,
        "temp_roll_7d_mean": 64, "temp_roll_7d_std": 3.5,
        "moist_roll_3d_mean": 59, "cn_ratio": 6.875,
        "temp_optimal": 0, "moist_optimal": 1, "ph_optimal": 1,
        "health_score": 2, "co2_risk_flag": 1, "compost_maturity": False
    }
}

for bin_name, sensor_data in bins.items():
    input_df = pd.DataFrame([sensor_data])[feature_cols]
    predicted_rate = model.predict(input_df)[0]
    days_remaining = max(1, int((100 - predicted_rate * 7) / predicted_rate))
    action = "✅ On track" if sensor_data["co2_risk_flag"] == 0 else "⚠️  Turn compost + add carbon"

    print(f"\n  {bin_name}")
    print(f"    Decomposition Rate : {predicted_rate:.1f}%/day")
    print(f"    Est. Days to Ready : {days_remaining} days")
    print(f"    Recommendation     : {action}")

print("\n" + "=" * 60)
print("  Pipeline complete. Data ready for mobile app + dashboard.")
print("=" * 60)
