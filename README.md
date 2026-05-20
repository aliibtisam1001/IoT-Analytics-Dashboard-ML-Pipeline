🔬 CompostIQ — IoT Analytics Dashboard & ML Pipeline
End-to-end AI + IoT system for smart composting optimization
Albukhary International University (AIU), Kedah, Malaysia
Student Research Assistant (Mobile App Developer) — Supporting Project
https://python.org
https://scikit-learn.org
https://pandas.pydata.org
https://numpy.org
https://www.chartjs.org
https://www.espressif.com/
🎯 What This Project Does
This repository contains the complete backend + analytics stack for the CompostIQ smart composting system:
📡 Simulates ESP32 sensor networks — Realistic IoT data generation
🔧 Engineers 20+ ML features — Time-series lag, rolling stats, domain-specific flags
🤖 Trains Random Forest model — Predicts compost decomposition rate (R² = 0.94)
📊 Serves live dashboard — Dark-themed analytics with real-time Chart.js visualizations
🔮 Predicts readiness — Estimates days-to-ready for each community bin
📁 Repository Structure
plain
Copy
compostiq-iot-ml-pipeline/
├── 📄 Project2_CompostIQ_IoT_Dashboard.html    # Interactive web dashboard
├── 📄 Project2_ML_Pipeline.py                  # Python ML simulation engine
├── 📄 README.md                                  # This file
├── 📂 assets/
│   └── demo-dashboard.png                        # Dashboard screenshot
└── 📂 data/
    └── sensor_readings.csv                       # Generated dataset (optional)
🚀 Quick Start
1. Run the ML Pipeline
bash
Copy
# Clone the repo
git clone https://github.com/aliibtisam1001/compostiq-iot-ml-pipeline.git
cd compostiq-iot-ml-pipeline

# Install dependencies
pip install numpy pandas scikit-learn matplotlib

# Run the ML engine
python Project2_ML_Pipeline.py
Expected output:
plain
Copy
============================================================
  CompostIQ — ML Pipeline v1.0
  AI + IoT Smart Composting Research · AIU
============================================================

[DATA] Loaded 500 sensor readings from ESP32 network
...
[RESULTS] Model Performance:
  R² Score : 0.9412
  MAE      : 2.34%/day
  RMSE     : 3.12%/day

[PREDICT] Running readiness predictions for active bins...

  Bin #1 (Baling)
    Decomposition Rate : 4.2%/day
    Est. Days to Ready : 14 days
    Recommendation     : ✅ On track

  Bin #2 (Sik)
    Decomposition Rate : 5.8%/day
    Est. Days to Ready : 11 days
    Recommendation     : ⚠️  Turn compost + add carbon
2. Open the Dashboard
bash
Copy
# Simple: open in browser
double-click Project2_CompostIQ_IoT_Dashboard.html

# Or serve locally
python -m http.server 8000
# Visit: https://aliibtisam1001.github.io/IoT-Analytics-Dashboard-ML-Pipeline/Project2_CompostIQ_IoT_Dashboard.html
Dashboard features:
⏱️ Live clock with pulsing "Live Data" badge
📈 Temperature line chart (dual bin comparison)
🕸️ Radar chart for compost health parameters
📊 Sensor data table with status pills
🤖 ML prediction panel (R², MAE, feature importance)
📡 ESP32 device status panel
🧠 Machine Learning Pipeline
Data Simulation
Python
Copy
# Realistic ESP32 sensor data
n_samples = 500
temperature = 40 + 15 * np.sin(days / 30) + noise   # 25-75°C range
moisture    = 55 + 10 * np.sin(days / 20 + 1) + noise # 30-90% range
ph_level    = 6.5 + 0.5 * np.sin(days / 15) + noise   # 5.5-8.5 range
co2_level   = 400 + 100 * abs(sin) + noise            # 300-800 ppm
Feature Engineering (20+ features)
Table
Feature Type	Examples	Purpose
Lag Features	temp_lag_1d, moist_lag_7d	Capture temporal dependencies
Rolling Stats	temp_roll_7d_mean, temp_roll_7d_std	Trend smoothing
Domain Flags	temp_optimal, moist_optimal, ph_optimal	Expert knowledge injection
Derived Metrics	cn_ratio, health_score, co2_risk_flag	Composting-specific indicators
Maturity Signal	compost_maturity	End-of-cycle detection
Model: Random Forest Regressor
Python
Copy
RandomForestRegressor(
    n_estimators=200,    # High ensemble for stability
    max_depth=12,        # Prevent overfitting
    min_samples_split=5, # Minimum split threshold
    n_jobs=-1            # Parallel processing
)
Performance:
Table
Metric	Value	Interpretation
R² Score	0.9412	94% variance explained
MAE	2.34%/day	Average prediction error
RMSE	3.12%/day	Standard deviation of errors
Top Features by Importance
plain
Copy
temperature_c        0.312  ████████████████████████████████
moisture_pct         0.198  ████████████████████
ph_level             0.156  ████████████████
health_score         0.089  █████████
temp_roll_7d_mean    0.067  ███████
cn_ratio             0.045  ████
co2_ppm              0.038  ████
moist_lag_1d         0.022  ██
📊 Dashboard Architecture
plain
Copy
┌─────────────────────────────────────────────┐
│  🌱 CompostIQ — IoT Dashboard               │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │ Sensors │ │  Temp   │ │   ML    │       │
│  │   6     │ │  54°C   │ │ R²=0.94 │       │
│  └─────────┘ └─────────┘ └─────────┘       │
│  ┌──────────────────┐ ┌─────────────────┐   │
│  │ Temperature Chart│ │  Health Radar   │   │
│  │ (Chart.js Line)  │ │  (Chart.js Rad) │   │
│  └──────────────────┘ └─────────────────┘   │
│  ┌──────────────┐ ┌──────────────────────┐   │
│  │ ML Predictions│ │  ESP32 Device Status │   │
│  │ Bin #1: 14d   │ │  ● Connected        │   │
│  │ Bin #2: 11d   │ │  ⚠ Alert            │   │
│  └──────────────┘ └──────────────────────┘   │
│  ┌────────────────────────────────────────┐   │
│  │ Recent Sensor Readings Table          │   │
│  │ Time | Device | Temp | Moist | pH | CO₂│   │
│  └────────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
Design choices:
🌑 Dark theme — GitHub-inspired color palette (#0d1117, #161b22)
🖥️ Monospace fonts — IBM Plex Mono for technical credibility
🎨 Color coding — Green (optimal), Orange (warning), Red (alert)
⚡ Live updates — Temperature simulates every 3 seconds
📱 Responsive — Works on desktop, tablet, and mobile browsers
🔗 Integration with Mobile App
This backend powers the CompostIQ Mobile App:
plain
Copy
Mobile App (Flutter)  ←──REST API──→  Flask/FastAPI Server  ←──→  ML Model
     │                                        │
     └──────── Firebase RTDB ←──────── ESP32 MQTT
API Endpoints (planned):
Table
Endpoint	Method	Description
/api/sensors	GET	Latest sensor readings
/api/predict	POST	Compost readiness prediction
/api/alerts	GET	Active alerts per bin
/api/history	GET	Historical data for charts
📸 Screenshots
ML Pipeline Output
Live Dashboard
Sensor Data Table
🧪 Testing
bash
Copy
# Run ML pipeline with custom parameters
python Project2_ML_Pipeline.py --samples 1000 --bins 5

# Validate model performance
python -c "from ml_pipeline import validate; validate()"
🗺️ Project Context
AI + IoT Smart Composting Research Project
Albukhary International University, Kedah, Malaysia
Pilot Sites:
🏘️ Baling Community — Rural composting initiative
🏘️ Sik Community — Agricultural waste management
🏫 AIU Campus — University sustainability program
Research Goals:
Deploy ESP32 sensor networks in community bins
Collect temperature, moisture, pH, CO₂ data
Train ML models for compost optimization
Build mobile app for community engagement
Reduce organic waste landfill contribution
📚 References
Composting Temperature Guide — EPA
IoT in Agriculture — MDPI Sensors Journal
Random Forest Regression — scikit-learn docs
ESP32 Datasheet — Espressif
👨‍💻 Author
Ali Ibtisam
Data Science Undergraduate @ AIU
CGPA: 3.89/4.0 | Fully Funded Scholarship
📧 aliibtisam1001@gmail.com
🔗 LinkedIn | GitHub

School of Computing and Informatics, AIU
📄 License
MIT License — open source for educational and research use.
<p align="center">
  <sub>🔬 Research-grade IoT + ML pipeline · Built for sustainable communities · AIU 2026</sub>
</p>