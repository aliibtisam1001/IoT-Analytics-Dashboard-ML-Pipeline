# 🔬 CompostIQ — IoT Analytics Dashboard & ML Pipeline

### End-to-end AI + IoT system for smart composting optimization
**Albukhary International University (AIU), Kedah, Malaysia** *Student Research Assistant (Mobile App Developer) — Supporting Project*

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-orange.svg)](https://scikit-learn.org)
[![pandas](https://img.shields.io/badge/pandas-Latest-darkblue.svg)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-Latest-blue.svg)](https://numpy.org)
[![Chart.js](https://img.shields.io/badge/Chart.js-Latest-lightgrey.svg)](https://www.chartjs.org)
[![Espressif](https://img.shields.io/badge/Espressif-ESP32-red.svg)](https://www.espressif.com/)

---

## 🎯 What This Project Does
This repository contains the complete backend + analytics stack for the CompostIQ smart composting system:
* **📡 Simulates ESP32 sensor networks** — Realistic IoT data generation
* **🔧 Engineers 20+ ML features** — Time-series lag, rolling stats, domain-specific flags
* **🤖 Trains Random Forest model** — Predicts compost decomposition rate ($R^2 = 0.94$)
* **📊 Serves live dashboard** — Dark-themed analytics with real-time Chart.js visualizations
* **🔮 Predicts readiness** — Estimates days-to-ready for each community bin

---

## 📁 Repository Structure

```text
IoT-Analytics-Dashboard-ML-Pipeline/
├── 📄 Project2_CompostIQ_IoT_Dashboard.html    # Interactive web dashboard
├── 📄 Project2_ML_Pipeline.py                  # Python ML simulation engine
├── 📄 README.md                                # This file
├── 📂 assets/
│   └── demo-dashboard.png                      # Dashboard screenshot
└── 📂 data/
    └── sensor_readings.csv                     # Generated dataset (optional)

```

---

## 🚀 Quick Start

### 1. Run the ML Pipeline

```bash
# Clone the repo
git clone [https://github.com/aliibtisam1001/IoT-Analytics-Dashboard-ML-Pipeline.git](https://github.com/aliibtisam1001/IoT-Analytics-Dashboard-ML-Pipeline.git)
cd IoT-Analytics-Dashboard-ML-Pipeline

# Install dependencies
pip install numpy pandas scikit-learn matplotlib

# Run the ML engine
python Project2_ML_Pipeline.py

```

#### Expected output:

```text
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

```

### 2. Open the Dashboard

```bash
# Simple: open in browser
double-click Project2_CompostIQ_IoT_Dashboard.html

# Or serve locally
python -m http.server 8000

```

👉 **Live Deployment Link:** [View Dashboard Live on GitHub Pages](https://aliibtisam1001.github.io/IoT-Analytics-Dashboard-ML-Pipeline/Project2_CompostIQ_IoT_Dashboard.html)

#### Dashboard features:

* ⏱️ Live clock with pulsing "Live Data" badge
* 📈 Temperature line chart (dual bin comparison)
* 🕸️ Radar chart for compost health parameters
* 📊 Sensor data table with status pills
* 🤖 ML prediction panel ($R^2$, MAE, feature importance)
* 📡 ESP32 device status panel

---

## 🧠 Machine Learning Pipeline

### Data Simulation

```python
# Realistic ESP32 sensor data
n_samples = 500
temperature = 40 + 15 * np.sin(days / 30) + noise   # 25-75°C range
moisture    = 55 + 10 * np.sin(days / 20 + 1) + noise # 30-90% range
ph_level    = 6.5 + 0.5 * np.sin(days / 15) + noise   # 5.5-8.5 range
co2_level   = 400 + 100 * abs(sin) + noise            # 300-800 ppm

```

### Feature Engineering (20+ features)

| Feature Type | Examples | Purpose |
| --- | --- | --- |
| **Lag Features** | `temp_lag_1d`, `moist_lag_7d` | Capture temporal dependencies |
| **Rolling Stats** | `temp_roll_7d_mean`, `temp_roll_7d_std` | Trend smoothing |
| **Domain Flags** | `temp_optimal`, `moist_optimal` | Expert knowledge injection |
| **Derived Metrics** | `cn_ratio`, `health_score`, `co2_risk_flag` | Composting-specific indicators |
| **Maturity Signal** | `compost_maturity` | End-of-cycle detection |

### Model: Random Forest Regressor

```python
RandomForestRegressor(
    n_estimators=200,    # High ensemble for stability
    max_depth=12,        # Prevent overfitting
    min_samples_split=5, # Minimum split threshold
    n_jobs=-1            # Parallel processing
)

```

#### Performance Metrics:

| Metric | Value | Interpretation |
| --- | --- | --- |
| **$R^2$ Score** | 0.9412 | 94% variance explained |
| **MAE** | 2.34%/day | Average prediction error |
| **RMSE** | 3.12%/day | Standard deviation of errors |

#### Top Features by Importance:

```text
temperature_c        0.312  ████████████████████████████████
moisture_pct         0.198  ████████████████████
ph_level             0.156  ████████████████
health_score         0.089  █████████
temp_roll_7d_mean    0.067  ███████
cn_ratio             0.045  ████
co2_ppm              0.038  ████
moist_lag_1d         0.022  ██

```

---

## 📊 Dashboard Architecture

```text
┌─────────────────────────────────────────────┐
│  🌱 CompostIQ — IoT Dashboard               │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐        │
│  │ Sensors │ │  Temp   │ │   ML    │        │
│  │   6     │ │  54°C   │ │ R²=0.94 │        │
│  └─────────┘ └─────────┘ └─────────┘        │
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

```

#### Design Choices:

* 🌑 **Dark theme** — GitHub-inspired color palette (`#0d1117`, `#161b22`)
* 🖥️ **Monospace fonts** — IBM Plex Mono for technical credibility
* 🎨 **Color coding** — Green (optimal), Orange (warning), Red (alert)
* ⚡ **Live updates** — Temperature simulates every 3 seconds
* 📱 **Responsive** — Works beautifully on desktop, tablet, and mobile browsers

---

## 🔗 Integration with Mobile App

This backend is designed to power the modular CompostIQ Mobile App infrastructure:

```text
Mobile App (Flutter)  ←──REST API──→  Flask/FastAPI Server  ←──→  ML Model
     │                                        │
     └──────── Firebase RTDB ←──────── ESP32 MQTT

```

### API Endpoints (Planned):

| Endpoint | Method | Description |
| --- | --- | --- |
| `/api/sensors` | `GET` | Latest real-time sensor readings |
| `/api/predict` | `POST` | Compost readiness time-series prediction |
| `/api/alerts` | `GET` | Active hazard/action alerts per community bin |
| `/api/history` | `GET` | Historical data compilation for custom charts |

---

## 🧪 Testing

```bash
# Run ML pipeline with custom parameters
python Project2_ML_Pipeline.py --samples 1000 --bins 5

# Validate model performance
python -c "from ml_pipeline import validate; validate()"

```

---

## 🗺️ Project Context

**AI + IoT Smart Composting Research Project** *Albukhary International University, Kedah, Malaysia*

#### Pilot Sites:

* 🏘️ **Baling Community** — Rural composting initiative
* 🏘️ **Sik Community** — Agricultural waste management
* 🏫 **AIU Campus** — University sustainability program

#### Research Goals:

1. Deploy robust ESP32 sensor networks in live community bins.
2. Collect high-fidelity temperature, moisture, pH, and $CO_2$ data.
3. Train predictive ML models for dynamic compost optimization.
4. Build a sleek mobile app layout for localized community engagement.
5. Significantly reduce organic waste landfill contributions.

---

## 📚 References

* Composting Temperature Guide — EPA
* IoT in Agriculture — MDPI Sensors Journal
* Random Forest Regression — scikit-learn documentation
* ESP32 Technical Datasheet — Espressif

---

## 👨‍💻 Author

### Ali Ibtisam

**Data Science Undergraduate @ AIU** *CGPA: 3.89/4.0 | Fully Funded Scholarship recipient*

📧 [aliibtisam1001@gmail.com]()

🔗 [LinkedIn](https://linkedin.com) | [GitHub](https://www.google.com/search?q=https://github.com/aliibtisam1001)

*School of Computing and Informatics, AIU*

---

## 📄 License

This project is licensed under the **MIT License** — open source and free for educational and research use.
