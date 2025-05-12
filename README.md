# Infrastructure Insight Platform

An interactive web app that uses time series forecasting, clustering, and machine learning to analyze U.S. infrastructure spending. The app visualizes trends, detects anomalies, and reveals macroeconomic patterns within datasets related to highway construction costs and public finance spending.

## 🌐 Website

➡️ [Live Website Link](https://cs163-senior-project.wl.r.appspot.com/)

## 🏗️ Project Overview

This project investigates infrastructure spending trends in the United States using two key datasets:

- **NHCCI (National Highway Construction Cost Index)**: Measures changes in highway construction costs over time, providing a benchmark for infrastructure inflation.
- **TPFS (Transportation Public Finance Statistics)**: Captures federal, state, and local government spending across different transportation modes like highways, transit, rail, water, and air.
- **Energy Infastructure**: Shows relationships between state energy infastructure and systems in order to predict indicators and dive deeper into future insights

By combining statistical modeling and machine learning, we aim to provide actionable insights for policy analysts, economists, and urban planners.

## 📦 Setup Instructions

To run locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/george-wilfert/CS163-Senior-Project.git
   cd CS163-Senior-Project
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:
   ```bash
   python appengine/app.py
   ```

⚠️ Ensure the Google Cloud storage bucket is linked properly to serve hosted images for the dashboard.

## 🔁 Pipeline Flow

Raw Datasets → Data Cleaning → Feature Engineering → Modeling & Analysis → Plot Generation → Interactive Web Dashboard

### ➤ NHCCI Pipeline:
- **Data Source**: National Highway Construction Cost Index (Quarterly)

**Modeling Techniques**:
- SARIMAX for forecasting  
- K-Means for clustering cost regimes  
- LassoCV for feature selection and macroeconomic regression

**Outputs**:
- Trend forecasts  
- Structural investment periods  
- Economic correlation heatmaps

### ➤ TPFS Pipeline:
- **Data Source**: DOT’s Public Finance Database (Capital/Non-Capital by transit mode)

**Modeling Techniques**:
- Ridge Regression for log-transformed budget prediction  
- Time Series Decomposition (Seasonal-Trend)  
- Isolation Forest for anomaly detection

**Outputs**:
- Macro-level spending signals  
- Seasonality trends by government level  
- Anomalies with transit-level drilldowns

## 📁 Repository Structure

```
.
├── app.py                      # Main Dash application entry point
├── pages/
│   ├── home.py                 # Homepage content
│   ├── objectives.py           # Project goals and dataset explanation
│   ├── analytics_method.py     # Technical explanation of models used
│   ├── major_findings.py       # Key results and interactive plots
│   └── more_findings.py        # Additional insights and minor results
├── data/
│   ├── NHCCI/                  # NHCCI cleaned CSVs and macroeconomic join
│   └── TPFS/                   # Public finance cleaned datasets
├── plots/
│   └── static_images/          # Hosted .png plot files (Google Cloud Bucket)
├── models/
│   ├── nhcci_sarima.py
│   ├── nhcci_clustering.py
│   ├── tpfs_ridge.py
│   └── tpfs_isolation.py
├── requirements.txt
└── README.md
```

## 📈 Technologies Used

- Python: Pandas, NumPy, scikit-learn, statsmodels, plotly  
- Dash/Plotly: Interactive dashboard and multi-page routing  
- Google Cloud Storage: Static image hosting  
- GitHub: Collaboration and version control  

## ✅ Authors

- Brendan Dishion  
- George Wilfert
