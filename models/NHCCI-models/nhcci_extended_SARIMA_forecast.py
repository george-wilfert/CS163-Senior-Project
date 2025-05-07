import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np

# Load and prepare data
from prepare_nhcci_data import load_and_merge_all

# Load dataset
nhcci_data = load_and_merge_all()
nhcci_data.set_index('quarter_period', inplace=True)

# Define exogenous variables
exog_vars = ["GDP", "TTLCONS"]

# Filter data: use only recent years (e.g., post-2014)
recent_data = nhcci_data[nhcci_data['datetime'] >= '2016-01-01'].copy()

# Normalize exogenous variables
scaler = StandardScaler()
recent_data[exog_vars] = scaler.fit_transform(recent_data[exog_vars])

# Define training targets
y_recent = recent_data["NHCCI-Seasonally-Adjusted"]
exog_recent = recent_data[exog_vars]

# Fit SARIMA model on recent data
model = SARIMAX(
    y_recent,
    exog=exog_recent,
    order=(1, 1, 2),
    seasonal_order=(1, 1, 1, 4),
    enforce_stationarity=False,
    enforce_invertibility=False
)
model_fit = model.fit(maxiter=500, disp=False)

# ---------- PRINT MODEL STATS ----------
print("AIC:", model_fit.aic)
print("BIC:", model_fit.bic)

# Compute in-sample RMSE for training fit
train_preds = model_fit.fittedvalues
rmse = np.sqrt(mean_squared_error(y_recent.iloc[1:], train_preds.iloc[1:]))  # skip first differenced value
print("In-sample RMSE:", round(rmse, 4))

# ---------- FORECAST ----------
n_steps = 12  # e.g., 3 years = 12 quarters
last_period = recent_data.index[-1]
future_periods = pd.period_range(start=last_period + 1, periods=n_steps, freq='Q')

# ---------- PROJECT FUTURE GDP AND TTLCONS USING LINEAR TREND ----------

def extrapolate_linear_trend(series, n_steps):
    # Use last 8 quarters to estimate trend
    recent = series[-12:]
    periods = np.arange(len(recent))
    coeffs = np.polyfit(periods, recent, 1)  # Linear fit
    future_periods = np.arange(len(recent), len(recent) + n_steps)
    return pd.Series(np.polyval(coeffs, future_periods), index=future_periods)

# De-normalize for extrapolation
gdp_actual = scaler.inverse_transform(recent_data[exog_vars])
recent_gdp = pd.Series(gdp_actual[:, 0], index=recent_data.index)
recent_ttl = pd.Series(gdp_actual[:, 1], index=recent_data.index)

# Project raw GDP and TTLCONS
gdp_proj_raw = extrapolate_linear_trend(recent_gdp, n_steps)
ttl_proj_raw = extrapolate_linear_trend(recent_ttl, n_steps)

# Re-normalize using existing scaler
future_exog_raw = pd.DataFrame({
    'GDP': gdp_proj_raw.values,
    'TTLCONS': ttl_proj_raw.values
})
future_exog_scaled = pd.DataFrame(scaler.transform(future_exog_raw), columns=exog_vars)

# Assign index for proper alignment
future_exog_scaled.index = pd.period_range(start=last_period + 1, periods=n_steps, freq='Q')
future_exog = future_exog_scaled

# Get forecast + confidence intervals
forecast_result = model_fit.get_forecast(steps=n_steps, exog=future_exog)
forecast_mean = forecast_result.predicted_mean
conf_int = forecast_result.conf_int()

# ---------- PLOT ----------
plt.figure(figsize=(14, 7))
plt.plot(recent_data['datetime'], y_recent, label="Recent NHCCI (Training)", color='black')
plt.plot(forecast_mean.index.to_timestamp(), forecast_mean, label="Forecasted NHCCI (Next 3 Years)", color='orange', linestyle='--')
plt.fill_between(
    forecast_mean.index.to_timestamp(),
    conf_int.iloc[:, 0],
    conf_int.iloc[:, 1],
    color='orange',
    alpha=0.3,
    label="95% Confidence Interval"
)
plt.title("NHCCI Forecast Using Recent Trends Only (w/ Confidence Intervals)")
plt.xlabel("Date")
plt.ylabel("NHCCI Seasonally Adjusted Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
