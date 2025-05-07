import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.dates as mdates
from prepare_nhcci_data import load_and_merge_all 

nhcci_data = load_and_merge_all()
nhcci_data['datetime'] = pd.PeriodIndex(nhcci_data['quarter_period'].astype(str), freq='Q').to_timestamp()

plt.figure(figsize=(14,7))
plt.plot(nhcci_data["datetime"], nhcci_data["NHCCI-Seasonally-Adjusted"], label='NHCCI Seasonally Adjusted Value')
plt.title('National Highway Construction Cost Index Over Time')
plt.xlabel('Date')
plt.ylabel('NHCCI Seasonally Adjusted Value')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

result_original = adfuller(nhcci_data["NHCCI-Seasonally-Adjusted"])
print(f"ADF Statistic (Original): {result_original[0]:.4f}")
print(f"p-value (Original): {result_original[1]:.4f}")
if result_original[1] < 0.05:
    print("Interpretation: The original series is Stationary.\n")
else:
    print("Interpretation: The original series is Non-Stationary.\n")

nhcci_data['NHCCI_Seasonally_Adjusted_Diff'] = nhcci_data['NHCCI-Seasonally-Adjusted'].diff()

result_diff = adfuller(nhcci_data["NHCCI_Seasonally_Adjusted_Diff"].dropna())
print(f"ADF Statistic (Differenced): {result_diff[0]:.4f}")
print(f"p-value (Differenced): {result_diff[1]:.4f}")
if result_diff[1] < 0.05:
    print("Interpretation: The differenced series is Stationary.")
else:
    print("Interpretation: The differenced series is Non-Stationary.")

plt.figure(figsize=(14, 7))
plt.plot(nhcci_data["datetime"], nhcci_data['NHCCI_Seasonally_Adjusted_Diff'], label='Differenced NHCCI Seasonally Adjusted Value', color='orange')
plt.title('Differenced NHCCI Seasonally Adjusted Value Over Time')
plt.xlabel('Date')
plt.ylabel('Differenced NHCCI Seasonally Adjusted Value')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(nhcci_data['NHCCI_Seasonally_Adjusted_Diff'].dropna(), lags=40, ax=axes[0])
axes[0].set_title('Autocorrelation Function (ACF)')
plot_pacf(nhcci_data['NHCCI_Seasonally_Adjusted_Diff'].dropna(), lags=40, ax=axes[1])
axes[1].set_title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()

for col in ["GDP", "TTLCONS"]:
    nhcci_data[col] = (nhcci_data[col] - nhcci_data[col].mean()) / nhcci_data[col].std()

train_size = int(len(nhcci_data) * 0.80)
train, test = nhcci_data.iloc[:train_size], nhcci_data.iloc[train_size:]
exog_vars = ["GDP", "TTLCONS"]
exog_train = train[exog_vars]
exog_test = test[exog_vars]

model = SARIMAX(
    train["NHCCI-Seasonally-Adjusted"],
    exog=exog_train,
    order=(1, 1, 2),
    seasonal_order=(1, 1, 1, 4),
    enforce_stationarity=False,
    enforce_invertibility=False
)
model_fit = model.fit(maxiter=500, disp=True)
print(model_fit.mle_retvals)

forecasting = model_fit.forecast(steps=len(test), exog=exog_test)

plt.figure(figsize=(14,7))
plt.plot(train["datetime"], train["NHCCI-Seasonally-Adjusted"], label='Train', color='#203147')
plt.plot(test["datetime"], test["NHCCI-Seasonally-Adjusted"], label='Test', color='#01ef63')
plt.plot(test["datetime"], forecasting, label='Forecast', color='orange')
plt.title('Close Price Forecast')
plt.xlabel('Date')
plt.ylabel('NHCCI Seasonally Adjusted Value')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

print(f"AIC: {model_fit.aic}")
print(f"BIC: {model_fit.bic}")
forecast = forecasting[:len(test)]
test_close = test["NHCCI-Seasonally-Adjusted"][:len(forecast)]
rmse = np.sqrt(mean_squared_error(test_close, forecast))
print(f"RMSE: {rmse:.4f}")

plt.figure(figsize=(14,7))
plt.plot(train["datetime"], train["NHCCI-Seasonally-Adjusted"], label='Train', color='black')
plt.plot(test["datetime"], test["NHCCI-Seasonally-Adjusted"], label='Actual (Test)', color='green')
plt.plot(test["datetime"], forecasting, label='Forecast', color='orange')
plt.fill_between(test["datetime"], 
                 forecasting - 1.96 * np.std(model_fit.resid), 
                 forecasting + 1.96 * np.std(model_fit.resid), 
                 color='orange', alpha=0.2, label='95% Confidence Interval')
plt.title("SARIMA Forecast vs Actual NHCCI Values")
plt.xlabel("Date")
plt.ylabel("NHCCI Seasonally Adjusted Value")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

residuals = test["NHCCI-Seasonally-Adjusted"][:len(forecasting)] - forecasting

residuals_df = pd.DataFrame({
    "datetime": test["datetime"],
    "residuals": residuals
})
#residuals_df.to_csv("nhcci_SARIMA_residuals.csv", index=False)

plot_df = test[["datetime", "NHCCI-Seasonally-Adjusted"]].copy()
plot_df["Forecast"] = forecasting.values
#plot_df.to_csv("nhcci_SARIMA_plot_df.csv", index=False)

plt.figure(figsize=(14,5))
plt.plot(test["datetime"], residuals, marker='o', linestyle='-')
plt.axhline(y=0, color='black', linestyle='--')
plt.title("Forecast Residuals (Actual - Forecast)")
plt.xlabel("Date")
plt.ylabel("Residual")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# print(test.head())
# print(train.head())
#train.to_csv("nhcci_SARIMA_train.csv", index=False)
#test.to_csv("nhcci_SARIMA_test.csv", index=False)

# forecast_df = pd.DataFrame({
#     "datetime": test["datetime"][:len(forecast)],
#     "NHCCI-Seasonally-Adjusted": test_close.values,
#     "Forecast": forecast
# })
# forecast_df.to_csv("nhcci_SARIMA_forecast_plot.csv", index=False)