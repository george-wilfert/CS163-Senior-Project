import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# Load data
path = r"https://storage.googleapis.com/databucket_seniorproj/NHCCI%20Data/nhcci_refined.csv"
nhcci_data = pd.read_csv(path)
nhcci_data['date'] = pd.PeriodIndex(nhcci_data['quarter_period'].str.replace(' ', ''), freq='Q').to_timestamp()
nhcci_data.set_index('date', inplace=True)

# Focused predictors aligning with hypotheses
macro_vars = ["TTLCONS", "PPIACO"]

# Create lagged versions and percent change (growth rates)
for col in macro_vars:
    nhcci_data[f"{col}_lag1"] = nhcci_data[col].shift(1)

# Drop rows with NaNs caused by shifting or % change
predictor_cols = [f"{col}_lag1" for col in macro_vars] 
nhcci_model_data = nhcci_data.dropna(subset=predictor_cols + ["NHCCI-Seasonally-Adjusted"])

# Define X and y
X_macro = nhcci_model_data[predictor_cols].values
y_macro = nhcci_model_data["NHCCI-Seasonally-Adjusted"].values

# Time-aware split (no shuffle)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_macro, y_macro, test_size=0.3, shuffle=False)

# Scaling
scaler_macro = StandardScaler()
X_train_m_scaled = scaler_macro.fit_transform(X_train_m)
X_test_m_scaled = scaler_macro.transform(X_test_m)

# LassoCV model
lasso_cv_macro = LassoCV(alphas=np.logspace(-4, 1, 100), cv=TimeSeriesSplit(n_splits=5), max_iter=50000, random_state=0)
lasso_cv_macro.fit(X_train_m_scaled, y_train_m)

print("\n[Macro Model Aligned with Hypotheses] LassoCV Results:")
print("Optimal alpha:", lasso_cv_macro.alpha_)
print("Train R²:", lasso_cv_macro.score(X_train_m_scaled, y_train_m))
print("Test R²:", lasso_cv_macro.score(X_test_m_scaled, y_test_m))

# Coefficient inspection
coef_series_macro = pd.Series(lasso_cv_macro.coef_, index=predictor_cols)
selected_macro = coef_series_macro[coef_series_macro != 0]
print("\nSelected Macro Features by LassoCV:")
print(selected_macro)

# Plot
selected_macro.sort_values().plot(kind='barh', figsize=(8, 6), title='LassoCV Coefficients (Macro Predictors)')
plt.xlabel('Coefficient Value')
plt.tight_layout()
plt.show()

# Predicted vs Actual
y_pred = lasso_cv_macro.predict(scaler_macro.transform(X_macro))

# Add to DataFrame for plotting
plot_df = nhcci_model_data.copy()
plot_df["Predicted_NHCCI"] = y_pred

#plot_df.to_csv("nhcci_plot_df.csv", index=False)

# Format x-axis with year labels
plt.figure(figsize=(10, 5))
plt.plot(plot_df.index, plot_df["NHCCI-Seasonally-Adjusted"], label="Actual", linewidth=2)
plt.plot(plot_df.index, plot_df["Predicted_NHCCI"], label="Predicted", linestyle='--')
plt.title("LassoCV Regression Model - Actual vs. Predicted NHCCI")
plt.ylabel("NHCCI (Seasonally Adjusted)")
plt.xlabel("Date")
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
plt.legend()
plt.tight_layout()
plt.show()

# Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(nhcci_model_data[[*predictor_cols, "NHCCI-Seasonally-Adjusted"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix: Predictors and Target")
plt.tight_layout()
plt.show()
