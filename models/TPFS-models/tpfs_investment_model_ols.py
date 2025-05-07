import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("https://storage.googleapis.com/databucket_seniorproj/TPFS_Data/TPFS_enriched.csv")

# Drop missing and non-positive values
df_clean = df.dropna(subset=[
    "chained_value", "gdp", "unemployment_rate", "mode", "year"
])
df_clean = df_clean[df_clean["chained_value"] > 0]

# Log-transform the dependent variable
df_clean["log_chained_value"] = np.log(df_clean["chained_value"])

# Optional: Drop low-count mode-year combos (e.g., keep those with >10 obs)
mode_year_counts = df_clean.groupby(["mode", "year"]).size().reset_index(name="count")
sufficient_data = mode_year_counts[mode_year_counts["count"] > 10]
df_clean = df_clean.merge(sufficient_data[["mode", "year"]], on=["mode", "year"], how="inner")

# Optional: Remove "Total" mode if it combines other modes
df_clean = df_clean[df_clean["mode"] != "Total"]

# Scale macroeconomic variables
scaler = StandardScaler()
df_clean[["gdp_scaled", "unemployment_scaled"]] = scaler.fit_transform(
    df_clean[["gdp", "unemployment_rate"]]
)

# Optional: Add numeric time trend if desired
df_clean["year_trend"] = df_clean["year"] - df_clean["year"].min()

# Fit OLS model
formula = "log_chained_value ~ gdp_scaled + unemployment_scaled + year_trend + C(mode)"
model = smf.ols(formula=formula, data=df_clean).fit(cov_type="HC3")
print(model.summary())

sns.barplot(data=df_clean, x='mode', y='log_chained_value', estimator='mean', errorbar='sd')
plt.title("Average Infrastructure Investment by Mode")
plt.ylabel("Log of Chained Value")
plt.xlabel("Transportation Mode")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(data=df_clean, x='year', y='log_chained_value', hue='mode', estimator='mean', errorbar=None)
plt.title("Investment Trends Over Time by Mode")
plt.ylabel("Log of Chained Value")
plt.xlabel("Year")
plt.legend(
    title='Mode',
    bbox_to_anchor=(1.05, 1), 
    loc='upper left',
    borderaxespad=0.
)
plt.tight_layout(rect=[0, 0, 0.85, 1])  
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clean, x='gdp', y='log_chained_value', hue='mode')
plt.title("GDP vs Infrastructure Investment")
plt.xlabel("GDP (in billions)")
plt.ylabel("Log of Chained Value")
plt.legend(
    title='Mode',
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clean, x='unemployment_rate', y='log_chained_value', hue='mode')
plt.title("Unemployment Rate vs Infrastructure Investment")
plt.xlabel("Unemployment Rate (percentage)")
plt.ylabel("Log of Chained Value")
plt.legend(
    title='Mode',
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()