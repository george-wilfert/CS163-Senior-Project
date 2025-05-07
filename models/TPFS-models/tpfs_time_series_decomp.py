import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import ccf
import seaborn as sns

# Load and prepare dataset
df_tpfs = pd.read_csv("https://storage.googleapis.com/databucket_seniorproj/TPFS_Data/TPFS_enriched.csv", parse_dates=["year"])
df_ts = df_tpfs.dropna(subset=["chained_value", "year"]).copy()
df_ts["year"] = df_ts["year"].dt.to_period("Y").dt.to_timestamp()

# Group and pivot by gov_level
ts_gov_level = df_ts.groupby(["year", "gov_level"])["chained_value"].sum().reset_index()
ts_pivot = ts_gov_level.pivot(index="year", columns="gov_level", values="chained_value")

# Loop through each government level
for gov_level in ts_pivot.columns:
    series = ts_pivot[gov_level].dropna()

    # Skip if there are fewer than 3 data points
    if len(series) < 3:
        print(f"Skipping {gov_level} — not enough data.")
        continue

    # Ensure sorted index with annual frequency
    series = series.sort_index()
    series.index = pd.date_range(start=series.index.min(), periods=len(series), freq="YE")

    # Decompose without forcing frequency
    decomp_result = seasonal_decompose(series, model="additive", period=1, extrapolate_trend='freq')

    # Plot
    fig = decomp_result.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle(f"Time Series Decomposition - {gov_level} Infrastructure Spending", fontsize=14)
    plt.tight_layout()
    plt.show()
    
df_highway = df_tpfs[df_tpfs["mode"] == "Highways"].copy()
df_highway = df_highway.groupby("year")[["chained_value", "ppi", "gdp", "total_construction_spending"]].sum().reset_index()
df_highway.to_csv("TPFS_economic_indicators_time_series.csv", index=False)
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot chained_value on primary y-axis
ax1.plot(df_highway["year"], df_highway["chained_value"], color="tab:blue", label="Chained Value")
ax1.set_xlabel("Year")
ax1.set_ylabel("Dollars (Chained)", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

# Secondary y-axis for ppi
ax2 = ax1.twinx()
ax2.plot(df_highway["year"], df_highway["ppi"], color="tab:orange", label="PPI")
ax2.set_ylabel("PPI", color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")

ax3 = ax1.twinx()
ax3.plot(df_highway["year"], df_highway["gdp"], color="tab:red", label="GDP")
ax3.set_ylabel("GDP", color="tab:red")
ax3.tick_params(axis="y", labelcolor="tab:red")

ax4 = ax1.twinx()
ax4.plot(df_highway["year"], df_highway["total_construction_spending"], color="tab:green", label="TTLCONS")
ax4.set_ylabel("TTLCONS", color="tab:green")
ax4.tick_params(axis="y", labelcolor="tab:green")

plt.title("Highways Spending vs Economic Metrics")
fig.tight_layout()
plt.show()

def plot_lagged_correlation(df, x_col, y_col, max_lag=5, title=None):
    lags = range(max_lag)
    corrs = [df[x_col].corr(df[y_col].shift(-l)) for l in lags]

    plt.figure(figsize=(6, 4))
    plt.bar(lags, corrs, color="steelblue")
    plt.title(title or f"Cross-Correlation: {x_col} leads {y_col}")
    plt.xlabel("Lag (years)")
    plt.ylabel("Correlation")
    plt.ylim(-1, 1)
    plt.xticks(lags)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_lagged_correlation(df_highway, "chained_value", "ppi", title="Cross-Correlation: Spending leads PPI")
plot_lagged_correlation(df_highway, "chained_value", "gdp", title="Cross-Correlation: Spending leads GDP")
plot_lagged_correlation(df_highway, "chained_value", "total_construction_spending", title="Cross-Correlation: Spending leads Construction")

# df_corr = df_highway[["chained_value", "gdp"]].rolling(window=3).corr().dropna()
# df_corr.loc[df_corr.index.get_level_values(1) == "gdp"]["chained_value"].plot(
#     title="3-Year Rolling Correlation: Spending vs GDP", figsize=(10, 4)
# )
# plt.axhline(0, linestyle="--", color="gray")
# plt.tight_layout()
# plt.show()
