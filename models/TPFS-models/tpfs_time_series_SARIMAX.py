import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler

# === CONFIG ===
DATA_PATH = "https://storage.googleapis.com/databucket_seniorproj/TPFS_Data/TPFS_enriched.csv"
ORDER = (1, 0, 1)
MODES = ["Highways", "Transit", "Air"]

# === LOAD ===
df = pd.read_csv(DATA_PATH, parse_dates=["year"])
df["year"] = pd.to_datetime(df["year"])
df.set_index("year", inplace=True)

# === LOOP THROUGH EACH MODE ===
for mode in MODES:
    print(f"\n===== MODE: {mode} =====")

    # Subset data
    ts = df[df["mode"] == mode].copy()
    ts = ts.loc["2012":]  # Cut to 2012–2022

    # Define exogenous vars (only 3 retained)
    exog_vars = [
        "ppi",
        "employment_street_construction",
    ]
    exog = ts[exog_vars]

    # Scale exogenous inputs
    exog_scaled = pd.DataFrame(
        StandardScaler().fit_transform(exog),
        index=exog.index,
        columns=exog.columns
    )

    # Model
    model = SARIMAX(
        ts["chained_value"],
        order=ORDER,
        exog=exog_scaled,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)

    # Output summary
    print(results.summary())
