# prepare_nhcci_data.py

import pandas as pd

def load_and_merge_all():
    # Load base NHCCI
    path = r"https://storage.googleapis.com/databucket_seniorproj/NHCCI%20Data/NHCCI.csv"
    nhcci_data = pd.read_csv(path)
    nhcci_data['quarter_fixed'] = nhcci_data['quarter'].str.replace(' ', '')
    nhcci_data['quarter_period'] = pd.PeriodIndex(nhcci_data['quarter_fixed'], freq='Q')
    nhcci_data.set_index('quarter_period', inplace=True)
    nhcci_data.drop(columns=['quarter', 'quarter_fixed'], inplace=True)
    nhcci_data.sort_index(inplace=True)

    # GDP
    gdp_df = pd.read_csv("GDP.csv")
    gdp_df['observation_date'] = pd.to_datetime(gdp_df['observation_date'])
    gdp_df['quarter_period'] = gdp_df['observation_date'].dt.to_period('Q')
    gdp_df.drop(columns=['observation_date'], inplace=True)
    gdp_df.set_index('quarter_period', inplace=True)
    nhcci_data = nhcci_data.join(gdp_df, how='inner')

    # TTLCONS
    ttlcons_df = pd.read_csv("TTLCONS.csv")
    ttlcons_df['observation_date'] = pd.to_datetime(ttlcons_df['observation_date'])
    ttlcons_df.set_index('observation_date', inplace=True)
    ttlcons_quarterly = ttlcons_df.resample('Q').mean()
    ttlcons_quarterly.index = ttlcons_quarterly.index.to_period('Q')
    ttlcons_quarterly.index.name = 'quarter_period'
    nhcci_data = nhcci_data.join(ttlcons_quarterly, how='inner')

    # UNRATE
    unrate_df = pd.read_csv("UNRATE.csv")
    unrate_df['observation_date'] = pd.to_datetime(unrate_df['observation_date'])
    unrate_df.set_index('observation_date', inplace=True)
    unrate_quarterly = unrate_df.resample('Q').mean()
    unrate_quarterly.index = unrate_quarterly.index.to_period('Q')
    unrate_quarterly.index.name = 'quarter_period'
    nhcci_data = nhcci_data.join(unrate_quarterly, how='inner')

    # PPI
    ppi_df = pd.read_csv("producer_price_index.csv")
    ppi_df['observation_date'] = pd.to_datetime(ppi_df['observation_date'])
    ppi_df['quarter_period'] = ppi_df['observation_date'].dt.to_period('Q')
    ppi_df = ppi_df.drop(columns=['observation_date'])
    ppi_df = ppi_df.groupby('quarter_period').mean()
    nhcci_data = nhcci_data.join(ppi_df, how='left')

    # Employment
    emp_df = pd.read_csv("employment_street_construction.csv")
    emp_df['observation_date'] = pd.to_datetime(emp_df['observation_date'])
    emp_df['quarter_period'] = emp_df['observation_date'].dt.to_period('Q')
    emp_quarterly = emp_df.groupby('quarter_period')['IPUDN2373W200000000'].mean()
    nhcci_data = nhcci_data.join(emp_quarterly.rename("Construction_Employment"), how='left')
    nhcci_data["Construction_Employment"].fillna(nhcci_data["Construction_Employment"].mean(), inplace=True)

    # Final prep
    nhcci_data.reset_index(inplace=True)  # restore quarter_period as a column
    nhcci_data['quarter_period'] = nhcci_data['quarter_period'].astype('period[Q]')
    nhcci_data['datetime'] = nhcci_data['quarter_period'].dt.to_timestamp()
    
    #nhcci_data.to_csv("nhcci_refined.csv", index=False)
    return nhcci_data