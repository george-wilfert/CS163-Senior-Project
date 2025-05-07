import pandas as pd


df_tpfs = pd.read_csv("https://storage.googleapis.com/databucket_seniorproj/TPFS_Data/TPFS.csv")
df_tpfs = df_tpfs.dropna(subset=['chained_value']).copy()
df_tpfs['year'] = df_tpfs['year'].astype(int)

# Helper function to convert monthly/quarterly to annual average 
def prepare_macro_data(path, date_col, value_col, agg='mean'):
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col])
    df['year'] = df[date_col].dt.year
    df_agg = df.groupby('year')[value_col].agg(agg).reset_index()
    
    readable_names = {
    'GDP': 'gdp',
    'PPIACO': 'ppi',
    'TTLCONS': 'total_construction_spending',
    'UNRATE': 'unemployment_rate',
    'IPUDN2373W200000000': 'employment_street_construction'
    }
    
    return df_agg.rename(columns={value_col: readable_names.get(value_col, value_col.lower())})

# Load and prepare external datasets 
df_gdp = prepare_macro_data("GDP.csv", "observation_date", "GDP", agg='mean')
df_ppi = prepare_macro_data("producer_price_index.csv", "observation_date", "PPIACO", agg='mean')
df_cons = prepare_macro_data("TTLCONS.csv", "observation_date", "TTLCONS", agg='mean')
df_unrate = prepare_macro_data("UNRATE.csv", "observation_date", "UNRATE", agg='mean')
df_emp = prepare_macro_data("employment_street_construction.csv", "observation_date", "IPUDN2373W200000000", agg='mean')

# Merge all indicators on year
macro = df_gdp \
    .merge(df_ppi, on='year') \
    .merge(df_cons, on='year') \
    .merge(df_unrate, on='year') \
    .merge(df_emp, on='year')

# Merge with TPFS 
df_merged = df_tpfs.merge(macro, on='year', how='left')

# Save final dataset for regression use 
#df_merged.to_csv("TPFS_enriched.csv", index=False)

#print("TPFS enriched dataset saved as 'TPFS_enriched.csv'")
