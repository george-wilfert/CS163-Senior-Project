import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings as wrngs

url = "https://storage.googleapis.com/databucket_seniorproj/NHCCI%20Data/NHCCI.csv"
df = pd.read_csv(url)

# Basic data insights
print(df.head(10))
print(f"Shape of the data \n {df.shape} \n")

print("Dataset Info: ")
df.info()
print("\n")

print(f"Description of numeric data: \n {df.describe()} \n")

print(f"Columns as list for reference: \n {df.columns.tolist()} \n")

df.replace(['-', 'N/A', 'NULL', '', 'None', 'NaN'], np.nan, inplace=True)
print("Counting missing entries")
print(df.isnull().sum())
print("\n")

print("Checking duplicate values: ")
duplicates = df.nunique()
print(duplicates)
print("\n")

plt.figure(figsize=(12, 6))
sns.lineplot(x='quarter', y='NHCCI', data=df, marker='o')
xtick_positions = np.arange(0, len(df), step=4)  
plt.xticks(df['quarter'][xtick_positions], rotation=90)
plt.title('National Highway Construction Cost Trend Over Time')
plt.xlabel('Quarter')
plt.ylabel('NHCCI Value')
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(df['NHCCI'], bins=30, kde=True)
plt.title('Distribution of NHCCI Values')
plt.xlabel('NHCCI Value')
plt.ylabel('Frequency')
plt.show()

num_cols = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(12, 8))
sns.heatmap(df[num_cols].corr(), annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of NHCCI and Cost Components')
plt.show()

cost_components = [
    'grp9-pct-change-Asphalt', 'grp9-pct-change-Concrete', 
    'grp9-pct-change-Base-Stone', 'grp9-pct-change-Drainage'
]
df_melted = df.melt(id_vars=['quarter'], value_vars=cost_components, 
                     var_name='Component', value_name='Percentage Change')

plt.figure(figsize=(14, 6))
sns.lineplot(x='quarter', y='Percentage Change', hue='Component', data=df_melted, marker='o')
xtick_positions = np.arange(0, len(df), step=4)  
plt.xticks(df['quarter'][xtick_positions], rotation=90)
plt.title('Cost Component Contributions to NHCCI Over Time')
plt.xlabel('Quarter')
plt.ylabel('Percentage Change')
plt.legend()
plt.grid()
plt.show()



