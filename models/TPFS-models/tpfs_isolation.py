from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd

df = pd.read_csv("https://storage.googleapis.com/databucket_seniorproj/TPFS_Data/TPFS.csv")
# Drop rows with missing chained_value
df_clean = df.dropna(subset=['chained_value']).copy()

# Initialize a new column to mark anomalies
df_clean['anomaly'] = 0

# Loop through each mode and gov_level segment
for (mode, gov_level), group in df_clean.groupby(['mode', 'gov_level']):
    if len(group) < 5:
        continue  # skip tiny segments
    
    # Prepare data for model
    X = group[['chained_value']].values

    # Isolation Forest Model
    clf = IsolationForest(contamination=0.1, random_state=22)
    clf.fit(X)
    preds = clf.predict(X)  # -1 is anomaly, 1 is normal

    # Apply predictions to the dataframe
    df_clean.loc[group.index, 'anomaly'] = preds

# Filter anomalies
anomalies = df_clean[df_clean['anomaly'] == -1]
anomalies_clean = anomalies.dropna(subset=['chained_value', 'value', 'deflator'])

# Show top N by chained_value
top_anomalies_sorted = anomalies_clean.sort_values(by='chained_value', ascending=False)[
    ['year', 'mode', 'gov_level', 'chained_value', 'desccription']
]

#top_anomalies_sorted.to_csv("TPFS_anomalies_top_sorted.csv", index=False)

#anomalies_clean.to_csv("TPFS_anomalies_clean.csv", index=False)

import matplotlib.pyplot as plt
import seaborn as sns

anom_summary = anomalies_clean.groupby(['year', 'mode', 'gov_level']).size().reset_index(name='anomaly_count')
anom_summary = anom_summary.sort_values('anomaly_count', ascending=False)
#anom_summary.to_csv("TPFS_anomalies_summary.csv", index=False)

plt.title("Boxplot of Gov't Level vs Chained Spending Value in Transportation Infrastructure")
sns.boxplot(data=anomalies_clean, x='gov_level', y='chained_value')
plt.show()

plt.title("Distribution Heatmap of Transit Type vs. Gov't Level Anomalies")
heat_data = anomalies_clean.groupby(['mode', 'gov_level']).size().unstack(fill_value=0)
sns.heatmap(heat_data, annot=True, fmt="d", cmap="YlGnBu")
plt.show()

heatmap_anom = anomalies_clean[anomalies_clean['anomaly'] == -1].groupby(['mode', 'year']).size().unstack(fill_value=0)
sns.heatmap(heatmap_anom, cmap="Reds", annot=True)
plt.title("Anomaly Count by Mode and Year")
plt.show()

df = pd.read_csv("TPFS_anomalies_top_sorted.csv")

# Convert year to string
df['year'] = df['year'].astype(str)
df['label'] = df['desccription'] + " (" + df['year'] + ")" # Combine description and year for more informative y-axis
df = df.sort_values(by='chained_value', ascending=False).head(15) # Sort by highest spending
plt.figure(figsize=(12, 8))
sns.barplot(data=df, x='chained_value', y='label', hue='mode', dodge=False)

plt.title("Top 15 Most Isolated Transportation Spending Anomalies", fontsize=14)
plt.xlabel("Inflation-Adjusted Spending (Chained $)", fontsize=12)
plt.ylabel("Project Description (Year)", fontsize=12)
plt.legend(title="Mode", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()





