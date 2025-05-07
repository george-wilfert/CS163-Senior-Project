import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from prepare_nhcci_data import load_and_merge_all 
import matplotlib.dates as mdates

nhcci_data = load_and_merge_all()
# Ensure quarter_period column is properly formatted (from the original CSV, not the index)
if 'quarter_period' not in nhcci_data.columns:
    nhcci_data['quarter_period'] = nhcci_data.index.astype(str)

# Convert quarter_period safely
nhcci_data['datetime'] = pd.PeriodIndex(nhcci_data['quarter_period'], freq='Q').to_timestamp()
print(nhcci_data.head())

# Step 1: Select percent change columns
component_features = [col for col in nhcci_data.columns if 'pct-change' in col]

kmeans_features = component_features + ['GDP', 'UNRATE', 'PPIACO', 'Construction_Employment']
cluster_input = nhcci_data[['datetime', 'quarter_period', 'NHCCI-Seasonally-Adjusted'] + kmeans_features].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_input[kmeans_features])

kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto')
kmeans.fit(X_scaled)
cluster_input['Cluster'] = kmeans.labels_

# Inverse transform cluster centers
cluster_centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers_df = pd.DataFrame(cluster_centers_original, columns=kmeans_features)
print("Cluster centers (in original scale):")
print(cluster_centers_df)

# Step 6: Compute average spending level per cluster and assign labels
cluster_averages = cluster_centers_df.mean(axis=1)
sorted_clusters = cluster_averages.sort_values().index
cluster_labels = {
    sorted_clusters[0]: "Low Investment",
    sorted_clusters[1]: "High Investment",
}

cluster_input['Spending Cluster Label'] = cluster_input['Cluster'].map(cluster_labels)

print(cluster_input['Cluster'].value_counts())

#cluster_input.to_csv("nhcci_cluster_input.csv", index=False)

plt.figure(figsize=(12, 6))
sns.scatterplot(x='datetime', y='Cluster', data=cluster_input, hue='Spending Cluster Label', palette='Set2')
plt.title("Cluster Assignment per Quarter (Labeled)")
plt.xlabel("Year")
plt.ylabel("Cluster")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)
plt.legend(title="Spending Cluster")
plt.tight_layout()
plt.show()

sns.boxplot(x='Spending Cluster Label', y='NHCCI-Seasonally-Adjusted', data=cluster_input)
plt.title("NHCCI Distribution by Investment Cluster")
plt.xlabel("Spending Cluster")
plt.ylabel("NHCCI (Seasonally Adjusted)")
plt.tight_layout()
plt.show()

sns.lineplot(
    x='datetime', 
    y='NHCCI-Seasonally-Adjusted', 
    data=cluster_input, 
    hue='Spending Cluster Label', 
    palette='Set2'
)

plt.title("NHCCI Over Time by Investment Cluster")
plt.xlabel("Year")
plt.ylabel("NHCCI (Seasonally Adjusted)")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Normalized Line Chart: Cluster Center Comparison ---

from sklearn.preprocessing import MinMaxScaler

# Select and normalize
features_to_plot = ['GDP', 'UNRATE', 'PPIACO', 'Construction_Employment']
normalized_centers = cluster_centers_df[features_to_plot].copy()
scaler = MinMaxScaler()
normalized_centers[features_to_plot] = scaler.fit_transform(normalized_centers[features_to_plot])

# Transpose and relabel
normalized_centers.index = ['Low Investment', 'High Investment']
normalized_centers_T = normalized_centers.T

# Plot
plt.figure(figsize=(10, 5))
for label in normalized_centers_T.columns:
    plt.plot(normalized_centers_T.index, normalized_centers_T[label], marker='o', label=label)

plt.title('Normalized Cluster Center Comparison Across Economic Indicators')
plt.ylabel('Normalized Value')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Filter for Most Informative Components ---

# Compute average % change for each component by cluster
component_means = cluster_input.groupby('Spending Cluster Label')[component_features].mean().T

# Compute absolute difference between High vs Low investment clusters
component_diff = (component_means.iloc[:, 0] - component_means.iloc[:, 1]).abs()

# Get top 10 components with largest differences
top_components = component_diff.sort_values(ascending=False).head(10).index

# Slice down to top components only
filtered_means = component_means.loc[top_components]
filtered_means.reset_index(inplace=True)
filtered_means = filtered_means.rename(columns={'index': 'Construction Component'})

# Melt to long format for Seaborn
filtered_long = filtered_means.melt(id_vars='Construction Component', var_name='Spending Cluster', value_name='Average % Change')

#filtered_long.to_csv("nhcci_filtered_long.csv", index=False)

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=filtered_long, 
            x='Construction Component', 
            y='Average % Change', 
            hue='Spending Cluster', 
            palette='Set2')

plt.title('Top 10 Component Cost Changes That Distinguish Investment Clusters')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Component')
plt.ylabel('Average % Change')
plt.legend(title='Investment Cluster')
plt.tight_layout()
plt.show()

#normalized_centers_T.to_csv("nhcci_normalized_centers_T.csv")
