import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

# Load dataset from CSV
file_path = "iris_data.csv"
df = pd.read_csv(file_path)

# Assign proper column names
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop(columns=['class']))

# Finding optimal clusters using the Elbow Method
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o', linestyle='-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

# Apply K-Means Clustering with optimal K (choosing K=3 based on Elbow Method)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

# Apply Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=3)
df['Hierarchical_Cluster'] = hierarchical.fit_predict(X_scaled)

# Compute evaluation metrics
silhouette_kmeans = silhouette_score(X_scaled, df['KMeans_Cluster'])
silhouette_hierarchical = silhouette_score(X_scaled, df['Hierarchical_Cluster'])
ari_kmeans = adjusted_rand_score(df['class'], df['KMeans_Cluster'])
ari_hierarchical = adjusted_rand_score(df['class'], df['Hierarchical_Cluster'])

# Print evaluation metrics
print(f"Silhouette Score (K-Means): {silhouette_kmeans:.2f}")
print(f"Silhouette Score (Hierarchical): {silhouette_hierarchical:.2f}")
print(f"Adjusted Rand Index (K-Means): {ari_kmeans:.2f}")
print(f"Adjusted Rand Index (Hierarchical): {ari_hierarchical:.2f}")

# Visualizing clusters using PCA
pca = PCA(n_components=2)
df_pca = pca.fit_transform(X_scaled)
df['PCA1'] = df_pca[:, 0]
df['PCA2'] = df_pca[:, 1]

# Plot K-Means clusters
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['PCA1'], y=df['PCA2'], hue=df['KMeans_Cluster'], palette='Set1', style=df['class'])
plt.title('K-Means Clustering (PCA Reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster', loc='best')
plt.show()

# Plot Hierarchical Clusters
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['PCA1'], y=df['PCA2'], hue=df['Hierarchical_Cluster'], palette='Set2', style=df['class'])
plt.title('Hierarchical Clustering (PCA Reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster', loc='best')
plt.show()

# Generate Dendrogram for Hierarchical Clustering
plt.figure(figsize=(10, 5))
linkage_matrix = linkage(X_scaled, method='ward')
dendrogram(linkage_matrix, labels=df['class'].values, leaf_rotation=90, leaf_font_size=8)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()

# Display the clustered data
print(df.head())