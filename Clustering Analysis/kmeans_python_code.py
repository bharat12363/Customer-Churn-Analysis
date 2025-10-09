import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Using only features (no target) for clustering
X_cluster = X

inertia = []       # sum of squared distances
K_range = range(1, 11)  # test k = 1 to 10

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia (Within-cluster Sum of Squares)')
plt.show()


optimal_k = 3 # elbow result

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_cluster)

# Adding cluster labels back to the DataFrame for interpretation
df_clusters = df_scaled.copy()  # or df_encoded.copy()
df_clusters['Cluster'] = clusters

print(df_clusters['Cluster'].value_counts())


# Save model to disk
import joblib
joblib.dump(kmeans, "kmeans_model.pkl")
print("K-Means model saved to kmeans_model.pkl")

# After fitting kmeans with n_clusters=3
df_clusters = df_scaled.copy()        # or df_encoded.copy()
df_clusters["Cluster"] = clusters     # from kmeans.fit_predict

# Checking the count showing how many customers per cluster
print(df_clusters["Cluster"].value_counts())


