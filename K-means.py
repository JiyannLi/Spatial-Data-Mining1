import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# 读取
file_path = r"D:\KJWj\EXP1_Bostondata.csv"
df = pd.read_csv(file_path)
features = ["crim", "zn", "indus", "nox", "rm", "age", "dis",
            "rad", "tax", "ptratio", "black", "lstat"]
X = df[features]

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 手肘 轮廓系数
inertia = []
sil_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# 可视化
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertia, marker='o', linestyle='--')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (SSE)')
plt.title('Elbow Method for Optimal K')

plt.subplot(1, 2, 2)
plt.plot(K_range, sil_scores, marker='s', linestyle='--', color='r')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal K')

plt.show()


optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

# 箱型图
plt.figure(figsize=(10, 6))
sns.boxplot(x="KMeans_Cluster", y="medv", data=df, palette="Set2")
plt.xlabel("Cluster")
plt.ylabel("Median House Value")
plt.title("K-means Cluster vs House Prices")
plt.grid(True)
plt.show()


cluster_means = df.groupby("KMeans_Cluster")[features + ["medv"]].mean()
print(cluster_means)

#  PCA 降维可视化
pca_vis = PCA(n_components=2)
X_vis = pca_vis.fit_transform(X_scaled)
centers_2d = pca_vis.transform(kmeans.cluster_centers_)

# PCA 2D 聚类结果图
plt.figure(figsize=(8, 6))
for cluster in range(optimal_k):
    plt.scatter(
        X_vis[df["KMeans_Cluster"] == cluster, 0],
        X_vis[df["KMeans_Cluster"] == cluster, 1],
        label=f"Cluster {cluster}", alpha=0.6
    )

# 聚类中心
plt.scatter(
    centers_2d[:, 0], centers_2d[:, 1],
    s=200, c='gold', edgecolors='black', marker='*', label='Final Centers'
)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("K-means Cluster Visualization (2D PCA)")
plt.legend()
plt.grid(True)
plt.show()


