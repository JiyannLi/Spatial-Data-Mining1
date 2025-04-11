import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import BisectingKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

#读取
file_path = r"D:\KJWj\EXP1_Bostondata.csv"  # 替换为你的实际路径
df = pd.read_csv(file_path)
features = ["crim", "zn", "indus", "nox", "rm", "age", "dis",
            "rad", "tax", "ptratio", "black", "lstat"]
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分裂聚类（Bisecting K-means）
bkm = BisectingKMeans(n_clusters=4, random_state=42)
df["Divisive_Cluster"] = bkm.fit_predict(X_scaled)

# 轮廓系数
sil_score = silhouette_score(X_scaled, df["Divisive_Cluster"])
print("Divisive Clustering (BisectingKMeans) Silhouette Score:", sil_score)

# 箱型图
plt.figure(figsize=(10, 6))
sns.boxplot(x="Divisive_Cluster", y="medv", data=df, palette="Set3")
plt.title("Divisive Clustering (Bisecting K-means) vs House Prices")
plt.xlabel("Cluster")
plt.ylabel("Median House Value")
plt.grid(True)
plt.show()

# 降维可视化
pca_vis = PCA(n_components=2)
X_vis = pca_vis.fit_transform(X_scaled)
centers_vis = pca_vis.transform(bkm.cluster_centers_)

plt.figure(figsize=(8, 6))
for cluster in range(4):
    plt.scatter(
        X_vis[df["Divisive_Cluster"] == cluster, 0],
        X_vis[df["Divisive_Cluster"] == cluster, 1],
        label=f"Cluster {cluster}", alpha=0.6
    )

plt.scatter(centers_vis[:, 0], centers_vis[:, 1], s=200, c='gold', edgecolors='black',
            marker='*', label='Cluster Centers')

plt.title("Divisive Clustering Result (2D PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.show()

