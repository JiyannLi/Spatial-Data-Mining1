import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# 读取
file_path = r"D:\KJWj\EXP1_Bostondata.csv"
df = pd.read_csv(file_path)
features = ["crim", "zn", "indus", "nox", "rm", "age", "dis",
            "rad", "tax", "ptratio", "black", "lstat"]
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 树状图
linked = linkage(X_scaled, method='ward')
plt.figure(figsize=(12, 6))
dendrogram(linked, truncate_mode='lastp', p=30, leaf_rotation=45., leaf_font_size=10.)
plt.title('Hierarchical Clustering Ward Dendrogram')
plt.xlabel('Cluster Index')
plt.ylabel('Distance')
plt.show()

# 聚类标签（设定为4）
labels = fcluster(linked, t=4, criterion='maxclust')
df["Ward_Cluster"] = labels

sil_score = silhouette_score(X_scaled, labels)
print("Hierarchical Clustering Silhouette Score:", sil_score)

# 可视化
plt.figure(figsize=(10, 6))
sns.boxplot(x="Ward_Cluster", y="medv", data=df, palette="Pastel1")
plt.title("Hierarchical Clusters Ward vs House Prices")
plt.xlabel("Cluster Label")
plt.ylabel("Median House Value")
plt.show()

# PCA 降维 可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

plt.figure(figsize=(8, 6))
for cluster in np.unique(df["Ward_Cluster"]):
    plt.scatter(
        df[df["Ward_Cluster"] == cluster]["PCA1"],
        df[df["Ward_Cluster"] == cluster]["PCA2"],
        label=f"Cluster {cluster}", alpha=0.6
    )
plt.title("True Ward Hierarchical Clustering (PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
