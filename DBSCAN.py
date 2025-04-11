import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

# 读取
file_path = r"D:\KJWj\EXP1_Bostondata.csv"
df = pd.read_csv(file_path)

# 特征选择与标准化
features = ["crim", "zn", "indus", "nox", "rm", "age", "dis",
            "rad", "tax", "ptratio", "black", "lstat"]
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#选择 eps（可视化 KNN 距离图）
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
distances = np.sort(distances[:, 4])
plt.plot(distances)
plt.title("5-NN Distance Graph (DBSCAN)")
plt.xlabel("Sample index")
plt.ylabel("Distance")
plt.show()

#  DBSCAN 聚类
dbscan = DBSCAN(eps=1.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)
df["DBSCAN_Cluster"] = labels
mask = labels != -1  # 排除噪声点
if len(set(labels[mask])) > 1:
    sil_score = silhouette_score(X_scaled[mask], labels[mask])
    print("DBSCAN Silhouette Score (excluding noise):", sil_score)
else:
    print("聚类数量不足，无法计算轮廓系数")

# 可视化
plt.figure(figsize=(10, 6))
sns.boxplot(x="DBSCAN_Cluster", y="medv", data=df, palette="Set3")
plt.title("DBSCAN Clusters vs House Prices")
plt.xlabel("Cluster Label")
plt.ylabel("Median House Value")
plt.show()
