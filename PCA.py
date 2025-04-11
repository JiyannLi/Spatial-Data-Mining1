import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#读取
file_path = r"D:\KJWj\EXP1_Bostondata.csv"
df = pd.read_csv(file_path)
features = ["crim", "zn", "indus", "nox", "rm", "age", "dis",
            "rad", "tax", "ptratio", "black", "lstat"]
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  PCA分析
pca = PCA(n_components=len(features))
X_pca = pca.fit_transform(X_scaled)

#绘制解释方差率
plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance by PCA Components")
plt.grid(True)
plt.show()


for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {ratio:.4f}")
