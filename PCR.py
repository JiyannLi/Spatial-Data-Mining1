import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#  读取
file_path = r"D:\KJWj\EXP1_Bostondata.csv"
df = pd.read_csv(file_path)
features = ["crim", "zn", "indus", "nox", "rm", "age", "dis",
            "rad", "tax", "ptratio", "black", "lstat"]
X = df[features]
y = df["medv"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA 降维（保留6）
pca = PCA(n_components=6)
X_pca = pca.fit_transform(X_scaled)

# 回归
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 评估
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"PCR 模型 R²: {r2:.4f}")
print(f"PCR 模型 MSE: {mse:.4f}")

# 可视化
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([0, 50], [0, 50], '--', color='red')
plt.xlabel("True Prices")
plt.ylabel("Predicted Prices")
plt.title("PCR: True vs Predicted House Prices")
plt.grid(True)
plt.show()
