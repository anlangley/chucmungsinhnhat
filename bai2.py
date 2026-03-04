import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Sinh 100 điểm dữ liệu ngẫu nhiên
np.random.seed(0)
X = np.random.rand(100, 2) * 10

# 2. K-means với 5 cụm
kmeans = KMeans(n_clusters=5, n_init=10, random_state=0)
kmeans.fit(X)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# In tọa độ tâm cụm
print("TỌA ĐỘ 5 TÂM CỤM:")
for i, c in enumerate(centroids):
    print(f"Tâm cụm {i+1}: (x = {c[0]:.2f}, y = {c[1]:.2f})")

plt.figure(figsize=(12, 5))

# Hình 1: Dữ liệu ban đầu
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], color='gray')
plt.title("Dữ liệu ban đầu")
plt.xlabel("X")
plt.ylabel("Y")

# Hình 2: Kết quả phân cụm
plt.subplot(1, 2, 2)

colors = plt.cm.tab10.colors  # bảng màu chuẩn

for i in range(5):
    cluster_points = X[labels == i]
    plt.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        color=colors[i],
        label=f'Cụm {i+1}'
    )

# vẽ tâm cụm
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker='X',
    s=20,
    color='red',
    label='Tâm cụm'
)

plt.legend()
plt.title("Kết quả K-means (k = 5)")
plt.xlabel("X")
plt.ylabel("Y")

plt.tight_layout()
plt.show()
