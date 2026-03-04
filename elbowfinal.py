import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Sinh dữ liệu
np.random.seed(0)
X = np.random.rand(100, 2) * 10

# Tính WCSS
wcss = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# ===== PHẦN TÌM k TỐI ƯU (ELBOW) =====
x = np.array(list(K_range))
y = np.array(wcss)

# nối điểm đầu và cuối
A = np.array([x[0], y[0]])
B = np.array([x[-1], y[-1]])

# tính khoảng cách
distances = [
    np.abs(np.cross(B - A, A - np.array([x[i], y[i]])))
    / np.linalg.norm(B - A)
    for i in range(len(x))
]

# k tối ưu
k_optimal = x[np.argmax(distances)]

print("k tối ưu =", k_optimal)

# ===== VẼ ELBOW + CHẤM ĐỎ =====
plt.plot(x, y, marker='o')
plt.scatter(k_optimal, y[k_optimal - 1], color='red', s=100)
plt.xlabel("k")
plt.ylabel("WCSS")
plt.title("Elbow Method – k tối ưu")
plt.grid(True)
plt.show()
