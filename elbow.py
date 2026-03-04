import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# =========================
# 1. Sinh dữ liệu
# =========================
np.random.seed(0)
X = np.random.rand(100, 2) * 10

# =========================
# 2. Tính WCSS cho nhiều k
# =========================
wcss = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# =========================
# 3. TỰ ĐỘNG TÌM ELBOW
# =========================
x = np.array(list(K_range))
y = np.array(wcss)

A = np.array([x[0], y[0]])
B = np.array([x[-1], y[-1]])

distances = [
    np.abs(np.cross(B - A, A - np.array([x[i], y[i]])))
    / np.linalg.norm(B - A)
    for i in range(len(x))
]

k_optimal = x[np.argmax(distances)]
print("👉 k tối ưu =", k_optimal)

# =========================
# 4. K-means với k tối ưu
# =========================
kmeans = KMeans(n_clusters=k_optimal, n_init=10, random_state=0)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# =========================
# 5. VẼ 2 BIỂU ĐỒ TRONG 1 FIGURE
# =========================
plt.figure(figsize=(12, 5))

# ---- Biểu đồ 1: Elbow ----
plt.subplot(1, 2, 1)
plt.plot(K_range, wcss, marker='o', label='WCSS')
plt.scatter(k_optimal, y[k_optimal - 1], color='red', s=120, label='k tối ưu')
plt.xlabel("Số cụm k")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.legend()
plt.grid(True)

# ---- Biểu đồ 2: K-means ----
plt.subplot(1, 2, 2)
colors = plt.cm.tab10.colors

for i in range(k_optimal):
    plt.scatter(
        X[labels == i, 0],
        X[labels == i, 1],
        color=colors[i],
        label=f'Cụm {i+1}'
    )

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker='X',
    s=200,
    color='red',
    label='Tâm cụm'
)

plt.xlabel("X")
plt.ylabel("Y")
plt.title(f"K-means (k = {k_optimal})")
plt.legend()

plt.tight_layout()
plt.show()
