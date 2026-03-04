import numpy as np
import matplotlib.pyplot as plt

points = np.array([
    [1, 1],  # A
    [2, 1],  # B
    [4, 3],  # C
    [5, 4]   # D
])

x1, y1 = map(float, input("Nhập tâm cụm 1 (x y): ").split())
x2, y2 = map(float, input("Nhập tâm cụm 2 (x y): ").split())

centroids = np.array([
    [x1, y1],
    [x2, y2]
])

def distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# K-means
for _ in range(10):
    clusters = {0: [], 1: []}

    for p in points:
        d0 = distance(p, centroids[0])
        d1 = distance(p, centroids[1])
        if d0 < d1:
            clusters[0].append(p)
        else:
            clusters[1].append(p)

    for i in range(2):
        if len(clusters[i]) > 0:
            centroids[i] = np.mean(clusters[i], axis=0)

# In kết quả
for i in range(2):
    print(f"Cụm {i+1}: {clusters[i]}")
    print(f"Tâm cụm {i+1}: {centroids[i]}")

colors = ['blue', 'green']

for i in range(2):
    cluster_points = np.array(clusters[i])
    if len(cluster_points) > 0:
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            color=colors[i],
            label=f'Cụm {i+1}'
        )

# Vẽ tâm cụm
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker='X',
    s=20,
    color='red',
    label='Tâm cụm'
)

plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("K-means với 4 điểm và 2 cụm")
plt.show()
