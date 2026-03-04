import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# =========================
# 1. Sinh dữ liệu ngẫu nhiên
# =========================
np.random.seed(42)
X = np.random.rand(100, 2) * 10

# =========================
# 2. Hierarchical Clustering
# =========================
Z = linkage(X, method='ward')
labels = fcluster(Z, t=5, criterion='maxclust')

# =========================
# 3. Trực quan hóa trên 1 figure
# =========================
plt.figure(figsize=(15, 6))

# --- Biểu đồ 1: Dữ liệu ban đầu ---
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1])
plt.title("Dữ liệu ban đầu")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)

# --- Biểu đồ 2: Kết quả phân cụm ---
plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.title("Hierarchical Clustering (5 cụm)")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)

# --- Biểu đồ 3: Cây phân cấp (Dendrogram) ---
plt.subplot(1, 3, 3)
dendrogram(
    Z,
    truncate_mode='lastp',   # rút gọn cây
    p=5,                     # hiển thị 5 cụm
    show_leaf_counts=True,   # hiện số điểm mỗi cụm
    show_contracted=True     # gom nhánh cho gọn
)
plt.title("Cây phân cấp (5 cụm)")
plt.xlabel("Cụm")
plt.ylabel("Khoảng cách")
plt.grid(True)

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# =========================
# 1. Sinh dữ liệu ngẫu nhiên
# =========================
np.random.seed(42)
X = np.random.rand(100, 2) * 10

# =========================
# 2. Hierarchical Clustering
# =========================
Z = linkage(X, method='ward')
labels = fcluster(Z, t=5, criterion='maxclust')

# =========================
# 3. Trực quan hóa trên 1 figure
# =========================
plt.figure(figsize=(15, 6))

# --- Biểu đồ 1: Dữ liệu ban đầu ---
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1])
plt.title("Dữ liệu ban đầu")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)

# --- Biểu đồ 2: Kết quả phân cụm ---
plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.title("Hierarchical Clustering (5 cụm)")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)

# --- Biểu đồ 3: Cây phân cấp (Dendrogram) ---
plt.subplot(1, 3, 3)
dendrogram(
    Z,
    truncate_mode='lastp',   # rút gọn cây
    p=5,                     # hiển thị 5 cụm
    show_leaf_counts=True,   # hiện số điểm mỗi cụm
    show_contracted=True     # gom nhánh cho gọn
)
plt.title("Cây phân cấp (5 cụm)")
plt.xlabel("Cụm")
plt.ylabel("Khoảng cách")
plt.grid(True)

plt.tight_layout()
plt.show()

