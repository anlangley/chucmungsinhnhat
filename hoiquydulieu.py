import numpy as np
import matplotlib.pyplot as plt

# ======================
# 1. Nạp dữ liệu đầu vào
# ======================
X = np.array([17, 21, 35, 39, 50, 65])   # kg thịt bò
Y = np.array([132, 150, 160, 162, 149, 170])  # giá $

# ======================
# 2. Hồi quy tuyến tính OLS
# ======================
x_mean = np.mean(X)
y_mean = np.mean(Y)

SS_xy = np.sum((X - x_mean) * (Y - y_mean))
SS_xx = np.sum((X - x_mean) ** 2)

beta_1 = SS_xy / SS_xx   # hệ số góc
beta_0 = y_mean - beta_1 * x_mean  # hệ số chặn

print("Phương trình hồi quy:")
print(f"Y = {beta_0:.2f} + {beta_1:.3f} * X")

# ======================
# 3. Dự báo
# ======================
x_pred = np.array([45, 55])
y_pred = beta_0 + beta_1 * x_pred

print("\nDự báo:")
print(f"45kg  -> {y_pred[0]:.2f} $")
print(f"55kg  -> {y_pred[1]:.2f} $")

# ======================
# 4. Trực quan hóa
# ======================
plt.figure()
plt.scatter(X, Y, label="Dữ liệu thực tế")
plt.plot(X, beta_0 + beta_1 * X, label="Đường hồi quy OLS")
plt.scatter(x_pred, y_pred, label="Giá dự báo")

plt.xlabel("Khối lượng (kg)")
plt.ylabel("Giá ($)")
plt.title("Hồi quy tuyến tính OLS - Giá thịt bò")
plt.legend()
plt.grid(True)
plt.show()
