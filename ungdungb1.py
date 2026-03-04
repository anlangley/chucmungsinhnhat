import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# Dữ liệu
X = np.array([10, 20, 30, 40]).reshape(-1, 1)  # Chi phí quảng cáo
y = np.array([100, 150, 200, 250])             # Doanh số

# Tạo mô hình
model = LinearRegression()
model.fit(X, y)

# Hệ số
a = model.coef_[0]
b = model.intercept_

print("He so a:", a)
print("He so b:", b)

# Dự báo khi chi 35 triệu
du_bao = model.predict([[35]])
print("Doanh so du kien khi chi 35 trieu:", du_bao[0])
# Tạo dữ liệu dự đoán để vẽ đường thẳng
X_line = np.linspace(0, 50, 100).reshape(-1, 1)
y_line = model.predict(X_line)

# Vẽ dữ liệu gốc
plt.scatter(X, y)
plt.plot(X_line, y_line)

plt.xlabel("Chi phi quang cao (trieu)")
plt.ylabel("Doanh so")
plt.title("Hoi quy tuyen tinh giua chi phi quang cao va doanh so")

plt.show()