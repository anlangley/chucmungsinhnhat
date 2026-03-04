import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# du lieu
x = np.array([20,22,25,28,30,32,35]).reshape(-1,1)  # nhiet do (°C)
y = np.array([20,24,30,36,40,44,50])               # dien tieu thu (kWh)

# train
model = LinearRegression()
model.fit(x,y)

a = model.coef_[0]
b = model.intercept_

# du bao
nhietdo = float(input("nhap nhiet do moi truong: "))
nhietdo_array = np.array([[nhietdo]])
dien = model.predict(nhietdo_array)

print(f"luong dien du doan la: {dien[0]:.2f} kWh")

# truc quan hoa
x_line = np.linspace(min(x), max(x), 100).reshape(-1,1)
y_line = model.predict(x_line)

plt.figure()
plt.scatter(x,y)
plt.plot(x_line,y_line)
plt.scatter(nhietdo,dien[0])
plt.xlabel("nhiet do (°C)")
plt.ylabel("dien tieu thu (kWh)")
plt.show()