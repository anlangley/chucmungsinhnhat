import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x= np.array([10,20,30,40,70]).reshape(-1,1)
y= np.array([15,28,36,60,100])

#train
model=LinearRegression()
model.fit(x,y)
a=model.coef_[0]
b=model.intercept_

#dubao
ngansach= float(input("nhap so tien dau tu: "))
ngansach_array=np.array([[ngansach]])
doanhthu=model.predict(ngansach_array)
print(f"doanh thu du doan la: {doanhthu[0]} ty")

#truc quan hoa
x_line = np.linspace(10,100,100).reshape(-1,1)
y_line = model.predict(x_line)
plt.figure()
plt.scatter(x,y)
plt.plot(x_line,y_line)
plt.scatter(ngansach,doanhthu[0])
plt.xlabel("ngan sach(ty)")
plt.ylabel("doanhthu(ty)")
plt.show()