import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x= np.array([1000,2000,3000,4000,10000]).reshape(-1,1)
y= np.array([10,20,30,40,100])

#train
model=LinearRegression()
model.fit(x,y)
a=model.coef_[0]
b=model.intercept_

#dubao
dung_luong= float(input("nhap dung luong pin: "))
dung_luong_array=np.array([[dung_luong]])
tgian=model.predict(dung_luong_array)
print(f"thoi gian su dung lien tuc la: {tgian[0]} phut")

#truc quan hoa
x_line = np.linspace(10,10000,100).reshape(-1,1)
y_line = model.predict(x_line)
plt.figure()
plt.scatter(x,y)
plt.plot(x_line,y_line)
plt.scatter(dung_luong,tgian[0])
plt.xlabel("dung luong mah")
plt.ylabel("thoi gian su dung")
plt.show()