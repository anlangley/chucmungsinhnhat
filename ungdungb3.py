import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x= np.array([10,20,30,40,60]).reshape(-1,1)
y= np.array([6,7,8,9,10])

#train
model=LinearRegression()
model.fit(x,y)
a=model.coef_[0]
b=model.intercept_

#dubao
gio=float(input("nhap gio hoc muon du bao: "))
gio_array=np.array([[gio]])
diem = model.predict(gio_array)
print("diem du bao: ",diem)
# TRUC QUAN HOA
# Tao duong hoi quy
x_line = np.linspace(10,60,100).reshape(-1,1)
y_line = model.predict(x_line)

plt.figure()
plt.scatter(x, y)              
plt.plot(x_line, y_line)     
plt.scatter(gio, diem[0])        

plt.xlabel("gio hoc/tuan")
plt.ylabel("diem")
plt.title("Hoi quy tuyen tinh: gio hoc vs diem")

plt.show()