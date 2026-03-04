import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x= np.array([50,60,70,80,100]).reshape(-1,1) # dien tich
y= np.array([1500,1700,2000,2300,3000]) # gia 1 met
#train model
model = LinearRegression()
model.fit(x,y)
a=model.coef_[0]
b=model.intercept_
print("he so a: ",a)
print("he so b: ",b)
#du bao
x_new=float(input("nhap dien tich muon du bao: "))
x_new_array = np.array([[x_new]])   # CHUYEN VE 2D
gia = model.predict(x_new_array)
print("gia nha du kien: ",gia)
# TRUC QUAN HOA
# Tao duong hoi quy
x_line = np.linspace(45,110,100).reshape(-1,1)
y_line = model.predict(x_line)

plt.figure()
plt.scatter(x, y)              # diem du lieu that
plt.plot(x_line, y_line)       # duong hoi quy
plt.scatter(x_new, gia)        # diem du bao 55m2

plt.xlabel("Dien tich (m2)")
plt.ylabel("Gia nha (trieu)")
plt.title("Hoi quy tuyen tinh: Dien tich vs Gia nha")

plt.show()