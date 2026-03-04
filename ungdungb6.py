import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array([1,2,3,4,5,6,7,8]).reshape(-1,1)   # tuoi cay (nam)
y = np.array([1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0])  # chieu cao (m)

#train
model=LinearRegression()
model.fit(x,y)
a=model.coef_[0]
b=model.intercept_

#dubao
tuoicay= float(input("nhap tuoi cay: "))
tuoicay_array=np.array([[tuoicay]])
chieucao=model.predict(tuoicay_array)
print(f"chieu cao cua cay du doan la: {chieucao[0]} met")

#truc quan hoa
x_line = np.linspace(1,8,100).reshape(-1,1)
y_line = model.predict(x_line)
plt.figure()
plt.scatter(x,y)
plt.plot(x_line,y_line)
plt.scatter(tuoicay,chieucao[0])
plt.xlabel("tuoi cay(nam)")
plt.ylabel("chieu cao cay(met)")
plt.show()