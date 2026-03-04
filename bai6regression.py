import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
#sinh 100 du lieu ngau nhien
m=100
tuoi = np.random.randint(1,21,m)
noise = np.random.normal(0,1,m)
chieucao= 1 + 0.7*tuoi + noise
data= pd.DataFrame({
    "tuoi":tuoi,
    "chieucao":chieucao
})

#chia train/test 8:2
train =data.sample(frac=0.8, random_state=42)
test = data.drop(train.index)

train.to_csv("train.csv",index=False)
test.to_csv("test.csv",index=False)
print("da luu du lieu train va test")

#train
x_train= train["tuoi"].values
y_train=train["chieucao"].values
n=len(x_train)
beta0=0
beta1=0
learning_rate= 0.001
epochs = 2000
for i in range(epochs):
    y_hat = beta0 + beta1 * x_train
    
    d_beta0 = (-2/n) * np.sum(y_train - y_hat)
    d_beta1 = (-2/n) * np.sum(x_train * (y_train - y_hat))
    
    beta0 -= learning_rate * d_beta0
    beta1 -= learning_rate * d_beta1

print("Phuong trinh hoi quy:")
print(f"y = {beta0:.4f} + {beta1:.4f}x")

# Đánh giá trên test
x_test = test["tuoi"].values
y_test = test["chieucao"].values

y_pred = beta0 + beta1 * x_test

# MAE
mae = np.mean(np.abs(y_test - y_pred))

# MSE
mse = np.mean((y_test - y_pred)**2)

# RMSE
rmse = math.sqrt(mse)

# R2
ss_total = np.sum((y_test - np.mean(y_test))**2)
ss_res = np.sum((y_test - y_pred)**2)
r2 = 1 - (ss_res / ss_total)

print("\nDanh gia tren test:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)

#truc quan hoa
x_line = np.linspace(min(x_test), max(x_test), 100)
y_line = beta0 + beta1 * x_line

plt.figure()
plt.scatter(x_test, y_test)
plt.plot(x_line, y_line, color='red')
plt.xlabel("Tuoi cay")
plt.ylabel("Chieu cao cay")
plt.title("Hoi quy Gradient Descent tren tap test")
plt.show()