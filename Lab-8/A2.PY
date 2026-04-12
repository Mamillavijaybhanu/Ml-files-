import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

w = np.array([0.2, -0.75])
b = 10
lr = 0.05

errors = []

for epoch in range(1000):
    total_error = 0

    for i in range(len(X)):
        z = np.dot(X[i], w) + b
        y_pred = 1 if z >= 0 else 0

        e = y[i] - y_pred
        total_error += e**2

        w = w + lr * e * X[i]
        b = b + lr * e

    errors.append(total_error)

    if total_error <= 0.002:
        break

plt.plot(errors)
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.show()

print("Epochs:", epoch)