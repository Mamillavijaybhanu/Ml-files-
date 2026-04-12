import numpy as np
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

w = np.random.rand(2)
b = np.random.rand()
lr = 0.05

for epoch in range(1000):
    total_error = 0

    for i in range(len(X)):
        z = np.dot(X[i], w) + b
        y_pred = 1 if z >= 0 else 0

        e = y[i] - y_pred
        total_error += e**2

        w += lr * e * X[i]
        b += lr * e

print("Final Error:", total_error)