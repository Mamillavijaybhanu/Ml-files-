import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[0],[0],[1]])

np.random.seed(1)

w1 = np.random.uniform(-0.5, 0.5, (2,2))
w2 = np.random.uniform(-0.5, 0.5, (2,1))

lr = 0.05
errors = []

for epoch in range(1000):
    total_error = 0

    for i in range(len(X)):
        x = X[i].reshape(1,2)
        target = y[i].reshape(1,1)

        h = sigmoid(np.dot(x, w1))
        o = sigmoid(np.dot(h, w2))

        error = target - o
        total_error += np.sum(error**2)

        d_o = error * sigmoid_derivative(o)
        d_h = sigmoid_derivative(h) * np.dot(d_o, w2.T)

        w2 += lr * np.dot(h.T, d_o)
        w1 += lr * np.dot(x.T, d_h)

    errors.append(total_error)

    if total_error <= 0.002:
        break

print("Epochs:", epoch)

for i in range(len(X)):
    h = sigmoid(np.dot(X[i], w1))
    o = sigmoid(np.dot(h, w2))
    print(X[i], "->", o.round())