import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

def bipolar_step(x):
    return 1 if x >= 0 else -1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return max(0, x)

def train(act_func, is_continuous=False):
    w = np.array([0.2, -0.75])
    b = 10
    lr = 0.05

    for epoch in range(1000):
        total_error = 0

        for i in range(len(X)):
            z = np.dot(X[i], w) + b

            if is_continuous:
                y_pred = act_func(z)
                y_pred_binary = 1 if y_pred >= 0.5 else 0
            else:
                y_pred_binary = act_func(z)

            e = y[i] - y_pred_binary
            total_error += e**2

            w = w + lr * e * X[i]
            b = b + lr * e

        if total_error <= 0.002:
            return epoch

    return 1000

print("Bipolar Step Epochs:", train(bipolar_step))
print("Sigmoid Epochs:", train(sigmoid, True))
print("ReLU Epochs:", train(relu, True))