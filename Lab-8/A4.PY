import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

learning_rates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
epochs_list = []

for lr in learning_rates:
    w = np.array([0.2, -0.75])
    b = 10

    for epoch in range(1000):
        total_error = 0

        for i in range(len(X)):
            z = np.dot(X[i], w) + b
            y_pred = 1 if z >= 0 else 0

            e = y[i] - y_pred
            total_error += e**2

            w = w + lr * e * X[i]
            b = b + lr * e

        if total_error <= 0.002:
            break

    epochs_list.append(epoch)

plt.plot(learning_rates, epochs_list, marker='o')
plt.xlabel("Learning Rate")
plt.ylabel("Epochs to Converge")
plt.title("Learning Rate vs Epochs")
plt.show()

print("Learning Rates:", learning_rates)
print("Epochs:", epochs_list)