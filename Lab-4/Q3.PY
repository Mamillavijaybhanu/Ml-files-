import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

X = np.random.uniform(1, 10, (20, 2))
y = (X[:, 0] + X[:, 1] > 10).astype(int)

colors = ['blue' if label == 0 else 'red' for label in y]

plt.scatter(X[:, 0], X[:, 1], c=colors)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Training Data")
plt.show()