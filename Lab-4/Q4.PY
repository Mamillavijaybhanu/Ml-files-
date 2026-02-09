import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(42)
train_X = np.random.uniform(1, 10, (20, 2))
train_y = (train_X[:, 0] + train_X[:, 1] > 10).astype(int)

grid = np.array([[x, y] for x in np.arange(0,10,0.1)
                          for y in np.arange(0,10,0.1)])

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_X, train_y)

pred = knn.predict(grid)

colors = ['blue' if p == 0 else 'red' for p in pred]

plt.scatter(grid[:,0], grid[:,1], c=colors, s=1)
plt.title("Decision Regions k=3")
plt.show()