import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(42)
train_X = np.random.uniform(1, 10, (20, 2))
train_y = (train_X[:, 0] + train_X[:, 1] > 10).astype(int)

grid = np.array([[x, y] for x in np.arange(0,10,0.2)
                          for y in np.arange(0,10,0.2)])

for k in [1, 3, 7]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_X, train_y)
    pred = knn.predict(grid)

    colors = ['blue' if p == 0 else 'red' for p in pred]

    plt.figure()
    plt.scatter(grid[:,0], grid[:,1], c=colors, s=1)
    plt.title(f"Decision Regions k={k}")
    plt.show()