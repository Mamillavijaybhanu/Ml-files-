import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("features_with_labels(in).csv")

numeric_data = data.select_dtypes(include=[np.number])
numeric_data = numeric_data.fillna(numeric_data.mean())

X_full = numeric_data.iloc[:, :-1].values
y_full = numeric_data.iloc[:, -1].values
y_full = (y_full > np.median(y_full)).astype(int)

X = X_full[:, :2]

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y_full)

plt.scatter(X[:,0], X[:,1], c=y_full)
plt.title("Project Data Classes")
plt.show()