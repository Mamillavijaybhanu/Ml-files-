import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_excel(
    r"C:\Users\nunna\OneDrive\Desktop\4th semester\Machine learning\lab3\features_with_labels.xlsx"
)

cols = list(df.columns)
start = cols.index("label")
end = cols.index("hnr")

y = df["label"].values
X = df.iloc[:, start+1:end+1]

X = X.fillna(X.mean())
X = X.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

sk_acc = knn.score(X_test, y_test)

def euclidean_distance(a, b):
    s = 0
    for i in range(len(a)):
        s += (a[i] - b[i]) ** 2
    return s ** 0.5

def knn_manual_predict(X_train, y_train, test_point, k):
    dist = []

    for i in range(len(X_train)):
        d = euclidean_distance(X_train[i], test_point)
        dist.append((d, y_train[i]))

    dist.sort(key=lambda x: x[0])

    labels = []
    for i in range(k):
        labels.append(dist[i][1])

    return max(set(labels), key=labels.count)


k = 3
correct = 0

for i in range(len(X_test)):
    pred = knn_manual_predict(X_train, y_train, X_test[i], k)
    if pred == y_test[i]:
        correct += 1

manual_acc = correct / len(X_test)

print("Sklearn kNN Accuracy:", sk_acc)
print("Manual kNN Accuracy:", manual_acc)
