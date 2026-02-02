import pandas as pd
import matplotlib.pyplot as plt
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

acc_vals = []
k_vals = []

for k in range(1, 12):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    k_vals.append(k)
    acc_vals.append(acc)
    print("k =", k, "Accuracy =", acc)

plt.plot(k_vals, acc_vals)
plt.show()
