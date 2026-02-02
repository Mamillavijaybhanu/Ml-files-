import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

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

print("kNN trained successfully")

acc = knn.score(X_test, y_test)
print("Accuracy:", acc)

y_pred = knn.predict(X_test)

print(
    precision_score(y_test, y_pred, average="weighted"),
    recall_score(y_test, y_pred, average="weighted"),
    f1_score(y_test, y_pred, average="weighted")
)
