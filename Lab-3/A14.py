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

# -------- kNN CLASSIFIER --------
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_acc = knn.score(X_test, y_test)

# -------- MATRIX INVERSION --------
X_train_aug = np.c_[np.ones(X_train.shape[0]), X_train]
X_test_aug = np.c_[np.ones(X_test.shape[0]), X_test]

theta = np.linalg.pinv(X_train_aug) @ y_train
y_pred_reg = X_test_aug @ theta

y_pred_reg = np.round(y_pred_reg).astype(int)

correct = 0
for i in range(len(y_test)):
    if y_pred_reg[i] == y_test[i]:
        correct += 1

reg_acc = correct / len(y_test)

print("kNN Accuracy:", knn_acc)
print("Matrix Inversion Accuracy:", reg_acc)
