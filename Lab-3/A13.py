import pandas as pd
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

y_pred = knn.predict(X_test)


def confusion_matrix_manual(y_true, y_pred):
    labels = list(set(y_true))
    matrix = [[0 for _ in labels] for _ in labels]

    for i in range(len(y_true)):
        r = labels.index(y_true[i])
        c = labels.index(y_pred[i])
        matrix[r][c] += 1

    return matrix, labels


def accuracy_manual(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / len(y_true)


def precision_manual(y_true, y_pred):
    tp = 0
    fp = 0
    for i in range(len(y_true)):
        if y_pred[i] == y_true[i]:
            tp += 1
        else:
            fp += 1
    return tp / (tp + fp)


def recall_manual(y_true, y_pred):
    tp = 0
    fn = 0
    for i in range(len(y_true)):
        if y_pred[i] == y_true[i]:
            tp += 1
        else:
            fn += 1
    return tp / (tp + fn)


def fbeta_manual(precision, recall, beta=1):
    return (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall)


cm, labels = confusion_matrix_manual(y_test, y_pred)

acc = accuracy_manual(y_test, y_pred)
prec = precision_manual(y_test, y_pred)
rec = recall_manual(y_test, y_pred)
f1 = fbeta_manual(prec, rec)

print("Labels:", labels)
print("Confusion Matrix:")
for row in cm:
    print(row)

print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)
