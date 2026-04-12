import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

data = pd.read_csv("features_with_labels(in).csv")

numeric_data = data.select_dtypes(include=[float,int])
numeric_data = numeric_data.fillna(numeric_data.mean())

X = numeric_data.iloc[:,:-1].values
y = numeric_data.iloc[:,-1].values

y = (y > y.mean()).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3
)

model = MLPClassifier(max_iter=500)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))