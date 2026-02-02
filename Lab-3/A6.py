import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_excel(
    r"C:\Users\nunna\OneDrive\Desktop\4th semester\Machine learning\lab3\features_with_labels.xlsx"
)

X = df.drop(columns=["label"]).values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print(len(X_train), len(X_test))
