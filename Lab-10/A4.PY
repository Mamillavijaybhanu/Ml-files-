import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier

df = pd.read_excel("C:/Users/mamill/OneDrive/Desktop/sem4/ML/features_with_labels.xlsx")

df = df.select_dtypes(include=["int64","float64"]).dropna()

X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

model = RandomForestClassifier()

sfs = SequentialFeatureSelector(model, n_features_to_select=10)
sfs.fit(X_train, y_train)

X_train_fs = sfs.transform(X_train)
X_test_fs = sfs.transform(X_test)

model.fit(X_train_fs, y_train)

print("Feature Selection Accuracy:", model.score(X_test_fs, y_test))
