import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_excel(
"C:/Users/nunna/OneDrive/Desktop/4th semester/Machine learning/features_with_labels.xlsx"
)

df = df.select_dtypes(include=["int64","float64"]).dropna()

X = df.drop(columns=["label"])
y = df["label"]

model = DecisionTreeClassifier()

model.fit(X,y)

print("Decision Tree Built")