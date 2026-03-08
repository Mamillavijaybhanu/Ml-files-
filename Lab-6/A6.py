import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_excel(
"C:/Users/nunna/OneDrive/Desktop/4th semester/Machine learning/features_with_labels.xlsx"
)

df = df.select_dtypes(include=["int64","float64"]).dropna()

X = df.drop(columns=["label"])
y = df["label"]

model = DecisionTreeClassifier()
model.fit(X,y)

plt.figure(figsize=(12,8))
plot_tree(model, filled=True)
plt.show()