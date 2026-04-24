import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_excel("C:/Users/mamill/OneDrive/Desktop/sem4/ML/features_with_labels.xlsx")

df = df.select_dtypes(include=["int64","float64"]).dropna()

X = df.drop(columns=["label"])
y = df["label"]

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print("PCA 95% Accuracy:", model.score(X_test, y_test))
