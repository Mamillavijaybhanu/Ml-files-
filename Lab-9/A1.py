import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def load_data(path):
    df = pd.read_excel(path)
    df = df.select_dtypes(include=["int64","float64"]).dropna()
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, y

def build_stacking_model():
    base_models = [
        ("knn", KNeighborsClassifier()),
        ("dt", DecisionTreeClassifier()),
        ("rf", RandomForestClassifier()),
        ("svm", SVC(probability=True))
    ]

    meta_model = LogisticRegression()

    model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model
    )

    return model

if __name__ == "__main__":
    X, y = load_data("C:/Users/mamill/OneDrive/Desktop/sem4/ML/features_with_labels.xlsx")

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

    model = build_stacking_model()
    model.fit(X_train,y_train)

    print("Stacking Accuracy:", model.score(X_test,y_test))
