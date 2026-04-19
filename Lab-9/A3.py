import pandas as pd
import lime
import lime.lime_tabular
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def load_data(path):
    df = pd.read_excel(path)
    df = df.select_dtypes(include=["int64","float64"]).dropna()
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, y

def build_pipeline():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier())
    ])
    return pipe

def explain_model(X_train, model):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns,
        class_names=["0","1"],
        mode="classification"
    )
    return explainer

if __name__ == "__main__":
    X, y = load_data(
        "C:/Users/mamill/OneDrive/Desktop/sem4/ML/features_with_labels.xlsx"
    )

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

    pipe = build_pipeline()
    pipe.fit(X_train,y_train)

    explainer = explain_model(X_train, pipe)

    exp = explainer.explain_instance(
        X_test.iloc[0].values,
        pipe.predict_proba
    )

    print(exp.as_list())
