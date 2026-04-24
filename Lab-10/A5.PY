import pandas as pd
import lime
import lime.lime_tabular
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# =======================
# LOAD DATA
# =======================
df = pd.read_excel(
    "C:/Users/mamill/OneDrive/Desktop/sem4/ML/features_with_labels.xlsx"
)

df = df.select_dtypes(include=["int64","float64"]).dropna()

X = df.drop(columns=["label"])
y = df["label"]

# =======================
# SPLIT DATA
# =======================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# =======================
# TRAIN MODEL
# =======================
model = RandomForestClassifier()
model.fit(X_train, y_train)

# =======================
# -------- LIME ---------
# =======================
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns,
    class_names=["0","1"],
    mode="classification"
)

lime_exp = explainer_lime.explain_instance(
    X_test.iloc[0].values,
    model.predict_proba
)

print("\n===== LIME EXPLANATION =====")
for item in lime_exp.as_list():
    print(item)

# =======================
# -------- SHAP ---------
# =======================
print("\n===== SHAP (FAST VERSION) =====")

explainer_shap = shap.TreeExplainer(model)

# 🔥 IMPORTANT: use small sample to avoid hanging
X_sample = X_train.sample(100)

shap_values = explainer_shap.shap_values(X_sample)

# Plot
shap.summary_plot(shap_values, X_sample)
