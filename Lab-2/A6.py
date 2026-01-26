import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

df = pd.read_excel(
    r"C:\Users\mamill\OneDrive\Desktop\sem4\Ml\l2\Lab_Session_Data.xlsx",
    sheet_name="thyroid0387_UCI"
)

df = df.replace('?', np.nan)

numeric_features = ['TSH','T3','TT4','T4U','FTI','TBG']
df[numeric_features] = df[numeric_features].apply(pd.to_numeric, errors='coerce')

df = df.fillna(0)

category = df.select_dtypes(include=['object','string']).columns
numeric = df.select_dtypes(include=['int64','float64']).columns

df[category] = df[category].astype(str)

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
category_enc = encoder.fit_transform(df[category])

X = np.hstack((df[numeric].values, category_enc))

A = X[0]
B = X[1]

dot = np.dot(A, B)
normA = np.linalg.norm(A)
normB = np.linalg.norm(B)
cos_sim = dot / (normA * normB)

print("length of vector A:", len(A))
print("length of vector B:", len(B))
print("cosine similarity =", cos_sim)
