import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt

df = pd.read_excel(
    r"C:\Users\mamill\OneDrive\Desktop\sem4\Ml\l2\Lab_Session_Data.xlsx", sheet_name="thyroid0387_UCI"
)
print(df.head())
print(df.info())
category = df.select_dtypes(include=['object','string']).columns
numerical = df.select_dtypes(include=['int64','float64']).columns
print("categorical", list(category))
print("numeric", list(numerical))
df.replace('?', np.nan, inplace=True)
nfeat= ['TSH','T3','TT4','T4U','FTI','TBG']
df[nfeat]=df[nfeat].apply(pd.to_numeric, errors='coerce')
print("missing values are", df.isnull().sum())
print("numeric statics", df[nfeat].describe())
zs = abs(zscore(df[nfeat], nan_policy='omit'))
ol = (zs > 3).any(axis=1)
print("outlier count", ol.sum())
for col in nfeat:
    print(f"{col}: Mean={df[col].mean()}, Variance={df[col].var()}, Std={df[col].std()}")
