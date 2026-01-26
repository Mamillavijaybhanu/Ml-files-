import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

df = pd.read_excel(
    r"C:\Users\mamilla\OneDrive\Desktop\sem4\Ml\l2\Lab_Session_Data.xlsx",
    sheet_name="thyroid0387_UCI"
)

df = df.replace('?', np.nan)

num_cols = ['age','TSH','T3','TT4','T4U','FTI','TBG']
cat_cols = df.select_dtypes(include=['object','string']).columns

df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

df['age'] = df['age'].fillna(df['age'].mean())
df[['TSH','T3','TT4','T4U','FTI','TBG']] = df[['TSH','T3','TT4','T4U','FTI','TBG']].fillna(df[['TSH','T3','TT4','T4U','FTI','TBG']].median())

for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode().iloc[0])
