import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel(
    r"C:\Users\mamill\OneDrive\Desktop\sem4\Ml\l2\Lab_Session_Data.xlsx",
    sheet_name="thyroid0387_UCI"
)

df = df.replace('?', 0)

num_cols = ['age','TSH','T3','TT4','T4U','FTI','TBG']
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

scaler = MinMaxScaler()
df_norm = df.copy()
df_norm[num_cols] = scaler.fit_transform(df_norm[num_cols])

print(df_norm[num_cols].head())
