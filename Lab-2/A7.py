import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel(
    r"C:\Users\mamill\OneDrive\Desktop\sem4\Ml\l2\Lab_Session_Data.xlsx",
    sheet_name="thyroid0387_UCI"
)

df = df.replace('?', np.nan)
df = df.fillna(0)

bin_cols = [
    'on thyroxine','query on thyroxine','on antithyroid medication','sick',
    'pregnant','thyroid surgery','I131 treatment','query hypothyroid',
    'query hyperthyroid','lithium','goitre','tumor','hypopituitary','psych',
    'TSH measured','T3 measured','TT4 measured','T4U measured','FTI measured',
    'TBG measured'
]

df_bin = df[bin_cols].replace({
    't':1,'T':1,'y':1,'Y':1,'yes':1,'1':1,
    'f':0,'F':0,'n':0,'N':0,'no':0,'0':0
}).astype(int)

df_bin = df_bin.iloc[:20]

n = 20
JC = np.zeros((n,n))
SMC = np.zeros((n,n))

for i in range(n):
    for j in range(n):
        v1 = df_bin.iloc[i].values
        v2 = df_bin.iloc[j].values
        f11 = np.sum((v1==1)&(v2==1))
        f10 = np.sum((v1==1)&(v2==0))
        f01 = np.sum((v1==0)&(v2==1))
        f00 = np.sum((v1==0)&(v2==0))
        JC[i,j] = f11/(f11+f10+f01) if (f11+f10+f01)!=0 else 0
        SMC[i,j] = (f11+f00)/(f11+f10+f01+f00)

cat = df.select_dtypes(include=['object','string']).astype(str)
num = df.select_dtypes(include=['int64','float64'])

enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit_transform(cat)

X = np.hstack((num.values, enc))
X = X[:20]

COS = cosine_similarity(X)

plt.figure(figsize=(8,6))
sns.heatmap(JC, annot=False, cmap='viridis')
plt.title("JC Similarity Heatmap")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(SMC, annot=False, cmap='viridis')
plt.title("SMC Similarity Heatmap")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(COS, annot=False, cmap='viridis')
plt.title("Cosine Similarity Heatmap")
plt.show()
