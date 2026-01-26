import pandas as pd
import numpy as np

df = pd.read_excel(r"C:\Users\mamill\OneDrive\Desktop\sem4\Ml\l2\Lab_Session_Data.xlsx",sheet_name="thyroid0387_UCI"
)
bin_cols = [
    'on thyroxine','query on thyroxine','on antithyroid medication','sick',
    'pregnant','thyroid surgery','I131 treatment','query hypothyroid',
    'query hyperthyroid','lithium','goitre','tumor','hypopituitary','psych',
    'TSH measured','T3 measured','TT4 measured','T4U measured','FTI measured',
    'TBG measured'
]

df_bin = df[bin_cols]


df_bin = df_bin.replace({'t':1,'T':1,'y':1,'Y':1,'yes':1,'1':1,
                         'f':0,'F':0,'n':0,'N':0,'no':0,'0':0})


v1 = df_bin.iloc[0].astype(int).values
v2 = df_bin.iloc[1].astype(int).values


f11 = np.sum((v1==1) & (v2==1))
f10 = np.sum((v1==1) & (v2==0))
f01 = np.sum((v1==0) & (v2==1))
f00 = np.sum((v1==0) & (v2==0))


JC = f11 / (f11 + f10 + f01)
SMC = (f11 + f00) / (f11 + f10 + f01 + f00)


print("Vector 1:", v1)
print("Vector 2:", v2)
print("f11 =", f11, "f10 =", f10, "f01 =", f01, "f00 =", f00)
print("Jaccard Coefficient (JC) =", JC)
print("Simple Matching Coefficient (SMC) =", SMC)

if JC < SMC:
    print("observation JC < SMC so SMC counts 0-0 matches, making vectors look more similar.")
else:
    print("\nObservation: JC >= SMC, dataset has balanced binary attributes.")
