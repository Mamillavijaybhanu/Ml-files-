import numpy as np
import pandas as pd
df=pd.read_excel(r"C:\Users\mamill\OneDrive\Desktop\sem4\Ml\l2\Lab_Session_Data.xlsx", sheet_name="Purchase data")
x=df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
y=df['Payment (Rs)'].values
rank = np.linalg.matrix_rank(x)
print("Rank of X =", rank)
xpinv = np.linalg.pinv(x)
cost = xpinv @ y
print("Candy cost ",cost[0])
print("Mango cost ",cost[1])
print("Milk cost ",cost[2])
