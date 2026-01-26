import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
df = pd.read_excel(r"C:\Users\mamill\OneDrive\Desktop\sem4\Ml\l2\Lab_Session_Data.xlsx", sheet_name="Purchase data")
X = df[['Candies (#)','Mangoes (Kg)','Milk Packets (#)']].values
df['Class'] = np.where(df['Payment (Rs)']>200,'rich','poor')
y = df['Class'].values
model = LogisticRegression()
model.fit(X, y)
pred=model.predict(X)
df['Predicted Class']=pred
print(df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)', 'Class', 'Predicted Class']])

