import pandas as pd
import numpy as np

df = pd.read_excel(
"C:/Users/nunna/OneDrive/Desktop/4th semester/Machine learning/features_with_labels.xlsx"
)

y = df["label"]

values = y.value_counts(normalize=True)

gini = 1 - np.sum(values**2)

print("Gini Index:", gini)