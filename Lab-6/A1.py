import pandas as pd
import numpy as np

df = pd.read_excel(
"C:/Users/nunna/OneDrive/Desktop/4th semester/Machine learning/features_with_labels.xlsx"
)

y = df["label"]

# entropy calculation
values = y.value_counts(normalize=True)

entropy = -np.sum(values * np.log2(values))

print("Entropy:", entropy)