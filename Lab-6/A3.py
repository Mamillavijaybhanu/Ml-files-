import pandas as pd
import numpy as np

df = pd.read_excel(
"C:/Users/nunna/OneDrive/Desktop/4th semester/Machine learning/features_with_labels.xlsx"
)

target = df["label"]

def entropy(col):
    p = col.value_counts(normalize=True)
    return -np.sum(p*np.log2(p))

base_entropy = entropy(target)

for feature in df.columns:
    if feature != "label":

        groups = df.groupby(feature)["label"]
        weighted_entropy = 0

        for name, group in groups:
            weighted_entropy += (len(group)/len(df))*entropy(group)

        info_gain = base_entropy - weighted_entropy
        print(feature,"Information Gain:",info_gain)