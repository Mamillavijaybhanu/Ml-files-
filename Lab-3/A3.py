import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel(
    r"C:\Users\nunna\OneDrive\Desktop\4th semester\Machine learning\lab3\features_with_labels.xlsx"
)

X = df["zcr_mean"].values

s = 0
for i in range(len(X)):
    s += X[i]
mean = s / len(X)

v = 0
for i in range(len(X)):
    v += (X[i] - mean) ** 2
variance = v / len(X)

print("Mean:", mean)
print("Variance:", variance)

plt.hist(X, bins=10)
plt.show()
