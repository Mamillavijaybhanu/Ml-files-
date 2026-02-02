import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel(
    r"C:\Users\nunna\OneDrive\Desktop\4th semester\Machine learning\lab3\features_with_labels.xlsx"
)

A = df["zcr_mean"].values
B = df["rms_mean"].values

p_vals = []
dist_vals = []

for p in range(1, 11):
    s = 0
    for i in range(len(A)):
        s += abs(A[i] - B[i]) ** p
    d = s ** (1 / p)
    p_vals.append(p)
    dist_vals.append(d)
    print("p =", p, "Distance =", d)

plt.plot(p_vals, dist_vals)
plt.show()
