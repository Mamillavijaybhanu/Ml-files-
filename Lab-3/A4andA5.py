import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski

df = pd.read_excel(
    r"C:\Users\nunna\OneDrive\Desktop\4th semester\Machine learning\lab3\features_with_labels.xlsx"
)

A = df["zcr_mean"].values
B = df["rms_mean"].values

p_vals = []
manual_vals = []

for p in range(1, 11):
    s = 0
    for i in range(len(A)):
        s += abs(A[i] - B[i]) ** p
    d_manual = s ** (1 / p)
    d_scipy = minkowski(A, B, p)

    p_vals.append(p)
    manual_vals.append(d_manual)

    print("p =", p, "Manual =", d_manual, "Scipy =", d_scipy)

plt.plot(p_vals, manual_vals)
plt.show()
