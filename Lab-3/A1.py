import pandas as pd
import numpy as np

df = pd.read_excel(
    r"C:\Users\nunna\OneDrive\Desktop\4th semester\Machine learning\lab3\features_with_labels.xlsx"
)

#A1
A = df["label"].values
B = df["rms_mean"].values
dp = 0
for i in range(len(A)):
    dp += A[i] * B[i]
sA = 0
for i in range(len(A)):
    sA += A[i] ** 2
normA = np.sqrt(sA)
sB = 0
for i in range(len(B)):
    sB += B[i] ** 2
normB = np.sqrt(sB)
print("Dot Product:", dp)
print("Length of A:", normA)
print("Length of B:", normB)

