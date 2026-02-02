import pandas as pd
import numpy as np

df = pd.read_excel(
    r"C:\Users\nunna\OneDrive\Desktop\4th semester\Machine learning\lab3\features_with_labels.xlsx"
)
A = df["label"].values
B = df["rms_mean"].values
sumA = 0
for i in range(len(A)):
    sumA += A[i]
meanA = sumA / len(A)
sumB = 0
for i in range(len(B)):
    sumB += B[i]
meanB = sumB / len(B)
varA = 0
for i in range(len(A)):
    varA += (A[i] - meanA) ** 2
varA = varA / len(A)
varB = 0
for i in range(len(B)):
    varB += (B[i] - meanB) ** 2
varB = varB / len(B)
stdA = np.sqrt(varA)
stdB = np.sqrt(varB)
dist = np.sqrt((meanA - meanB) ** 2)
print("Mean of A:", meanA)
print("Variance of A:", varA)
print("Std Dev of A:", stdA)
print("Mean of B:", meanB)
print("Variance of B:", varB)
print("Std Dev of B:", stdB)
print("Euclidean Distance between means:", dist)
