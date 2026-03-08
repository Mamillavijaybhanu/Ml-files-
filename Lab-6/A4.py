import pandas as pd

df = pd.read_excel(
"C:/Users/nunna/OneDrive/Desktop/4th semester/Machine learning/features_with_labels.xlsx"
)

df["rms_mean_bin"] = pd.cut(df["rms_mean"], bins=4)

print(df[["rms_mean","rms_mean_bin"]].head())