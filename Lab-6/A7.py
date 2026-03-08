import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

df = pd.read_excel(
"C:/Users/nunna/OneDrive/Desktop/4th semester/Machine learning/features_with_labels.xlsx"
)

df = df[["rms_mean","zcr_mean","label"]].dropna()

X = df[["rms_mean","zcr_mean"]].values
y = df["label"].values

model = DecisionTreeClassifier()
model.fit(X,y)

x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1

xx, yy = np.meshgrid(np.arange(x_min,x_max,0.01),np.arange(y_min,y_max,0.01))

Z = model.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx,yy,Z,alpha=0.3)
plt.scatter(X[:,0],X[:,1],c=y)
plt.xlabel("rms_mean")
plt.ylabel("zcr_mean")
plt.title("Decision Boundary")
plt.show()