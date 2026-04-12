import pandas as pd
import numpy as np
data = pd.DataFrame({
    "candies":[20,16,27,19,24,22,15,18,21,16],
    "mangoes":[6,3,6,1,4,1,4,4,1,2],
    "milk":[2,6,2,2,2,5,2,2,4,4],
    "payment":[386,289,393,110,280,167,271,274,148,198],
    "label":[1,1,1,0,1,0,1,1,0,0]
})

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

w = np.random.rand(4)
b = np.random.rand()
lr = 0.01

for epoch in range(500):
    for i in range(len(X)):
        z = np.dot(X[i], w) + b
        y_pred = 1/(1+np.exp(-z))

        e = y[i] - y_pred

        w += lr * e * X[i]
        b += lr * e

print("Training done")