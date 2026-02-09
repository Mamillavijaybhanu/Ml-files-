import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

y_true = np.array([100, 120, 130, 150, 170])
y_pred = np.array([110, 115, 140, 145, 165])

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
r2 = r2_score(y_true, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("MAPE:", mape)
print("R2 Score:", r2)