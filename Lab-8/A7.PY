import numpy as np

# AND gate data
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

# Add bias
X_bias = np.c_[np.ones(len(X)), X]

# Compute weights using pseudo-inverse
w = np.linalg.pinv(X_bias).dot(y)

# Predictions
pred = X_bias.dot(w)

# Convert to binary output
pred_binary = (pred >= 0.5).astype(int)

print("Weights:", w)
print("Predictions:", pred_binary)