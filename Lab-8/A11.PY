from sklearn.neural_network import MLPClassifier

X = [[0,0],[0,1],[1,0],[1,1]]
y_and = [0,0,0,1]
y_xor = [0,1,1,0]

model = MLPClassifier(hidden_layer_sizes=(4,), max_iter=5000)

model.fit(X, y_and)
print("AND:", model.predict(X))

model.fit(X, y_xor)
print("XOR:", model.predict(X))