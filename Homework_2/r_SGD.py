import numpy as np
import matplotlib.pyplot as plt

lr = 0.00001
for i in range(10):
    X = np.random.normal(size=(5000, 2304))
    y = np.random.normal(size=(5000, 1))
    W = np.random.normal(size=(2304, 1))
    b = np.random.normal(size=(1, 1))
    errors = []
    for j in range(1000):
        y_pred = np.dot(X, W) + b
        error = y - y_pred
        W += X.T.dot(error) * lr
        b += np.mean(y - y_pred) * lr
        # errors.append(np.mean(np.abs(error)))
        errors.append(np.mean(np.abs(W)))
    print(errors)
    plt.plot(errors/np.mean(errors))
plt.savefig('plot.png')