import numpy as np

data = np.loadtxt('Datasets/MiniBatch.csv', delimiter=',')
X = data[:,:-1]
y = data[:, -1]
print(X.min(), X.max())