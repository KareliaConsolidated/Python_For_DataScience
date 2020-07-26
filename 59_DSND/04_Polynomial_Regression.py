import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

train_data = pd.read_csv('Datasets/PolyData.csv')
X = train_data['Var_X'].values.reshape(-1,1)
y = train_data['Var_Y'].values

poly_feat = PolynomialFeatures(degree = 4)
X_poly = poly_feat.fit_transform(X)

poly_model = LinearRegression().fit(X_poly, y)