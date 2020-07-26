# Import Packages
import pandas as pd
from sklearn.linear_model import LinearRegression

bmi = pd.read_csv('Datasets/BMI.csv')

model = LinearRegression()
model.fit(bmi[['BMI']], bmi[['Life expectancy']])

predict_life = model.predict([[21.07931]])
print(predict_life)