import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset into a Pandas DataFrame
data = pd.read_csv('microbial_growth_data.csv')

# Split the data into training and testing sets
X = data.drop('growth_rate', axis=1)
y = data['growth_rate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model on the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Use the trained model to make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
from sklearn.metrics import r2_score, mean_squared_error
print("R^2 score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
