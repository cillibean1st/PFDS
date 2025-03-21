# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# read the datasets
dataset = pd.read_csv("weight-height.csv")

# check the dataset
print(dataset.head())

# check if null values is present in dataset or not....
print(dataset.isnull().sum())

# checking the dimensions of the dataset
print(dataset.shape)

# plot gender vs weight
x1 = dataset.iloc[:, 0].values
y1 = dataset.iloc[:, 1].values
plt.scatter(x1, y1, label='Gender', color='green')
plt.xlabel("Gender")
plt.ylabel("Weight")
plt.title("Gender vs Weight")
plt.legend()
plt.show()

# plot height vs weight
x2 = dataset.iloc[:, 1].values
y2 = dataset.iloc[:, 2].values
plt.scatter(x2, y2, label='Height', color='blue')
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Height vs Weight")
plt.legend(loc='lower right')
plt.show()

# separating the dependent and independent values
# X-independent variable
X = dataset.iloc[:, 1:2].values
print(X)

# y-dependent on target
y = dataset.iloc[:, 2].values
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# creating linear regression model

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting the test set
y_pred = regressor.predict(X_test)


print(y_pred)
print('Coefficients: ', regressor.coef_)
# the mean squared error
print("Mean squared error: %.2f" % np.mean((regressor.predict(X_test) - y_test)**2))
# explained variance score
print('Variance score: %.2f' % regressor.score(X_test, y_test))

