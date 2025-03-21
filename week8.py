# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

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

