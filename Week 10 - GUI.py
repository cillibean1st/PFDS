import numpy as np
import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")

# get data
data = pd.read_csv("data.csv")
print(data.head())

# get variables
x = data.drop(['Outcome'], axis=1)
y = data['Outcome']

# split the data
x_train, x_test, y_train, y_test = train_test_split(x,y)

# scale the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

# Instantiate and train the model
model

