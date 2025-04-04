import numpy as np
import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import warnings

warnings.filterwarnings("ignore")

# get data
data = pd.read_csv("diabetes(1).csv")
print(data.head())

# get variables
x = data.drop(['Outcome'], axis=1)
y = data['Outcome']

# split the data
x_train, x_test, y_train, y_test = train_test_split(x, y)

# scale the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

# Instantiate and train the model
model = MLPClassifier(max_iter=1000, alpha=1)
model.fit(x_train, y_train)
print("Model Accuracy: on training set:", model.score(x_train, y_train))
print("Model Accuracy: on test set:", model.score(x_test, y_test))

# create the function for gradio
print(data.columns)


def diabetes(pregnancies, glucose, blood_pressure, skinthickness, insulin, bmi, diabetespedigreefunction, age):
    x = np.array([pregnancies, glucose, blood_pressure, skinthickness, insulin, bmi, diabetespedigreefunction, age])
    prediction = model.predict(x.reshape(1, -1))
    return prediction


outputs = gr.Textbox()
app = gr.Interface(fn=diabetes,
                   inputs=['number', 'number', 'number', 'number', 'number', 'number', 'number', 'number'],
                   outputs=outputs, description="This is a diabetes model")

app.launch(share=True)
