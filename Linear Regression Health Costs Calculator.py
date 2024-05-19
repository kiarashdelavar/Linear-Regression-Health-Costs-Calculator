import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

data = pd.read_csv('insurance.csv')

le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])
data['smoker'] = le.fit_transform(data['smoker'])
data['region'] = le.fit_transform(data['region'])

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

train_labels = train_data.pop('expenses')
test_labels = test_data.pop('expenses')

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_data.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

model.compile(loss='mse', optimizer='adam', metrics=['mae'])

history = model.fit(train_data, train_labels, epochs=100, validation_split=0.2, verbose=0)

test_loss, test_mae = model.evaluate(test_data, test_labels, verbose=0)
print("Mean Absolute Error: ${:.2f}".format(test_mae))

test_predictions = model.predict(test_data).flatten()

plt.figure(figsize=(8, 6))
plt.scatter(test_labels, test_predictions, alpha=0.6)
plt.xlabel('True Expenses ($)')
plt.ylabel('Predicted Expenses ($)')
plt.title('Actual vs. Predicted Expenses')
plt.plot([0, max(test_labels)], [0, max(test_predictions)], color='red', linestyle='--')
plt.show()


model.compile(loss='mse', optimizer='adam', metrics=['mae'])

history = model.fit(train_data, train_labels, epochs=100, validation_split=0.2, verbose=0)

test_loss, test_mae = model.evaluate(test_data, test_labels, verbose=0)

print("Mean Absolute Error: ${:.2f}".format(test_mae))

import matplotlib.pyplot as plt

test_predictions = model.predict(test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Expenses ($)')
plt.ylabel('Predicted Expenses ($)')
plt.title('Actual vs. Predicted Expenses')
plt.show()
