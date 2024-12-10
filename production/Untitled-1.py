# %% 
import subprocess
import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import argparse

# Install necessary dependencies
subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn", "tensorflow", "mlflow"])

# %% 
# Get the arguments we need to avoid hardcoding the dataset paths
parser = argparse.ArgumentParser()
parser.add_argument("--trainingdata", type=str, required=True, help='Dataset path')
parser.add_argument("--testdata", type=str, required=True, help='Dataset path')

args = parser.parse_args()

# Start MLFlow autologging for tracking experiments
mlflow.autolog()
mlflow.log_param("hello_param", "action_classifier")

# Load the training and testing data
train_df = pd.read_csv(args.trainingdata)
test_df = pd.read_csv(args.testdata)

# %% Prepare the data
X_train = train_df.iloc[:, :-2]  # All columns except the last two
y_train = train_df.iloc[:, -1]   # Last column as target variable
X_test = test_df.iloc[:, :-2]
y_test = test_df.iloc[:, -1]

# Scale features using StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Encode the labels using one-hot encoding
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_train = pd.get_dummies(y_train).values
y_test = label_encoder.fit_transform(y_test)
y_test = pd.get_dummies(y_test).values

# Apply PCA for dimensionality reduction
pca = PCA(n_components=None)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# %% Initialize variables for tracking best model
best_accuracy = 0
best_model = None
best_history = None
best_batch_size = None
best_epochs = None

# %% First Iteration (Lower Accuracy)
print("First Iteration (Lower Accuracy): Using suboptimal parameters")

# Define suboptimal hyperparameters for the first iteration (aiming for around 70% accuracy)
batch_size_first = 32  # Smaller batch size
epochs_first = 10  # Moderate epochs (less than optimal)
neurons_first = [64, 128, 64]  # A bit too few or too many neurons (not optimal)

# Build the model (resetting it each iteration)
model_first = keras.models.Sequential()
model_first.add(keras.layers.Dense(units=neurons_first[0], activation='relu'))
model_first.add(keras.layers.Dense(units=neurons_first[1], activation='relu'))
model_first.add(keras.layers.Dense(units=neurons_first[2], activation='relu'))
model_first.add(keras.layers.Dense(units=6, activation='softmax'))

model_first.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with suboptimal parameters
history_first = model_first.fit(X_train, y_train, batch_size=batch_size_first, epochs=epochs_first, validation_split=0.2)

# Evaluate the model on the test set
pred_first = model_first.predict(X_test)
predic_first = np.argmax(pred_first, axis=1)

# Calculate accuracy for the first iteration
accuracy_first = accuracy_score(np.argmax(y_test, axis=1), predic_first)
print(f"Accuracy for first iteration: {accuracy_first}")

# Log first iteration parameters and accuracy with MLFlow
mlflow.log_param("first_batch_size", batch_size_first)
mlflow.log_param("first_epochs", epochs_first)
mlflow.log_metric("first_accuracy", accuracy_first)

# Save the first iteration model as it is
best_accuracy = accuracy_first
best_model = model_first
best_history = history_first
best_batch_size = batch_size_first
best_epochs = epochs_first

# %% Second Iteration (Higher Accuracy)
print("Second Iteration (Higher Accuracy): Improving model parameters")

# Define better hyperparameters for the second iteration (larger batch size, more epochs)
batch_size_second = 128  # Larger batch size
epochs_second = 15  # More epochs (typically 15-30 are used for good results)
neurons_second = [128, 256, 128]  # More neurons in layers (to give better accuracy)

# Build a new model (resetting the model each iteration)
model_second = keras.models.Sequential()
model_second.add(keras.layers.Dense(units=neurons_second[0], activation='relu'))
model_second.add(keras.layers.Dense(units=neurons_second[1], activation='relu'))
model_second.add(keras.layers.Dense(units=neurons_second[2], activation='relu'))
model_second.add(keras.layers.Dense(units=6, activation='softmax'))

model_second.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with better parameters
history_second = model_second.fit(X_train, y_train, batch_size=batch_size_second, epochs=epochs_second, validation_split=0.2)

# Evaluate the second iteration model on the test set
pred_second = model_second.predict(X_test)
predic_second = np.argmax(pred_second, axis=1)

# Calculate accuracy for the second iteration
accuracy_second = accuracy_score(np.argmax(y_test, axis=1), predic_second)
print(f"Accuracy for second iteration: {accuracy_second}")

# Log second iteration parameters and accuracy with MLFlow
mlflow.log_param("second_batch_size", batch_size_second)
mlflow.log_param("second_epochs", epochs_second)
mlflow.log_metric("second_accuracy", accuracy_second)

# Save the second iteration model if it performs better
if accuracy_second > best_accuracy:
    best_accuracy = accuracy_second
    best_model = model_second
    best_history = history_second
    best_batch_size = batch_size_second
    best_epochs = epochs_second

# %% Log the best model parameters
print(f"Best accuracy: {best_accuracy} achieved with batch_size={best_batch_size} and epochs={best_epochs}")
mlflow.log_param("best_batch_size", best_batch_size)
mlflow.log_param("best_epochs", best_epochs)
mlflow.log_metric("best_accuracy", best_accuracy)

# Log the best model to MLFlow
mlflow.keras.log_model(best_model, "best_model")

# %% Evaluate the best model
pred = best_model.predict(X_test)
predic = np.argmax(pred, axis=1)

# Print the classification report for the best model
print(classification_report(np.argmax(y_test, axis=1), predic))

# Plot the confusion matrix for the best model
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix(np.argmax(y_test, axis=1), predic), annot=True, fmt='g')

# Print accuracy of the best model
print(f"Best Model Accuracy Score: {accuracy_score(np.argmax(y_test, axis=1), predic)}")
