# %% 
import argparse
import mlflow
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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings("ignore")

# Get the arguments for dataset path
parser = argparse.ArgumentParser()
parser.add_argument("--trainingdata", type=str, required=True, help='Path to the training dataset (CSV file)')
parser.add_argument("--testdata", type=str, required=True, help='Path to the testing dataset (CSV file)')
args = parser.parse_args()

# Start mlflow autologging
mlflow.autolog()

# %% [markdown]
# ## Load the data
# Now, let's load the datasets based on the provided paths.

# %% 
train_df = pd.read_csv(args.trainingdata)
test_df = pd.read_csv(args.testdata)

# Check the first few rows of the training dataset
print(train_df.head())

# Check the first few rows of the test dataset
print(test_df.head())

# %% 
# Split features and labels for training and test data
X_train = train_df.iloc[:, :-1]  # Features (all columns except the last one)
y_train = train_df.iloc[:, -1]   # Labels (last column)

X_test = test_df.iloc[:, :-1]    # Features (all columns except the last one)
y_test = test_df.iloc[:, -1]     # Labels (last column)

# %% 
# Visualize class distribution using Seaborn
class_label = y_train.value_counts()
plt.figure(figsize=(10, 10))
plt.xticks(rotation=75)
sns.barplot(x=class_label.index, y=class_label.values)

# %% 
# Standardize features using StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# %% 
# Encode labels using LabelEncoder and convert to one-hot encoding
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_train = pd.get_dummies(y_train).values  # One-hot encoding
y_test = label_encoder.transform(y_test)
y_test = pd.get_dummies(y_test).values  # One-hot encoding

# %% 
# Apply PCA for dimensionality reduction (optional)
pca = PCA(n_components=None)  # You can set the number of components as needed
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# %% 
# Build the neural network model
model = keras.models.Sequential([
    keras.layers.Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dense(units=y_train.shape[1], activation='softmax')  # Output layer with softmax for classification
])

# %% 
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# %% 
# Train the model
history = model.fit(X_train, y_train, batch_size=128, epochs=15, validation_split=0.2)

# %% 
# Plot Loss vs. Epochs
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train Loss', 'Validation Loss'])
plt.title("Loss vs. Epochs")
plt.show()

# %% 
# Plot Accuracy vs. Epochs
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train Accuracy', 'Validation Accuracy'])
plt.title("Accuracy vs. Epochs")
plt.show()

# %% 
# Make predictions on the test data
pred = model.predict(X_test)
predic = np.argmax(pred, axis=1)  # Convert to class labels (not probabilities)

# %% 
# Convert one-hot encoded labels back to integer labels
y_test_label = np.argmax(y_test, axis=1)

# %% 
# Classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test_label, predic))

# %% 
# Plot confusion matrix
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix(y_test_label, predic), annot=True, fmt='g', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# %% 
# Accuracy score
print("Accuracy Score: ", accuracy_score(y_test_label, predic))
