# %%
import argparse
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

# Get the arugments we need to avoid fixing the dataset path in code
parser = argparse.ArgumentParser()
parser.add_argument("--trainingdata", type=str, required=True, help='Dataset for training')
args = parser.parse_args()
mlflow.autolog()
# %% [markdown]
# ## First load the data
# The first thing we need to do is load the data we're going to work with and have a quick look at a summary of it.
# Pandas gives us a function to read CSV files.
# **You might need update the location of the dataset to point to the correct place you saved it to!**
# "../" means "back one directory from where we are now"
# "./" means "from where we are now"

# %%
df = pd.read_csv(args.trainingdata)
print(df)
# %%
train_df = pd.read_csv('../../train.csv')
train_df.head()
print("hello")

# %%
train_df.shape

# %%
test_df = pd.read_csv('../../test.csv')
test_df.head()

# %%
test_df.shape

# %%
X_train = train_df.iloc[:, :-2]
y_train = train_df.iloc[:, -1]
X_test = test_df.iloc[:, :-2]
y_test = test_df.iloc[:, -1]

# %%
class_label = y_train.value_counts()
plt.figure(figsize=(10, 10))
plt.xticks(rotation=75)
sns.barplot(class_label)

# %%
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# %%
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_train=pd.get_dummies(y_train).values
y_test = label_encoder.fit_transform(y_test)
y_test=pd.get_dummies(y_test).values

# %%
y_train = np.array(y_train)
y_test = np.array(y_test)

# %%
pca = PCA(n_components=None)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# %%
model = keras.models.Sequential()
model.add(keras.layers.Dense(units=64,activation='relu'))
model.add(keras.layers.Dense(units=128,activation='relu'))
model.add(keras.layers.Dense(units=64,activation='relu'))
model.add(keras.layers.Dense(units=6,activation='softmax'))

# %%
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# %%
history = model.fit(X_train, y_train, batch_size=128, epochs=15, validation_split=0.2)

# %%
## Loss Vs. Epochs

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train Loss', 'Validation Loss']);

# %%
## Accuracy Vs. Epochs

plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train Accuracy', 'Validation Accuracy']);

# %%
pred = model.predict(X_test)
predic = []
for p in pred:
    p = np.argmax(p)
    predic.append(p)
predic = np.array(predic)

# %%
y_test[0]
y_test_label = []
for i in range(len(y_test)):
    for ind, j in enumerate(y_test[i]):
        if j == 1:
            y_test_label.append(ind)
y_test_label = np.array(y_test_label)

# %%
print(classification_report(y_test_label, predic))

# %%
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix(y_test_label, predic), annot=True, fmt='g')

# %%
print("Accuracy Score: ", accuracy_score(y_test_label, predic))


