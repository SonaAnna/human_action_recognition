# %%
import subprocess
import sys

# Ensure seaborn is installed
subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])

# Ensure tensorflow is installed
subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, accuracy_score
import argparse
import mlflow

# Get the arugments we need to avoid fixing the dataset path in code
parser = argparse.ArgumentParser()
parser.add_argument("--trainingdata", type=str, required=True, help='Dataset path')

args = parser.parse_args()
mlflow.autolog()
mlflow.log_param("hello_param", "action_classifier")

data_csv = pd.read_csv(args.trainingdata)
print(data_csv.head())
# %%
# Step 2: Load the dataset
df_full = data_csv
df_full.shape
df_full.head()

# %%
# Display the data types of the columns
df_full.dtypes.value_counts()


# %%
# Exploratory Data Analysis (EDA)

# Distribution of the target variable
plt.figure(figsize=(8, 6))
sns.countplot(x='Activity', data=df_full, palette='viridis')
plt.title('Distribution of Activity')
plt.xticks(rotation=45)
plt.show()


# %%
# Distribution of some of the numerical features
features = ['tBodyAcc-mean()-X', 'tBodyAcc-mean()-Y', 'tBodyAcc-mean()-Z', 
            'tBodyAcc-std()-X', 'tBodyAcc-std()-Y', 'tBodyAcc-std()-Z', 
            'tBodyAcc-mad()-X', 'tBodyAcc-mad()-Y', 'tBodyAcc-mad()-Z']

plt.figure(figsize=(15, 10))
for i, feature in enumerate(features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df_full[feature], kde=True, color='blue', bins=30)
    plt.title(f'Distribution of {feature}')
    plt.xlabel('')
    plt.ylabel('')
plt.tight_layout()
plt.show()


# %%
# Correlation heatmap
plt.figure(figsize=(12, 10))
corr = df_full.corr().iloc[:20, :20]  # Selecting the first 20 features
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()



# %%
# Step 4: Dimensionality Reduction (PCA)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Separate features and target variable
X = df_full.drop('Activity', axis=1)
y = df_full['Activity']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Check the number of components after PCA
print("Number of components after PCA:", pca.n_components_)


# %%
# Step 5: SVM Model Development

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import RandomizedSearchCV

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Define the SVM model
svm_model = SVC()

param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf', 'linear']}

# Randomized search with reduced parameter grid and n_iter
random_search = RandomizedSearchCV(estimator=svm_model, param_distributions=param_grid, n_iter=5, cv=5, verbose=2, random_state=42)

# Fit the randomized search to the data
random_search.fit(X_train, y_train)

# Make predictions
y_pred = random_search.predict(X_test)

# %% Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# For multi-class classification, use one-vs-rest (ovr) for ROC AUC
y_prob = random_search.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')

# %% Log metrics to MLflow
mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("precision", precision)
mlflow.log_metric("recall", recall)
mlflow.log_metric("f1_score", f1)
mlflow.log_metric("roc_auc", roc_auc)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

