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
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import argparse
import mlflow

# Get the arugments we need to avoid fixing the dataset path in code
parser = argparse.ArgumentParser()
parser.add_argument("--trainingdata", type=str, required=True, help='Dataset path')
parser.add_argument("--retraining", type=float, required=True, help='setiing retraining parameter true or false')

args = parser.parse_args()
mlflow.autolog()
mlflow.log_param("hello_param", "action_classifier")

data_csv = pd.read_csv(args.trainingdata)

print(data_csv.head())
# %%
# Step 2: dataset loading
df_full = data_csv
df_full.shape
df_full.head()

# %%
# find the data types 
df_full.dtypes.value_counts()


# %%
# Exploratory Data Analysis (EDA)

plt.figure(figsize=(8, 6))
sns.countplot(x='Activity', data=df_full, palette='viridis')
plt.title('Activity Distibution')
plt.xticks(rotation=45)
plt.show()


# %%
# Distribution of numerical vaues
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
# Correlation matrix
plt.figure(figsize=(12, 10))
corr = df_full.corr().iloc[:20, :20]  # choosing the 20 features
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()



# %%
# Dimensionality Reduction (PCA)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Separating target variables and features
X = df_full.drop('Activity', axis=1)
y = df_full['Activity']

# Standardization of features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying pca value 
if args.retraining == True:
    pcavalue = 0.95
else:
    pcavalue = 0.7
pca = PCA(n_components=pcavalue)
X_pca = pca.fit_transform(X_scaled)

# Checking components number
print("components number after PCA:", pca.n_components_)


# %%
#SVM Model 

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import RandomizedSearchCV

# Spliting train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# SVM model
svm_model = SVC()

param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf', 'linear']}

# Randomized search 
random_search = RandomizedSearchCV(estimator=svm_model, param_distributions=param_grid, n_iter=5, cv=5, verbose=2, random_state=42)

# Fitting the data
random_search.fit(X_train, y_train)

#predictions
y_pred = random_search.predict(X_test)

# %% Calculating metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

lb = LabelBinarizer()
y_test_binarized = lb.fit_transform(y_test)

# decision scores & ROC AUC
decision_scores = random_search.decision_function(X_test)
roc_auc = roc_auc_score(y_test_binarized, decision_scores, average='weighted', multi_class='ovr')

# %% Logging metrics to MLflow
mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("precision", precision)
mlflow.log_metric("recall", recall)
mlflow.log_metric("f1_score", f1)
mlflow.log_metric("roc_auc", roc_auc)


#model evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# %% Print metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC AUC Score: {roc_auc}")
