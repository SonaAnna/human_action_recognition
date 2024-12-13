import argparse
import numpy as np
import pandas as pd
import mlflow
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Function to load data
def loadData(file_path):
    """
    Load the dataset from a CSV file and separate features and target labels.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        X (DataFrame): Features.
        y (Series): Target labels.
    """
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1]  # All columns except the last one
    y = df.iloc[:, -1]   # Last column as target
    return X, y

# Function to preprocess data
def preprocessData(X, pca_components=0.95):
    """
    Preprocess features by scaling and applying PCA for dimensionality reduction.
    Args:
        X (DataFrame): Features.
        pca_components (float): Variance to retain for PCA.
    Returns:
        X_transformed (ndarray): Preprocessed features.
        scaler (StandardScaler): Fitted scaler object.
        pca (PCA): Fitted PCA object.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=pca_components)
    X_transformed = pca.fit_transform(X_scaled)
    
    return X_transformed, scaler, pca

# Function to build and train the model
def trainModel(X_train, y_train, model_save_path=None):
    """
    Train an SVM model with the given training data.
    Args:
        X_train (ndarray): Training features.
        y_train (Series): Training target labels.
        model_save_path (str): Path to save the trained model.
    Returns:
        model (SVC): Trained model.
    """
    model = SVC(kernel='rbf', C=1, gamma=0.1, probability=True)
    model.fit(X_train, y_train)

    if model_save_path:
        print(f"Saving model to {model_save_path}")
        mlflow.sklearn.save_model(model, model_save_path)
    
    return model

# Function to evaluate the model
def evaluateModel(model, X_test, y_test):
    """
    Evaluate the model on the test dataset.
    Args:
        model (SVC): Trained model.
        X_test (ndarray): Test features.
        y_test (Series): Test target labels.
    Returns:
        metrics (dict): Dictionary containing evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(pd.get_dummies(y_test), y_pred_proba, average='weighted', multi_class='ovr')
    classification_rep = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("ROC AUC Score:", roc_auc)
    print("Classification Report:\n", classification_rep)
    
    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "classification_report": classification_rep
    }

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train an SVM model with PCA preprocessing.")
    parser.add_argument('--data', type=str, required=True, help="Path to the dataset CSV file.")
    parser.add_argument('--model', type=str, default=None, help="Path to save the trained model.")
    args = parser.parse_args()

    # Enable MLflow autologging
    mlflow.autolog()

    # Load dataset
    X, y = loadData(args.data)

    # Preprocess data
    X_transformed, scaler, pca = preprocessData(X, pca_components=0.95)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

    # Train model
    model = trainModel(X_train, y_train, model_save_path=args.model)

    # Evaluate model
    metrics = evaluateModel(model, X_test, y_test)

    # Log metrics
    for metric_name, metric_value in metrics.items():
        if metric_name != "classification_report":  # Log only numerical metrics
            mlflow.log_metric(metric_name, metric_value)
