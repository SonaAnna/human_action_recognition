import unittest
import tempfile
import os
import mlflow
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from mlflow.sklearn import load_model

class TestModelPipelineIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Simulate data for testing
        np.random.seed(42)
        cls.data = pd.DataFrame(
            np.random.randn(100, 565),  # 100 samples, 565 features (including target)
            columns=[f"feature_{i}" for i in range(565)]
        )
        cls.data['Activity'] = np.random.choice(['Walking', 'Running', 'Standing'], size=100)

    def test_pipeline_integration(self):
        """Test end-to-end pipeline integration."""
        features = self.data.drop('Activity', axis=1)
        target = self.data['Activity']

        # Temporary directory for saving model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model")

            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)

            # Apply PCA
            pca = PCA(n_components=0.95)
            reduced_features = pca.fit_transform(scaled_features)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(reduced_features, target, test_size=0.2, random_state=42)

            # Train model
            model = SVC(kernel='rbf', C=1, gamma=0.1, probability=True)
            model.fit(X_train, y_train)

            # Save model using MLflow
            mlflow.sklearn.save_model(model, model_path)

            # Assert model file exists
            self.assertTrue(os.path.exists(model_path))

            # Load model back
            loaded_model = load_model(model_path)

            # Assert loaded model is of correct type
            self.assertIsInstance(loaded_model, SVC)

            # Predict with the loaded model
            y_pred = loaded_model.predict(X_test)

            # Check prediction accuracy
            accuracy = accuracy_score(y_test, y_pred)

            # Assert accuracy is within valid range
            self.assertGreaterEqual(accuracy, 0)
            self.assertLessEqual(accuracy, 1)

if __name__ == '__main__':
    unittest.main()
