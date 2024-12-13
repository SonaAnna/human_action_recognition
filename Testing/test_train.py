import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.decomposition import PCA

class TestModelPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Simulate data for testing
        np.random.seed(42)
        cls.data = pd.DataFrame(
            np.random.randn(100, 565),  # 100 samples, 565 features (including target)
            columns=[f"feature_{i}" for i in range(565)]
        )
        cls.data['Activity'] = np.random.choice(['Walking', 'Running', 'Standing'], size=100)

        cls.features = cls.data.drop('Activity', axis=1)
        cls.target = cls.data['Activity']

        cls.scaler = StandardScaler()
        cls.lb = LabelBinarizer()

    def test_scaling(self):
        """Test feature scaling with StandardScaler"""
        scaled_features = self.scaler.fit_transform(self.features)
        self.assertEqual(scaled_features.shape, self.features.shape)
        self.assertAlmostEqual(np.mean(scaled_features, axis=0).sum(), 0, delta=1e-5)

    def test_pca(self):
        """Test dimensionality reduction using PCA"""
        scaled_features = self.scaler.fit_transform(self.features)

        pca = PCA(n_components=0.95)
        reduced_features = pca.fit_transform(scaled_features)

        self.assertLess(reduced_features.shape[1], self.features.shape[1])
        self.assertGreaterEqual(pca.explained_variance_ratio_.sum(), 0.95)

    def test_train_test_split(self):
        """Test splitting of data into train and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.target, test_size=0.2, random_state=42)

        self.assertEqual(len(X_train) + len(X_test), len(self.features))
        self.assertEqual(len(y_train) + len(y_test), len(self.target))

    def test_model_training_and_metrics(self):
        """Test SVM model training and metrics"""
        # Scale features
        scaled_features = self.scaler.fit_transform(self.features)

        # Apply PCA
        pca = PCA(n_components=0.95)
        reduced_features = pca.fit_transform(scaled_features)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(reduced_features, self.target, test_size=0.2, random_state=42)

        # SVM model
        model = SVC(kernel='rbf', C=1, gamma=0.1, probability=True)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_test_binarized = self.lb.fit_transform(y_test)

        decision_scores = model.decision_function(X_test)
        roc_auc = roc_auc_score(y_test_binarized, decision_scores, average='weighted', multi_class='ovr')

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Asserting metrics are within valid range
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)
        self.assertGreaterEqual(precision, 0)
        self.assertLessEqual(precision, 1)
        self.assertGreaterEqual(recall, 0)
        self.assertLessEqual(recall, 1)
        self.assertGreaterEqual(f1, 0)
        self.assertLessEqual(f1, 1)
        self.assertGreaterEqual(roc_auc, 0)
        self.assertLessEqual(roc_auc, 1)

if __name__ == '__main__':
    unittest.main()
