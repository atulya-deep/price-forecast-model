import unittest
import pandas as pd
from src.model_training import train_model

class TestModelTraining(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.X_train = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'day': [1, 2, 3, 4, 5]
        })
        self.y_train = pd.Series([10, 15, 20, 25, 30])

    def test_train_model(self):
        model = train_model(self.X_train, self.y_train)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'predict'))

    def test_model_performance(self):
        model = train_model(self.X_train, self.y_train)
        y_pred = model.predict(self.X_train)
        self.assertEqual(len(y_pred), len(self.y_train))

if __name__ == '__main__':
    unittest.main()