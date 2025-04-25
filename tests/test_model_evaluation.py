import unittest
from src.model_evaluation import mean_squared_error, mean_absolute_percentage_error

class TestModelEvaluation(unittest.TestCase):

    def test_mean_squared_error(self):
        y_true = [3, -0.5, 2, 7]
        y_pred = [2.5, 0.0, 2, 8]
        mse = mean_squared_error(y_true, y_pred)
        self.assertAlmostEqual(mse, 0.375)

    def test_mean_absolute_percentage_error(self):
        y_true = [100, 200, 300]
        y_pred = [90, 210, 330]
        mape = mean_absolute_percentage_error(y_true, y_pred)
        self.assertAlmostEqual(mape, 0.03333333333333333)  # 3.33%

if __name__ == '__main__':
    unittest.main()