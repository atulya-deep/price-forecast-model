from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import pandas as pd
import numpy as np

def train_model(df):
    # Separate features (X) and target variable (y)
    X = df.drop(columns=['Price'])
    y = df['Price']

    # Split the data into train and test sets based on the 'Date' column
    train_data = df[df['Date'] < '2022-12-31']
    test_data = df[df['Date'] >= '2022-12-31']

    X_train = train_data.drop(columns=['Price', 'Date'])
    y_train = train_data['Price']
    X_test = test_data.drop(columns=['Price', 'Date'])
    y_test = test_data['Price']

    # Interpolate missing values in the test set
    X_test[:] = np.nan
    X_test = X_test.interpolate(method='linear', axis=0)

    # Convert the 'day' column to a categorical type
    X_train['day'] = X_train['day'].astype('category')
    X_test['day'] = X_test['day'].astype('category')

    # Build and train the XGBoost regression model
    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, enable_categorical=True)
    xgb_model.fit(X_train, y_train)

    return xgb_model, X_test, y_test

def forecast_prices(model, X_test):
    # Make predictions
    y_pred = model.predict(X_test)
    return y_pred

def evaluate_model(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = mse ** 0.5
    return mse, rmse, mape

# Example usage:
# df = pd.read_csv('path_to_your_data.csv')
# model, X_test, y_test = train_model(df)
# y_pred = forecast_prices(model, X_test)
# mse, rmse, mape = evaluate_model(y_test, y_pred)
# print(f"Mean Absolute Percentage Error: {mape * 100:.2f}%")
# print(f"Mean Squared Error: {mse}")
# print(f"Root Mean Squared Error: {rmse}")