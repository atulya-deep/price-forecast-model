from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = mse ** 0.5
    
    evaluation_results = {
        "Mean Absolute Percentage Error": mape * 100,
        "Mean Squared Error": mse,
        "Root Mean Squared Error": rmse
    }
    
    return evaluation_results