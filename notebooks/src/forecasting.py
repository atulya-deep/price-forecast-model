from xgboost import XGBRegressor
import pandas as pd
import numpy as np

def load_model(model_path):
    """Load the trained XGBoost model from the specified path."""
    import joblib
    return joblib.load(model_path)

def generate_forecasts(model, X, horizons=4):
    """Generate price forecasts for the next specified time horizons."""
    forecasts = []
    for _ in range(horizons):
        forecast = model.predict(X)
        forecasts.append(forecast)
        # Prepare the next input for the model (this is a placeholder logic)
        X = X.shift(-1)  # Shift the input data for the next forecast
        X.iloc[-1] = forecast  # Update the last row with the forecasted value
    return np.array(forecasts)

def save_forecasts(forecasts, output_path):
    """Save the generated forecasts to a CSV file."""
    forecast_df = pd.DataFrame(forecasts, columns=[f'Forecast_Horizon_{i+1}' for i in range(forecasts.shape[0])])
    forecast_df.to_csv(output_path, index=False)

def main(model_path, input_data_path, output_path):
    """Main function to load the model, generate forecasts, and save them."""
    model = load_model(model_path)
    input_data = pd.read_csv(input_data_path)
    
    # Assuming the input data is preprocessed and ready for prediction
    forecasts = generate_forecasts(model, input_data)
    save_forecasts(forecasts, output_path)

# Example usage (uncomment to use):
# main('path/to/trained_model.joblib', 'path/to/input_data.csv', 'path/to/output_forecasts.csv')