# Price Forecast Model - Weekly Extrapolation and Prediction

This script performs weekly extrapolation of data and generates predictions for commodity prices using an XGBoost regression model. The predictions are saved as CSV files for each week.

## Folder Structure
price-forecast-model/
├── data/
│   ├── raw/                     # Raw data files
│   ├── processed/               # Processed data files
│   ├── predictions/             # Weekly prediction CSV files
├── notebooks/
│   ├── [price_forecast.ipynb](http://_vscodecontentref_/0)     # Jupyter Notebook for forecasting
├── src/
│   ├── forecasting.py           # Script for weekly extrapolation and prediction

## Workflow

1. **Initialization**:
   - The script starts from a specific date (`2023-05-05`) and iterates weekly until the end of December 2023.
   - A forecasting horizon of 32 weeks is defined.

2. **Data Preparation**:
   - For each week, the script uses historical data up to the current date for training.
   - Missing values are handled using a rolling mean (100-day window) or overall mean for extrapolation.

3. **Extrapolation**:
   - Forecasted values for features such as `Price_diff`, `volume`, `open_price`, and others are calculated using a rolling mean or overall mean.
   - A new row is appended to the dataset for each week in the forecasting horizon.

4. **Model Training**:
   - The XGBoost regression model is trained using historical data up to the current date.
   - Features include `volume`, `open_price`, `USDZAR Curncy`, `W 1 Comdty`, and others.
   - The target variable is `Price`.

5. **Prediction**:
   - The model predicts prices for the extrapolated data beyond the current date.
   - Predictions are saved in a CSV file named `predictions_<current_date>.csv`.

6. **Evaluation**:
   - The model's performance is evaluated using Mean Squared Error (MSE) and Mean Absolute Percentage Error (MAPE).

## Key Features

- **Rolling Mean for Extrapolation**:
  - Uses a 100-day rolling mean for features if sufficient data is available.
  - Falls back to the overall mean if the rolling window is not applicable.

- **Dynamic Forecasting Horizon**:
  - The forecasting horizon decreases by 1 week after each iteration.

- **Weekly Predictions**:
  - Predictions are saved weekly in separate CSV files for easy tracking.

## Output

- **CSV Files**:
  - Each CSV file contains the predicted prices for the forecasting horizon starting from the current date.
  - File format: `predictions_<current_date>.csv`.

- **Performance Metrics**:
  - MSE and MAPE are printed for each iteration to evaluate the model's accuracy.

## Example CSV Output

| Date       | Predicted_Price | Horizon |
|------------|-----------------|---------|
| 2023-05-12 | 150.25          | 1       |
| 2023-05-19 | 152.30          | 2       |
| 2023-05-26 | 148.90          | 3       |

## Notes

- Ensure the dataset (`df`) is preprocessed and contains all required columns before running the script.
- Adjust the `horizon` variable as needed for longer or shorter forecasting periods.
- Install required libraries such as `XGBoost`, `pandas`, `numpy`, and `statsmodels`.

---

## How to Run

1. Place the script in the appropriate directory with access to the dataset.
2. Ensure all dependencies are installed:
   ```bash
   pip install xgboost pandas numpy statsmodels
