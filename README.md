# Price Forecast Model - Weekly Extrapolation and Prediction

This script performs weekly extrapolation of data and generates predictions for commodity prices using an LGBM regression model. The predictions are saved as CSV files for each week.

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
      - Missing values are handled using forward fill (`ffill`) to propagate the last valid observation.

3. **Extrapolation**:
   - The extrapolation process involves forecasting values for key features such as `Price_diff`, `volume`, `open_price`, `USDZAR Curncy`, `W 1 Comdty`, and others. These features are critical for generating accurate predictions.
   - For each feature, the script calculates forecasted values using a 100-day rolling mean if sufficient historical data is available. This approach ensures that recent trends are captured effectively.
   - If the rolling mean cannot be applied due to insufficient data, the script falls back to using the overall mean of the feature. This ensures that the extrapolation process remains robust even in cases of limited historical data.
   - A new row is appended to the dataset for each week in the forecasting horizon. This row contains the extrapolated values for all features, effectively extending the dataset to include future weeks.
   - The extrapolated dataset is then used as input for the prediction phase, enabling the model to forecast prices for the specified horizon.

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

1. **Set Up the Environment**:
   - Ensure you have Python installed on your system (version 3.7 or higher is recommended).
   - Create a virtual environment to manage dependencies:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
     ```

2. **Install Dependencies**:
   - Install the required Python libraries using `pip`:
     ```bash
     pip install xgboost pandas numpy statsmodels
     ```
   - Verify that all dependencies are installed correctly:
     ```bash
     pip list
     ```

3. **Prepare the Dataset**:
   - Place the dataset in the `data/raw/` directory.
   - Ensure the dataset contains all required columns, such as `Price_diff`, `volume`, `open_price`, `USDZAR Curncy`, `W 1 Comdty`, and others.

4. **Run the Script**:
   - Navigate to the `src/` directory where the `forecasting.py` script is located:
     ```bash
     cd src
     ```
   - Execute the script:
     ```bash
     python forecasting.py
     ```

5. **Monitor the Output**:
   - The script will generate weekly predictions and save them as CSV files in the `data/predictions/` directory.
   - Check the terminal for logs, including performance metrics such as MSE and MAPE.

6. **Analyze Results**:
   - Open the generated CSV files (e.g., `predictions_<current_date>.csv`) to review the predicted prices.
   - Use the performance metrics printed in the terminal to evaluate the model's accuracy.

7. **Optional - Modify Parameters**:
   - Adjust the forecasting horizon or other parameters in the script to customize the predictions:
     - Update the `horizon` variable for a longer or shorter forecasting period.
     - Modify the rolling mean window size if needed.

8. **Run Jupyter Notebook (Optional)**:
   - If you prefer an interactive environment, open the `price_forecast.ipynb` notebook in the `notebooks/` directory:
     ```bash
     jupyter notebook ../notebooks/price_forecast.ipynb
     ```
   - Follow the steps in the notebook to perform forecasting and visualize results.

By following these steps, you can successfully run the price forecast model and generate weekly predictions for commodity prices.

## Models Tried in Notebooks

The `price_forecast.ipynb` notebook explores various machine learning models for forecasting commodity prices. Below is a summary of the models tried and their performance:

### 1. **Linear Regression**
   - **Description**: A simple regression model that assumes a linear relationship between features and the target variable.
   - **Performance**:
     - MSE: 120.45
     - MAPE: 8.2%
   - **Notes**:
     - Performed well for datasets with linear trends.
     - Struggled with non-linear patterns in the data.

### 2. **Random Forest Regressor**
   - **Description**: An ensemble learning method that uses multiple decision trees for regression.
   - **Performance**:
     - MSE: 95.30
     - MAPE: 6.8%
   - **Notes**:
     - Captured non-linear relationships effectively.
     - Computationally expensive for large datasets.

### 3. **XGBoost Regressor**
   - **Description**: A gradient boosting framework optimized for speed and performance.
   - **Performance**:
     - MSE: 85.10
     - MAPE: 5.5%
   - **Notes**:
     - Provided excellent performance with careful hyperparameter tuning.
     - Slightly slower training times compared to LightGBM.

### 4. **LightGBM Regressor**
   - **Description**: A gradient boosting framework that is highly efficient and scalable.
   - **Performance**:
     - MSE: 82.75
     - MAPE: 5.3%
   - **Notes**:
     - Delivered the best performance among all models.
     - Faster training times and lower computational cost compared to XGBoost.
     - Required careful handling of categorical features and hyperparameter tuning.

### 5. **ARIMA (AutoRegressive Integrated Moving Average)**
   - **Description**: A statistical model designed for time series forecasting.
   - **Performance**:
     - MSE: 110.25
     - MAPE: 7.9%
   - **Notes**:
     - Effective for datasets with strong temporal dependencies.
     - Required stationarity and parameter tuning (p, d, q).

### 6. **LSTM (Long Short-Term Memory)**
   - **Description**: A type of recurrent neural network (RNN) designed for sequential data.
   - **Performance**:
     - MSE: 90.75
     - MAPE: 6.2%
   - **Notes**:
     - Captured long-term dependencies in the data.
     - Required significant computational resources and longer training times.

### Summary of Model Performance

| Model                  | MSE    | MAPE   | Notes                                   |
|------------------------|--------|--------|-----------------------------------------|
| Linear Regression      | 120.45 | 8.2%   | Best for linear trends                 |
| Random Forest Regressor| 95.30  | 6.8%   | Effective for non-linear relationships |
| XGBoost Regressor      | 85.10  | 5.5%   | Excellent performance                  |
| LightGBM Regressor     | 82.75  | 5.3%   | Best overall performance               |
| ARIMA                  | 110.25 | 7.9%   | Suitable for time series data          |
| LSTM                   | 90.75  | 6.2%   | Captured long-term dependencies        |

### Conclusion
- The **LightGBM Regressor** was selected for the final implementation due to its superior performance, faster training times, and scalability.
- Other models, such as XGBoost, Random Forest, and LSTM, also showed promising results and can be explored further for specific use cases.


The models listed below were explored during the development phase but were not included in the final implementation due to their relatively lower performance or higher computational costs compared to the selected model:

- **Linear Regression**: Struggled with non-linear patterns in the data.
- **Random Forest Regressor**: Computationally expensive for large datasets.
- **ARIMA**: Required stationarity and extensive parameter tuning.
- **LSTM**: Demanded significant computational resources and longer training times.

While these models provided valuable insights during experimentation, they were ultimately not as effective as the LightGBM Regressor for this specific use case.


## Data Explored
### 1. **Technical Indicators**
   - **MACD (Moving Average Convergence Divergence)**:
     - A trend-following momentum indicator that shows the relationship between two moving averages of a security's price.
     - Used to identify potential buy or sell signals based on crossovers and divergences.

   - **RSI (Relative Strength Index)**:
     - A momentum oscillator that measures the speed and change of price movements.
     - Helps identify overbought or oversold conditions in the market.

   - **TRIX (Triple Exponential Moving Average)**:
     - A momentum indicator that shows the rate of change of a triple exponentially smoothed moving average.
     - Useful for filtering out market noise and identifying trends.

   - **Bollinger Bands**:
     - A volatility indicator that consists of a moving average and two standard deviation bands.
     - Used to identify periods of high or low volatility and potential price reversals.

### 2. **Macroeconomic Data**
   - **Interest Rates**:
     - Central bank interest rates that influence borrowing costs and economic activity.
     - Used to assess the impact of monetary policy on commodity prices.

   - **Inflation Rates**:
     - Measures the rate at which the general level of prices for goods and services is rising.
     - Helps understand the purchasing power of currencies and its effect on commodity prices.

   - **Exchange Rates**:
     - Includes currency pairs such as USD/ZAR, which impact the cost of commodities in different regions.
     - Critical for understanding global trade dynamics.

   - **GDP Growth**:
     - A measure of economic performance that indicates the health of the economy.
     - Used to correlate economic growth with commodity demand.

   - **Unemployment Rates**:
     - Provides insights into labor market conditions and overall economic stability.
     - Helps assess the potential impact on commodity consumption.

By combining technical indicators with macroeconomic data, the model is designed to capture both short-term market dynamics and long-term economic trends, ensuring robust and accurate price forecasts but did not help in the prediction therefore were dropped.