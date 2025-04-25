from src.data_preprocessing import load_data, preprocess_data
import pandas as pd
import pytest

def test_load_data():
    # Test loading data from the raw data directory
    df = load_data('data/raw/sample_data.csv')
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_preprocess_data():
    # Test preprocessing of data
    raw_data = pd.DataFrame({
        'Price': [100, 200, None, 400],
        'Feature1': [1, 2, 3, 4],
        'Feature2': [None, 5, 6, 7]
    })
    processed_data = preprocess_data(raw_data)
    
    # Check if missing values are handled
    assert processed_data['Price'].isnull().sum() == 0
    assert processed_data['Feature2'].isnull().sum() == 0
    assert 'Date' in processed_data.columns  # Assuming 'Date' is added during preprocessing

def test_preprocess_data_shape():
    # Test the shape of the processed data
    raw_data = pd.DataFrame({
        'Price': [100, 200, None, 400],
        'Feature1': [1, 2, 3, 4],
        'Feature2': [None, 5, 6, 7]
    })
    processed_data = preprocess_data(raw_data)
    assert processed_data.shape[1] == 3  # Assuming two features and one target variable

@pytest.mark.parametrize("input_data, expected_output", [
    (pd.DataFrame({'Price': [None, 200, 300], 'Feature1': [1, 2, 3]}), 2),
    (pd.DataFrame({'Price': [100, 200, 300], 'Feature1': [1, 2, 3]}), 3),
])
def test_price_column(input_data, expected_output):
    processed_data = preprocess_data(input_data)
    assert processed_data['Price'].count() == expected_output