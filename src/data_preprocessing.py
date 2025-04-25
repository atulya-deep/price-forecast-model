from pathlib import Path
import pandas as pd

def load_data(file_path):
    """Load raw data from a specified file path."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Clean the data by handling missing values and outliers."""
    df = df.dropna()  # Drop rows with missing values
    # Additional cleaning steps can be added here
    return df

def transform_data(df):
    """Transform the data into a suitable format for analysis."""
    # Example transformation: converting date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    # Additional transformation steps can be added here
    return df

def preprocess_data(file_path):
    """Load, clean, and transform the data."""
    raw_data = load_data(file_path)
    cleaned_data = clean_data(raw_data)
    processed_data = transform_data(cleaned_data)
    return processed_data

def save_processed_data(df, output_path):
    """Save the processed data to a specified output path."""
    df.to_csv(output_path, index=False)