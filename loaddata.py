import pandas as pd
import numpy as np
from pathlib import Path

def load_interaction_data(file_path, nrows=None):
    """
    Load interaction data from CSV file
    
    Args:
        file_path: Path to the CSV file
        nrows: Number of rows to load (None to load all)
        
    Returns:
        DataFrame with interaction data
    """
    return pd.read_csv(file_path, nrows=nrows)

def load_item_categories(file_path):
    """
    Load item categories data
    
    Returns:
        DataFrame with item categories
    """
    return pd.read_csv(file_path)

def load_user_features(file_path):
    """
    Load user features data
    
    Returns:
        DataFrame with user features
    """
    return pd.read_csv(file_path)

def print_dataset_info(df, name, nrows=5):
    """
    Print basic information about a dataset
    
    Args:
        df: DataFrame to describe
        name: Name of the dataset
        nrows: Number of sample rows to display
    """
    print(f"\n{'-'*50}\nDataset: {name}\n{'-'*50}")
    print(f"Shape: {df.shape} ({df.shape[0]} rows, {df.shape[1]} columns)")
    print(f"\nColumns: {', '.join(df.columns)}")
    print(f"\nSample data:")
    print(df.head(nrows))
    print(f"\nData types:")
    print(df.dtypes)
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\nMissing values:")
        print(missing[missing > 0])
    
    return df 
