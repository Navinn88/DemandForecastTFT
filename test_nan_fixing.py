import pandas as pd
import numpy as np
from example_usage import load_and_prepare_data, analyze_and_fix_nans, engineer_features

def test_nan_analysis():
    """
    Test script to analyze and fix NaN values in the dataset.
    """
    print("=== Testing NaN Analysis and Fixing ===")
    
    # Load data with smaller subset for testing
    print("Loading data...")
    df = load_and_prepare_data(subset_size=25_000, random_state=42)
    
    print(f"\nBEFORE feature engineering:")
    print(f"Shape: {df.shape}")
    print(f"NaN count: {df.isnull().sum().sum()}")
    
    # Show which columns have NaN values
    nan_counts = df.isnull().sum().sort_values(ascending=False)
    columns_with_nans = nan_counts[nan_counts > 0]
    
    if len(columns_with_nans) > 0:
        print(f"\nColumns with NaN values:")
        for col, count in columns_with_nans.items():
            print(f"  {col}: {count} NaNs ({count/len(df)*100:.2f}%)")
    
    # Fix NaN values
    print(f"\nFixing NaN values...")
    df = analyze_and_fix_nans(df)
    
    # Engineer features
    print(f"\nEngineering features...")
    df = engineer_features(df)
    
    print(f"\nAFTER feature engineering:")
    print(f"Shape: {df.shape}")
    print(f"NaN count: {df.isnull().sum().sum()}")
    
    # Check specific lag/rolling features
    lag_features = ['lag_7', 'lag_28', 'rolling_mean_7', 'rolling_mean_28', 'rolling_std_7']
    print(f"\nNaN counts for lag/rolling features:")
    for feature in lag_features:
        if feature in df.columns:
            nan_count = df[feature].isnull().sum()
            print(f"{feature}: {nan_count} NaNs")
    
    # Final cleanup if needed
    if df.isnull().sum().sum() > 0:
        print(f"\nFinal cleanup needed...")
        df = df.fillna(0)
        print(f"Final NaN count: {df.isnull().sum().sum()}")
    
    print("=== Test Complete ===")
    return df

if __name__ == "__main__":
    df = test_nan_analysis() 