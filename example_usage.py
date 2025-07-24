import pandas as pd
import numpy as np
from data_preprocessing import TFTDataPreprocessor
from pytorch_forecasting import TimeSeriesDataSet
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(subset_size=100_000, random_state=42):
    """
    Load and prepare data from CSV files, using a subset for development.
    
    Args:
        subset_size: Number of rows to sample from train.csv (default: 100,000)
        random_state: Random seed for reproducible sampling
    
    Returns:
        pd.DataFrame: Prepared DataFrame with time_idx and target
    """
    print(f"Loading data with subset size: {subset_size:,}")
    
    # Load train data with subset
    train = pd.read_csv("train.csv", parse_dates=["date"])
    print(f"Original train shape: {train.shape}")
    
    # Sample subset for development
    train_subset = train.sample(n=subset_size, random_state=random_state).copy()
    print(f"Subset train shape: {train_subset.shape}")
    
    # Load other data
    test = pd.read_csv("test.csv", parse_dates=["date"])
    items = pd.read_csv("items.csv")
    stores = pd.read_csv("stores.csv")
    transactions = pd.read_csv("transactions.csv", parse_dates=["date"])
    oil = pd.read_csv("oil.csv", parse_dates=["date"])
    holidays = pd.read_csv("holidays_events.csv", parse_dates=["date"])
    
    print("Data loaded successfully!")
    
    # Create time_idx for train subset
    train_subset = train_subset.sort_values(["store_nbr", "item_nbr", "date"])
    train_subset["time_idx"] = (train_subset["date"] - train_subset["date"].min()).dt.days
    
    # Merge with other datasets
    print("Merging datasets...")
    
    # Merge with items
    df = train_subset.merge(items, on="item_nbr", how="left")
    
    # Merge with stores
    df = df.merge(stores, on="store_nbr", how="left")
    
    # Merge with transactions (store-level)
    df = df.merge(transactions, on=["store_nbr", "date"], how="left")
    
    # Merge with oil (date-level)
    df = df.merge(oil, on="date", how="left")
    
    # Merge with holidays (date-level)
    df = df.merge(holidays, on="date", how="left")
    
    print(f"Final merged shape: {df.shape}")
    
    # Convert categorical columns to string type for TFT
    categorical_columns = ["store_nbr", "item_nbr", "family", "class", "perishable", 
                          "city", "state", "type_x", "cluster"]
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    print("Data preparation completed!")
    
    return df

def analyze_and_fix_nans(df: pd.DataFrame):
    """
    Comprehensive analysis and fixing of NaN values in the dataset.
    """
    print("=== NaN Analysis and Fixing ===")
    
    # Before fixing
    print(f"BEFORE fixing:")
    print(f"Shape: {df.shape}")
    print(f"NaN count: {df.isnull().sum().sum()}")
    
    # Analyze NaN by column
    nan_counts = df.isnull().sum().sort_values(ascending=False)
    columns_with_nans = nan_counts[nan_counts > 0]
    
    if len(columns_with_nans) > 0:
        print(f"\nColumns with NaN values:")
        for col, count in columns_with_nans.items():
            print(f"  {col}: {count} NaNs ({count/len(df)*100:.2f}%)")
    
    # Fix NaN values systematically
    print(f"\nFixing NaN values...")
    
    # 1. Fix categorical columns
    categorical_cols = ['family', 'class', 'perishable', 'city', 'state', 'type_x', 'cluster']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            print(f"  ✓ {col}: filled with 'Unknown'")
    
    # 2. Fix holiday-related columns
    holiday_cols = ['type_y', 'locale', 'locale_name', 'description', 'transferred']
    for col in holiday_cols:
        if col in df.columns:
            if col == 'transferred':
                df[col] = df[col].fillna(False)
            else:
                df[col] = df[col].fillna('None')
            print(f"  ✓ {col}: filled appropriately")
    
    # 3. Fix numeric columns with appropriate strategies
    numeric_fixes = {
        'transactions': 0,  # No transactions = 0
        'dcoilwtico': df['dcoilwtico'].median() if 'dcoilwtico' in df.columns else 0,  # Oil price median
        'unit_sales': 0,  # No sales = 0
        'onpromotion': 0,  # Not on promotion = 0
    }
    
    for col, fill_value in numeric_fixes.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)
            print(f"  ✓ {col}: filled with {fill_value}")
    
    # 4. Fix any remaining numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(0)
            print(f"  ✓ {col}: filled with 0")
    
    # 5. Fix any remaining object columns
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna('Unknown')
            print(f"  ✓ {col}: filled with 'Unknown'")
    
    # After fixing
    print(f"\nAFTER fixing:")
    print(f"Shape: {df.shape}")
    print(f"NaN count: {df.isnull().sum().sum()}")
    
    if df.isnull().sum().sum() == 0:
        print("  ✓ All NaN values have been fixed!")
    else:
        print("  ⚠️ Some NaN values remain:")
        remaining_nans = df.isnull().sum()[df.isnull().sum() > 0]
        for col, count in remaining_nans.items():
            print(f"    {col}: {count} NaNs")
    
    print("=" * 50)
    return df

def engineer_features(df: pd.DataFrame):
    """
    Apply feature engineering to create lag features, rolling statistics, and other engineered features efficiently for large datasets.
    """
    print("Engineering features efficiently...")
    
    # Create target column from unit_sales
    df["target"] = df["unit_sales"].copy()
    # Handle negative sales (returns) by setting to 0
    df["target"] = np.maximum(df["target"], 0)
    
    # 1. Sort for group-wise operations
    df = df.sort_values(["store_nbr", "item_nbr", "date"])
    
    # 2. Efficient lag and rolling features
    print("Creating lag and rolling features per group...")
    df["lag_7"] = np.nan
    df["lag_28"] = np.nan
    df["rolling_mean_7"] = np.nan
    df["rolling_mean_28"] = np.nan
    df["rolling_std_7"] = np.nan
    
    for (store, item), group in df.groupby(["store_nbr", "item_nbr"]):
        idx = group.index
        sales = group["unit_sales"]
        
        # Only calculate features if we have enough data
        if len(sales) >= 7:
            df.loc[idx, "lag_7"] = sales.shift(7)
            df.loc[idx, "rolling_mean_7"] = sales.shift(1).rolling(7, min_periods=1).mean()
            df.loc[idx, "rolling_std_7"] = sales.shift(1).rolling(7, min_periods=1).std()
        
        if len(sales) >= 28:
            df.loc[idx, "lag_28"] = sales.shift(28)
            df.loc[idx, "rolling_mean_28"] = sales.shift(1).rolling(28, min_periods=1).mean()
    
    # 3. Calendar features
    print("Creating calendar features...")
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["month"] = df["date"].dt.month
    
    # 4. Promotion flag
    df["onpromotion"] = df["onpromotion"].fillna(0).astype(int)
    
    # 5. Holiday indicator and related features
    # Check if holiday type column exists (could be 'type_x', 'type_y', or 'holiday_type')
    holiday_type_col = None
    if 'type_y' in df.columns:  # This is the holiday type from holidays dataset
        holiday_type_col = 'type_y'
    elif 'type_x' in df.columns:  # This is the store type from stores dataset
        holiday_type_col = 'type_x'
    elif 'type' in df.columns:
        holiday_type_col = 'type'
    elif 'holiday_type' in df.columns:
        holiday_type_col = 'holiday_type'
    
    if holiday_type_col:
        df["is_holiday"] = (~df[holiday_type_col].isna()).astype(int)
        df["holiday_type"] = df[holiday_type_col].fillna("None")
    else:
        # If no holiday type column, create default values
        df["is_holiday"] = 0
        df["holiday_type"] = "None"
    
    # Handle other holiday-related columns
    holiday_cols = ['locale', 'locale_name', 'description', 'transferred', 'type_y']
    for col in holiday_cols:
        if col in df.columns:
            if col == 'transferred':
                df[col] = df[col].fillna(False)
            else:
                df[col] = df[col].fillna('None')
            print(f"  ✓ {col}: filled appropriately")
    
    # 6. External signals (transactions, oil)
    print("Processing external signals...")
    for col, mask in [("transactions", "transactions_missing"), ("dcoilwtico", "oil_missing")]:
        df[mask] = df[col].isna().astype(int)
        df[col] = df[col].fillna(method="ffill").fillna(method="bfill").fillna(0)
    
    # 7. Missingness masks for real features and fill NaN values
    print("Creating missingness masks and filling NaN values...")
    real_features = ["lag_7", "lag_28", "rolling_mean_7", "rolling_mean_28", "rolling_std_7"]
    for col in real_features:
        mask_col = f"{col}_missing"
        df[mask_col] = df[col].isna().astype(int)
        # Fill NaN with 0 for lag features (no historical data)
        df[col] = df[col].fillna(0)
    
    print("Feature engineering completed!")
    return df

def debug_dataframe_info(df: pd.DataFrame):
    """
    Debug function to check DataFrame columns and missing values.
    """
    print("=== DataFrame Debug Info ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nMissing values per column:")
    missing_counts = df.isnull().sum().sort_values(ascending=False)
    print(missing_counts[missing_counts > 0])
    print(f"\nTotal missing values: {df.isnull().sum().sum()}")
    print("=" * 30)

def analyze_dataset_columns(df: pd.DataFrame):
    """
    Analyze the actual columns in the dataset and identify missing ones.
    """
    print("=== Dataset Column Analysis ===")
    print(f"Total columns: {len(df.columns)}")
    print(f"Columns: {list(df.columns)}")
    
    # Check which expected columns are missing
    expected_static_categoricals = ['store_nbr', 'item_nbr', 'type_x', 'cluster', 'family', 'class', 'perishable', 'city', 'state']
    expected_time_varying_known = ['time_idx', 'dayofweek', 'is_weekend', 'month', 'onpromotion', 'transactions', 'transactions_missing', 'dcoilwtico', 'oil_missing']
    expected_time_varying_unknown = ['target', 'unit_sales', 'lag_7', 'lag_28', 'rolling_mean_7', 'rolling_mean_28', 'rolling_std_7', 'lag_7_missing', 'lag_28_missing', 'rolling_mean_7_missing', 'rolling_mean_28_missing', 'rolling_std_7_missing']
    
    print(f"\nMissing static categoricals:")
    for col in expected_static_categoricals:
        if col not in df.columns:
            print(f"  ❌ {col}")
        else:
            print(f"  ✅ {col}")
    
    print(f"\nMissing time-varying known reals:")
    for col in expected_time_varying_known:
        if col not in df.columns:
            print(f"  ❌ {col}")
        else:
            print(f"  ✅ {col}")
    
    print(f"\nMissing time-varying unknown reals:")
    for col in expected_time_varying_unknown:
        if col not in df.columns:
            print(f"  ❌ {col}")
        else:
            print(f"  ✅ {col}")
    
    print(f"\nMissing values per column:")
    missing_counts = df.isnull().sum().sort_values(ascending=False)
    print(missing_counts[missing_counts > 0])
    print("=" * 50)

def create_tft_dataset(df: pd.DataFrame):
    """
    Create TFT dataset using the preprocessor.
    """
    # Define column categories based on the actual columns in the dataset
    static_categoricals = [
        'store_nbr', 'item_nbr', 'type_x', 'cluster', 
        'family', 'class', 'perishable',
        'city', 'state'
    ]
    
    static_reals = []  # No static real columns in our dataset
    
    time_varying_known_reals = [
        'time_idx', 'dayofweek', 'is_weekend', 'month',
        'onpromotion', 'transactions', 'transactions_missing',
        'dcoilwtico', 'oil_missing'
    ]
    
    time_varying_known_categoricals = [
        'is_holiday', 'holiday_type'
    ]
    
    time_varying_unknown_reals = [
        'target', 'unit_sales', 'lag_7', 'lag_28',
        'rolling_mean_7', 'rolling_mean_28', 'rolling_std_7',
        'lag_7_missing', 'lag_28_missing', 'rolling_mean_7_missing',
        'rolling_mean_28_missing', 'rolling_std_7_missing'
    ]
    
    # Initialize preprocessor
    preprocessor = TFTDataPreprocessor(
        static_categoricals=static_categoricals,
        static_reals=static_reals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_column='target',
        time_idx_column='time_idx',
        group_ids=['store_nbr', 'item_nbr']
    )
    
    # Preprocess data
    df_processed = preprocessor.load_and_preprocess_data(df)
    
    # Create TimeSeriesDataSet
    dataset = preprocessor.create_timeseries_dataset(
        df_processed,
        max_prediction_length=14,  # 2 weeks forecast
        max_encoder_length=90,     # 3 months of history
        allow_missing_timesteps=True,
        impute_mode='mask_value'
    )
    
    return dataset, df_processed

def main(subset_size=100_000, random_state=42):
    """
    Complete pipeline: load data, engineer features, and create TFT dataset.
    
    Args:
        subset_size: Number of rows to sample from train.csv (default: 100,000)
        random_state: Random seed for reproducible sampling
    
    Returns:
        tuple: (TimeSeriesDataSet, pd.DataFrame) - dataset and processed DataFrame
    """
    print("=== TFT Demand Forecasting Pipeline ===")
    print(f"Using subset size: {subset_size:,}")
    
    # Step 1: Load and prepare data
    print("\n1. Loading and preparing data...")
    df = load_and_prepare_data(subset_size=subset_size, random_state=random_state)
    
    # Step 2: Analyze and fix NaNs BEFORE feature engineering
    print("\n2. Analyzing and fixing NaNs...")
    df = analyze_and_fix_nans(df)
    
    # Step 3: Engineer features
    print("\n3. Engineering features...")
    df = engineer_features(df)
    
    # Step 4: Final NaN check after feature engineering
    print("\n4. Final NaN check after feature engineering...")
    final_nan_count = df.isnull().sum().sum()
    if final_nan_count > 0:
        print(f"⚠️ {final_nan_count} NaN values remain after feature engineering")
        print("Columns with remaining NaNs:")
        remaining_nans = df.isnull().sum()[df.isnull().sum() > 0]
        for col, count in remaining_nans.items():
            print(f"  {col}: {count} NaNs")
        # Final cleanup
        df = df.fillna(0)  # Fill any remaining NaNs with 0
        print("✓ Final cleanup: filled remaining NaNs with 0")
    else:
        print("✓ No NaN values remaining after feature engineering")
    
    # Step 5: Analyze columns and missing values
    print("\n5. Analyzing dataset columns...")
    analyze_dataset_columns(df)
    
    # Step 6: Debug info before preprocessing
    print("\n6. Debug info before preprocessing...")
    debug_dataframe_info(df)
    
    # Step 7: Create TFT dataset
    print("\n7. Creating TFT dataset...")
    dataset, df_processed = create_tft_dataset(df)
    
    print(f"\n=== Pipeline Complete ===")
    print(f"Final dataset shape: {df_processed.shape}")
    print(f"TimeSeriesDataSet samples: {len(dataset)}")
    print(f"Max encoder length: {dataset.max_encoder_length}")
    print(f"Max prediction length: {dataset.max_prediction_length}")
    
    return dataset, df_processed

if __name__ == "__main__":
    dataset, df_processed = main() 