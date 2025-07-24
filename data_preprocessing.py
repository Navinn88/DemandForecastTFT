import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from pytorch_forecasting import TimeSeriesDataSet
import warnings
warnings.filterwarnings('ignore')

class TFTDataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for Temporal Fusion Transformer (TFT) models.
    Handles missing value imputation, feature engineering, and TimeSeriesDataSet creation.
    """
    
    def __init__(self, 
                 static_categoricals: List[str],
                 static_reals: List[str],
                 time_varying_known_reals: List[str],
                 time_varying_unknown_reals: List[str],
                 target_column: str = 'target',
                 time_idx_column: str = 'time_idx',
                 group_ids: List[str] = None):
        """
        Initialize the TFT data preprocessor.
        
        Args:
            static_categoricals: List of static categorical features
            static_reals: List of static real-valued features
            time_varying_known_reals: List of time-varying known real features
            time_varying_unknown_reals: List of time-varying unknown real features
            target_column: Name of the target variable
            time_idx_column: Name of the time index column
            group_ids: List of group identifier columns (e.g., ['store_id', 'item_id'])
        """
        self.static_categoricals = static_categoricals
        self.static_reals = static_reals
        self.time_varying_known_reals = time_varying_known_reals
        self.time_varying_unknown_reals = time_varying_unknown_reals
        self.target_column = target_column
        self.time_idx_column = time_idx_column
        self.group_ids = group_ids or ['store_nbr', 'item_nbr']
        
        # Store processed columns for TimeSeriesDataSet
        self.processed_static_categoricals = []
        self.processed_static_reals = []
        self.processed_time_varying_known_reals = []
        self.processed_time_varying_unknown_reals = []
        
    def load_and_preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main preprocessing pipeline that handles all missing value imputation.
        
        Args:
            df: Input DataFrame with raw data
            
        Returns:
            Preprocessed DataFrame ready for TimeSeriesDataSet
        """
        print("Starting TFT data preprocessing...")
        
        # Create a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Validate required columns exist
        self._validate_required_columns(df_processed)
        
        # Ensure proper data types
        df_processed = self._ensure_proper_data_types(df_processed)
        
        # 1. Handle static categoricals
        df_processed = self._handle_static_categoricals(df_processed)
        
        # 2. Handle static reals
        df_processed = self._handle_static_reals(df_processed)
        
        # 3. Handle time-varying known reals
        df_processed = self._handle_time_varying_known_reals(df_processed)
        
        # 4. Handle time-varying unknown reals
        df_processed = self._handle_time_varying_unknown_reals(df_processed)
        
        # 5. Final validation
        df_processed = self._final_validation(df_processed)
        
        print("Data preprocessing completed!")
        return df_processed
    
    def _validate_required_columns(self, df: pd.DataFrame):
        """Validate that required columns exist."""
        required_cols = [self.time_idx_column, self.target_column] + self.group_ids
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print(f"✓ All required columns present: {required_cols}")
    
    def _ensure_proper_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure proper data types for all columns."""
        print("Ensuring proper data types...")
        
        # Ensure time_idx is integer
        if self.time_idx_column in df.columns:
            df[self.time_idx_column] = df[self.time_idx_column].astype(int)
        
        # Ensure target is numeric
        if self.target_column in df.columns:
            df[self.target_column] = pd.to_numeric(df[self.target_column], errors='coerce')
        
        # Ensure group IDs are string (for categorical handling)
        for group_id in self.group_ids:
            if group_id in df.columns:
                df[group_id] = df[group_id].astype(str)
        
        return df
    
    def _handle_static_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in static categorical features."""
        print("Processing static categoricals...")
        
        for col in self.static_categoricals:
            if col in df.columns:
                # Convert to string first to handle mixed types
                df[col] = df[col].astype(str)
                
                # Fill NaNs with 'Unknown'
                df[col] = df[col].replace(['nan', 'None', 'null', ''], 'Unknown')
                df[col] = df[col].fillna('Unknown')
                
                self.processed_static_categoricals.append(col)
                print(f"  ✓ {col}: filled NaNs with 'Unknown'")
            else:
                print(f"  ⚠️ {col}: column not found in dataset")
        
        return df
    
    def _handle_static_reals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in static real features."""
        print("Processing static reals...")
        
        for col in self.static_reals:
            if col in df.columns:
                # Convert to numeric, handling errors
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Group by series_id and fill NaNs with group median
                # Handle case where all values in group are NaN
                df[col] = df.groupby(self.group_ids)[col].transform(
                    lambda x: x.fillna(x.median()) if not x.isna().all() else x.fillna(0)
                )
                
                # Add mask column for missing values
                mask_col = f"{col}_missing"
                df[mask_col] = df[col].isna().astype(int)
                
                # Final fill for any remaining NaNs
                df[col] = df[col].fillna(0)
                
                self.processed_static_reals.extend([col, mask_col])
                print(f"  ✓ {col}: filled NaNs with group median, added mask column")
            else:
                print(f"  ⚠️ {col}: column not found in dataset")
        
        return df
    
    def _handle_time_varying_known_reals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in time-varying known real features."""
        print("Processing time-varying known reals...")
        
        for col in self.time_varying_known_reals:
            if col in df.columns:
                # Convert to numeric, handling errors
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Forward-fill then backward-fill NaNs within groups
                df[col] = df.groupby(self.group_ids)[col].transform(
                    lambda x: x.fillna(method='ffill').fillna(method='bfill')
                )
                
                # Add mask column for missing values
                mask_col = f"{col}_missing"
                df[mask_col] = df[col].isna().astype(int)
                
                # Final fill for any remaining NaNs
                df[col] = df[col].fillna(0)
                
                self.processed_time_varying_known_reals.extend([col, mask_col])
                print(f"  ✓ {col}: forward/backward filled NaNs, added mask column")
            else:
                print(f"  ⚠️ {col}: column not found in dataset")
        
        return df
    
    def _handle_time_varying_unknown_reals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in time-varying unknown real features."""
        print("Processing time-varying unknown reals...")
        
        for col in self.time_varying_unknown_reals:
            if col in df.columns:
                # Convert to numeric, handling errors
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Forward-fill then backward-fill NaNs within groups
                df[col] = df.groupby(self.group_ids)[col].transform(
                    lambda x: x.fillna(method='ffill').fillna(method='bfill')
                )
                
                # Add mask column for missing values
                mask_col = f"{col}_missing"
                df[mask_col] = df[col].isna().astype(int)
                
                # Final fill for any remaining NaNs
                df[col] = df[col].fillna(0)
                
                self.processed_time_varying_unknown_reals.extend([col, mask_col])
                print(f"  ✓ {col}: forward/backward filled NaNs, added mask column")
            else:
                print(f"  ⚠️ {col}: column not found in dataset")
        
        return df
    
    def _final_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final validation and cleanup."""
        print("Performing final validation...")
        
        # Check for any remaining NaN values
        remaining_nans = df.isna().sum().sum()
        if remaining_nans > 0:
            print(f"  ⚠️ Warning: {remaining_nans} NaN values remaining")
            # Fill any remaining NaNs with 0 for numeric, 'Unknown' for object
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna('Unknown')
        else:
            print("  ✓ No NaN values remaining")
        
        # Ensure all mask columns are binary
        mask_cols = [col for col in df.columns if col.endswith('_missing')]
        for col in mask_cols:
            df[col] = df[col].astype(int)
        
        print("  ✓ Final validation completed")
        return df
    
    def create_timeseries_dataset(self, 
                                df: pd.DataFrame,
                                max_prediction_length: int = 14,
                                max_encoder_length: int = 90,
                                min_prediction_idx: Optional[int] = None,
                                min_prediction_length: int = 1,
                                min_encoder_length: int = 1,
                                allow_missing_timesteps: bool = True,
                                impute_mode: str = 'mask_value') -> TimeSeriesDataSet:
        """
        Create a TimeSeriesDataSet for TFT training.
        
        Args:
            df: Preprocessed DataFrame
            max_prediction_length: Maximum prediction horizon
            max_encoder_length: Maximum encoder length
            min_prediction_idx: Minimum prediction index
            min_prediction_length: Minimum prediction length
            min_encoder_length: Minimum encoder length
            allow_missing_timesteps: Whether to allow missing timesteps
            impute_mode: Imputation mode for missing values
            
        Returns:
            Configured TimeSeriesDataSet
        """
        print("Creating TimeSeriesDataSet...")
        
        # Ensure time_idx is sorted
        df = df.sort_values(self.group_ids + [self.time_idx_column])
        
        # Validate that all required columns exist
        all_required_cols = (
            self.processed_static_categoricals + 
            self.processed_static_reals + 
            self.processed_time_varying_known_reals + 
            self.processed_time_varying_unknown_reals
        )
        
        missing_cols = [col for col in all_required_cols if col not in df.columns]
        if missing_cols:
            print(f"  ⚠️ Warning: Missing columns in dataset: {missing_cols}")
        
        try:
            dataset = TimeSeriesDataSet(
                data=df,
                time_idx=self.time_idx_column,
                target=self.target_column,
                group_ids=self.group_ids,
                min_encoder_length=min_encoder_length,
                max_encoder_length=max_encoder_length,
                min_prediction_idx=min_prediction_idx,
                min_prediction_length=min_prediction_length,
                max_prediction_length=max_prediction_length,
                static_categoricals=self.processed_static_categoricals,
                static_reals=self.processed_static_reals,
                time_varying_known_reals=self.processed_time_varying_known_reals,
                time_varying_unknown_reals=self.processed_time_varying_unknown_reals,
                allow_missing_timesteps=allow_missing_timesteps,
                impute_mode=impute_mode,
                target_normalizer=None  # TFT handles normalization internally
            )
            
            print(f"  ✓ TimeSeriesDataSet created with {len(dataset)} samples")
            return dataset
            
        except Exception as e:
            print(f"  ❌ Error creating TimeSeriesDataSet: {str(e)}")
            raise

# Example usage function
def create_tft_dataset_example():
    """
    Example function showing how to use the TFTDataPreprocessor.
    """
    # Define column categories
    static_categoricals = ['store_id', 'item_id', 'store_type', 'item_category']
    static_reals = ['avg_price', 'store_size', 'item_weight']
    time_varying_known_reals = ['time_idx', 'promo', 'month', 'day_of_week', 'holiday']
    time_varying_unknown_reals = ['target', 'price', 'demand', 'sales']
    
    # Initialize preprocessor
    preprocessor = TFTDataPreprocessor(
        static_categoricals=static_categoricals,
        static_reals=static_reals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_column='target',
        time_idx_column='time_idx',
        group_ids=['store_id', 'item_id']
    )
    
    # Load your data (replace with your actual data loading)
    # df = pd.read_csv('your_data.csv')
    
    # Preprocess data
    # df_processed = preprocessor.load_and_preprocess_data(df)
    
    # Create TimeSeriesDataSet
    # dataset = preprocessor.create_timeseries_dataset(
    #     df_processed,
    #     max_prediction_length=14,
    #     max_encoder_length=90
    # )
    
    # return dataset

if __name__ == "__main__":
    # Example usage
    create_tft_dataset_example() 