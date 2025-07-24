import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pytorch_forecasting import TimeSeriesDataSet
import warnings
warnings.filterwarnings('ignore')

class TFTDataPreprocessorAnalysis:
    """
    Comprehensive analysis and improved preprocessing pipeline for TFT models.
    Identifies and fixes missing value handling issues and type errors.
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
        Initialize the TFT data preprocessor with analysis capabilities.
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
        
        # Analysis results
        self.missing_value_report = {}
        self.type_error_report = {}
        self.data_quality_report = {}
    
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive data quality analysis.
        """
        print("=== Data Quality Analysis ===")
        
        analysis = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': {},
            'data_types': {},
            'unique_values': {},
            'potential_issues': []
        }
        
        # Analyze each column
        for col in df.columns:
            # Missing values
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            analysis['missing_values'][col] = {
                'count': missing_count,
                'percentage': missing_pct
            }
            
            # Data types
            analysis['data_types'][col] = str(df[col].dtype)
            
            # Unique values (for categorical analysis)
            if df[col].dtype == 'object' or df[col].nunique() < 50:
                analysis['unique_values'][col] = df[col].nunique()
            
            # Identify potential issues
            if missing_pct > 50:
                analysis['potential_issues'].append(f"High missing values in {col}: {missing_pct:.1f}%")
            
            if df[col].dtype == 'object' and df[col].nunique() > 1000:
                analysis['potential_issues'].append(f"High cardinality categorical in {col}: {df[col].nunique()} unique values")
        
        self.data_quality_report = analysis
        return analysis
    
    def load_and_preprocess_data_improved(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Improved preprocessing pipeline with better error handling and validation.
        """
        print("Starting improved TFT data preprocessing...")
        
        # Create a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Validate required columns exist
        self._validate_required_columns(df_processed)
        
        # Ensure proper data types
        df_processed = self._ensure_proper_data_types(df_processed)
        
        # 1. Handle static categoricals with improved logic
        df_processed = self._handle_static_categoricals_improved(df_processed)
        
        # 2. Handle static reals with improved logic
        df_processed = self._handle_static_reals_improved(df_processed)
        
        # 3. Handle time-varying known reals with improved logic
        df_processed = self._handle_time_varying_known_reals_improved(df_processed)
        
        # 4. Handle time-varying unknown reals with improved logic
        df_processed = self._handle_time_varying_unknown_reals_improved(df_processed)
        
        # 5. Final validation
        df_processed = self._final_validation(df_processed)
        
        print("Improved data preprocessing completed!")
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
    
    def _handle_static_categoricals_improved(self, df: pd.DataFrame) -> pd.DataFrame:
        """Improved handling of static categorical features."""
        print("Processing static categoricals (improved)...")
        
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
    
    def _handle_static_reals_improved(self, df: pd.DataFrame) -> pd.DataFrame:
        """Improved handling of static real features."""
        print("Processing static reals (improved)...")
        
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
    
    def _handle_time_varying_known_reals_improved(self, df: pd.DataFrame) -> pd.DataFrame:
        """Improved handling of time-varying known real features."""
        print("Processing time-varying known reals (improved)...")
        
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
    
    def _handle_time_varying_unknown_reals_improved(self, df: pd.DataFrame) -> pd.DataFrame:
        """Improved handling of time-varying unknown real features."""
        print("Processing time-varying unknown reals (improved)...")
        
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
    
    def create_timeseries_dataset_improved(self, 
                                         df: pd.DataFrame,
                                         max_prediction_length: int = 14,
                                         max_encoder_length: int = 90,
                                         min_prediction_idx: Optional[int] = None,
                                         min_prediction_length: int = 1,
                                         min_encoder_length: int = 1,
                                         allow_missing_timesteps: bool = True,
                                         impute_mode: str = 'mask_value') -> TimeSeriesDataSet:
        """
        Create a TimeSeriesDataSet with improved error handling.
        """
        print("Creating improved TimeSeriesDataSet...")
        
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
    
    def generate_analysis_report(self) -> Dict:
        """Generate comprehensive analysis report."""
        report = {
            'data_quality': self.data_quality_report,
            'missing_value_handling': self.missing_value_report,
            'type_errors': self.type_error_report,
            'recommendations': []
        }
        
        # Add recommendations based on analysis
        if self.data_quality_report:
            for issue in self.data_quality_report.get('potential_issues', []):
                report['recommendations'].append(f"Address: {issue}")
        
        return report

# Example usage with analysis
def analyze_and_preprocess_example():
    """
    Example showing how to use the improved preprocessor with analysis.
    """
    # Define column categories
    static_categoricals = ['store_nbr', 'item_nbr', 'store_type', 'item_family']
    static_reals = ['store_size', 'item_weight']
    time_varying_known_reals = ['time_idx', 'onpromotion', 'transactions']
    time_varying_unknown_reals = ['target', 'unit_sales']
    
    # Initialize improved preprocessor
    preprocessor = TFTDataPreprocessorAnalysis(
        static_categoricals=static_categoricals,
        static_reals=static_reals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_column='target',
        time_idx_column='time_idx',
        group_ids=['store_nbr', 'item_nbr']
    )
    
    # Load your data (replace with your actual data loading)
    # df = pd.read_csv('your_data.csv')
    
    # Analyze data quality
    # analysis = preprocessor.analyze_data_quality(df)
    
    # Preprocess data with improved handling
    # df_processed = preprocessor.load_and_preprocess_data_improved(df)
    
    # Create TimeSeriesDataSet
    # dataset = preprocessor.create_timeseries_dataset_improved(df_processed)
    
    # Generate analysis report
    # report = preprocessor.generate_analysis_report()
    
    # return dataset, df_processed, report

if __name__ == "__main__":
    analyze_and_preprocess_example() 