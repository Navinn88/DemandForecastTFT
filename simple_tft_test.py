import pandas as pd
import numpy as np
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
import warnings
warnings.filterwarnings('ignore')

from example_usage import load_and_prepare_data, analyze_and_fix_nans, engineer_features

def analyze_data_distribution():
    """
    Analyze the data distribution to understand the time series structure.
    """
    print("=== Data Distribution Analysis ===")
    
    # Load and prepare data
    print("1. Loading and preparing data...")
    df = load_and_prepare_data(subset_size=5_000, random_state=42)
    
    print("2. Fixing NaN values...")
    df = analyze_and_fix_nans(df)
    
    print("3. Engineering features...")
    df = engineer_features(df)
    
    # Convert categorical columns
    categorical_columns = ["store_nbr", "item_nbr", "family", "class", "perishable", 
                          "city", "state", "type_x", "cluster"]
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    print(f"Final dataset shape: {df.shape}")
    
    # Analyze time series structure
    print("\n=== Time Series Analysis ===")
    
    # Check time_idx range
    print(f"Time index range: {df['time_idx'].min()} to {df['time_idx'].max()}")
    print(f"Total time steps: {df['time_idx'].max() - df['time_idx'].min() + 1}")
    
    # Check group distribution
    groups = df.groupby(['store_nbr', 'item_nbr']).size()
    print(f"Number of unique groups: {len(groups)}")
    print(f"Average time steps per group: {groups.mean():.2f}")
    print(f"Min time steps per group: {groups.min()}")
    print(f"Max time steps per group: {groups.max()}")
    
    # Check groups with sufficient data
    sufficient_groups = groups[groups >= 10]  # At least 10 time steps
    print(f"Groups with >= 10 time steps: {len(sufficient_groups)}")
    
    if len(sufficient_groups) > 0:
        print(f"Average time steps for sufficient groups: {sufficient_groups.mean():.2f}")
        print(f"Min time steps for sufficient groups: {sufficient_groups.min()}")
        print(f"Max time steps for sufficient groups: {sufficient_groups.max()}")
    
    return df, groups

def create_minimal_dataset(df, groups):
    """
    Create a minimal dataset with only groups that have sufficient data.
    """
    print("\n=== Creating Minimal Dataset ===")
    
    # Filter to groups with sufficient data
    sufficient_groups = groups[groups >= 15]  # At least 15 time steps
    print(f"Using {len(sufficient_groups)} groups with >= 15 time steps")
    
    if len(sufficient_groups) == 0:
        print("❌ No groups have sufficient data!")
        return None, None
    
    # Get the group names
    group_names = sufficient_groups.index.tolist()
    
    # Filter the dataframe
    df_filtered = df[df.set_index(['store_nbr', 'item_nbr']).index.isin(group_names)]
    print(f"Filtered dataset shape: {df_filtered.shape}")
    
    # Check time range again
    print(f"Filtered time index range: {df_filtered['time_idx'].min()} to {df_filtered['time_idx'].max()}")
    
    # Create training/validation split
    max_time = df_filtered['time_idx'].max()
    min_time = df_filtered['time_idx'].min()
    total_time_steps = max_time - min_time + 1
    
    print(f"Total time steps: {total_time_steps}")
    
    # Use very conservative parameters
    max_encoder_length = 5
    max_prediction_length = 2
    training_cutoff = max_time - max_prediction_length
    
    print(f"Training cutoff: {training_cutoff}")
    print(f"Max encoder length: {max_encoder_length}")
    print(f"Max prediction length: {max_prediction_length}")
    
    # Split data
    training_data = df_filtered[df_filtered['time_idx'] <= training_cutoff]
    validation_data = df_filtered[df_filtered['time_idx'] > training_cutoff]
    
    print(f"Training data shape: {training_data.shape}")
    print(f"Validation data shape: {validation_data.shape}")
    
    # Check if we have enough data
    if len(training_data) == 0 or len(validation_data) == 0:
        print("❌ Not enough data for training/validation split!")
        return None, None
    
    # Create dataset
    print("Creating TimeSeriesDataSet...")
    
    # Use minimal features
    static_categoricals = ['store_nbr', 'item_nbr']
    time_varying_known_reals = ['time_idx', 'dayofweek', 'is_weekend', 'month']
    time_varying_unknown_reals = ['target']
    
    # Filter columns that exist
    static_categoricals = [col for col in static_categoricals if col in training_data.columns]
    time_varying_known_reals = [col for col in time_varying_known_reals if col in training_data.columns]
    time_varying_unknown_reals = [col for col in time_varying_unknown_reals if col in training_data.columns]
    
    print(f"Static categoricals: {static_categoricals}")
    print(f"Time-varying known reals: {time_varying_known_reals}")
    print(f"Time-varying unknown reals: {time_varying_unknown_reals}")
    
    try:
        training_dataset = TimeSeriesDataSet(
            data=training_data,
            time_idx="time_idx",
            target="target",
            group_ids=["store_nbr", "item_nbr"],
            min_encoder_length=3,
            max_encoder_length=max_encoder_length,
            min_prediction_idx=training_cutoff + 1,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=static_categoricals,
            static_reals=[],
            time_varying_known_reals=time_varying_known_reals,
            time_varying_known_categoricals=[],
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=None,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        
        validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, validation_data, predict=True)
        
        print(f"✓ Training dataset samples: {len(training_dataset)}")
        print(f"✓ Validation dataset samples: {len(validation_dataset)}")
        
        return training_dataset, validation_dataset
        
    except Exception as e:
        print(f"❌ Error creating dataset: {str(e)}")
        return None, None

def test_minimal_model(training_dataset, validation_dataset):
    """
    Test creating and training a minimal TFT model.
    """
    if training_dataset is None or validation_dataset is None:
        print("❌ Cannot test model - no valid datasets!")
        return False
    
    print("\n=== Testing Minimal TFT Model ===")
    
    try:
        # Create model
        print("1. Creating TFT model...")
        model = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=0.001,
            hidden_size=4,  # Very small
            attention_head_size=2,
            dropout=0.1,
            hidden_continuous_size=2,
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
            use_learning_rate_finder=False,
        )
        
        print(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Create dataloaders
        print("2. Creating dataloaders...")
        train_dataloader = training_dataset.to_dataloader(
            train=True, batch_size=4, num_workers=0
        )
        
        val_dataloader = validation_dataset.to_dataloader(
            train=False, batch_size=4, num_workers=0
        )
        
        # Test forward pass
        print("3. Testing forward pass...")
        batch = next(iter(train_dataloader))
        output = model(batch)
        print(f"✓ Forward pass successful! Output shape: {output[0].shape}")
        
        # Test prediction
        print("4. Testing prediction...")
        predictions = model.predict(val_dataloader, return_x=False)
        print(f"✓ Prediction successful! Shape: {predictions.shape}")
        
        print("🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return False

def main():
    """
    Main function to run the minimal TFT test.
    """
    print("=== Minimal TFT Test ===")
    
    # Step 1: Analyze data distribution
    df, groups = analyze_data_distribution()
    
    # Step 2: Create minimal dataset
    training_dataset, validation_dataset = create_minimal_dataset(df, groups)
    
    # Step 3: Test minimal model
    success = test_minimal_model(training_dataset, validation_dataset)
    
    if success:
        print("\n🎉 TFT model is working! Ready for full training.")
    else:
        print("\n❌ TFT model test failed.")
    
    return success

if __name__ == "__main__":
    main() 