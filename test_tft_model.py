import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
import warnings
warnings.filterwarnings('ignore')

from example_usage import load_and_prepare_data, analyze_and_fix_nans, engineer_features

def test_tft_creation():
    """
    Test that we can create a TFT model with our preprocessed data.
    """
    print("=== Testing TFT Model Creation ===")
    
    # Load and prepare data
    print("1. Loading and preparing data...")
    df = load_and_prepare_data(subset_size=10_000, random_state=42)  # Smaller subset for testing
    
    print("2. Fixing NaN values...")
    df = analyze_and_fix_nans(df)
    
    print("3. Engineering features...")
    df = engineer_features(df)
    
    # Final NaN check
    final_nan_count = df.isnull().sum().sum()
    if final_nan_count > 0:
        print(f"⚠️ {final_nan_count} NaN values remain, filling with 0")
        df = df.fillna(0)
    else:
        print("✓ No NaN values remaining")
    
    # Convert categorical columns to string type for TFT
    categorical_columns = ["store_nbr", "item_nbr", "family", "class", "perishable", 
                          "city", "state", "type_x", "cluster"]
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    print(f"Final dataset shape: {df.shape}")
    
    # Create TimeSeriesDataSet
    print("4. Creating TimeSeriesDataSet...")
    
    # Define column categories (simplified for testing)
    static_categoricals = ['store_nbr', 'item_nbr']
    time_varying_known_reals = ['time_idx', 'dayofweek', 'is_weekend', 'month', 'onpromotion']
    time_varying_unknown_reals = ['target', 'lag_7', 'rolling_mean_7']
    
    # Filter columns that exist
    static_categoricals = [col for col in static_categoricals if col in df.columns]
    time_varying_known_reals = [col for col in time_varying_known_reals if col in df.columns]
    time_varying_unknown_reals = [col for col in time_varying_unknown_reals if col in df.columns]
    
    print(f"Static categoricals: {static_categoricals}")
    print(f"Time-varying known reals: {time_varying_known_reals}")
    print(f"Time-varying unknown reals: {time_varying_unknown_reals}")
    
    # Create training cutoff
    max_encoder_length = 10  # Smaller for testing
    max_prediction_length = 3  # Smaller for testing
    training_cutoff = df['time_idx'].max() - max_prediction_length
    
    training_data = df[lambda x: x.time_idx <= training_cutoff]
    validation_data = df[lambda x: x.time_idx > training_cutoff]
    
    print(f"Training data shape: {training_data.shape}")
    print(f"Validation data shape: {validation_data.shape}")
    
    # Create dataset
    training_dataset = TimeSeriesDataSet(
        data=training_data,
        time_idx="time_idx",
        target="target",
        group_ids=["store_nbr", "item_nbr"],
        min_encoder_length=5,  # Smaller for testing
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
    
    print(f"Training dataset samples: {len(training_dataset)}")
    print(f"Validation dataset samples: {len(validation_dataset)}")
    
    # Create model
    print("5. Creating TFT model...")
    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.001,
        hidden_size=16,  # Small for testing
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
        use_learning_rate_finder=False,
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dataloaders
    print("6. Creating dataloaders...")
    train_dataloader = training_dataset.to_dataloader(
        train=True, batch_size=16, num_workers=0
    )
    
    val_dataloader = validation_dataset.to_dataloader(
        train=False, batch_size=16, num_workers=0
    )
    
    # Test a single training step
    print("7. Testing single training step...")
    try:
        # Get a batch
        batch = next(iter(train_dataloader))
        
        # Forward pass
        output = model(batch)
        
        print(f"✓ Model forward pass successful!")
        print(f"Output shape: {output[0].shape}")
        
        # Test prediction
        print("8. Testing prediction...")
        predictions = model.predict(val_dataloader, return_x=False)
        print(f"✓ Predictions successful! Shape: {predictions.shape}")
        
        print("=== All Tests Passed! ===")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return False

def test_simple_training():
    """
    Test a simple training run with minimal epochs.
    """
    print("\n=== Testing Simple Training ===")
    
    # Load and prepare data
    print("1. Loading and preparing data...")
    df = load_and_prepare_data(subset_size=5_000, random_state=42)  # Very small subset
    
    print("2. Fixing NaN values...")
    df = analyze_and_fix_nans(df)
    
    print("3. Engineering features...")
    df = engineer_features(df)
    
    # Final NaN check
    final_nan_count = df.isnull().sum().sum()
    if final_nan_count > 0:
        df = df.fillna(0)
    
    # Convert categorical columns to string type for TFT
    categorical_columns = ["store_nbr", "item_nbr", "family", "class", "perishable", 
                          "city", "state", "type_x", "cluster"]
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # Create TimeSeriesDataSet
    print("4. Creating TimeSeriesDataSet...")
    
    static_categoricals = ['store_nbr', 'item_nbr']
    time_varying_known_reals = ['time_idx', 'dayofweek', 'is_weekend', 'month', 'onpromotion']
    time_varying_unknown_reals = ['target', 'lag_7', 'rolling_mean_7']
    
    # Filter columns that exist
    static_categoricals = [col for col in static_categoricals if col in df.columns]
    time_varying_known_reals = [col for col in time_varying_known_reals if col in df.columns]
    time_varying_unknown_reals = [col for col in time_varying_unknown_reals if col in df.columns]
    
    # Create training cutoff
    max_encoder_length = 20
    max_prediction_length = 5
    training_cutoff = df['time_idx'].max() - max_prediction_length
    
    training_data = df[lambda x: x.time_idx <= training_cutoff]
    validation_data = df[lambda x: x.time_idx > training_cutoff]
    
    # Create dataset
    training_dataset = TimeSeriesDataSet(
        data=training_data,
        time_idx="time_idx",
        target="target",
        group_ids=["store_nbr", "item_nbr"],
        min_encoder_length=3,  # Smaller for testing
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
    
    # Create model
    print("5. Creating TFT model...")
    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.001,
        hidden_size=8,  # Very small for testing
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=4,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
        use_learning_rate_finder=False,
    )
    
    # Create dataloaders
    train_dataloader = training_dataset.to_dataloader(
        train=True, batch_size=8, num_workers=0
    )
    
    val_dataloader = validation_dataset.to_dataloader(
        train=False, batch_size=8, num_workers=0
    )
    
    # Create trainer
    print("6. Creating trainer...")
    trainer = pl.Trainer(
        max_epochs=2,  # Just 2 epochs for testing
        accelerator="auto",
        devices=1,
        enable_model_summary=False,
        gradient_clip_val=0.1,
        enable_progress_bar=True,
    )
    
    # Train the model
    print("7. Starting training...")
    try:
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        
        print("✓ Training completed successfully!")
        
        # Test prediction
        print("8. Testing prediction...")
        predictions = model.predict(val_dataloader, return_x=False)
        print(f"✓ Predictions successful! Shape: {predictions.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== TFT Model Testing ===")
    
    # Test 1: Model creation
    test1_passed = test_tft_creation()
    
    # Test 2: Simple training
    test2_passed = test_simple_training()
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! TFT model is ready for full training.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.") 