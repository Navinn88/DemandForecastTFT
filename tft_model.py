import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import warnings
warnings.filterwarnings('ignore')

from example_usage import load_and_prepare_data, analyze_and_fix_nans, engineer_features

class TFTDemandForecaster:
    """
    Complete TFT-based demand forecasting pipeline.
    """
    
    def __init__(self, 
                 subset_size=25_000,
                 max_encoder_length=90,
                 max_prediction_length=14,
                 batch_size=64,
                 learning_rate=0.001,
                 hidden_size=64,
                 attention_head_size=4,
                 dropout=0.1,
                 hidden_continuous_size=32,
                 loss=QuantileLoss(),
                 log_interval=10,
                 reduce_on_plateau_patience=4):
        """
        Initialize the TFT forecaster.
        
        Args:
            subset_size: Number of samples to use for training
            max_encoder_length: Maximum length of encoder sequence
            max_prediction_length: Maximum length of prediction sequence
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            hidden_size: Hidden size for the model
            attention_head_size: Number of attention heads
            dropout: Dropout rate
            hidden_continuous_size: Hidden size for continuous variables
            loss: Loss function
            log_interval: Logging interval
            reduce_on_plateau_patience: Patience for learning rate reduction
        """
        self.subset_size = subset_size
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size
        self.loss = loss
        self.log_interval = log_interval
        self.reduce_on_plateau_patience = reduce_on_plateau_patience
        
        # Model and data attributes
        self.model = None
        self.trainer = None
        self.training_cutoff = None
        
    def prepare_data(self, random_state=42):
        """
        Prepare the dataset for TFT training.
        """
        print("=== Preparing Data for TFT ===")
        
        # Load and prepare data
        print("1. Loading and preparing data...")
        df = load_and_prepare_data(subset_size=self.subset_size, random_state=random_state)
        
        # Fix NaN values
        print("2. Fixing NaN values...")
        df = analyze_and_fix_nans(df)
        
        # Engineer features
        print("3. Engineering features...")
        df = engineer_features(df)
        
        # Final NaN check
        final_nan_count = df.isnull().sum().sum()
        if final_nan_count > 0:
            print(f"⚠️ {final_nan_count} NaN values remain, filling with 0")
            df = df.fillna(0)
        else:
            print("✓ No NaN values remaining")
        
        print(f"Final dataset shape: {df.shape}")
        return df
    
    def create_timeseries_dataset(self, df):
        """
        Create TimeSeriesDataSet for TFT training.
        """
        print("=== Creating TimeSeriesDataSet ===")
        
        # Define column categories
        static_categoricals = [
            'store_nbr', 'item_nbr', 'family', 'class', 'perishable',
            'city', 'state', 'type_x', 'cluster'
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
        
        # Filter columns that exist in the dataset
        static_categoricals = [col for col in static_categoricals if col in df.columns]
        time_varying_known_reals = [col for col in time_varying_known_reals if col in df.columns]
        time_varying_known_categoricals = [col for col in time_varying_known_categoricals if col in df.columns]
        time_varying_unknown_reals = [col for col in time_varying_unknown_reals if col in df.columns]
        
        print(f"Static categoricals: {static_categoricals}")
        print(f"Time-varying known reals: {time_varying_known_reals}")
        print(f"Time-varying known categoricals: {time_varying_known_categoricals}")
        print(f"Time-varying unknown reals: {time_varying_unknown_reals}")
        
        # Create training cutoff (use 80% of data for training)
        self.training_cutoff = df['time_idx'].max() - self.max_prediction_length
        
        # Create dataset
        training_cutoff = self.training_cutoff
        
        training_data = df[lambda x: x.time_idx <= training_cutoff]
        validation_data = df[lambda x: x.time_idx > training_cutoff]
        
        print(f"Training data shape: {training_data.shape}")
        print(f"Validation data shape: {validation_data.shape}")
        
        # Create datasets
        training_dataset = TimeSeriesDataSet(
            data=training_data,
            time_idx="time_idx",
            target="target",
            group_ids=["store_nbr", "item_nbr"],
            min_encoder_length=self.max_encoder_length // 2,
            max_encoder_length=self.max_encoder_length,
            min_prediction_idx=training_cutoff + 1,
            min_prediction_length=1,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=static_categoricals,
            static_reals=static_reals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_known_categoricals=time_varying_known_categoricals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=None,  # TFT handles normalization internally
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        
        validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, validation_data, predict=True)
        
        print(f"Training dataset samples: {len(training_dataset)}")
        print(f"Validation dataset samples: {len(validation_dataset)}")
        
        return training_dataset, validation_dataset
    
    def create_model(self, training_dataset):
        """
        Create the TFT model.
        """
        print("=== Creating TFT Model ===")
        
        # Create dataloaders
        train_dataloader = training_dataset.to_dataloader(
            train=True, batch_size=self.batch_size, num_workers=0
        )
        
        # Create model
        self.model = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
            loss=self.loss,
            log_interval=self.log_interval,
            reduce_on_plateau_patience=self.reduce_on_plateau_patience,
            use_learning_rate_finder=False,
        )
        
        print(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
        return train_dataloader
    
    def train_model(self, train_dataloader, validation_dataset, max_epochs=30):
        """
        Train the TFT model.
        """
        print("=== Training TFT Model ===")
        
        # Create validation dataloader
        val_dataloader = validation_dataset.to_dataloader(
            train=False, batch_size=self.batch_size, num_workers=0
        )
        
        # Create callbacks
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=10,
            verbose=False,
            mode="min"
        )
        
        lr_logger = LearningRateMonitor()
        
        # Create trainer
        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices=1,
            enable_model_summary=True,
            gradient_clip_val=0.1,
            callbacks=[early_stop_callback, lr_logger],
            enable_progress_bar=True,
        )
        
        # Train the model
        print("Starting training...")
        self.trainer.fit(
            self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        
        print("Training completed!")
        return self.model
    
    def evaluate_model(self, validation_dataset):
        """
        Evaluate the trained model.
        """
        print("=== Evaluating Model ===")
        
        # Create validation dataloader
        val_dataloader = validation_dataset.to_dataloader(
            train=False, batch_size=self.batch_size, num_workers=0
        )
        
        # Evaluate
        predictions = self.model.predict(val_dataloader, return_x=False)
        actuals = torch.cat([y[0] for x, y in iter(val_dataloader)], dim=0)
        
        # Calculate metrics
        mae = torch.mean(torch.abs(predictions - actuals))
        mse = torch.mean((predictions - actuals) ** 2)
        rmse = torch.sqrt(mse)
        
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        
        return {
            'mae': mae.item(),
            'mse': mse.item(),
            'rmse': rmse.item(),
            'predictions': predictions,
            'actuals': actuals
        }
    
    def predict_future(self, df, prediction_length=14):
        """
        Make future predictions.
        """
        print("=== Making Future Predictions ===")
        
        # Get the latest data for each group
        latest_data = df.groupby(['store_nbr', 'item_nbr']).tail(self.max_encoder_length)
        
        # Create prediction dataset
        predict_dataset = TimeSeriesDataSet.from_dataset(
            self.model.training_dataset, latest_data, predict=True
        )
        
        # Create prediction dataloader
        predict_dataloader = predict_dataset.to_dataloader(
            train=False, batch_size=self.batch_size, num_workers=0
        )
        
        # Make predictions
        predictions = self.model.predict(predict_dataloader, return_x=False)
        
        print(f"Predictions shape: {predictions.shape}")
        return predictions
    
    def run_complete_pipeline(self, random_state=42, max_epochs=30):
        """
        Run the complete TFT forecasting pipeline.
        """
        print("=== TFT Demand Forecasting Pipeline ===")
        
        # Step 1: Prepare data
        df = self.prepare_data(random_state=random_state)
        
        # Step 2: Create datasets
        training_dataset, validation_dataset = self.create_timeseries_dataset(df)
        
        # Step 3: Create model and dataloader
        train_dataloader = self.create_model(training_dataset)
        
        # Step 4: Train model
        self.train_model(train_dataloader, validation_dataset, max_epochs=max_epochs)
        
        # Step 5: Evaluate model
        evaluation_results = self.evaluate_model(validation_dataset)
        
        # Step 6: Make future predictions
        future_predictions = self.predict_future(df)
        
        print("=== Pipeline Complete ===")
        return {
            'model': self.model,
            'trainer': self.trainer,
            'evaluation_results': evaluation_results,
            'future_predictions': future_predictions,
            'training_dataset': training_dataset,
            'validation_dataset': validation_dataset
        }

def main():
    """
    Main function to run the TFT forecasting pipeline.
    """
    # Initialize forecaster
    forecaster = TFTDemandForecaster(
        subset_size=25_000,  # Use smaller subset for faster training
        max_encoder_length=90,
        max_prediction_length=14,
        batch_size=32,  # Smaller batch size for memory efficiency
        learning_rate=0.001,
        hidden_size=32,  # Smaller model for faster training
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=16,
        max_epochs=20  # Fewer epochs for faster training
    )
    
    # Run complete pipeline
    results = forecaster.run_complete_pipeline(random_state=42, max_epochs=20)
    
    print("Pipeline completed successfully!")
    return results

if __name__ == "__main__":
    results = main() 