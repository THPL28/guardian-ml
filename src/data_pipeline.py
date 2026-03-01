"""
Guardian-ML: Data Pipeline Module
Handles data generation, validation, and preparation for model training.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Tuple, Dict, Any
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDataGenerator:
    """
    Generates synthetic fraud detection dataset at scale.
    
    Simulates realistic transaction patterns:
    - 0.3-0.5% fraud rate (realistic for payments)
    - Feature distributions based on real-world patterns
    - Temporal patterns (time-of-day, day-of-week)
    - User and merchant patterns
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize data generator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.n_transactions = self.config['data']['n_transactions']
        self.fraud_rate = self.config['data']['fraud_rate']
        self.random_seed = self.config['training']['random_seed']
        np.random.seed(self.random_seed)
        
        logger.info(f"Initialized FraudDataGenerator: {self.n_transactions:,} transactions, "
                   f"{self.fraud_rate:.2%} fraud rate")
    
    def generate_transactions(self) -> pd.DataFrame:
        """
        Generate synthetic transaction dataset.
        
        Returns:
            DataFrame with transaction features
        """
        logger.info("Generating transaction data...")
        
        n = self.n_transactions
        n_fraud = int(n * self.fraud_rate)
        
        # Create transaction IDs and timestamps
        transaction_ids = np.arange(n)
        
        # Temporal features
        start_date = datetime(2024, 1, 1)
        timestamps = [start_date + timedelta(seconds=int(i * 86400 / n)) 
                     for i in range(n)]
        dates = pd.to_datetime(timestamps)
        
        # User demographics
        user_ids = np.random.randint(10000, 50000, n)
        user_ages = np.random.normal(35, 15, n).clip(18, 80).astype(int)
        user_countries = np.random.choice(['US', 'UK', 'DE', 'FR', 'BR', 'IN', 'CN'], n)
        
        # Merchant features
        merchant_ids = np.random.randint(1000, 5000, n)
        merchant_categories = np.random.choice(
            ['grocery', 'retail', 'entertainment', 'travel', 'dining', 'utilities'], n
        )
        
        # Transaction features
        amounts = np.random.lognormal(3.5, 1.5, n)  # Log-normal distribution
        amounts = np.clip(amounts, 0.5, 10000)  # Realistic range
        
        # Device features
        device_types = np.random.choice(['mobile', 'desktop', 'tablet'], n, p=[0.5, 0.35, 0.15])
        
        # Create base dataframe
        df = pd.DataFrame({
            'transaction_id': transaction_ids,
            'timestamp': dates,
            'user_id': user_ids,
            'user_age': user_ages,
            'user_country': user_countries,
            'merchant_id': merchant_ids,
            'merchant_category': merchant_categories,
            'amount': amounts,
            'device_type': device_types,
        })
        
        # Generate fraud label
        fraud_mask = np.zeros(n, dtype=bool)
        fraud_indices = np.random.choice(n, n_fraud, replace=False)
        fraud_mask[fraud_indices] = True
        df['is_fraud'] = fraud_mask.astype(int)
        
        # Add correlations between features and fraud for realism
        # Frauds tend to have higher amounts, unusual devices, etc.
        df.loc[df['is_fraud'] == 1, 'amount'] *= np.random.uniform(1.5, 3.0, n_fraud)
        df.loc[df['is_fraud'] == 1, 'user_age'] -= np.random.uniform(5, 15, n_fraud)
        
        logger.info(f"Generated {n:,} transactions with {n_fraud:,} frauds ({self.fraud_rate:.2%})")
        
        return df
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features for modeling.
        
        Args:
            df: Transaction dataframe
            
        Returns:
            DataFrame with derived features
        """
        logger.info("Adding derived features...")
        
        # Temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)
        
        # Aggregated features (computed at this point in time)
        df['user_transaction_count_30d'] = df.groupby('user_id')['transaction_id'].transform('count')
        df['user_fraud_rate_30d'] = df.groupby('user_id')['is_fraud'].transform('mean')
        
        df['merchant_transaction_count_30d'] = df.groupby('merchant_id')['transaction_id'].transform('count')
        df['merchant_fraud_rate_30d'] = df.groupby('merchant_id')['is_fraud'].transform('mean')
        
        # Interaction features
        df['user_merchant_interaction_count'] = df.groupby(['user_id', 'merchant_id'])['transaction_id'].transform('count')
        
        # Amount features
        df['amount_log'] = np.log1p(df['amount'])
        df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate generated data quality.
        
        Args:
            df: Transaction dataframe
            
        Returns:
            True if validation passes
        """
        logger.info("Validating data...")
        
        checks = {
            'no_missing_values': df.isnull().sum().sum() == 0,
            'fraud_label_binary': df['is_fraud'].isin([0, 1]).all(),
            'fraud_rate_realistic': 0.001 < df['is_fraud'].mean() < 0.01,
            'all_amounts_positive': (df['amount'] > 0).all(),
            'valid_timestamps': (df['timestamp'] >= pd.Timestamp('2024-01-01')).all(),
        }
        
        for check_name, result in checks.items():
            logger.info(f"  {check_name}: {'✓' if result else '✗'}")
            if not result:
                return False
        
        return True


class DataSplitter:
    """
    Handles train/val/test splitting with temporal integrity.
    
    For time-series-like data, use temporal split to avoid look-ahead bias:
    - Training: Oldest data
    - Validation: Middle data
    - Test: Most recent data
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize splitter with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.train_size = self.config['data']['train_size']
        self.val_size = self.config['data']['val_size']
        self.test_size = self.config['data']['test_size']
        self.random_seed = self.config['training']['random_seed']
        
        assert abs(self.train_size + self.val_size + self.test_size - 1.0) < 1e-6
    
    def temporal_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally (no look-ahead bias).
        
        Args:
            df: Transaction dataframe (must have 'timestamp' column)
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Performing temporal split...")
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        n = len(df)
        train_idx = int(n * self.train_size)
        val_idx = int(n * (self.train_size + self.val_size))
        
        train_df = df.iloc[:train_idx]
        val_df = df.iloc[train_idx:val_idx]
        test_df = df.iloc[val_idx:]
        
        logger.info(f"  Train: {len(train_df):,} samples ({len(train_df)/n:.1%})")
        logger.info(f"  Val:   {len(val_df):,} samples ({len(val_df)/n:.1%})")
        logger.info(f"  Test:  {len(test_df):,} samples ({len(test_df)/n:.1%})")
        
        # Verify no data leakage
        assert train_df['timestamp'].max() < val_df['timestamp'].min(), "Temporal leakage in train/val"
        assert val_df['timestamp'].max() < test_df['timestamp'].min(), "Temporal leakage in val/test"
        
        return train_df, val_df, test_df
    
    def stratified_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data with stratification on fraud label (maintains fraud rate).
        
        Used for cross-validation within model training.
        
        Args:
            df: Transaction dataframe
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        from sklearn.model_selection import train_test_split
        
        logger.info("Performing stratified split...")
        
        # First split: train + validation vs test
        train_val, test = train_test_split(
            df,
            test_size=self.test_size,
            stratify=df['is_fraud'],
            random_state=self.random_seed
        )
        
        # Second split: train vs validation
        train, val = train_test_split(
            train_val,
            test_size=self.val_size / (self.train_size + self.val_size),
            stratify=train_val['is_fraud'],
            random_state=self.random_seed
        )
        
        logger.info(f"  Train: {len(train):,} samples ({train['is_fraud'].mean():.2%} fraud)")
        logger.info(f"  Val:   {len(val):,} samples ({val['is_fraud'].mean():.2%} fraud)")
        logger.info(f"  Test:  {len(test):,} samples ({test['is_fraud'].mean():.2%} fraud)")
        
        return train, val, test


def main():
    """
    Main pipeline: Generate → Validate → Split → Save
    """
    logger.info("=" * 80)
    logger.info("GUARDIAN-ML: DATA PIPELINE")
    logger.info("=" * 80)
    
    # Generate data
    generator = FraudDataGenerator("config/config.yaml")
    df = generator.generate_transactions()
    df = generator.add_derived_features(df)
    
    # Validate
    if not generator.validate_data(df):
        logger.error("Data validation failed!")
        return
    
    # Split
    splitter = DataSplitter("config/config.yaml")
    train_df, val_df, test_df = splitter.temporal_split(df)
    
    # Save
    logger.info("\nSaving data...")
    train_df.to_parquet("data/processed/train.parquet", index=False)
    val_df.to_parquet("data/processed/val.parquet", index=False)
    test_df.to_parquet("data/processed/test.parquet", index=False)
    
    logger.info("✓ Data pipeline complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
