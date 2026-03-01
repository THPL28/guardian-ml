"""
Guardian-ML: Feature Engineering Module

Implements advanced feature engineering techniques:
- Feature normalization and scaling
- Feature selection with statistical tests
- Feature interactions and polynomial features
- Feature store versioning and validation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import logging
from typing import Dict, List, Tuple, Any
import pickle
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FeatureEngineer:
    """
    Advanced feature engineering for fraud detection.
    
    Responsibilities:
    - Categorical encoding
    - Numerical scaling
    - Feature selection
    - Feature interactions
    - Feature store management
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize feature engineer."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.encoders: Dict[str, LabelEncoder] = {}
        self.scalers: Dict[str, Any] = {}
        self.selected_features: List[str] = []
        self.feature_importances: Dict[str, float] = {}
        
        logger.info("Initialized FeatureEngineer")
    
    # ========================================================================
    # CATEGORICAL ENCODING
    # ========================================================================
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                    fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features to numerical values.
        
        Strategy:
        - Label encoding for ordinal categories (country, device_type)
        - One-hot encoding for low-cardinality features
        
        Args:
            df: Input dataframe
            fit: If True, fit new encoders. If False, use existing ones.
            
        Returns:
            DataFrame with encoded categorical features
        """
        logger.info("Encoding categorical features...")
        
        df_encoded = df.copy()
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col in ['timestamp']:  # Skip temporal columns
                continue
            
            if fit:
                self.encoders[col] = LabelEncoder()
                df_encoded[col] = self.encoders[col].fit_transform(df[col])
            else:
                # Use existing encoder
                df_encoded[col] = self.encoders[col].transform(df[col])
        
        logger.info(f"  Encoded {len(categorical_cols)} categorical features")
        
        return df_encoded
    
    # ========================================================================
    # NUMERICAL SCALING
    # ========================================================================
    
    def scale_numerical_features(self, df: pd.DataFrame, 
                                fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features to zero mean and unit variance.
        
        Method: RobustScaler (resistant to outliers)
        Formula: x_scaled = (x - median) / IQR
        
        Args:
            df: Input dataframe
            fit: If True, fit new scaler. If False, use existing one.
            
        Returns:
            DataFrame with scaled numerical features
        """
        logger.info("Scaling numerical features...")
        
        df_scaled = df.copy()
        
        # Identify numerical columns (exclude IDs and labels)
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['transaction_id', 'user_id', 'merchant_id', 'is_fraud']
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        if fit:
            self.scalers['robust_scaler'] = RobustScaler()
            df_scaled[numerical_cols] = self.scalers['robust_scaler'].fit_transform(
                df[numerical_cols]
            )
        else:
            df_scaled[numerical_cols] = self.scalers['robust_scaler'].transform(
                df[numerical_cols]
            )
        
        logger.info(f"  Scaled {len(numerical_cols)} numerical features")
        logger.info(f"    Columns: {numerical_cols[:5]}..." if len(numerical_cols) > 5 
                   else f"    Columns: {numerical_cols}")
        
        return df_scaled
    
    # ========================================================================
    # FEATURE SELECTION
    # ========================================================================
    
    def select_features_statistical(self, df: pd.DataFrame, target: pd.Series,
                                   k: int = 30, method: str = 'f_classif') -> List[str]:
        """
        Select top-k features using statistical tests.
        
        Methods:
        - f_classif: ANOVA F-value (linear relationship)
        - mutual_info_classif: Mutual information (non-linear)
        
        Math:
        F = (between-group variance) / (within-group variance)
        Higher F → stronger feature-label relationship
        
        Args:
            df: Feature matrix
            target: Target variable
            k: Number of features to select
            method: Feature selection method
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting top-{k} features using {method}...")
        
        if method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=min(k, df.shape[1]))
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=min(k, df.shape[1]))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Extract features and target matrix
        X = df.select_dtypes(include=[np.number]).drop(
            columns=['transaction_id', 'user_id', 'merchant_id'], errors='ignore'
        )
        
        selector.fit(X, target)
        
        # Get selected features
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        # Store importances
        scores = selector.scores_
        self.feature_importances = dict(zip(X.columns, scores))
        
        # Sort by importance
        selected_features_sorted = sorted(
            selected_features, 
            key=lambda x: self.feature_importances[x], 
            reverse=True
        )
        
        self.selected_features = selected_features_sorted
        
        logger.info(f"  Selected {len(selected_features)} features")
        logger.info(f"  Top 5: {selected_features_sorted[:5]}")
        
        return selected_features_sorted
    
    # ========================================================================
    # FEATURE INTERACTIONS
    # ========================================================================
    
    def create_feature_interactions(self, df: pd.DataFrame, 
                                   feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create interaction features (feature1 * feature2).
        
        Example:
        - amount * is_night (large amounts at night → more suspicious)
        - user_fraud_rate * merchant_fraud_rate (trusted user × risky merchant)
        
        Args:
            df: Input dataframe
            feature_pairs: List of (feature1, feature2) tuples
            
        Returns:
            DataFrame with interaction features added
        """
        logger.info(f"Creating {len(feature_pairs)} feature interactions...")
        
        df_interact = df.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                interaction_name = f"{feat1}_x_{feat2}"
                df_interact[interaction_name] = df[feat1] * df[feat2]
        
        logger.info(f"  Created {len(feature_pairs)} interactions")
        
        return df_interact
    
    # ========================================================================
    # FEATURE STORE
    # ========================================================================
    
    def save_feature_store(self, df: pd.DataFrame, version: str = "1.0.0"):
        """
        Save processed features to versioned feature store.
        
        Structure:
        feature_store/
        ├── v1.0.0/
        │   ├── features.parquet
        │   └── metadata.json
        └── v2.0.0/
            └── ...
        """
        logger.info(f"Saving feature store v{version}...")
        
        feature_dir = f"data/feature_store/v{version}"
        df.to_parquet(f"{feature_dir}/features.parquet", index=False)
        
        # Save metadata
        metadata = {
            'version': version,
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'columns': df.columns.tolist(),
            'feature_importances': self.feature_importances,
            'selected_features': self.selected_features,
            'timestamp': pd.Timestamp.now().isoformat(),
        }
        
        import json
        with open(f"{feature_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save encoders and scalers
        with open(f"{feature_dir}/encoders.pkl", 'wb') as f:
            pickle.dump(self.encoders, f)
        
        with open(f"{feature_dir}/scalers.pkl", 'wb') as f:
            pickle.dump(self.scalers, f)
        
        logger.info(f"  ✓ Saved to {feature_dir}")
        
        return metadata
    
    def load_feature_store(self, version: str = "1.0.0"):
        """Load encoders and scalers from feature store."""
        logger.info(f"Loading feature store v{version}...")
        
        feature_dir = f"data/feature_store/v{version}"
        
        with open(f"{feature_dir}/encoders.pkl", 'rb') as f:
            self.encoders = pickle.load(f)
        
        with open(f"{feature_dir}/scalers.pkl", 'rb') as f:
            self.scalers = pickle.load(f)
        
        logger.info(f"  ✓ Loaded from {feature_dir}")


def main():
    """Example feature engineering pipeline."""
    logger.info("=" * 80)
    logger.info("GUARDIAN-ML: FEATURE ENGINEERING")
    logger.info("=" * 80)
    
    # Load data
    logger.info("\nLoading training data...")
    train_df = pd.read_parquet("data/processed/train.parquet")
    
    # Initialize engineer
    engineer = FeatureEngineer("config/config.yaml")
    
    # Encode categorical features
    train_encoded = engineer.encode_categorical_features(train_df, fit=True)
    
    # Scale numerical features
    train_scaled = engineer.scale_numerical_features(train_encoded, fit=True)
    
    # Feature selection
    target = train_scaled['is_fraud']
    X = train_scaled.drop(columns=['is_fraud', 'timestamp'])
    
    selected_features = engineer.select_features_statistical(X, target, k=30)
    
    # Create interactions
    interactions = [
        ('amount_log', 'is_night'),
        ('user_transaction_count_30d', 'user_fraud_rate_30d'),
        ('merchant_fraud_rate_30d', 'merchant_transaction_count_30d'),
    ]
    train_interact = engineer.create_feature_interactions(train_scaled, interactions)
    
    # Save feature store
    engineer.save_feature_store(train_interact, version="1.0.0")
    
    logger.info("✓ Feature engineering complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
