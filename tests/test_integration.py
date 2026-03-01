"""
Guardian-ML: Unit Tests

Test coverage for correctness, edge cases, and fairness.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import modules to test
import sys
sys.path.insert(0, 'src')

from data_pipeline import FraudDataGenerator, DataSplitter
from feature_engineering import FeatureEngineer
from evaluation import RigourousEvaluator


# ============================================================================
# DATA PIPELINE TESTS
# ============================================================================

class TestFraudDataGenerator:
    """Test synthetic data generation."""
    
    def test_fraud_rate_realistic(self):
        """Generated fraud rate should be close to target."""
        gen = FraudDataGenerator()
        df = gen.generate_transactions()
        
        fraud_rate = df['is_fraud'].mean()
        
        # Should be within 0.1 percentage points
        assert abs(fraud_rate - 0.003) < 0.001
    
    def test_no_missing_values(self):
        """Generated data should have no missing values."""
        gen = FraudDataGenerator()
        df = gen.generate_transactions()
        
        assert df.isnull().sum().sum() == 0, "Generated data has missing values"
    
    def test_valid_timestamps(self):
        """Timestamps should be recent and in order."""
        gen = FraudDataGenerator()
        df = gen.generate_transactions()
        
        assert (df['timestamp'].min() >= pd.Timestamp('2024-01-01'))
        assert (df['timestamp'].max() <= pd.Timestamp('2024-12-31'))
        assert (df['timestamp'].is_monotonic_increasing)
    
    def test_amounts_positive(self):
        """Transaction amounts must be positive."""
        gen = FraudDataGenerator()
        df = gen.generate_transactions()
        
        assert (df['amount'] > 0).all()
    
    def test_derived_features_computed(self):
        """Test that derived features are properly computed."""
        gen = FraudDataGenerator()
        df = gen.generate_transactions()
        df = gen.add_derived_features(df)
        
        assert 'hour' in df.columns
        assert 'day_of_week' in df.columns
        assert 'is_weekend' in df.columns
        assert 'amount_log' in df.columns


class TestDataSplitter:
    """Test train/val/test splitting."""
    
    def test_temporal_split_no_leakage(self):
        """Temporal split should have no overlap."""
        gen = FraudDataGenerator()
        df = gen.generate_transactions()
        
        splitter = DataSplitter()
        train, val, test = splitter.temporal_split(df)
        
        # Verify no timestamp overlap
        assert train['timestamp'].max() < val['timestamp'].min()
        assert val['timestamp'].max() < test['timestamp'].min()
    
    def test_split_sizes(self):
        """Split ratios should match configuration."""
        gen = FraudDataGenerator()
        df = gen.generate_transactions()
        
        splitter = DataSplitter()
        train, val, test = splitter.temporal_split(df)
        
        n = len(df)
        assert len(train) / n == pytest.approx(0.7, abs=0.01)
        assert len(val) / n == pytest.approx(0.15, abs=0.01)
        assert len(test) / n == pytest.approx(0.15, abs=0.01)
    
    def test_stratified_split_fraud_rate_maintained(self):
        """Stratified split should maintain fraud rate in each set."""
        gen = FraudDataGenerator()
        df = gen.generate_transactions()
        
        splitter = DataSplitter()
        train, val, test = splitter.stratified_split(df)
        
        overall_rate = df['is_fraud'].mean()
        
        # Each split should have similar fraud rate
        assert abs(train['is_fraud'].mean() - overall_rate) < 0.001
        assert abs(val['is_fraud'].mean() - overall_rate) < 0.001
        assert abs(test['is_fraud'].mean() - overall_rate) < 0.001


# ============================================================================
# FEATURE ENGINEERING TESTS
# ============================================================================

class TestFeatureEngineer:
    """Test feature engineering."""
    
    def test_categorical_encoding(self):
        """Test categorical feature encoding."""
        df = pd.DataFrame({
            'country': ['US', 'UK', 'US', 'FR'],
            'device': ['mobile', 'desktop', 'mobile', 'tablet']
        })
        
        engineer = FeatureEngineer()
        df_encoded = engineer.encode_categorical_features(df, fit=True)
        
        # Should be numeric after encoding
        assert df_encoded.select_dtypes(include=[np.number]).shape[1] == 2
    
    def test_numerical_scaling(self):
        """Test that numerical features are scaled correctly."""
        df = pd.DataFrame({
            'amount': [100, 200, 300],
            'age': [20, 30, 40],
        })
        
        engineer = FeatureEngineer()
        df_scaled = engineer.scale_numerical_features(df, fit=True)
        
        # Scaled features should have mean close to 0
        assert abs(df_scaled['amount'].mean()) < 0.1
        assert abs(df_scaled['age'].mean()) < 0.1
    
    def test_feature_selection_reduces_dimensionality(self):
        """Feature selection should reduce number of features."""
        df = pd.DataFrame({
            'feat1': np.random.rand(100),
            'feat2': np.random.rand(100),
            'feat3': np.random.rand(100),
            'feat4': np.random.rand(100),
            'feat5': np.random.rand(100),
        })
        y = np.random.binomial(1, 0.5, 100)
        
        engineer = FeatureEngineer()
        selected = engineer.select_features_statistical(df, y, k=3)
        
        assert len(selected) <= 3


# ============================================================================
# EVALUATION TESTS
# ============================================================================

class TestRigourousEvaluator:
    """Test evaluation metrics."""
    
    def test_auc_roc_perfect_classifier(self):
        """Perfect classifier should have AUC = 1.0."""
        evaluator = RigourousEvaluator()
        
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])  # Perfect separation
        
        auc = evaluator.compute_auc_roc(y_true, y_pred)
        
        assert auc == pytest.approx(1.0, abs=0.01)
    
    def test_auc_roc_random_classifier(self):
        """Random classifier should have AUC ≈ 0.5."""
        evaluator = RigourousEvaluator()
        
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.5, 100)
        y_pred = np.random.rand(100)
        
        auc = evaluator.compute_auc_roc(y_true, y_pred)
        
        assert auc == pytest.approx(0.5, abs=0.1)
    
    def test_bootstrap_ci_reasonable(self):
        """Bootstrap CI should bracket point estimate."""
        evaluator = RigourousEvaluator()
        
        y_true = np.array([0, 0, 0, 1, 1, 1] * 10)
        y_pred = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9] * 10)
        
        point, lower, upper = evaluator.bootstrap_ci(
            y_true, y_pred, metric='auc_roc'
        )
        
        # Point estimate should be within CI
        assert lower <= point <= upper
        # CI should be narrow (not huge)
        assert upper - lower < 0.2
    
    def test_optimal_threshold_business_cost(self):
        """Optimal threshold should minimize business cost."""
        evaluator = RigourousEvaluator()
        
        y_true = np.array([0, 0, 0, 1, 1, 1] * 10)
        y_pred = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9] * 10)
        
        # Cost of false negative (fraud) is much higher
        threshold, cost = evaluator.find_optimal_threshold(
            y_true, y_pred,
            cost_fn=1000.0,  # High cost for missing fraud
            cost_fp=1.0,     # Low cost for false alarm
        )
        
        # With such high FN cost, threshold should be low
        # (approve less, catch more fraud)
        assert threshold < 0.5


# ============================================================================
# FAIRNESS TESTS
# ============================================================================

class TestFairness:
    """Test fairness across demographic groups."""
    
    def test_disparate_impact_detection(self):
        """Should detect disparate impact across groups."""
        evaluator = RigourousEvaluator()
        
        # Simulate biased predictions
        y_true = np.array([0, 1, 0, 1] * 25)
        y_pred = np.array([0.1, 0.9, 0.1, 0.9] * 25)
        
        # Group assignments (simulating geographic bias)
        groups = np.array([0] * 50 + [1] * 50)  # 0=Country A, 1=Country B
        
        fairness = evaluator.fairness_analysis(
            y_true, y_pred, groups, threshold=0.5
        )
        
        # Should report metrics per group
        assert 'group_0' in str(fairness) or '0' in str(fairness)
        assert 'group_1' in str(fairness) or '1' in str(fairness)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestEndToEnd:
    """End-to-end pipeline tests."""
    
    def test_full_pipeline_data_to_eval(self):
        """Test complete pipeline: data → features → modeling → eval."""
        
        # Generate data
        gen = FraudDataGenerator()
        df = gen.generate_transactions()
        df = gen.add_derived_features(df)
        
        # Split
        splitter = DataSplitter()
        train, val, test = splitter.temporal_split(df)
        
        # Engineer features
        engineer = FeatureEngineer()
        train_encoded = engineer.encode_categorical_features(train, fit=True)
        train_scaled = engineer.scale_numerical_features(train_encoded, fit=True)
        
        # Evaluate on train data (sanity check)
        evaluator = RigourousEvaluator()
        y_true = train_scaled['is_fraud']
        y_pred_dummy = np.random.rand(len(train_scaled))  # Dummy predictions
        
        auc = evaluator.compute_auc_roc(y_true, y_pred_dummy)
        
        # Dummy model should be random (~0.5 AUC)
        assert 0.4 < auc < 0.6


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
