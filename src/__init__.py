"""
Guardian-ML: Real-Time Fraud Detection System

A comprehensive, production-ready fraud detection system demonstrating
excellence in ML engineering, statistical rigor, and MLOps.
"""

__version__ = "1.0.0"
__author__ = "Data Science Team"

from .data_pipeline import FraudDataGenerator, DataSplitter
from .feature_engineering import FeatureEngineer
from .models import BaselineModel, XGBoostModel, NeuralNetworkModel
from .evaluation import RigourousEvaluator

__all__ = [
    'FraudDataGenerator',
    'DataSplitter',
    'FeatureEngineer',
    'BaselineModel',
    'XGBoostModel',
    'NeuralNetworkModel',
    'RigourousEvaluator',
]
