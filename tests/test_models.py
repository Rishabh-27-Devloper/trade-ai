
"""
Tests for ML models
"""
import pytest
import pandas as pd
import numpy as np
from app.models.xgb_model import XGBoostTradingModel
from app.models.baseline import BaselineModel
from app.config import Config

@pytest.fixture
def config():
    return Config()

@pytest.fixture
def sample_data():
    """Create sample training data"""
    np.random.seed(42)
    n_samples = 100

    data = {
        'sma_20': np.random.uniform(45000, 50000, n_samples),
        'ema_20': np.random.uniform(45000, 50000, n_samples),
        'rsi_14': np.random.uniform(20, 80, n_samples),
        'macd': np.random.uniform(-100, 100, n_samples),
        'macd_signal': np.random.uniform(-100, 100, n_samples),
        'bb_upper': np.random.uniform(50000, 52000, n_samples),
        'bb_lower': np.random.uniform(44000, 46000, n_samples),
        'bb_middle': np.random.uniform(47000, 49000, n_samples),
        'atr_14': np.random.uniform(500, 1500, n_samples),
        'volatility_20': np.random.uniform(0.01, 0.05, n_samples),
        'direction_1h': np.random.randint(0, 2, n_samples)
    }

    return pd.DataFrame(data)

def test_xgboost_model_initialization(config):
    """Test XGBoost model initialization"""
    model = XGBoostTradingModel(config)

    assert model.config == config
    assert model.model is None
    assert not model.is_trained
    assert len(model.feature_names) > 0

def test_xgboost_model_training(config, sample_data):
    """Test XGBoost model training"""
    model = XGBoostTradingModel(config)

    success = model.train(sample_data)

    assert success
    assert model.is_trained
    assert model.model is not None
    assert len(model.training_metrics) > 0

def test_xgboost_prediction(config, sample_data):
    """Test XGBoost prediction"""
    model = XGBoostTradingModel(config)
    model.train(sample_data)

    # Test single prediction
    features_dict = {
        'sma_20': 47000,
        'ema_20': 47100,
        'rsi_14': 65,
        'macd': 50,
        'macd_signal': 40,
        'bb_upper': 48000,
        'bb_lower': 46000,
        'bb_middle': 47000,
        'atr_14': 800,
        'volatility_20': 0.02
    }

    result = model.predict_single(features_dict)

    assert result is not None
    assert 'prediction' in result
    assert 'probability' in result
    assert result['prediction'] in [0, 1]
    assert 0 <= result['probability'] <= 1

def test_baseline_model(config, sample_data):
    """Test baseline model functionality"""
    model = BaselineModel(config, 'logistic')

    success = model.train(sample_data)
    assert success
    assert model.is_trained

    # Test prediction
    features_dict = {'rsi_14': 30, 'close_price': 47000, 'sma_20': 46500}
    result = model.predict_single(features_dict)

    assert result is not None
    assert 'prediction' in result
    assert 'probability' in result

def test_rule_based_model(config):
    """Test rule-based model"""
    model = BaselineModel(config, 'rule_based')
    model.is_trained = True  # Rule-based doesn't need training

    # Test oversold condition
    features = {'rsi_14': 25, 'close_price': 47000, 'sma_20': 46500}
    result = model.predict_single(features)

    assert result['prediction'] == 1  # Should be bullish
    assert result['probability'] > 0.5
