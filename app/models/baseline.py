"""
Baseline model implementation for Trading AI System
Simple rule-based and logistic regression models
"""
import numpy as np
import pandas as pd
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

class BaselineModel:
    """Simple baseline models for comparison"""

    def __init__(self, config, model_type='logistic'):
        self.config = config
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = [
            'sma_20', 'ema_20', 'rsi_14', 'macd', 'bb_upper', 'bb_lower'
        ]

    def train(self, training_data):
        """Train baseline model"""
        try:
            logger.info(f"🧠 Training baseline model ({self.model_type})...")

            X, y = self._prepare_data(training_data)
            if X is None or len(X) == 0:
                return False

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            if self.model_type == 'logistic':
                self.model = LogisticRegression(random_state=42)
                self.model.fit(X_scaled, y)
            else:  # rule-based
                self.model = 'rule_based'

            self.is_trained = True

            # Calculate accuracy
            if self.model_type == 'logistic':
                y_pred = self.model.predict(X_scaled)
                accuracy = accuracy_score(y, y_pred)
                logger.info(f"✅ Baseline model accuracy: {accuracy:.4f}")

            return True

        except Exception as e:
            logger.error(f"❌ Baseline training failed: {e}")
            return False

    def predict_single(self, features_dict):
        """Make single prediction"""
        try:
            if not self.is_trained:
                return None

            if self.model_type == 'rule_based':
                return self._rule_based_prediction(features_dict)
            else:
                return self._logistic_prediction(features_dict)

        except Exception as e:
            logger.error(f"❌ Baseline prediction failed: {e}")
            return None

    def _rule_based_prediction(self, features):
        """Simple rule-based prediction"""
        rsi = features.get('rsi_14', 50)
        close_price = features.get('close_price', 0)
        sma_20 = features.get('sma_20', 0)

        # Simple rules
        if rsi < 30 and close_price > sma_20:  # Oversold and above SMA
            return {'prediction': 1, 'probability': 0.7}
        elif rsi > 70 and close_price < sma_20:  # Overbought and below SMA
            return {'prediction': 0, 'probability': 0.3}
        else:
            return {'prediction': 1 if rsi < 50 else 0, 'probability': 0.5}

    def _logistic_prediction(self, features):
        """Logistic regression prediction"""
        feature_values = [features.get(f, 0) for f in self.feature_names]
        X = np.array([feature_values])
        X_scaled = self.scaler.transform(X)

        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0, 1]

        return {'prediction': int(prediction), 'probability': float(probability)}

    def _prepare_data(self, df):
        """Prepare data for training"""
        if df.empty:
            return None, None

        available_features = [col for col in self.feature_names if col in df.columns]
        X = df[available_features].fillna(0)
        y = df['direction_1h'].fillna(0) if 'direction_1h' in df.columns else None

        return X, y