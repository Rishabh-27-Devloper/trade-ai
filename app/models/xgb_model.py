"""
XGBoost model implementation for Trading AI System
Optimized for risk-reward trading objectives
"""
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timezone
import joblib
import json
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

logger = logging.getLogger(__name__)

class XGBoostTradingModel:
    """XGBoost model for trading signal generation"""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.feature_names = [
            'sma_20', 'ema_20', 'rsi_14', 'macd', 'macd_signal',
            'bb_upper', 'bb_lower', 'bb_middle', 'atr_14', 'volatility_20'
        ]
        self.is_trained = False
        self.training_metrics = {}

        # XGBoost hyperparameters optimized for financial data
        self.params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'n_estimators': 100,
            'random_state': 42,
            'verbosity': 0
        }

    def prepare_data(self, df):
        """Prepare data for training/prediction"""
        try:
            if df.empty:
                return None, None

            # Select feature columns
            available_features = [col for col in self.feature_names if col in df.columns]
            X = df[available_features].copy()

            # Handle missing values
            X = X.fillna(0)

            # Prepare labels if available
            y = None
            if 'direction_1h' in df.columns:
                y = df['direction_1h'].copy()
                # Remove rows where y is NaN
                valid_idx = ~y.isna()
                X = X[valid_idx]
                y = y[valid_idx]

            return X, y

        except Exception as e:
            logger.error(f"❌ Error preparing data: {e}")
            return None, None

    def train(self, training_data):
        """Train the XGBoost model"""
        try:
            logger.info("🧠 Training XGBoost model...")

            X, y = self.prepare_data(training_data)

            if X is None or y is None or len(X) == 0:
                logger.error("❌ No valid training data available")
                return False

            logger.info(f"📊 Training on {len(X)} samples with {len(X.columns)} features")

            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=3)

            # Create XGBoost model
            self.model = xgb.XGBClassifier(**self.params)

            # Cross-validation scores
            cv_scores = cross_val_score(self.model, X, y, cv=tscv, scoring='accuracy')

            # Train on full dataset
            self.model.fit(X, y)

            # Generate predictions for metrics
            y_pred = self.model.predict(X)
            y_pred_proba = self.model.predict_proba(X)[:, 1]

            # Calculate metrics
            self.training_metrics = {
                'accuracy': float(accuracy_score(y, y_pred)),
                'precision': float(precision_score(y, y_pred, zero_division=0)),
                'recall': float(recall_score(y, y_pred, zero_division=0)),
                'f1_score': float(f1_score(y, y_pred, zero_division=0)),
                'cv_accuracy_mean': float(np.mean(cv_scores)),
                'cv_accuracy_std': float(np.std(cv_scores)),
                'training_samples': len(X),
                'feature_count': len(X.columns)
            }

            self.is_trained = True

            logger.info(f"✅ Model training completed:")
            logger.info(f"   Accuracy: {self.training_metrics['accuracy']:.4f}")
            logger.info(f"   Precision: {self.training_metrics['precision']:.4f}")
            logger.info(f"   Recall: {self.training_metrics['recall']:.4f}")
            logger.info(f"   F1-Score: {self.training_metrics['f1_score']:.4f}")
            logger.info(f"   CV Accuracy: {self.training_metrics['cv_accuracy_mean']:.4f} ± {self.training_metrics['cv_accuracy_std']:.4f}")

            return True

        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            return False

    def predict(self, features_data):
        """Make predictions on new data"""
        try:
            if not self.is_trained or self.model is None:
                logger.warning("⚠️ Model not trained, cannot make predictions")
                return None

            X, _ = self.prepare_data(features_data)

            if X is None or len(X) == 0:
                logger.warning("⚠️ No valid data for prediction")
                return None

            # Get prediction probabilities
            probabilities = self.model.predict_proba(X)[:, 1]  # Probability of positive class
            predictions = self.model.predict(X)

            return {
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist(),
                'timestamps': features_data.index.tolist()
            }

        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return None

    def predict_single(self, features_dict):
        """Make single prediction from feature dictionary"""
        try:
            if not self.is_trained or self.model is None:
                return None

            # Create DataFrame from dictionary
            feature_values = []
            for feature in self.feature_names:
                value = features_dict.get(feature, 0)
                feature_values.append(value)

            X = np.array([feature_values])

            probability = self.model.predict_proba(X)[0, 1]
            prediction = self.model.predict(X)[0]

            return {
                'prediction': int(prediction),
                'probability': float(probability),
                'signal_strength': float(abs(probability - 0.5) * 2)  # 0 to 1 scale
            }

        except Exception as e:
            logger.error(f"❌ Single prediction failed: {e}")
            return None

    def get_feature_importance(self):
        """Get feature importance scores"""
        try:
            if not self.is_trained or self.model is None:
                return {}

            importance_scores = self.model.feature_importances_
            available_features = [col for col in self.feature_names if hasattr(self.model, 'feature_names_in_') and col in self.model.feature_names_in_]

            if not available_features:
                available_features = self.feature_names[:len(importance_scores)]

            importance_dict = {}
            for i, feature in enumerate(available_features):
                if i < len(importance_scores):
                    importance_dict[feature] = float(importance_scores[i])

            return importance_dict

        except Exception as e:
            logger.error(f"❌ Error getting feature importance: {e}")
            return {}

    def save_model(self, filepath):
        """Save trained model to file"""
        try:
            if not self.is_trained or self.model is None:
                logger.error("❌ No trained model to save")
                return False

            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'training_metrics': self.training_metrics,
                'params': self.params,
                'config': self.config.to_dict(),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

            joblib.dump(model_data, filepath)
            logger.info(f"✅ Model saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
            return False

    def load_model(self, filepath):
        """Load trained model from file"""
        try:
            model_data = joblib.load(filepath)

            self.model = model_data['model']
            self.feature_names = model_data.get('feature_names', self.feature_names)
            self.training_metrics = model_data.get('training_metrics', {})
            self.params = model_data.get('params', self.params)

            self.is_trained = True

            logger.info(f"✅ Model loaded from {filepath}")
            logger.info(f"   Training accuracy: {self.training_metrics.get('accuracy', 'N/A')}")

            return True

        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False

    def get_model_info(self):
        """Get model information and metrics"""
        return {
            'model_type': 'XGBoost',
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names),
            'parameters': self.params,
            'training_metrics': self.training_metrics,
            'feature_names': self.feature_names
        }

    def calculate_trading_metrics(self, predictions, actual_returns):
        """Calculate trading-specific metrics"""
        try:
            if len(predictions) != len(actual_returns):
                logger.error("Mismatch between predictions and returns length")
                return {}

            # Convert to numpy arrays
            preds = np.array(predictions)
            returns = np.array(actual_returns)

            # Calculate strategy returns
            strategy_returns = returns * (preds * 2 - 1)  # Long when pred=1, short when pred=0

            # Trading metrics
            total_return = np.sum(strategy_returns)
            win_rate = np.mean(strategy_returns > 0)

            # Risk metrics
            sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) > 0 else 0

            # Calculate drawdown
            cumulative_returns = np.cumsum(strategy_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = running_max - cumulative_returns
            max_drawdown = np.max(drawdowns)

            # Sortino ratio
            downside_returns = strategy_returns[strategy_returns < 0]
            sortino_ratio = np.mean(strategy_returns) / np.std(downside_returns) if len(downside_returns) > 0 and np.std(downside_returns) > 0 else 0

            return {
                'total_return': float(total_return),
                'win_rate': float(win_rate),
                'sharpe_ratio': float(sharpe_ratio),
                'sortino_ratio': float(sortino_ratio),
                'max_drawdown': float(max_drawdown),
                'avg_return': float(np.mean(strategy_returns)),
                'return_volatility': float(np.std(strategy_returns)),
                'total_trades': len(predictions)
            }

        except Exception as e:
            logger.error(f"❌ Error calculating trading metrics: {e}")
            return {}