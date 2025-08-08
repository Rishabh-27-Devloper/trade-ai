"""
Model training orchestrator for Trading AI System
Handles continuous training and model management
"""
import os
import time
import logging
import threading
from datetime import datetime, timezone
import json

from app.db import get_db_session, ModelCheckpoint, log_system_event
from app.features import FeatureGenerator
from app.models.xgb_model import XGBoostTradingModel
from app.models.baseline import BaselineModel
from app.models.lstm_model import LSTMTradingWrapper

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Orchestrates model training and management"""

    def __init__(self, db_manager, config):
        self.db_manager = db_manager
        self.config = config
        self.feature_generator = FeatureGenerator(db_manager, config)

        self.running = False
        self.current_model = None
        self.model_version = "1.0.0"

        # Create models directory
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)

        # Training configuration
        self.retrain_interval = config.RETRAIN_INTERVAL_SECONDS
        self.min_data_points = config.MIN_NEW_ROWS_FOR_RETRAIN

        # Initialize model
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the selected model type"""
        try:
            model_type = self.config.MODEL_TYPE.lower()

            if model_type == 'xgboost':
                self.current_model = XGBoostTradingModel(self.config)
            elif model_type == 'baseline':
                self.current_model = BaselineModel(self.config, 'logistic')
            elif model_type == 'lstm':
                self.current_model = LSTMTradingWrapper(self.config)
            else:
                logger.warning(f"⚠️ Unknown model type {model_type}, defaulting to XGBoost")
                self.current_model = XGBoostTradingModel(self.config)

            logger.info(f"🧠 Initialized {model_type} model")

            # Try to load existing model
            self._load_latest_model()

        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            # Fallback to baseline model
            self.current_model = BaselineModel(self.config, 'rule_based')

    def run_continuously(self):
        """Run continuous training loop"""
        logger.info("🔄 Starting continuous model training...")
        self.running = True

        last_feature_generation = 0
        last_training = 0

        while self.running:
            try:
                current_time = time.time()

                # Generate features periodically (every 5 minutes)
                if current_time - last_feature_generation >= 300:
                    logger.info("🔧 Generating features...")
                    success = self.feature_generator.generate_features()
                    if success:
                        last_feature_generation = current_time
                        log_system_event('INFO', 'trainer', 'Features generated successfully')
                    else:
                        logger.warning("⚠️ Feature generation failed")

                # Check if retraining is needed
                if current_time - last_training >= self.retrain_interval:
                    if self._should_retrain():
                        logger.info("🧠 Starting model retraining...")
                        success = self.train_model()
                        if success:
                            last_training = current_time
                            log_system_event('INFO', 'trainer', 'Model retrained successfully')
                        else:
                            logger.warning("⚠️ Model retraining failed")

                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"❌ Error in training loop: {e}")
                time.sleep(60)  # Wait longer on error

    def stop(self):
        """Stop continuous training"""
        logger.info("🛑 Stopping model trainer...")
        self.running = False

    def train_model(self, force=False):
        """Train the current model"""
        try:
            start_time = time.time()
            logger.info("🧠 Starting model training...")

            # Get training data
            training_data = self.feature_generator.get_training_data()

            if training_data.empty:
                logger.warning("⚠️ No training data available")
                return False

            logger.info(f"📊 Training data shape: {training_data.shape}")

            # Train the model
            success = self.current_model.train(training_data)

            if not success:
                logger.error("❌ Model training failed")
                return False

            training_duration = time.time() - start_time

            # Save model checkpoint
            checkpoint_saved = self._save_model_checkpoint(training_duration)

            if checkpoint_saved:
                logger.info(f"✅ Model training completed in {training_duration:.2f}s")
                log_system_event('INFO', 'trainer', f'Model trained successfully in {training_duration:.2f}s')
                return True
            else:
                logger.warning("⚠️ Model training completed but checkpoint save failed")
                return False

        except Exception as e:
            logger.error(f"❌ Model training error: {e}")
            log_system_event('ERROR', 'trainer', f'Model training failed: {e}')
            return False

    def _should_retrain(self):
        """Check if model should be retrained"""
        try:
            session = get_db_session()

            # Count new data points since last training
            last_checkpoint = session.query(ModelCheckpoint).filter(
                ModelCheckpoint.is_active == True
            ).order_by(ModelCheckpoint.created_at.desc()).first()

            if not last_checkpoint:
                session.close()
                return True  # No previous training

            # Count features created since last training
            from app.db import Features
            new_features_count = session.query(Features).filter(
                Features.created_at > last_checkpoint.created_at,
                Features.symbol == self.config.TRADING_PAIR
            ).count()

            session.close()

            should_retrain = new_features_count >= self.min_data_points

            if should_retrain:
                logger.info(f"📈 {new_features_count} new features available, triggering retrain")

            return should_retrain

        except Exception as e:
            logger.error(f"❌ Error checking retrain condition: {e}")
            return False

    def _save_model_checkpoint(self, training_duration):
        """Save model checkpoint to database and filesystem"""
        try:
            # Generate unique model filename
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            model_filename = f"{self.config.MODEL_TYPE}_{timestamp}.joblib"
            model_filepath = os.path.join(self.models_dir, model_filename)

            # Save model file
            if hasattr(self.current_model, 'save_model'):
                if not self.current_model.save_model(model_filepath):
                    return False
            else:
                logger.warning("⚠️ Model doesn't support saving")
                return False

            # Get model info and metrics
            model_info = self.current_model.get_model_info()
            training_metrics = model_info.get('training_metrics', {})
            feature_importance = {}

            if hasattr(self.current_model, 'get_feature_importance'):
                feature_importance = self.current_model.get_feature_importance()

            # Calculate trading metrics if possible
            trading_metrics = self._calculate_trading_metrics()

            # Save checkpoint to database
            session = get_db_session()

            # Deactivate previous checkpoints
            session.query(ModelCheckpoint).filter(
                ModelCheckpoint.is_active == True
            ).update({'is_active': False})

            # Create new checkpoint record
            checkpoint = ModelCheckpoint(
                model_type=self.config.MODEL_TYPE,
                version=self.model_version,
                file_path=model_filepath,
                train_accuracy=training_metrics.get('accuracy'),
                val_accuracy=training_metrics.get('cv_accuracy_mean'),
                train_loss=training_metrics.get('train_loss'),
                val_loss=training_metrics.get('val_loss'),
                sharpe_ratio=trading_metrics.get('sharpe_ratio'),
                sortino_ratio=trading_metrics.get('sortino_ratio'),
                max_drawdown=trading_metrics.get('max_drawdown'),
                win_rate=trading_metrics.get('win_rate'),
                total_returns=trading_metrics.get('total_return'),
                training_samples=training_metrics.get('training_samples'),
                training_duration_seconds=training_duration,
                feature_importance=json.dumps(feature_importance),
                is_active=True
            )

            session.add(checkpoint)
            session.commit()
            session.close()

            logger.info(f"✅ Model checkpoint saved: {model_filename}")
            return True

        except Exception as e:
            logger.error(f"❌ Error saving model checkpoint: {e}")
            return False

    def _calculate_trading_metrics(self):
        """Calculate trading performance metrics"""
        try:
            # Get recent data for backtesting
            features_data = self.feature_generator.get_latest_features(500)

            if features_data.empty or 'return_1h' not in features_data.columns:
                return {}

            # Make predictions
            predictions_result = self.current_model.predict(features_data)

            if not predictions_result:
                return {}

            predictions = predictions_result['predictions']
            actual_returns = features_data['return_1h'].fillna(0).tolist()

            # Align predictions and returns
            min_len = min(len(predictions), len(actual_returns))
            predictions = predictions[:min_len]
            actual_returns = actual_returns[:min_len]

            # Calculate trading metrics
            if hasattr(self.current_model, 'calculate_trading_metrics'):
                return self.current_model.calculate_trading_metrics(predictions, actual_returns)
            else:
                return {}

        except Exception as e:
            logger.error(f"❌ Error calculating trading metrics: {e}")
            return {}

    def _load_latest_model(self):
        """Load the latest trained model"""
        try:
            session = get_db_session()

            latest_checkpoint = session.query(ModelCheckpoint).filter(
                ModelCheckpoint.is_active == True
            ).order_by(ModelCheckpoint.created_at.desc()).first()

            session.close()

            if not latest_checkpoint:
                logger.info("📝 No existing model found, will train from scratch")
                return False

            if not os.path.exists(latest_checkpoint.file_path):
                logger.warning(f"⚠️ Model file not found: {latest_checkpoint.file_path}")
                return False

            # Load the model
            if hasattr(self.current_model, 'load_model'):
                success = self.current_model.load_model(latest_checkpoint.file_path)
                if success:
                    logger.info(f"✅ Loaded model from {latest_checkpoint.file_path}")
                    return True

            return False

        except Exception as e:
            logger.error(f"❌ Error loading latest model: {e}")
            return False

    def get_model_prediction(self, features_dict):
        """Get prediction from current model"""
        try:
            if not self.current_model:
                return None

            return self.current_model.predict_single(features_dict)

        except Exception as e:
            logger.error(f"❌ Error getting model prediction: {e}")
            return None

    def get_model_status(self):
        """Get current model status"""
        try:
            status = {
                'model_type': self.config.MODEL_TYPE,
                'is_trained': self.current_model.is_trained if self.current_model else False,
                'running': self.running,
                'models_dir': self.models_dir
            }

            if self.current_model and hasattr(self.current_model, 'get_model_info'):
                model_info = self.current_model.get_model_info()
                status.update(model_info)

            # Get latest checkpoint info
            try:
                session = get_db_session()
                latest_checkpoint = session.query(ModelCheckpoint).filter(
                    ModelCheckpoint.is_active == True
                ).order_by(ModelCheckpoint.created_at.desc()).first()
                session.close()

                if latest_checkpoint:
                    status['latest_checkpoint'] = {
                        'version': latest_checkpoint.version,
                        'created_at': latest_checkpoint.created_at.isoformat(),
                        'accuracy': latest_checkpoint.train_accuracy,
                        'sharpe_ratio': latest_checkpoint.sharpe_ratio,
                        'win_rate': latest_checkpoint.win_rate
                    }
            except Exception:
                pass

            return status

        except Exception as e:
            logger.error(f"❌ Error getting model status: {e}")
            return {'error': str(e)}

    def manual_retrain(self):
        """Manually trigger model retraining"""
        logger.info("🔄 Manual retrain triggered")
        return self.train_model(force=True)