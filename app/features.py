"""
Feature generation module for Trading AI System
Calculates technical indicators and prepares data for ML models
"""
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timezone, timedelta
from sqlalchemy import desc
import ta

from app.db import get_db_session, MarketData, Features, log_system_event

logger = logging.getLogger(__name__)

class FeatureGenerator:
    """Generate technical analysis features from market data"""

    def __init__(self, db_manager, config):
        self.db_manager = db_manager
        self.config = config
        self.window_size = config.FEATURE_WINDOW_SIZE

    def generate_features(self, lookback_hours=48):
        """Generate features from recent market data"""
        try:
            logger.info(f"🔧 Generating features for {self.config.TRADING_PAIR}...")

            # Fetch recent market data
            df = self._fetch_market_data(lookback_hours)

            if df.empty:
                logger.warning("⚠️ No market data available for feature generation")
                return False

            # Calculate technical indicators
            df_features = self._calculate_indicators(df)

            # Calculate labels (future returns)
            df_features = self._calculate_labels(df_features)

            # Store features in database
            stored_count = self._store_features(df_features)

            logger.info(f"✅ Generated and stored {stored_count} feature records")
            log_system_event('INFO', 'features', f'Generated {stored_count} features')

            return True

        except Exception as e:
            logger.error(f"❌ Feature generation failed: {e}")
            log_system_event('ERROR', 'features', f'Feature generation failed: {e}')
            return False

    def _fetch_market_data(self, lookback_hours):
        """Fetch market data from database"""
        try:
            session = get_db_session()

            # Calculate cutoff time
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

            # Query market data
            query = session.query(MarketData).filter(
                MarketData.symbol == self.config.TRADING_PAIR,
                MarketData.timestamp >= cutoff_time
            ).order_by(MarketData.timestamp)

            # Convert to DataFrame
            data = []
            for record in query:
                data.append({
                    'timestamp': record.timestamp,
                    'open': record.open_price,
                    'high': record.high_price,
                    'low': record.low_price,
                    'close': record.close_price,
                    'volume': record.volume
                })

            session.close()

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)

            logger.debug(f"📊 Fetched {len(df)} market data points")
            return df

        except Exception as e:
            logger.error(f"❌ Error fetching market data: {e}")
            return pd.DataFrame()

    def _calculate_indicators(self, df):
        """Calculate technical indicators"""
        try:
            df = df.copy()

            # Price-based indicators
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
            df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)

            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()

            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'], window=20)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()

            # Average True Range
            df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)

            # Volume indicators
            df['volume_sma_20'] = ta.volume.VolumeSMAIndicator(df['close'], df['volume'], window=20).volume_sma()
            df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])

            # Volatility
            df['volatility_20'] = df['close'].rolling(window=20).std()

            # Additional features
            df['price_change'] = df['close'].pct_change()
            df['volume_change'] = df['volume'].pct_change()
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_to_high'] = df['close'] / df['high']
            df['close_to_low'] = df['close'] / df['low']

            # Time-based features
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['is_weekend'] = df.index.dayofweek >= 5

            logger.debug("✅ Technical indicators calculated")
            return df

        except Exception as e:
            logger.error(f"❌ Error calculating indicators: {e}")
            return df

    def _calculate_labels(self, df):
        """Calculate future returns as labels"""
        try:
            # Calculate future returns
            df['return_1h'] = df['close'].shift(-60).pct_change(60)  # 1 hour return (60 minutes)
            df['return_4h'] = df['close'].shift(-240).pct_change(240)  # 4 hour return
            df['return_24h'] = df['close'].shift(-1440).pct_change(1440)  # 24 hour return

            # Binary direction labels
            df['direction_1h'] = (df['return_1h'] > 0).astype(int)

            logger.debug("✅ Labels calculated")
            return df

        except Exception as e:
            logger.error(f"❌ Error calculating labels: {e}")
            return df

    def _store_features(self, df):
        """Store features in database"""
        try:
            session = get_db_session()
            stored_count = 0

            # Define feature columns
            feature_columns = [
                'sma_20', 'ema_20', 'rsi_14', 'macd', 'macd_signal',
                'bb_upper', 'bb_lower', 'bb_middle', 'atr_14',
                'volume_sma_20', 'obv', 'volatility_20',
                'return_1h', 'return_4h', 'return_24h', 'direction_1h'
            ]

            for timestamp, row in df.iterrows():
                # Skip rows with NaN values in critical features
                if pd.isna(row[['sma_20', 'ema_20', 'rsi_14']]).any():
                    continue

                # Check if features already exist
                existing = session.query(Features).filter(
                    Features.timestamp == timestamp,
                    Features.symbol == self.config.TRADING_PAIR
                ).first()

                if existing:
                    continue  # Skip existing records

                # Create feature record
                feature_record = Features(
                    timestamp=timestamp,
                    symbol=self.config.TRADING_PAIR,
                    sma_20=float(row['sma_20']) if not pd.isna(row['sma_20']) else None,
                    ema_20=float(row['ema_20']) if not pd.isna(row['ema_20']) else None,
                    rsi_14=float(row['rsi_14']) if not pd.isna(row['rsi_14']) else None,
                    macd=float(row['macd']) if not pd.isna(row['macd']) else None,
                    macd_signal=float(row['macd_signal']) if not pd.isna(row['macd_signal']) else None,
                    bb_upper=float(row['bb_upper']) if not pd.isna(row['bb_upper']) else None,
                    bb_lower=float(row['bb_lower']) if not pd.isna(row['bb_lower']) else None,
                    bb_middle=float(row['bb_middle']) if not pd.isna(row['bb_middle']) else None,
                    atr_14=float(row['atr_14']) if not pd.isna(row['atr_14']) else None,
                    volume_sma_20=float(row['volume_sma_20']) if not pd.isna(row['volume_sma_20']) else None,
                    obv=float(row['obv']) if not pd.isna(row['obv']) else None,
                    volatility_20=float(row['volatility_20']) if not pd.isna(row['volatility_20']) else None,
                    return_1h=float(row['return_1h']) if not pd.isna(row['return_1h']) else None,
                    return_4h=float(row['return_4h']) if not pd.isna(row['return_4h']) else None,
                    return_24h=float(row['return_24h']) if not pd.isna(row['return_24h']) else None,
                    direction_1h=int(row['direction_1h']) if not pd.isna(row['direction_1h']) else None
                )

                session.add(feature_record)
                stored_count += 1

            session.commit()
            session.close()

            return stored_count

        except Exception as e:
            logger.error(f"❌ Error storing features: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return 0

    def get_latest_features(self, limit=100):
        """Get latest features for model input"""
        try:
            session = get_db_session()

            query = session.query(Features).filter(
                Features.symbol == self.config.TRADING_PAIR
            ).order_by(desc(Features.timestamp)).limit(limit)

            data = []
            for record in query:
                data.append({
                    'timestamp': record.timestamp,
                    'sma_20': record.sma_20,
                    'ema_20': record.ema_20,
                    'rsi_14': record.rsi_14,
                    'macd': record.macd,
                    'macd_signal': record.macd_signal,
                    'bb_upper': record.bb_upper,
                    'bb_lower': record.bb_lower,
                    'bb_middle': record.bb_middle,
                    'atr_14': record.atr_14,
                    'volume_sma_20': record.volume_sma_20,
                    'obv': record.obv,
                    'volatility_20': record.volatility_20,
                    'return_1h': record.return_1h,
                    'return_4h': record.return_4h,
                    'return_24h': record.return_24h,
                    'direction_1h': record.direction_1h
                })

            session.close()

            df = pd.DataFrame(data)
            if not df.empty:
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)

            return df

        except Exception as e:
            logger.error(f"❌ Error getting latest features: {e}")
            return pd.DataFrame()

    def get_training_data(self, lookback_days=30):
        """Get training data with features and labels"""
        try:
            session = get_db_session()

            # Calculate cutoff time
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=lookback_days)

            query = session.query(Features).filter(
                Features.symbol == self.config.TRADING_PAIR,
                Features.timestamp >= cutoff_time,
                Features.return_1h.isnot(None)  # Only complete records with labels
            ).order_by(Features.timestamp)

            data = []
            for record in query:
                # Skip records with missing critical features
                if any(x is None for x in [record.sma_20, record.ema_20, record.rsi_14]):
                    continue

                data.append({
                    'timestamp': record.timestamp,
                    'sma_20': record.sma_20,
                    'ema_20': record.ema_20,
                    'rsi_14': record.rsi_14,
                    'macd': record.macd or 0,
                    'macd_signal': record.macd_signal or 0,
                    'bb_upper': record.bb_upper or 0,
                    'bb_lower': record.bb_lower or 0,
                    'bb_middle': record.bb_middle or 0,
                    'atr_14': record.atr_14 or 0,
                    'volatility_20': record.volatility_20 or 0,
                    'return_1h': record.return_1h,
                    'direction_1h': record.direction_1h
                })

            session.close()

            df = pd.DataFrame(data)
            if not df.empty:
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)

                # Fill any remaining NaN values
                df.fillna(0, inplace=True)

            logger.info(f"📊 Retrieved {len(df)} training samples")
            return df

        except Exception as e:
            logger.error(f"❌ Error getting training data: {e}")
            return pd.DataFrame()

    def calculate_reward_function(self, returns, drawdowns, volatility, lambda_dd=0.5, gamma_vol=0.1):
        """Calculate risk-reward objective function"""
        try:
            mean_return = np.mean(returns)
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
            vol = np.std(returns)

            # Risk-adjusted reward: return - lambda * max_drawdown - gamma * volatility
            reward = mean_return - lambda_dd * max_drawdown - gamma_vol * vol

            return reward

        except Exception as e:
            logger.error(f"❌ Error calculating reward: {e}")
            return 0.0

    def get_feature_importance(self, model=None):
        """Get feature importance if model supports it"""
        feature_names = [
            'sma_20', 'ema_20', 'rsi_14', 'macd', 'macd_signal',
            'bb_upper', 'bb_lower', 'bb_middle', 'atr_14', 'volatility_20'
        ]

        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return dict(zip(feature_names, importance))
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
            return dict(zip(feature_names, importance))
        else:
            return {}