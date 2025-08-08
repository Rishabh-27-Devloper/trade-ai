"""
Configuration management for Trading AI System
Handles environment variables and application settings
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration class"""

    def __init__(self):
        # Flask Configuration
        self.SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
        self.DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
        self.ENV = os.getenv('FLASK_ENV', 'development')

        # Database Configuration
        self.DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///trade_ai.db')
        self.SQLALCHEMY_TRACK_MODIFICATIONS = False

        # Trading Configuration
        self.TRADING_PAIR = os.getenv('TRADING_PAIR', 'BTCUSDT')
        self.TRADE_AMOUNT_USDT = float(os.getenv('TRADE_AMOUNT_USDT', '100.0'))
        self.RISK_PERCENT = float(os.getenv('RISK_PERCENT', '0.01'))
        self.MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', '5'))
        self.COOLDOWN_SECONDS = int(os.getenv('COOLDOWN_SECONDS', '10'))

        # Binance API Configuration (optional for demo)
        self.BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
        self.BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')

        # Model Training Configuration
        self.RETRAIN_INTERVAL_SECONDS = int(os.getenv('RETRAIN_INTERVAL_SECONDS', '3600'))
        self.MIN_NEW_ROWS_FOR_RETRAIN = int(os.getenv('MIN_NEW_ROWS_FOR_RETRAIN', '100'))
        self.FEATURE_WINDOW_SIZE = int(os.getenv('FEATURE_WINDOW_SIZE', '20'))
        self.MODEL_TYPE = os.getenv('MODEL_TYPE', 'xgboost')

        # WebSocket Configuration
        self.BINANCE_WS_URL = 'wss://stream.binance.com:9443/ws'
        self.RECONNECTION_ATTEMPTS = 5
        self.RECONNECTION_DELAY = 30

        # Logging Configuration
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        self.LOG_FILE = os.getenv('LOG_FILE', 'logs/trade_ai.log')

        # Dashboard Authentication
        self.DASHBOARD_USERNAME = os.getenv('DASHBOARD_USERNAME', 'admin')
        self.DASHBOARD_PASSWORD = os.getenv('DASHBOARD_PASSWORD', 'admin')

        # Risk Management
        self.MAX_DRAWDOWN_PERCENT = 0.10  # 10% max drawdown
        self.SHARPE_RATIO_THRESHOLD = 0.5
        self.WIN_RATE_THRESHOLD = 0.4

        # Slippage and Fees
        self.SLIPPAGE_BPS = 5  # 0.05% slippage
        self.MAKER_FEE_BPS = 10  # 0.10% maker fee
        self.TAKER_FEE_BPS = 10  # 0.10% taker fee

    def to_dict(self):
        """Convert configuration to dictionary (excluding sensitive data)"""
        return {
            'trading_pair': self.TRADING_PAIR,
            'trade_amount_usdt': self.TRADE_AMOUNT_USDT,
            'risk_percent': self.RISK_PERCENT,
            'max_positions': self.MAX_POSITIONS,
            'model_type': self.MODEL_TYPE,
            'feature_window_size': self.FEATURE_WINDOW_SIZE,
            'retrain_interval_seconds': self.RETRAIN_INTERVAL_SECONDS,
            'environment': self.ENV,
            'debug': self.DEBUG
        }

    def validate(self):
        """Validate configuration settings"""
        errors = []

        if self.RISK_PERCENT <= 0 or self.RISK_PERCENT > 0.1:
            errors.append("RISK_PERCENT must be between 0 and 0.1 (10%)")

        if self.MAX_POSITIONS < 1:
            errors.append("MAX_POSITIONS must be at least 1")

        if self.FEATURE_WINDOW_SIZE < 5:
            errors.append("FEATURE_WINDOW_SIZE must be at least 5")

        if self.MODEL_TYPE not in ['xgboost', 'lightgbm', 'lstm', 'baseline']:
            errors.append("MODEL_TYPE must be one of: xgboost, lightgbm, lstm, baseline")

        return errors

def get_config():
    """Factory function to get validated configuration"""
    config = Config()
    errors = config.validate()

    if errors:
        raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

    return config