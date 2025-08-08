"""
Database management and models for Trading AI System
Uses SQLAlchemy with SQLite in WAL mode for concurrent access
"""
import os
import sqlite3
import logging
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

Base = declarative_base()

class MarketData(Base):
    """Market data table for storing OHLCV data"""
    __tablename__ = 'market_data'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    trades_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class Features(Base):
    """Features table for storing processed technical indicators"""
    __tablename__ = 'features'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)

    # Price-based features
    sma_20 = Column(Float)
    ema_20 = Column(Float)
    rsi_14 = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    bb_upper = Column(Float)
    bb_lower = Column(Float)
    bb_middle = Column(Float)
    atr_14 = Column(Float)

    # Volume features
    volume_sma_20 = Column(Float)
    obv = Column(Float)

    # Volatility features
    volatility_20 = Column(Float)

    # Labels for ML
    return_1h = Column(Float)
    return_4h = Column(Float)
    return_24h = Column(Float)
    direction_1h = Column(Integer)  # 1 for up, 0 for down

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class ModelCheckpoint(Base):
    """Model checkpoints and metadata"""
    __tablename__ = 'model_checkpoints'

    id = Column(Integer, primary_key=True)
    model_type = Column(String(50), nullable=False)
    version = Column(String(20), nullable=False)
    file_path = Column(String(255), nullable=False)

    # Training metrics
    train_accuracy = Column(Float)
    val_accuracy = Column(Float)
    train_loss = Column(Float)
    val_loss = Column(Float)

    # Trading metrics
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    total_returns = Column(Float)

    # Training info
    training_samples = Column(Integer)
    training_duration_seconds = Column(Float)
    feature_importance = Column(Text)  # JSON string

    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class SimulatedTrade(Base):
    """Simulated trading records"""
    __tablename__ = 'simulated_trades'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)

    # Entry details
    entry_timestamp = Column(DateTime(timezone=True), nullable=False)
    entry_price = Column(Float, nullable=False)
    entry_signal = Column(Float, nullable=False)  # Model prediction
    position_size = Column(Float, nullable=False)
    side = Column(String(10), nullable=False)  # 'long' or 'short'

    # Exit details
    exit_timestamp = Column(DateTime(timezone=True))
    exit_price = Column(Float)
    exit_reason = Column(String(50))  # 'take_profit', 'stop_loss', 'time_exit'

    # Performance
    pnl_gross = Column(Float)
    pnl_net = Column(Float)  # After fees and slippage
    return_percent = Column(Float)
    holding_period_minutes = Column(Integer)

    # Risk management
    stop_loss_price = Column(Float)
    take_profit_price = Column(Float)
    max_adverse_excursion = Column(Float)
    max_favorable_excursion = Column(Float)

    # Fees and costs
    entry_fee = Column(Float, default=0.0)
    exit_fee = Column(Float, default=0.0)
    slippage_cost = Column(Float, default=0.0)

    is_closed = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class SystemLog(Base):
    """System logs and events"""
    __tablename__ = 'system_logs'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    level = Column(String(10), nullable=False)  # INFO, WARNING, ERROR, etc.
    component = Column(String(50), nullable=False)  # ingester, trainer, simulator, etc.
    message = Column(Text, nullable=False)
    data = Column(Text)  # JSON string for additional data

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class DatabaseManager:
    """Database connection and session management"""

    def __init__(self, database_url):
        self.database_url = database_url
        self.engine = None
        self.SessionLocal = None
        self.session = None

    def connect(self):
        """Create database connection with WAL mode for SQLite"""
        try:
            # Create engine
            if 'sqlite' in self.database_url.lower():
                # Enable WAL mode for SQLite concurrent access
                self.engine = create_engine(
                    self.database_url,
                    echo=False,
                    pool_pre_ping=True,
                    connect_args={
                        'check_same_thread': False,
                        'timeout': 30
                    }
                )

                # Enable WAL mode
                def set_sqlite_pragma(dbapi_connection, connection_record):
                    cursor = dbapi_connection.cursor()
                    cursor.execute('PRAGMA journal_mode=WAL;')
                    cursor.execute('PRAGMA synchronous=NORMAL;')
                    cursor.execute('PRAGMA cache_size=10000;')
                    cursor.execute('PRAGMA temp_store=MEMORY;')
                    cursor.close()

                from sqlalchemy import event
                event.listen(self.engine, 'connect', set_sqlite_pragma)

            else:
                # PostgreSQL or other databases
                self.engine = create_engine(database_url, echo=False)

            # Create session factory
            self.SessionLocal = scoped_session(sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            ))

            logger.info(f"✅ Database connected: {self.database_url}")
            return True

        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False

    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("✅ Database tables created/verified")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create tables: {e}")
            return False

    def get_session(self):
        """Get database session"""
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized. Call connect() first.")
        return self.SessionLocal()

    def close(self):
        """Close database connections"""
        if self.SessionLocal:
            self.SessionLocal.remove()
        if self.engine:
            self.engine.dispose()
        logger.info("Database connections closed")

# Global database manager instance
db_manager = DatabaseManager("")

def init_db(database_url):
    """Initialize database with given URL"""
    global db_manager
    db_manager = DatabaseManager(database_url)

    if not db_manager.connect():
        raise RuntimeError("Failed to connect to database")

    return db_manager

def create_tables(db_manager):
    """Create database tables"""
    if not db_manager.create_tables():
        raise RuntimeError("Failed to create database tables")

def get_db_session():
    """Get database session (for use in application)"""
    return db_manager.get_session()

def log_system_event(level, component, message, data=None):
    """Log system event to database"""
    try:
        session = get_db_session()
        log_entry = SystemLog(
            level=level,
            component=component,
            message=message,
            data=data
        )
        session.add(log_entry)
        session.commit()
        session.close()
    except Exception as e:
        logger.error(f"Failed to log system event: {e}")