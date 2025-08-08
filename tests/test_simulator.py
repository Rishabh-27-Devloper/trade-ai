
"""
Tests for trading simulator
"""
import pytest
from unittest.mock import Mock, patch
from app.simulator import TradingSimulator
from app.config import Config

@pytest.fixture
def config():
    return Config()

@pytest.fixture
def mock_db_manager():
    return Mock()

def test_simulator_initialization(mock_db_manager, config):
    """Test simulator initialization"""
    simulator = TradingSimulator(mock_db_manager, config)

    assert simulator.db_manager == mock_db_manager
    assert simulator.config == config
    assert not simulator.running
    assert len(simulator.open_positions) == 0
    assert simulator.total_trades == 0

def test_position_size_calculation(mock_db_manager, config):
    """Test position size calculation"""
    simulator = TradingSimulator(mock_db_manager, config)

    price = 47000.0
    position_size = simulator._calculate_position_size(price)

    assert position_size > 0
    assert position_size == config.TRADE_AMOUNT_USDT / price

def test_slippage_calculation(mock_db_manager, config):
    """Test slippage application"""
    simulator = TradingSimulator(mock_db_manager, config)

    price = 47000.0

    # Long entry should increase price (slippage against us)
    long_entry_price = simulator._apply_slippage(price, 'long', 'entry')
    assert long_entry_price > price

    # Short entry should decrease price (slippage against us)
    short_entry_price = simulator._apply_slippage(price, 'short', 'entry')
    assert short_entry_price < price

def test_fee_calculation(mock_db_manager, config):
    """Test trading fee calculation"""
    simulator = TradingSimulator(mock_db_manager, config)

    trade_value = 1000.0

    maker_fee = simulator._calculate_fee(trade_value, 'maker')
    taker_fee = simulator._calculate_fee(trade_value, 'taker')

    assert maker_fee > 0
    assert taker_fee > 0
    assert taker_fee >= maker_fee  # Taker fees are usually higher

def test_portfolio_status(mock_db_manager, config):
    """Test portfolio status reporting"""
    simulator = TradingSimulator(mock_db_manager, config)

    status = simulator.get_portfolio_status()

    assert 'current_balance' in status
    assert 'total_pnl' in status
    assert 'total_trades' in status
    assert 'win_rate' in status
    assert 'open_positions' in status

    assert status['current_balance'] == simulator.initial_balance
    assert status['total_trades'] == 0
