
"""
Tests for data ingestion module
"""
import pytest
import time
from unittest.mock import Mock, patch
from app.ingest import BinanceDataIngester
from app.config import Config

@pytest.fixture
def config():
    return Config()

@pytest.fixture
def mock_db_manager():
    return Mock()

def test_ingester_initialization(mock_db_manager, config):
    """Test ingester initialization"""
    ingester = BinanceDataIngester(mock_db_manager, config)

    assert ingester.db_manager == mock_db_manager
    assert ingester.config == config
    assert not ingester.running
    assert ingester.reconnect_attempts == 0

def test_health_check(mock_db_manager, config):
    """Test health check functionality"""
    ingester = BinanceDataIngester(mock_db_manager, config)

    health = ingester.health_check()

    assert 'running' in health
    assert 'websocket_connected' in health
    assert 'time_since_last_ping' in health
    assert 'reconnect_attempts' in health
    assert 'buffer_size' in health

@patch('requests.get')
def test_fetch_historical_data(mock_get, mock_db_manager, config):
    """Test historical data fetching"""
    # Mock successful API response
    mock_response = Mock()
    mock_response.json.return_value = [
        [1640995200000, "47000.0", "47500.0", "46800.0", "47200.0", "100.5", 0, 0, 1500]
    ]
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    ingester = BinanceDataIngester(mock_db_manager, config)

    with patch.object(ingester, '_data_exists', return_value=False):
        with patch('app.ingest.get_db_session') as mock_session:
            mock_session.return_value.commit.return_value = None
            mock_session.return_value.close.return_value = None

            ingester._fetch_historical_data(limit=1)

            mock_get.assert_called_once()
            mock_session.return_value.add.assert_called_once()
