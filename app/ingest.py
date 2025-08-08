"""
Data ingestion module for Trading AI System
Handles Binance WebSocket streams and REST API data collection
"""
import json
import time
import logging
import requests
import threading
from datetime import datetime, timezone, timedelta
from websocket import WebSocketApp
from sqlalchemy.exc import SQLAlchemyError

from app.db import get_db_session, MarketData, log_system_event

logger = logging.getLogger(__name__)

class BinanceDataIngester:
    """Binance data ingestion via WebSocket and REST API"""

    def __init__(self, db_manager, config):
        self.db_manager = db_manager
        self.config = config
        self.ws = None
        self.running = False
        self.reconnect_attempts = 0
        self.last_ping = time.time()
        self.ws_connected = False  # Track connection state explicitly

        # Binance endpoints
        self.rest_base_url = 'https://api.binance.com/api/v3'
        self.ws_url = f"{config.BINANCE_WS_URL}/{config.TRADING_PAIR.lower()}@kline_1m"

        # Data buffer for batch inserts
        self.data_buffer = []
        self.buffer_lock = threading.Lock()
        self.max_buffer_size = 100

    def start(self):
        """Start data ingestion"""
        logger.info(f"Starting data ingestion for {self.config.TRADING_PAIR}")
        self.running = True

        # Fetch historical data first
        self._fetch_historical_data()

        # Start WebSocket connection
        self._connect_websocket()

    def stop(self):
        """Stop data ingestion"""
        logger.info("Stopping data ingestion...")
        self.running = False
        self.ws_connected = False

        if self.ws:
            self.ws.close()

        # Flush remaining buffer
        self._flush_buffer()

        log_system_event('INFO', 'ingester', 'Data ingestion stopped')

    def _fetch_historical_data(self, limit=1000):
        """Fetch historical kline data from REST API"""
        try:
            logger.info(f"Fetching historical data for {self.config.TRADING_PAIR}...")

            url = f"{self.rest_base_url}/klines"
            params = {
                'symbol': self.config.TRADING_PAIR,
                'interval': '1m',
                'limit': limit
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            klines = response.json()
            logger.info(f"Received {len(klines)} historical candles")

            # Process and store historical data
            session = get_db_session()
            stored_count = 0

            for kline in klines:
                market_data = self._parse_kline(kline, is_historical=True)
                if market_data and not self._data_exists(session, market_data.timestamp, market_data.symbol):
                    session.add(market_data)
                    stored_count += 1

            session.commit()
            session.close()

            logger.info(f"Stored {stored_count} new historical data points")
            log_system_event('INFO', 'ingester', f'Historical data loaded: {stored_count} points')

        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            log_system_event('ERROR', 'ingester', f'Historical data fetch failed: {e}')

    def _connect_websocket(self):
        """Connect to Binance WebSocket"""
        try:
            logger.info(f"Connecting to WebSocket: {self.ws_url}")

            self.ws = WebSocketApp(
                self.ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_ping=self._on_ping,
                on_pong=self._on_pong
            )

            # Start WebSocket in a separate thread
            ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
            ws_thread.start()

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self._schedule_reconnect()

    def _on_open(self, ws):
        """WebSocket connection opened"""
        logger.info("WebSocket connected")
        self.ws_connected = True
        self.reconnect_attempts = 0
        log_system_event('INFO', 'ingester', 'WebSocket connected')

    def _on_message(self, ws, message):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)

            if 'k' in data:  # Kline data
                kline_data = data['k']

                # Only process closed candles for consistency
                if kline_data['x']:  # x=true means kline is closed
                    market_data = self._parse_kline_ws(kline_data)
                    if market_data:
                        self._buffer_data(market_data)

            # Update last activity
            self.last_ping = time.time()

        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")

    def _on_error(self, ws, error):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {error}")
        self.ws_connected = False
        log_system_event('ERROR', 'ingester', f'WebSocket error: {error}')
        self._schedule_reconnect()

    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        logger.warning(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.ws_connected = False
        log_system_event('WARNING', 'ingester', f'WebSocket closed: {close_status_code}')

        if self.running:
            self._schedule_reconnect()

    def _on_ping(self, ws, message):
        """Handle ping from server"""
        self.last_ping = time.time()

    def _on_pong(self, ws, message):
        """Handle pong from server"""
        self.last_ping = time.time()

    def _schedule_reconnect(self):
        """Schedule WebSocket reconnection"""
        if not self.running:
            return

        if self.reconnect_attempts >= self.config.RECONNECTION_ATTEMPTS:
            logger.error("Max reconnection attempts reached")
            log_system_event('ERROR', 'ingester', 'Max WebSocket reconnection attempts reached')
            return

        self.reconnect_attempts += 1
        delay = min(self.config.RECONNECTION_DELAY * self.reconnect_attempts, 300)  # Max 5 min

        logger.info(f"Scheduling reconnection attempt {self.reconnect_attempts} in {delay}s")

        def delayed_reconnect():
            time.sleep(delay)
            if self.running:
                self._connect_websocket()

        threading.Thread(target=delayed_reconnect, daemon=True).start()

    def _parse_kline(self, kline, is_historical=False):
        """Parse kline data from REST API response"""
        try:
            return MarketData(
                timestamp=datetime.fromtimestamp(int(kline[0]) / 1000, tz=timezone.utc),
                symbol=self.config.TRADING_PAIR,
                open_price=float(kline[1]),
                high_price=float(kline[2]),
                low_price=float(kline[3]),
                close_price=float(kline[4]),
                volume=float(kline[5]),
                trades_count=int(kline[8])
            )
        except Exception as e:
            logger.error(f"Error parsing kline: {e}")
            return None

    def _parse_kline_ws(self, kline_data):
        """Parse kline data from WebSocket"""
        try:
            return MarketData(
                timestamp=datetime.fromtimestamp(int(kline_data['t']) / 1000, tz=timezone.utc),
                symbol=kline_data['s'],
                open_price=float(kline_data['o']),
                high_price=float(kline_data['h']),
                low_price=float(kline_data['l']),
                close_price=float(kline_data['c']),
                volume=float(kline_data['v']),
                trades_count=int(kline_data['n'])
            )
        except Exception as e:
            logger.error(f"Error parsing WebSocket kline: {e}")
            return None

    def _buffer_data(self, market_data):
        """Add data to buffer for batch processing"""
        with self.buffer_lock:
            self.data_buffer.append(market_data)

            if len(self.data_buffer) >= self.max_buffer_size:
                self._flush_buffer()

    def _flush_buffer(self):
        """Flush data buffer to database"""
        if not self.data_buffer:
            return

        try:
            session = get_db_session()

            # Add all buffered data
            for data in self.data_buffer:
                if not self._data_exists(session, data.timestamp, data.symbol):
                    session.add(data)

            session.commit()
            session.close()

            logger.debug(f"Flushed {len(self.data_buffer)} data points to database")
            self.data_buffer.clear()

        except Exception as e:
            logger.error(f"Error flushing data buffer: {e}")

    def _data_exists(self, session, timestamp, symbol):
        """Check if data point already exists"""
        try:
            return session.query(MarketData).filter(
                MarketData.timestamp == timestamp,
                MarketData.symbol == symbol
            ).first() is not None
        except Exception:
            return False

    def get_latest_price(self):
        """Get latest price from REST API"""
        try:
            url = f"{self.rest_base_url}/ticker/price"
            params = {'symbol': self.config.TRADING_PAIR}

            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()

            return float(response.json()['price'])
        except Exception as e:
            logger.error(f"Error fetching latest price: {e}")
            return None

    def health_check(self):
        """Check ingester health"""
        now = time.time()
        time_since_ping = now - self.last_ping

        # Check if WebSocket is connected using our explicit flag
        websocket_connected = self.ws_connected and self.ws is not None

        return {
            'running': self.running,
            'websocket_connected': websocket_connected,
            'time_since_last_ping': time_since_ping,
            'reconnect_attempts': self.reconnect_attempts,
            'buffer_size': len(self.data_buffer)
        }

class DataIngester:
    """Main data ingester orchestrator"""

    def __init__(self, db_manager, config):
        self.db_manager = db_manager
        self.config = config
        self.binance_ingester = BinanceDataIngester(db_manager, config)
        self.running = False

        # Periodic tasks
        self.flush_interval = 30  # Flush buffer every 30 seconds
        self.health_check_interval = 60  # Health check every minute

    def run_continuously(self):
        """Run data ingestion continuously"""
        logger.info("Starting continuous data ingestion...")
        self.running = True

        # Start Binance ingester
        self.binance_ingester.start()

        # Periodic maintenance tasks
        last_flush = time.time()
        last_health_check = time.time()

        while self.running:
            try:
                current_time = time.time()

                # Periodic buffer flush
                if current_time - last_flush >= self.flush_interval:
                    self.binance_ingester._flush_buffer()
                    last_flush = current_time

                # Periodic health check
                if current_time - last_health_check >= self.health_check_interval:
                    health = self.binance_ingester.health_check()
                    if not health['websocket_connected'] and self.running:
                        logger.warning("WebSocket disconnected, attempting reconnection...")
                        self.binance_ingester._connect_websocket()
                    last_health_check = current_time

                time.sleep(1)  # Check every second

            except Exception as e:
                logger.error(f"Error in data ingestion loop: {e}")
                time.sleep(5)

    def stop(self):
        """Stop data ingestion"""
        logger.info("Stopping data ingester...")
        self.running = False
        self.binance_ingester.stop()

    def get_status(self):
        """Get ingester status"""
        return self.binance_ingester.health_check()