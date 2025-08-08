#!/usr/bin/env python3
"""
Production-Grade Trading AI System
Main entry point with fixed logging configuration
"""
import os
import sys
import signal
import time
import logging
from threading import Thread
from flask import Flask
from flask_socketio import SocketIO

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.config import Config
from app.db import init_db, create_tables
from app.api import create_app
from app.ingest import DataIngester
from app.trainer import ModelTrainer
from app.simulator import TradingSimulator

def setup_logging():
    """Configure logging with UTF-8 support"""
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Custom formatter to handle emojis gracefully
    class SafeFormatter(logging.Formatter):
        def format(self, record):
            # Replace problematic emojis with safe alternatives
            emoji_replacements = {
                '🚀': '[START]',
                '✅': '[OK]',
                '❌': '[ERROR]',
                '⚠️': '[WARN]',
                '🔧': '[CONFIG]',
                '📊': '[DATA]',
                '🧠': '[AI]',
                '💰': '[TRADE]',
                '🔗': '[CONNECT]',
                '🔌': '[DISCONNECT]',
                '📈': '[MARKET]',
                '🛑': '[STOP]',
                '🔄': '[RETRY]',
                '⏳': '[WAIT]',
                '🌐': '[WEB]',
                '🔐': '[AUTH]',
                '📝': '[NOTE]',
                '⚡': '[EXEC]',
                '🏁': '[FINISH]',
                '🤖': '[BOT]'
            }
            
            # Format the message normally first
            formatted = super().format(record)
            
            # Replace emojis with safe alternatives
            for emoji, replacement in emoji_replacements.items():
                formatted = formatted.replace(emoji, replacement)
                
            return formatted

    # Set up formatter
    formatter = SafeFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler(
        os.path.join(log_dir, 'trade_ai.log'),
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Console handler with UTF-8 support
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()  # Clear existing handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Suppress some noisy third-party loggers
    logging.getLogger('websocket').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class TradingAIApplication:
    """Main application orchestrator"""

    def __init__(self):
        self.config = Config()
        self.running = False
        self.threads = []

        # Initialize components
        self.db = None
        self.flask_app = None
        self.socketio = None
        self.ingester = None
        self.trainer = None
        self.simulator = None

    def initialize(self):
        """Initialize all application components"""
        try:
            logger.info("[START] Initializing Trading AI Application...")

            # Initialize database
            self.db = init_db(self.config.DATABASE_URL)
            create_tables(self.db)
            logger.info("[OK] Database initialized")

            # Create Flask app
            self.flask_app, self.socketio = create_app(self.config)
            logger.info("[OK] Flask application created")

            # Initialize data ingester
            self.ingester = DataIngester(self.db, self.config)
            logger.info("[OK] Data ingester initialized")

            # Initialize model trainer
            self.trainer = ModelTrainer(self.db, self.config)
            logger.info("[OK] Model trainer initialized")

            # Initialize trading simulator
            self.simulator = TradingSimulator(self.db, self.config)
            self.simulator.set_trainer(self.trainer)  # Connect trainer to simulator
            logger.info("[OK] Trading simulator initialized")

            # Connect components to Flask app for API access
            self.flask_app.config['ingester'] = self.ingester
            self.flask_app.config['trainer'] = self.trainer
            self.flask_app.config['simulator'] = self.simulator

            return True

        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize application: {e}")
            return False

    def start_background_services(self):
        """Start all background services"""
        logger.info("[CONFIG] Starting background services...")

        # Start data ingestion
        ingestion_thread = Thread(target=self.ingester.run_continuously, daemon=True)
        ingestion_thread.start()
        self.threads.append(ingestion_thread)
        logger.info("[OK] Data ingestion service started")

        # Start model training
        training_thread = Thread(target=self.trainer.run_continuously, daemon=True)
        training_thread.start() 
        self.threads.append(training_thread)
        logger.info("[OK] Model training service started")

        # Start trading simulation
        simulation_thread = Thread(target=self.simulator.run_continuously, daemon=True)
        simulation_thread.start()
        self.threads.append(simulation_thread)
        logger.info("[OK] Trading simulation service started")

    def run(self):
        """Run the complete application"""
        if not self.initialize():
            logger.error("[ERROR] Failed to initialize application, exiting...")
            sys.exit(1)

        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # Start background services
        self.start_background_services()

        # Start Flask app
        self.running = True
        logger.info("[WEB] Starting Flask web server...")
        logger.info(f"[DATA] Dashboard will be available at http://localhost:5000")
        logger.info(f"[AUTH] Dashboard credentials: {self.config.DASHBOARD_USERNAME} / {self.config.DASHBOARD_PASSWORD}")

        try:
            self.socketio.run(
                self.flask_app,
                host='0.0.0.0',
                port=5000,
                debug=self.config.DEBUG,
                use_reloader=False  # Disable reloader to prevent APScheduler double execution
            )
        except Exception as e:
            logger.error(f"[ERROR] Flask server error: {e}")
        finally:
            self.shutdown()

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"[STOP] Received signal {signum}, shutting down...")
        self.shutdown()

    def shutdown(self):
        """Gracefully shutdown all services"""
        if not self.running:
            return

        self.running = False
        logger.info("[RETRY] Shutting down services...")

        # Stop background services
        if self.ingester:
            self.ingester.stop()
        if self.trainer:
            self.trainer.stop()
        if self.simulator:
            self.simulator.stop()

        # Wait for threads to finish
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=5)

        logger.info("[OK] Application shutdown complete")
        sys.exit(0)

def main():
    """Main entry point"""
    # Set up logging first
    setup_logging()
    
    logger.info("=" * 60)
    logger.info("[BOT] Production-Grade Trading AI System")
    logger.info("=" * 60)

    # Create and run the application
    app = TradingAIApplication()
    app.run()

if __name__ == "__main__":
    main()