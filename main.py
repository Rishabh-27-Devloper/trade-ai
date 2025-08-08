#!/usr/bin/env python3
"""
Production-Grade Trading AI System
Main entry point for the application
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trade_ai.log'),
        logging.StreamHandler()
    ]
)

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
            logger.info("🚀 Initializing Trading AI Application...")

            # Initialize database
            self.db = init_db(self.config.DATABASE_URL)
            create_tables(self.db)
            logger.info("✅ Database initialized")

            # Create Flask app
            self.flask_app, self.socketio = create_app(self.config)
            logger.info("✅ Flask application created")

            # Initialize data ingester
            self.ingester = DataIngester(self.db, self.config)
            logger.info("✅ Data ingester initialized")

            # Initialize model trainer
            self.trainer = ModelTrainer(self.db, self.config)
            logger.info("✅ Model trainer initialized")

            # Initialize trading simulator
            self.simulator = TradingSimulator(self.db, self.config)
            self.simulator.set_trainer(self.trainer)  # Connect trainer to simulator
            logger.info("✅ Trading simulator initialized")

            # Connect components to Flask app for API access
            self.flask_app.config['ingester'] = self.ingester
            self.flask_app.config['trainer'] = self.trainer
            self.flask_app.config['simulator'] = self.simulator

            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize application: {e}")
            return False

    def start_background_services(self):
        """Start all background services"""
        logger.info("🔧 Starting background services...")

        # Start data ingestion
        ingestion_thread = Thread(target=self.ingester.run_continuously, daemon=True)
        ingestion_thread.start()
        self.threads.append(ingestion_thread)
        logger.info("✅ Data ingestion service started")

        # Start model training
        training_thread = Thread(target=self.trainer.run_continuously, daemon=True)
        training_thread.start() 
        self.threads.append(training_thread)
        logger.info("✅ Model training service started")

        # Start trading simulation
        simulation_thread = Thread(target=self.simulator.run_continuously, daemon=True)
        simulation_thread.start()
        self.threads.append(simulation_thread)
        logger.info("✅ Trading simulation service started")

    def run(self):
        """Run the complete application"""
        if not self.initialize():
            logger.error("❌ Failed to initialize application, exiting...")
            sys.exit(1)

        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # Start background services
        self.start_background_services()

        # Start Flask app
        self.running = True
        logger.info("🌐 Starting Flask web server...")
        logger.info(f"📊 Dashboard will be available at http://localhost:5000")
        logger.info(f"🔐 Dashboard credentials: {self.config.DASHBOARD_USERNAME} / {self.config.DASHBOARD_PASSWORD}")

        try:
            self.socketio.run(
                self.flask_app,
                host='0.0.0.0',
                port=5000,
                debug=self.config.DEBUG,
                use_reloader=False  # Disable reloader to prevent APScheduler double execution
            )
        except Exception as e:
            logger.error(f"❌ Flask server error: {e}")
        finally:
            self.shutdown()

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"🛑 Received signal {signum}, shutting down...")
        self.shutdown()

    def shutdown(self):
        """Gracefully shutdown all services"""
        if not self.running:
            return

        self.running = False
        logger.info("🔄 Shutting down services...")

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

        logger.info("✅ Application shutdown complete")
        sys.exit(0)

def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("🤖 Production-Grade Trading AI System")
    logger.info("=" * 60)

    # Create and run the application
    app = TradingAIApplication()
    app.run()

if __name__ == "__main__":
    main()