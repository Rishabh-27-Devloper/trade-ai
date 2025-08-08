"""
Flask API and dashboard for Trading AI System
Provides REST API endpoints and real-time web dashboard
"""
import json
import logging
from datetime import datetime, timezone
from functools import wraps
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_socketio import SocketIO, emit
import threading
import time, os

from app.config import Config
from app.db import get_db_session, SystemLog, MarketData, Features, ModelCheckpoint, SimulatedTrade

logger = logging.getLogger(__name__)

def create_app(config):
    """Create and configure Flask application"""
    template_dir = os.path.join(os.path.dirname(__file__), 'ui', 'templates')
    static_dir = os.path.join(os.path.dirname(__file__), 'ui', 'static')
    app = Flask(__name__,template_folder=template_dir)
    app.config['SECRET_KEY'] = config.SECRET_KEY
    app.config['DEBUG'] = config.DEBUG

    # Initialize SocketIO
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

    # Store references for real-time updates
    app.config['ingester'] = None
    app.config['trainer'] = None
    app.config['simulator'] = None
    app.config['config'] = config

    def require_auth(f):
        """Simple authentication decorator"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            auth = request.authorization
            if not auth or not (auth.username == config.DASHBOARD_USERNAME and 
                               auth.password == config.DASHBOARD_PASSWORD):
                return ('Authentication required', 401, 
                       {'WWW-Authenticate': 'Basic realm="Trading AI Dashboard"'})
            return f(*args, **kwargs)
        return decorated_function

    # Routes
    @app.route('/')
    def index():
        """Main dashboard"""
        return render_template('dashboard.html', config=config.to_dict())

    @app.route('/api/status')
    def api_status():
        """Get system status"""
        try:
            status = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'system': {
                    'running': True,
                    'version': '1.0.0',
                    'environment': config.ENV
                },
                'ingester': _get_ingester_status(app),
                'trainer': _get_trainer_status(app),
                'simulator': _get_simulator_status(app),
                'database': _get_database_status()
            }
            return jsonify(status)
        except Exception as e:
            logger.error(f"❌ Error getting system status: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/portfolio')
    def api_portfolio():
        """Get portfolio status"""
        try:
            simulator = app.config.get('simulator')
            if simulator:
                portfolio = simulator.get_portfolio_status()
                return jsonify(portfolio)
            else:
                return jsonify({'error': 'Simulator not available'}), 503
        except Exception as e:
            logger.error(f"❌ Error getting portfolio: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/trades')
    def api_trades():
        """Get recent trades"""
        try:
            limit = request.args.get('limit', 20, type=int)
            simulator = app.config.get('simulator')

            if simulator:
                trades = simulator.get_recent_trades(limit)
                return jsonify(trades)
            else:
                return jsonify([])
        except Exception as e:
            logger.error(f"❌ Error getting trades: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/open-trades', methods=['GET'])
    def get_open_trades():
        sim = app.config['simulator']
        trades = sim.open_positions        
        return jsonify(trades)

    @app.route('/api/close-trade/<int:trade_id>', methods=['GET','POST'])
    def close_trade_api(trade_id):
        sim = app.config['simulator']

        # Find position in open_positions
        position = next((p for p in sim.open_positions if p['id'] == trade_id), None)
        if not position:
            return jsonify({"error": "Trade not found or already closed"}), 404

        current_price = _get_latest_price()
        sim._close_position(position, reason="Manual Close", exit_price=current_price)

        return jsonify({"success": True, "trade_id": trade_id})

    @app.route('/api/market-data')
    def api_market_data():
        """Get recent market data"""
        try:
            limit = request.args.get('limit', 100, type=int)
            session = get_db_session()

            market_data = session.query(MarketData).filter(
                MarketData.symbol == config.TRADING_PAIR
            ).order_by(MarketData.timestamp.desc()).limit(limit)

            data = []
            for record in market_data:
                data.append({
                    'timestamp': record.timestamp.isoformat(),
                    'open': record.open_price,
                    'high': record.high_price,
                    'low': record.low_price,
                    'close': record.close_price,
                    'volume': record.volume
                })

            session.close()
            return jsonify(data)

        except Exception as e:
            logger.error(f"❌ Error getting market data: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/model/info')
    def api_model_info():
        """Get model information"""
        try:
            trainer = app.config.get('trainer')
            if trainer:
                model_status = trainer.get_model_status()
                return jsonify(model_status)
            else:
                return jsonify({'error': 'Trainer not available'}), 503
        except Exception as e:
            logger.error(f"❌ Error getting model info: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/model/retrain', methods=['POST'])
    @require_auth
    def api_retrain_model():
        """Manually trigger model retraining"""
        try:
            trainer = app.config.get('trainer')
            if not trainer:
                return jsonify({'error': 'Trainer not available'}), 503

            # Trigger retraining in background
            def retrain_background():
                success = trainer.manual_retrain()
                logger.info(f"Manual retrain result: {success}")

            threading.Thread(target=retrain_background, daemon=True).start()

            return jsonify({'message': 'Retraining started', 'status': 'started'})

        except Exception as e:
            logger.error(f"❌ Error triggering retrain: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/trade/manual', methods=['POST'])
    @require_auth
    def api_manual_trade():
        """Manually trigger a trade"""
        try:
            data = request.get_json()
            direction = data.get('direction', 'long')

            if direction not in ['long', 'short']:
                return jsonify({'error': 'Invalid direction'}), 400

            simulator = app.config.get('simulator')
            if not simulator:
                return jsonify({'error': 'Simulator not available'}), 503

            success = simulator.manual_trade(direction, force=False)

            if success:
                return jsonify({'message': f'Manual {direction} trade executed', 'status': 'success'})
            else:
                return jsonify({'error': 'Trade execution failed'}), 400

        except Exception as e:
            logger.error(f"❌ Error executing manual trade: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/logs')
    def api_logs():
        """Get recent system logs"""
        try:
            limit = request.args.get('limit', 50, type=int)
            level = request.args.get('level', None)

            session = get_db_session()
            query = session.query(SystemLog)

            if level:
                query = query.filter(SystemLog.level == level.upper())

            logs = query.order_by(SystemLog.timestamp.desc()).limit(limit)

            log_data = []
            for log in logs:
                log_data.append({
                    'timestamp': log.timestamp.isoformat(),
                    'level': log.level,
                    'component': log.component,
                    'message': log.message,
                    'data': log.data
                })

            session.close()
            return jsonify(log_data)

        except Exception as e:
            logger.error(f"❌ Error getting logs: {e}")
            return jsonify({'error': str(e)}), 500

    # WebSocket events
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection"""
        logger.info("🔗 Dashboard client connected")
        emit('status', {'message': 'Connected to Trading AI System'})

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection"""
        logger.info("🔌 Dashboard client disconnected")

    @socketio.on('request_update')
    def handle_update_request():
        """Handle update request from client"""
        try:
            # Send current status
            status_data = _get_system_status(app)
            emit('status_update', status_data)

            # Send portfolio data
            simulator = app.config.get('simulator')
            if simulator:
                portfolio_data = simulator.get_portfolio_status()
                emit('portfolio_update', portfolio_data)

        except Exception as e:
            logger.error(f"❌ Error handling update request: {e}")

    # Start background task for real-time updates
    def background_updates():
        """Send periodic updates to connected clients"""
        while True:
            try:
                with app.app_context():
                    # Send system status every 30 seconds
                    status_data = _get_system_status(app)
                    socketio.emit('status_update', status_data)

                    # Send portfolio updates every 10 seconds
                    simulator = app.config.get('simulator')
                    if simulator:
                        portfolio_data = simulator.get_portfolio_status()
                        socketio.emit('portfolio_update', portfolio_data)

                    # Send latest price every 5 seconds
                    latest_price = _get_latest_price()
                    if latest_price:
                        socketio.emit('price_update', {'price': latest_price, 'timestamp': datetime.now().isoformat()})

                time.sleep(10)  # Update every 10 seconds

            except Exception as e:
                logger.error(f"❌ Error in background updates: {e}")
                time.sleep(30)

    # Start background thread
    update_thread = threading.Thread(target=background_updates, daemon=True)
    update_thread.start()

    return app, socketio

def _get_ingester_status(app):
    """Get ingester status"""
    try:
        ingester = app.config.get('ingester')
        if ingester:
            return ingester.get_status()
        return {'running': False, 'error': 'Not initialized'}
    except Exception as e:
        return {'running': False, 'error': str(e)}

def _get_trainer_status(app):
    """Get trainer status"""
    try:
        trainer = app.config.get('trainer')
        if trainer:
            return trainer.get_model_status()
        return {'running': False, 'error': 'Not initialized'}
    except Exception as e:
        return {'running': False, 'error': str(e)}

def _get_simulator_status(app):
    """Get simulator status"""
    try:
        simulator = app.config.get('simulator')
        if simulator:
            return {
                'running': simulator.running,
                'open_positions': len(simulator.open_positions),
                'total_trades': simulator.total_trades,
                'balance': simulator.current_balance
            }
        return {'running': False, 'error': 'Not initialized'}
    except Exception as e:
        return {'running': False, 'error': str(e)}

def _get_database_status():
    """Get database status"""
    try:
        session = get_db_session()

        # Count records in key tables
        market_data_count = session.query(MarketData).count()
        features_count = session.query(Features).count()
        trades_count = session.query(SimulatedTrade).count()

        session.close()

        return {
            'connected': True,
            'market_data_records': market_data_count,
            'feature_records': features_count,
            'trade_records': trades_count
        }
    except Exception as e:
        return {'connected': False, 'error': str(e)}

def _get_system_status(app):
    """Get complete system status"""
    return {
        'ingester': _get_ingester_status(app),
        'trainer': _get_trainer_status(app),
        'simulator': _get_simulator_status(app),
        'database': _get_database_status(),
        'timestamp': datetime.now().isoformat()
    }

def _get_latest_price():
    """Get latest market price"""
    try:
        session = get_db_session()
        latest = session.query(MarketData).order_by(MarketData.timestamp.desc()).first()
        session.close()

        if latest:
            return latest.close_price
        return None
    except Exception as e:
        logger.error(f"❌ Error getting latest price: {e}")
        return None
