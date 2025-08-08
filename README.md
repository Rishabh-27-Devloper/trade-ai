# Production-Grade Trading AI System 🤖

A complete, production-ready trading AI system that continuously ingests Binance market data, trains ML models, and executes simulated trades with comprehensive performance tracking.

## 🚀 Features

### Core Functionality
- **Real-time Data Ingestion**: Binance WebSocket streams with automatic reconnection
- **Advanced ML Models**: XGBoost, LSTM, and baseline models with risk-reward optimization
- **Trading Simulation**: Realistic P&L calculation with slippage, fees, and risk management
- **Live Dashboard**: Real-time web interface with charts, metrics, and controls
- **Production Deployment**: Docker containers with systemd service management

### Technical Highlights
- **Concurrent SQLite**: WAL mode for high-performance concurrent access
- **Continuous Training**: Automated model retraining with performance tracking
- **Risk Management**: Stop-loss, take-profit, position sizing, and drawdown controls
- **Real-time Updates**: WebSocket-based dashboard with live data streaming
- **Comprehensive Logging**: Structured logging with database storage and rotation

## 📊 Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Ingester  │───▶│   Database      │◀───│ Feature Engine  │
│  (WebSocket)    │    │   (SQLite WAL)  │    │ (Tech Analysis) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                         │
┌─────────────────┐           │                         ▼
│  Flask Dashboard│◀──────────┼─────────────┐    ┌─────────────────┐
│  (Real-time UI) │           │             │    │  Model Trainer  │
└─────────────────┘           │             │    │  (XGBoost/LSTM) │
                              ▼             │    └─────────────────┘
                       ┌─────────────────┐  │             │
                       │ Trade Simulator │◀─┘             │
                       │ (Risk/Reward)   │◀───────────────┘
                       └─────────────────┘
```

## 🛠 Installation & Setup

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- 4GB+ RAM (8GB recommended)
- Modern web browser

### Local Development

1. **Clone and setup**:
```bash
git clone https://github.com/Rishabh-27-Devloper/trade-ai.git
cd trade-ai
cp .env.example .env
```

2. **Configure environment**:
Edit `.env` file with your preferences:
```bash
TRADING_PAIR=BTCUSDT
TRADE_AMOUNT_USDT=100.0
DASHBOARD_USERNAME=admin
DASHBOARD_PASSWORD=your-secure-password
```

3. **Run with Docker**:
```bash
docker-compose up --build
```

4. **Access dashboard**:
Open http://localhost:5000 in your browser

### Production Deployment (AWS EC2)

1. **Launch EC2 instance**:
   - Ubuntu 22.04 LTS
   - t3.medium or larger
   - Security group: ports 22, 80, 443

2. **Run provision script**:
```bash
# Copy files to EC2
scp -r trade-ai/* ubuntu@YOUR-EC2-IP:/opt/trade-ai/

# SSH into EC2 and run provision
ssh ubuntu@YOUR-EC2-IP
sudo /opt/trade-ai/deploy/ec2-provision.sh
```

3. **Start service**:
```bash
sudo systemctl start trade-ai
sudo systemctl status trade-ai
```

4. **Monitor logs**:
```bash
sudo journalctl -u trade-ai -f
```

## 📱 Dashboard Features

### Real-time Monitoring
- **Portfolio Status**: Balance, P&L, returns, drawdown
- **Trading Stats**: Win rate, total trades, open positions
- **System Health**: All components status and metrics
- **Live Price Chart**: Real-time price visualization

### Interactive Controls
- **Manual Trading**: Execute long/short trades instantly
- **Model Retraining**: Trigger ML model updates
- **System Logs**: Real-time log streaming
- **Performance Metrics**: Sharpe ratio, Sortino ratio, etc.

## 🧠 ML Models

### XGBoost Model (Default)
- **Objective**: Binary classification (long/short signals)
- **Features**: 20+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Optimization**: Risk-reward function with drawdown penalties
- **Training**: Continuous retraining with time-series cross-validation

### Technical Indicators
- Moving Averages (SMA, EMA)
- Momentum (RSI, MACD, Stochastic)
- Volatility (Bollinger Bands, ATR)
- Volume (OBV, Volume SMA)
- Time-based features

### Risk-Reward Objective
```python
reward = mean_return - λ * max_drawdown - γ * volatility
```

## 💰 Trading Simulation

### Realistic Market Conditions
- **Slippage**: 0.05% (configurable)
- **Fees**: 0.1% maker/taker fees
- **Risk Management**: 2% stop-loss, 4% take-profit (2:1 R:R)
- **Position Sizing**: Fixed USD amount with risk controls

### Performance Tracking
- Real-time P&L calculation
- Win rate and return statistics
- Maximum drawdown monitoring
- Sharpe and Sortino ratios
- Trade-by-trade analysis

## 🗄️ Database Schema

### Key Tables
- **market_data**: OHLCV price data
- **features**: Technical indicators and labels
- **model_checkpoints**: Trained model versions
- **simulated_trades**: Complete trading history
- **system_logs**: Application events and errors

### Concurrent Access
- SQLite WAL mode for multi-reader/single-writer
- Connection pooling and timeout handling
- Automatic backup and log rotation

## 🔧 Configuration

### Environment Variables
```bash
# Trading Settings
TRADING_PAIR=BTCUSDT
TRADE_AMOUNT_USDT=100.0
RISK_PERCENT=0.01
MAX_POSITIONS=5

# Model Settings
MODEL_TYPE=xgboost
RETRAIN_INTERVAL_SECONDS=3600
FEATURE_WINDOW_SIZE=20

# Security
DASHBOARD_USERNAME=admin
DASHBOARD_PASSWORD=secure-password
```

### Model Parameters
```python
# XGBoost hyperparameters
params = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0
}
```

## 🚨 Risk Management

### Position Management
- Maximum 5 concurrent positions (configurable)
- 1% risk per trade (of total balance)
- 24-hour maximum holding period
- Cooldown between trades

### Safety Features
- Stop-loss: 2% from entry price
- Take-profit: 4% from entry price (2:1 reward:risk)
- Maximum drawdown: 10% portfolio protection
- Circuit breakers for extreme market conditions

## 📊 Performance Metrics

### Trading Metrics
- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Win Rate**: Percentage of profitable trades

### System Metrics
- Data ingestion rate and health
- Model prediction accuracy
- WebSocket connection stability
- Database performance and size

## 🧪 Testing

### Run Tests
```bash
# Unit tests
pytest tests/

# Coverage report
pytest --cov=app tests/

# Integration tests
pytest tests/test_integration.py -v
```

### Test Coverage
- Data ingestion and WebSocket handling
- Feature generation and technical indicators
- ML model training and prediction
- Trading simulation and risk management
- API endpoints and dashboard functionality

## 🔍 Monitoring & Maintenance

### Health Checks
- `/api/status`: Complete system status
- WebSocket connection monitoring
- Database connectivity checks
- Model performance tracking

### Log Management
```bash
# View real-time logs
sudo journalctl -u trade-ai -f

# Application logs
tail -f /opt/trade-ai/logs/trade_ai.log

# Database logs
sqlite3 /opt/trade-ai/data/trade_ai.db "SELECT * FROM system_logs ORDER BY timestamp DESC LIMIT 20;"
```

### Backup & Recovery
- Automated daily database backups
- Model checkpoint versioning
- Configuration backup
- AWS S3 integration (optional)

## 🔮 Scaling & Extensions

### Horizontal Scaling
1. **Replace SQLite with PostgreSQL**
   - Update DATABASE_URL
   - Modify connection settings
   - Test concurrent performance

2. **Add Message Queue (Redis/RabbitMQ)**
   - Separate data ingestion
   - Distribute model training
   - Scale trading logic

3. **Container Orchestration (ECS/EKS)**
   - Multi-instance deployment
   - Load balancing
   - Auto-scaling based on metrics

### Feature Extensions
- Multiple trading pairs
- Live trading with real funds
- Advanced ML models (Transformers, RL)
- Social sentiment analysis
- Options and derivatives trading

## 🔐 Security Considerations

### Production Hardening
- Change default credentials
- Enable HTTPS with Let's Encrypt
- Implement rate limiting
- Add input validation
- Use AWS Secrets Manager
- Enable fail2ban

### Network Security
```bash
# Basic firewall setup
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

## 📈 Performance Optimization

### Database Optimization
- SQLite PRAGMA settings tuned for performance
- Index optimization for time-series queries
- Regular VACUUM and ANALYZE operations

### Model Optimization
- Feature selection and dimensionality reduction
- Hyperparameter tuning with Optuna
- Model ensemble techniques
- GPU acceleration for deep learning

## 🚀 Quick Start Checklist

- [ ] Clone repository and review code
- [ ] Configure `.env` with your settings
- [ ] Run `docker-compose up --build`
- [ ] Access dashboard at http://localhost:5000
- [ ] Monitor logs and system status
- [ ] Review trading performance
- [ ] Customize models and strategies
- [ ] Deploy to production when ready

## 🆘 Troubleshooting

### Common Issues

**WebSocket disconnections:**
```bash
# Check network connectivity
curl -I https://stream.binance.com

# Review connection logs
grep "WebSocket" /opt/trade-ai/logs/trade_ai.log
```

**Model training failures:**
```bash
# Check feature data
sqlite3 trade_ai.db "SELECT COUNT(*) FROM features;"

# Review training logs
grep "training" /opt/trade-ai/logs/trade_ai.log
```

**Dashboard not loading:**
```bash
# Check Flask service
sudo systemctl status trade-ai

# Test API endpoints
curl http://localhost:5000/api/status
```

## 📞 Support & Contributing

### Getting Help
- Review logs for error details
- Check system status endpoint
- Verify configuration settings
- Monitor resource usage (CPU, memory, disk)

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request
5. Follow code style guidelines

## 📄 License

MIT License - see LICENSE file for details.

## ⚠️ Disclaimer

This system is for educational and research purposes. Trading involves significant financial risk. Past performance does not guarantee future results. Always thoroughly test strategies before deploying with real funds.

---

**Built with ❤️ for the trading and AI community**

*For questions, issues, or feature requests, please open a GitHub issue or contact the maintainers.*
