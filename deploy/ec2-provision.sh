#!/bin/bash
# EC2 Provision Script for Trading AI System
# Run this script on a fresh Ubuntu EC2 instance

set -e

echo "🚀 Starting EC2 provision for Trading AI System..."

# Update system
sudo apt-get update -y
sudo apt-get upgrade -y

# Install required packages
sudo apt-get install -y \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv \
    nginx \
    htop \
    unzip

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
rm get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Create application directory
sudo mkdir -p /opt/trade-ai
sudo chown ubuntu:ubuntu /opt/trade-ai

# Create data directories
mkdir -p /opt/trade-ai/{data,logs,models}

echo "📁 Created application directories"

# Clone repository (if using git - modify as needed)
# git clone YOUR_REPO_URL /opt/trade-ai

echo "📦 Copy your trade-ai application files to /opt/trade-ai/"
echo "   You can use: scp -r trade-ai/* ubuntu@YOUR-EC2-IP:/opt/trade-ai/"

# Create environment file
cat > /opt/trade-ai/.env << EOF
# Production Environment Configuration
FLASK_ENV=production
FLASK_DEBUG=false
SECRET_KEY=$(openssl rand -hex 32)

# Database Configuration
DATABASE_URL=sqlite:///data/trade_ai.db

# Trading Configuration
TRADING_PAIR=BTCUSDT
TRADE_AMOUNT_USDT=100.0
RISK_PERCENT=0.01
MAX_POSITIONS=5
COOLDOWN_SECONDS=10

# Model Configuration
MODEL_TYPE=xgboost
RETRAIN_INTERVAL_SECONDS=3600
MIN_NEW_ROWS_FOR_RETRAIN=100
FEATURE_WINDOW_SIZE=20

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/trade_ai.log

# Dashboard Authentication
DASHBOARD_USERNAME=admin
DASHBOARD_PASSWORD=$(openssl rand -base64 12)
EOF

echo "🔐 Created environment configuration with secure passwords"

# Install systemd service
sudo cp /opt/trade-ai/deploy/systemd/trade-ai.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable trade-ai.service

echo "⚙️ Installed systemd service"

# Configure nginx (optional - for reverse proxy)
sudo tee /etc/nginx/sites-available/trade-ai << EOF
server {
    listen 80;
    server_name localhost;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location /socket.io/ {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/trade-ai /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo systemctl enable nginx
sudo systemctl restart nginx

echo "🌐 Configured nginx reverse proxy"

# Set up log rotation
sudo tee /etc/logrotate.d/trade-ai << EOF
/opt/trade-ai/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    copytruncate
    notifempty
}
EOF

echo "📝 Configured log rotation"

# Create backup script
sudo tee /opt/trade-ai/backup.sh << EOF
#!/bin/bash
# Backup script for Trading AI System

BACKUP_DIR="/opt/trade-ai/backups"
DATE=\$(date +%Y%m%d_%H%M%S)

mkdir -p \$BACKUP_DIR

# Backup database
cp /opt/trade-ai/data/trade_ai.db \$BACKUP_DIR/trade_ai_\$DATE.db

# Backup models
tar -czf \$BACKUP_DIR/models_\$DATE.tar.gz -C /opt/trade-ai models/

# Keep only last 7 days of backups
find \$BACKUP_DIR -name "*.db" -mtime +7 -delete
find \$BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

echo "Backup completed: \$DATE"
EOF

chmod +x /opt/trade-ai/backup.sh

# Add backup to cron (daily at 2 AM)
echo "0 2 * * * /opt/trade-ai/backup.sh >> /opt/trade-ai/logs/backup.log 2>&1" | sudo crontab -

echo "💾 Configured automated backups"

# Security - Basic firewall
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw --force enable

echo "🔒 Configured basic firewall"

echo ""
echo "✅ EC2 provisioning completed!"
echo ""
echo "Next steps:"
echo "1. Copy your trade-ai application files to /opt/trade-ai/"
echo "2. Start the service: sudo systemctl start trade-ai"
echo "3. Check status: sudo systemctl status trade-ai"
echo "4. View logs: sudo journalctl -u trade-ai -f"
echo "5. Access dashboard at: http://YOUR-EC2-PUBLIC-IP/"
echo ""
echo "Dashboard credentials:"
cat /opt/trade-ai/.env | grep DASHBOARD_
echo ""
echo "🎉 Your Trading AI system is ready!"
