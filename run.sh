#!/bin/bash
# Simple run script for Trading AI System

echo "🤖 Starting Trading AI System..."
echo "=================================="

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️ Please edit .env file with your configuration."
fi

# Build and run with Docker Compose
echo "🐳 Building and starting containers..."
docker-compose up --build
