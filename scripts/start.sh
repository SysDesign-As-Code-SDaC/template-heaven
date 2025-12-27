#!/bin/bash
# Template Heaven API Startup Script

set -e

echo "🚀 Starting Template Heaven API..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Copying from env.example..."
    cp env.example .env
    echo "📝 Please edit .env file with your configuration before running again."
    exit 1
fi

# Create necessary directories
mkdir -p data logs templates stacks

# Start services with Docker Compose
echo "🐳 Starting services with Docker Compose..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check API health
echo "🔍 Checking API health..."
curl -f http://localhost:8000/api/v1/health || {
    echo "❌ API health check failed"
    docker-compose logs templateheaven
    exit 1
}

echo "✅ Template Heaven API is ready!"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🔍 Health Check: http://localhost:8000/api/v1/health"
