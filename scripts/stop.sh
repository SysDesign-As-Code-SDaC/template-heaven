#!/bin/bash
# Template Heaven API Shutdown Script

echo "🛑 Stopping Template Heaven API..."

# Stop all services
docker-compose down

echo "✅ Template Heaven API stopped successfully!"
