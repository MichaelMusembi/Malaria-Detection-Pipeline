#!/bin/bash
# 🚀 Complete Deployment Script for Malaria Detection System
# Builds and tests both API and UI for production deployment

echo "🦠 Malaria Detection System - Deployment Setup"
echo "==============================================="

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker to proceed."
    exit 1
fi

echo "🐳 Building Docker containers..."

# Build API container
echo "📡 Building API container..."
docker build -f docker/Dockerfile -t malaria-api .

# Build UI container  
echo "🎨 Building UI container..."
docker build -f docker/Dockerfile.ui -t malaria-ui .

echo "✅ Containers built successfully!"

# Test containers locally
echo ""
echo "🧪 Testing containers locally..."

# Start API container in background
echo "🚀 Starting API container on port 8000..."
docker run -d --name malaria-api-test -p 8000:8000 malaria-api

# Wait for API to start
sleep 15

# Test API health
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ API container is healthy"
else
    echo "❌ API container health check failed"
    docker logs malaria-api-test
fi

# Start UI container in background
echo "🎨 Starting UI container on port 8501..."
docker run -d --name malaria-ui-test -p 8501:8501 -e API_BASE_URL=http://localhost:8000 malaria-ui

# Wait for UI to start
sleep 20

# Test UI health
if curl -s http://localhost:8501/_stcore/health > /dev/null; then
    echo "✅ UI container is healthy"
else
    echo "❌ UI container health check failed"
    docker logs malaria-ui-test
fi

echo ""
echo "🎉 Deployment test complete!"
echo ""
echo "🌐 Access your application:"
echo "   API:  http://localhost:8000"
echo "   UI:   http://localhost:8501"
echo ""
echo "🧹 To clean up test containers:"
echo "   docker stop malaria-api-test malaria-ui-test"
echo "   docker rm malaria-api-test malaria-ui-test"
echo ""
echo "☁️ Ready for cloud deployment on Render:"
echo "   1. Push to GitHub"
echo "   2. Connect Render to your GitHub repository"
echo "   3. Render will automatically use render.yaml"
echo "   4. Both API and UI will be deployed with public URLs"
