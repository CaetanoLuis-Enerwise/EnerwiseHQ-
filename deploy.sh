#!/bin/bash

# Enerwise Grid AI Agent - Deployment Script
# Production deployment automation

set -e  # Exit on error

echo "Starting Enerwise Grid AI Agent deployment..."

# Configuration
APP_NAME="enerwise-grid-ai"
IMAGE_NAME="enerwise/grid-ai-agent"
VERSION="2.0.0"
PORT=8000

# Build Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME:$VERSION -t $IMAGE_NAME:latest .

# Test the image
echo "Testing the image..."
docker run -d --name ${APP_NAME}-test -p $PORT:8000 $IMAGE_NAME:latest

# Wait for service to start
echo "Waiting for service to start..."
sleep 10

# Health check
HEALTH_CHECK=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/health || true)

if [ "$HEALTH_CHECK" = "200" ]; then
    echo "Health check passed: Service is running correctly"
    
    # Stop test container
    docker stop ${APP_NAME}-test
    docker rm ${APP_NAME}-test
    
    # Deployment commands for different environments
    echo ""
    echo "Deployment successful!"
    echo "Image: $IMAGE_NAME:$VERSION"
    echo ""
    echo "To deploy to production:"
    echo "1. Push to container registry:"
    echo "   docker push $IMAGE_NAME:$VERSION"
    echo "   docker push $IMAGE_NAME:latest"
    echo ""
    echo "2. Deploy to Kubernetes:"
    echo "   kubectl set image deployment/enerwise-grid-ai grid-ai-agent=$IMAGE_NAME:$VERSION"
    echo ""
    echo "3. Or deploy with Docker Compose:"
    echo "   docker-compose up -d"
    
else
    echo "Health check failed: HTTP $HEALTH_CHECK"
    echo "Stopping test container..."
    docker stop ${APP_NAME}-test
    docker rm ${APP_NAME}-test
    exit 1
fi

echo "Deployment script completed successfully."
