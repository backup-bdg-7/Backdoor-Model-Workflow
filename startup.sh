#!/bin/bash

# Startup script for AI Model Training Platform
# This script helps start the different services for local development

# Print help information
function print_help {
    echo "AI Model Training Platform Startup Script"
    echo ""
    echo "Usage: ./startup.sh [options]"
    echo ""
    echo "Options:"
    echo "  --api          Start the API service (default port: 8000)"
    echo "  --training     Start the training worker"
    echo "  --monitor      Start the monitoring service (default port: 8081)"
    echo "  --all          Start all services"
    echo "  --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./startup.sh --api              # Start only the API service"
    echo "  ./startup.sh --monitor --port 8082  # Start monitoring service on port 8082"
    echo "  ./startup.sh --all              # Start all services"
    echo ""
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "Error: pip is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source .venv/Scripts/activate
else
    # Unix/macOS
    source .venv/bin/activate
fi

# Install dependencies if not already installed
if ! python -c "import torch" &> /dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Default values
START_API=false
START_TRAINING=false
START_MONITOR=false
API_PORT=8000
MONITOR_PORT=8081
DEBUG=false

# Parse command line arguments
if [ $# -eq 0 ]; then
    # No arguments provided, show help
    print_help
    exit 0
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --api)
            START_API=true
            shift
            ;;
        --training)
            START_TRAINING=true
            shift
            ;;
        --monitor)
            START_MONITOR=true
            shift
            ;;
        --all)
            START_API=true
            START_TRAINING=true
            START_MONITOR=true
            shift
            ;;
        --api-port)
            API_PORT=$2
            shift 2
            ;;
        --monitor-port)
            MONITOR_PORT=$2
            shift 2
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --help)
            print_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_help
            exit 1
            ;;
    esac
done

# Create necessary directories
mkdir -p /tmp/model-trainer
mkdir -p /tmp/model-trainer/models
mkdir -p /tmp/model-trainer/datasets
mkdir -p /tmp/model-trainer/logs
mkdir -p /tmp/model-trainer/exports

# Start services
if [ "$START_API" = true ]; then
    echo "Starting API service on port $API_PORT..."
    if [ "$DEBUG" = true ]; then
        python -m app.api.main --port $API_PORT --debug &
    else
        python -m app.api.main --port $API_PORT &
    fi
    API_PID=$!
    echo "API service started with PID $API_PID"
fi

if [ "$START_TRAINING" = true ]; then
    echo "Starting training worker..."
    python -m app.services.training_worker &
    TRAINING_PID=$!
    echo "Training worker started with PID $TRAINING_PID"
fi

if [ "$START_MONITOR" = true ]; then
    echo "Starting monitoring service on port $MONITOR_PORT..."
    if [ "$DEBUG" = true ]; then
        python -m uvicorn app.services.monitor:app --port $MONITOR_PORT --reload &
    else
        python -m uvicorn app.services.monitor:app --port $MONITOR_PORT &
    fi
    MONITOR_PID=$!
    echo "Monitoring service started with PID $MONITOR_PID"
fi

# Wait for all started services
echo ""
echo "All services started. Press Ctrl+C to stop all services."

# Handle SIGINT (Ctrl+C)
trap 'echo -e "\nStopping services..."; kill $API_PID $TRAINING_PID $MONITOR_PID 2>/dev/null; exit 0' INT

# Keep script running
wait
