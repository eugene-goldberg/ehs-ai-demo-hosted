#!/bin/bash

# Start the EHS AI Demo Backend Server
# This script ensures environment variables are properly loaded

echo "Starting EHS AI Demo Backend Server..."

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -E -v '^#' .env | xargs)
    echo "Environment variables loaded from .env"
else
    echo "Warning: .env file not found"
fi

# Check if LANGCHAIN_API_KEY is set
if [ -z "$LANGCHAIN_API_KEY" ]; then
    echo "Error: LANGCHAIN_API_KEY not found in environment"
    echo "Please check your .env file"
    exit 1
fi

echo "LANGCHAIN_API_KEY: ${LANGCHAIN_API_KEY:0:10}..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Activated Python virtual environment"
fi

# Start the server
echo "Starting FastAPI server on http://0.0.0.0:8001"
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8001