#!/bin/bash

# EHS AI Demo - Data Foundation Ingestion Runner
# This script sets up the proper environment and runs the document ingestion process

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_FOUNDATION_DIR="$(dirname "$SCRIPT_DIR")"

echo "Setting up environment for document ingestion..."
echo "Data Foundation Directory: $DATA_FOUNDATION_DIR"

# Change to the data-foundation directory
cd "$DATA_FOUNDATION_DIR"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found at .venv"
    echo "Please create a virtual environment first with: python3 -m venv .venv"
    exit 1
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Set PYTHONPATH to include the backend/src directory
export PYTHONPATH="$DATA_FOUNDATION_DIR/backend/src:$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"

# Check if the ingestion script exists
INGESTION_SCRIPT="$DATA_FOUNDATION_DIR/scripts/ingest_all_documents.py"
if [ ! -f "$INGESTION_SCRIPT" ]; then
    echo "Error: Ingestion script not found at $INGESTION_SCRIPT"
    exit 1
fi

echo "Running document ingestion with arguments: $@"
echo "----------------------------------------"

# Execute the Python script with all command line arguments passed through
python3 "$INGESTION_SCRIPT" "$@"

# Capture the exit code from the Python script
EXIT_CODE=$?

echo "----------------------------------------"
echo "Ingestion completed with exit code: $EXIT_CODE"

# Return the same exit code as the Python script
exit $EXIT_CODE