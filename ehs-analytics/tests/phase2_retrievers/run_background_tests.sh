#!/bin/bash

# Phase 2 Retriever Background Test Runner
echo "🚀 Starting Phase 2 Retriever Tests in Background"

# Activate virtual environment
source .venv/bin/activate

# Set environment variables
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export LOG_LEVEL=INFO

# Create log directory
mkdir -p tests/phase2_retrievers/logs

# Run tests in background with logging
cd tests/phase2_retrievers

echo "📋 Running comprehensive Phase 2 retriever tests..."
echo "📄 Logs will be saved to tests/phase2_retrievers/logs/"

python3 test_comprehensive_phase2_retrievers.py > logs/test_output_$(date +%Y%m%d_%H%M%S).log 2>&1 &

TEST_PID=$!
echo "🔄 Tests running in background with PID: $TEST_PID"
echo "📊 Monitor progress with: tail -f tests/phase2_retrievers/logs/test_output_*.log"
echo "⏹️  Stop tests with: kill $TEST_PID"

# Save PID for reference
echo $TEST_PID > logs/test_pid.txt

echo "✅ Background test execution started"
