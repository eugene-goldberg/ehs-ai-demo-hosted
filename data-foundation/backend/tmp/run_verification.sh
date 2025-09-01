#!/bin/bash

# Environmental Endpoints Verification Runner
# This script ensures the verification runs in the correct environment

echo "Environmental Endpoints Verification Runner"
echo "=========================================="

# Check if we're in the right directory
BACKEND_DIR="/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend"
if [ ! -d "$BACKEND_DIR" ]; then
    echo "Error: Backend directory not found at $BACKEND_DIR"
    exit 1
fi

cd "$BACKEND_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Please create one first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if server is running
echo "Checking if FastAPI server is running on localhost:8000..."
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "Warning: FastAPI server doesn't appear to be running on localhost:8000"
    echo "Please start the server first:"
    echo "  cd $BACKEND_DIR"
    echo "  source venv/bin/activate"
    echo "  python3 app.py"
    echo ""
    echo "Continuing with verification anyway..."
fi

# Install required packages if not present
echo "Ensuring required packages are installed..."
pip install requests > /dev/null 2>&1

# Run the verification script
echo "Running comprehensive endpoint verification..."
echo ""

python3 tmp/verify_all_environmental_endpoints.py

# Check results
VERIFICATION_EXIT_CODE=$?
echo ""
echo "Verification completed with exit code: $VERIFICATION_EXIT_CODE"

if [ $VERIFICATION_EXIT_CODE -eq 0 ]; then
    echo "✅ All endpoints verified successfully!"
else
    echo "❌ Some endpoints failed verification."
fi

echo ""
echo "Results files:"
echo "  - Summary: tmp/verification_summary.json"
echo "  - Logs: tmp/endpoint_verification_results.log"

exit $VERIFICATION_EXIT_CODE
