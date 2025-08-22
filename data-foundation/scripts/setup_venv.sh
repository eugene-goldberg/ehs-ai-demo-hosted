#!/bin/bash
# Setup virtual environment for EHS AI Data Foundation

echo "🔧 Setting up EHS AI Data Foundation virtual environment..."

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first."
    echo "Visit: https://github.com/astral-sh/uv"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment with Python 3.11..."
    uv venv --python 3.11
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
uv pip install -r backend/requirements.txt --index-strategy unsafe-best-match

# Note about torch CPU versions
echo ""
echo "⚠️  Note: torch CPU versions are not available for macOS ARM64"
echo "    The regular torch version has been installed instead."
echo ""

# Test Neo4j connection
echo "🔍 Testing Neo4j connection..."
python scripts/test_neo4j_connection.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the virtual environment in the future, run:"
echo "    source .venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "    deactivate"