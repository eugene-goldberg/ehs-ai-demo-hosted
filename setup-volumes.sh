#!/bin/bash

# Setup script for EHS AI Demo Docker environment
# This script creates the necessary directories and sets proper permissions

echo "Setting up EHS AI Demo Docker environment..."

# Create base directories
mkdir -p volumes/neo4j/{data,logs,import,plugins}
mkdir -p volumes/app-data
mkdir -p volumes/logs/{backend,analytics}
mkdir -p input-documents

# Create sample input directory structure
mkdir -p input-documents/{pdfs,reports,regulations,permits}

# Set permissions (adjust as needed for your system)
echo "Setting directory permissions..."

# Neo4j directories
chmod -R 755 volumes/neo4j/
chmod -R 777 volumes/neo4j/data  # Neo4j needs write access
chmod -R 777 volumes/neo4j/logs  # Neo4j needs write access
chmod -R 755 volumes/neo4j/import
chmod -R 755 volumes/neo4j/plugins

# Application data and logs
chmod -R 755 volumes/app-data
chmod -R 755 volumes/logs/

# Input documents (read-only for containers)
chmod -R 755 input-documents/

# Create sample .env file for EHS Analytics if it doesn't exist
if [ ! -f ehs-analytics/.env ]; then
    echo "Creating sample .env file for EHS Analytics..."
    cat > ehs-analytics/.env << 'EOF'
# EHS Analytics Environment Configuration
# Copy this file and customize for your environment

# Database Configuration
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password123

# API Configuration
PORT=8000
LOG_LEVEL=INFO
ENVIRONMENT=development

# API Keys (add your actual keys)
# OPENAI_API_KEY=your_openai_key_here
# ANTHROPIC_API_KEY=your_anthropic_key_here
# GOOGLE_API_KEY=your_google_key_here

# Data Foundation Integration
DATA_FOUNDATION_API_URL=http://data-foundation-backend:8000

# Feature Flags
ENABLE_TEXT2CYPHER=true
ENABLE_VECTOR_SEARCH=true
ENABLE_HYBRID_SEARCH=true
ENABLE_RISK_ASSESSMENT=true
ENABLE_COMPLIANCE_CHECK=true

# Performance Settings
BATCH_SIZE=10
MAX_CONCURRENT_REQUESTS=5
CACHE_TTL_SECONDS=3600
EOF
fi

echo "Creating sample README for input documents..."
cat > input-documents/README.md << 'EOF'
# Input Documents Directory

This directory is mounted read-only into all containers to provide access to input documents.

## Structure

- `pdfs/` - PDF documents for processing
- `reports/` - EHS reports and compliance documents  
- `regulations/` - Regulatory documents and standards
- `permits/` - Environmental permits and certifications

## Usage

1. Place your documents in the appropriate subdirectories
2. The containers will have read-only access to these files
3. Use the Data Foundation or EHS Analytics APIs to process the documents

## Supported File Types

- PDF documents
- Text files (.txt, .md)
- CSV data files
- Excel files (.xlsx, .xls)
- Word documents (.docx)

## Volume Mounting

This directory is mounted as:
- Data Foundation Backend: `/code/input` (read-only)
- EHS Analytics: `/app/input` (read-only)
- Neo4j: `/var/lib/neo4j/input` (read-only)
EOF

echo "Creating docker-compose override file for development..."
cat > docker-compose.override.yml << 'EOF'
# Docker Compose override for development
# This file is automatically loaded with docker-compose.yml

version: '3.8'

services:
  data-foundation-backend:
    volumes:
      # Override for development with live code reloading
      - ./data-foundation/backend:/code:cached
    environment:
      - DEBUG=true
      - RELOAD=true

  ehs-analytics:
    volumes:
      # Override for development with live code reloading  
      - ./ehs-analytics:/app:cached
    environment:
      - DEBUG=true
      - RELOAD=true
    command: ["python", "-m", "uvicorn", "ehs_analytics.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]

  neo4j:
    environment:
      # Development settings
      - NEO4J_dbms_logs_debug_level=INFO
EOF

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Review and customize the .env files in each service directory"
echo "2. Add your API keys to ehs-analytics/.env"
echo "3. Place input documents in the input-documents/ directory"
echo "4. Run: docker-compose up -d"
echo ""
echo "Service URLs:"
echo "- Data Foundation Frontend: http://localhost:8080"
echo "- Data Foundation Backend: http://localhost:8000"
echo "- EHS Analytics API: http://localhost:8001"
echo "- Neo4j Browser: http://localhost:7474"
echo ""
echo "Default Neo4j credentials: neo4j/password123"