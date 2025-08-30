#!/bin/bash

# EHS AI Demo - Service Restart Script
# This script restarts both FastAPI services for the EHS AI Demo

echo "=========================================="
echo "EHS AI Demo - Restarting Services"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BACKEND_DIR="$SCRIPT_DIR/backend"
WEBAPP_BACKEND_DIR="$SCRIPT_DIR/web-app/backend"

# Function to kill processes on a specific port
kill_port() {
    local port=$1
    local service_name=$2
    echo -e "${YELLOW}Stopping $service_name on port $port...${NC}"
    
    # Find and kill processes on the port
    local pids=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$pids" ]; then
        echo "$pids" | xargs kill -9 2>/dev/null
        echo -e "${GREEN}✓ Stopped $service_name${NC}"
        sleep 1
    else
        echo -e "${GREEN}✓ No process running on port $port${NC}"
    fi
}

# Function to start a service
start_service() {
    local service_name=$1
    local service_dir=$2
    local command=$3
    local port=$4
    
    echo -e "\n${YELLOW}Starting $service_name...${NC}"
    
    # Check if directory exists
    if [ ! -d "$service_dir" ]; then
        echo -e "${RED}✗ Directory not found: $service_dir${NC}"
        return 1
    fi
    
    # Navigate to directory
    cd "$service_dir" || {
        echo -e "${RED}✗ Failed to navigate to $service_dir${NC}"
        return 1
    }
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo -e "${RED}✗ Virtual environment not found in $service_dir${NC}"
        return 1
    fi
    
    # Start the service in background
    nohup bash -c "$command" > /tmp/${service_name// /_}.log 2>&1 &
    
    # Wait a moment for the service to start
    sleep 3
    
    # Check if service started successfully
    if lsof -i:$port >/dev/null 2>&1; then
        echo -e "${GREEN}✓ $service_name started successfully on port $port${NC}"
        echo -e "  Log file: /tmp/${service_name// /_}.log"
        return 0
    else
        echo -e "${RED}✗ Failed to start $service_name${NC}"
        echo -e "  Check log file: /tmp/${service_name// /_}.log"
        return 1
    fi
}

# Main execution
echo -e "\n${YELLOW}Step 1: Stopping existing services${NC}"
echo "=========================================="

# Kill existing services
kill_port 8000 "Main EHS Extraction API"
kill_port 8001 "Web App Backend API"

echo -e "\n${YELLOW}Step 2: Starting services${NC}"
echo "=========================================="

# Start Main EHS Extraction API
MAIN_API_CMD="source venv/bin/activate && \
export PYTHONPATH=$BACKEND_DIR:$BACKEND_DIR/src && \
export AUDIT_TRAIL_STORAGE_PATH='/tmp/audit_trail_storage' && \
python src/ehs_extraction_api.py"

start_service "Main EHS Extraction API" "$BACKEND_DIR" "$MAIN_API_CMD" 8000

# Start Web App Backend API
WEBAPP_API_CMD="source venv/bin/activate && python main.py"

start_service "Web App Backend API" "$WEBAPP_BACKEND_DIR" "$WEBAPP_API_CMD" 8001

# Summary
echo -e "\n=========================================="
echo -e "${GREEN}Service Status Summary:${NC}"
echo "=========================================="

if lsof -i:8000 >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Main EHS Extraction API${NC} - Running on http://localhost:8000"
else
    echo -e "${RED}✗ Main EHS Extraction API${NC} - Not running"
fi

if lsof -i:8001 >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Web App Backend API${NC} - Running on http://localhost:8001"
else
    echo -e "${RED}✗ Web App Backend API${NC} - Not running"
fi

echo -e "\n${GREEN}Frontend:${NC} http://localhost:3000 (start separately with 'npm start')"
echo -e "\n${YELLOW}Logs:${NC}"
echo "  Main API: /tmp/Main_EHS_Extraction_API.log"
echo "  Web App API: /tmp/Web_App_Backend_API.log"
echo -e "\n${YELLOW}To monitor logs in real-time:${NC}"
echo "  tail -f /tmp/Main_EHS_Extraction_API.log"
echo "  tail -f /tmp/Web_App_Backend_API.log"
echo -e "\n=========================================="