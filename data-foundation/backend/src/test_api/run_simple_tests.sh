#!/bin/bash

# Simple Test Runner Script
# Starts the simple test API server, runs quick tests, and handles cleanup

set -e  # Exit on any error

# Configuration
API_HOST="localhost"
API_PORT="8000"
API_BASE="http://${API_HOST}:${API_PORT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/test_reports/simple_test_run.log"
API_PID_FILE="${SCRIPT_DIR}/test_reports/api.pid"
MAX_STARTUP_WAIT=30
HEALTH_CHECK_INTERVAL=1
VENV_DIR="${SCRIPT_DIR}/test_venv"

# Colors for output
BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

# Print colored output
print_status() {
    local color="$1"
    local message="$2"
    echo -e "${color}${message}${NC}"
    log "INFO" "$message"
}

# Virtual environment management
setup_virtual_environment() {
    print_status "$BLUE" "Setting up virtual environment..."
    
    # Check if virtual environment exists
    if [ -d "$VENV_DIR" ]; then
        print_status "$GREEN" "Virtual environment found at $VENV_DIR"
    else
        print_status "$YELLOW" "Creating virtual environment at $VENV_DIR"
        python3 -m venv "$VENV_DIR"
        if [ $? -ne 0 ]; then
            print_status "$RED" "ERROR: Failed to create virtual environment"
            exit 1
        fi
        print_status "$GREEN" "Virtual environment created successfully"
    fi
    
    # Activate virtual environment
    print_status "$BLUE" "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    if [ $? -ne 0 ]; then
        print_status "$RED" "ERROR: Failed to activate virtual environment"
        exit 1
    fi
    print_status "$GREEN" "Virtual environment activated"
    
    # Check if requirements need to be installed
    if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
        print_status "$BLUE" "Installing dependencies from requirements.txt..."
        pip install -r "${SCRIPT_DIR}/requirements.txt"
        if [ $? -ne 0 ]; then
            print_status "$YELLOW" "WARNING: Some dependencies may have failed to install"
        else
            print_status "$GREEN" "Dependencies installed successfully"
        fi
    else
        # Install basic dependencies that are commonly needed
        print_status "$BLUE" "Installing basic dependencies..."
        pip install fastapi uvicorn requests
        if [ $? -ne 0 ]; then
            print_status "$YELLOW" "WARNING: Some basic dependencies may have failed to install"
        else
            print_status "$GREEN" "Basic dependencies installed successfully"
        fi
    fi
    
    # Verify Python is now using the venv
    local python_path=$(which python)
    if [[ "$python_path" == *"$VENV_DIR"* ]]; then
        print_status "$GREEN" "Using Python from virtual environment: $python_path"
    else
        print_status "$YELLOW" "WARNING: Python may not be using the virtual environment"
    fi
}

# Error handling function
handle_error() {
    local exit_code=$?
    local line_number=$1
    print_status "$RED" "ERROR: Script failed at line $line_number with exit code $exit_code"
    cleanup
    exit $exit_code
}

# Cleanup function
cleanup() {
    print_status "$YELLOW" "Performing cleanup..."
    
    # Stop the API server if it's running
    if [ -f "$API_PID_FILE" ]; then
        local api_pid=$(cat "$API_PID_FILE")
        if ps -p "$api_pid" > /dev/null 2>&1; then
            print_status "$YELLOW" "Stopping API server (PID: $api_pid)..."
            kill "$api_pid" 2>/dev/null || true
            
            # Wait for process to terminate
            local wait_count=0
            while ps -p "$api_pid" > /dev/null 2>&1 && [ $wait_count -lt 10 ]; do
                sleep 1
                ((wait_count++))
            done
            
            # Force kill if still running
            if ps -p "$api_pid" > /dev/null 2>&1; then
                print_status "$YELLOW" "Force killing API server..."
                kill -9 "$api_pid" 2>/dev/null || true
            fi
            
            print_status "$GREEN" "API server stopped"
        fi
        rm -f "$API_PID_FILE"
    fi
    
    # Check for any remaining processes on the port
    local port_pid=$(lsof -ti:$API_PORT 2>/dev/null || true)
    if [ -n "$port_pid" ]; then
        print_status "$YELLOW" "Killing remaining process on port $API_PORT (PID: $port_pid)..."
        kill -9 "$port_pid" 2>/dev/null || true
    fi
}

# Check if required files exist
check_prerequisites() {
    print_status "$BLUE" "Checking prerequisites..."
    
    # Create test_reports directory if it doesn't exist
    mkdir -p "${SCRIPT_DIR}/test_reports"
    
    # Check if simple_test_api.py exists
    if [ ! -f "${SCRIPT_DIR}/simple_test_api.py" ]; then
        print_status "$RED" "ERROR: simple_test_api.py not found in $SCRIPT_DIR"
        exit 1
    fi
    
    # Check if quick_test.sh exists
    if [ ! -f "${SCRIPT_DIR}/quick_test.sh" ]; then
        print_status "$RED" "ERROR: quick_test.sh not found in $SCRIPT_DIR"
        exit 1
    fi
    
    # Make quick_test.sh executable
    chmod +x "${SCRIPT_DIR}/quick_test.sh"
    
    # Check if Python 3 is available
    if ! command -v python3 >/dev/null 2>&1; then
        print_status "$RED" "ERROR: Python 3 is not installed or not in PATH"
        exit 1
    fi
    
    # Check if port is already in use
    if lsof -i:$API_PORT >/dev/null 2>&1; then
        print_status "$RED" "ERROR: Port $API_PORT is already in use"
        print_status "$YELLOW" "Please stop any services using this port and try again"
        exit 1
    fi
    
    print_status "$GREEN" "Prerequisites check passed"
}

# Start the API server
start_api_server() {
    print_status "$BLUE" "Starting API server..."
    
    cd "$SCRIPT_DIR"
    
    # Start the API server in the background using the venv python
    python simple_test_api.py > "${SCRIPT_DIR}/test_reports/api_server.log" 2>&1 &
    local api_pid=$!
    
    # Save PID for cleanup
    echo "$api_pid" > "$API_PID_FILE"
    
    print_status "$BLUE" "API server started with PID: $api_pid"
    print_status "$BLUE" "Waiting for server to be ready..."
    
    # Wait for server to be ready
    local wait_count=0
    while [ $wait_count -lt $MAX_STARTUP_WAIT ]; do
        if curl -s "$API_BASE/health" >/dev/null 2>&1; then
            print_status "$GREEN" "API server is ready!"
            return 0
        fi
        
        # Check if process is still running
        if ! ps -p "$api_pid" > /dev/null 2>&1; then
            print_status "$RED" "ERROR: API server process died during startup"
            cat "${SCRIPT_DIR}/test_reports/api_server.log"
            exit 1
        fi
        
        sleep $HEALTH_CHECK_INTERVAL
        ((wait_count++))
        
        if [ $((wait_count % 5)) -eq 0 ]; then
            print_status "$YELLOW" "Still waiting for server... ($wait_count/${MAX_STARTUP_WAIT}s)"
        fi
    done
    
    print_status "$RED" "ERROR: API server failed to start within $MAX_STARTUP_WAIT seconds"
    print_status "$YELLOW" "Server log output:"
    cat "${SCRIPT_DIR}/test_reports/api_server.log"
    exit 1
}

# Run the quick tests
run_tests() {
    print_status "$BLUE" "Running quick tests..."
    
    local test_output_file="${SCRIPT_DIR}/test_reports/quick_test_output.log"
    
    # Run the quick test script and capture output
    if "${SCRIPT_DIR}/quick_test.sh" > "$test_output_file" 2>&1; then
        print_status "$GREEN" "Tests completed successfully!"
        local test_exit_code=0
    else
        print_status "$RED" "Tests completed with errors!"
        local test_exit_code=1
    fi
    
    # Display test results
    print_status "$BLUE" "Test Results:"
    echo "========================================"
    cat "$test_output_file"
    echo "========================================"
    
    return $test_exit_code
}

# Display summary
display_summary() {
    local test_result=$1
    
    print_status "$BLUE" "Test Run Summary"
    echo "========================================"
    echo "Start Time: $(head -n 1 "$LOG_FILE" | cut -d']' -f1 | tr -d '[')"
    echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "API Base URL: $API_BASE"
    echo "Virtual Environment: $VENV_DIR"
    echo "Log File: $LOG_FILE"
    echo "Test Output: ${SCRIPT_DIR}/test_reports/quick_test_output.log"
    echo "API Server Log: ${SCRIPT_DIR}/test_reports/api_server.log"
    
    if [ $test_result -eq 0 ]; then
        print_status "$GREEN" "Overall Result: SUCCESS"
    else
        print_status "$RED" "Overall Result: FAILED"
    fi
    echo "========================================"
}

# Main function
main() {
    # Set up error handling
    trap 'handle_error $LINENO' ERR
    trap cleanup EXIT INT TERM
    
    print_status "$BOLD$BLUE" "Simple API Test Runner"
    print_status "$BLUE" "Starting test execution..."
    
    # Initialize log file
    echo "Test run started at $(date)" > "$LOG_FILE"
    
    # Setup virtual environment
    setup_virtual_environment
    
    # Check prerequisites
    check_prerequisites
    
    # Start API server
    start_api_server
    
    # Run tests
    local test_result=0
    run_tests || test_result=$?
    
    # Display summary
    display_summary $test_result
    
    # Return test result
    exit $test_result
}

# Run main function with all arguments
main "$@"