#!/usr/bin/env python3
"""
Simple FastAPI Test Server
A minimal FastAPI application for basic testing purposes.
"""

import logging
import sys
from typing import Dict, Any
from datetime import datetime

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError as e:
    print(f'Missing required dependencies: {e}')
    print('Please install: pip install fastapi uvicorn')
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI instance
app = FastAPI(
    title='Simple Test API',
    description='A minimal FastAPI server for testing',
    version='1.0.0'
)

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f'Global exception: {exc}')
    return JSONResponse(
        status_code=500,
        content={'error': 'Internal server error', 'detail': str(exc)}
    )

# Health check endpoint
@app.get('/health')
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'simple-test-api'
    }

# Root endpoint
@app.get('/')
async def root() -> Dict[str, str]:
    """Root endpoint"""
    return {'message': 'Simple Test API is running'}

# Simple echo endpoint
@app.post('/echo')
async def echo(data: Dict[str, Any]) -> Dict[str, Any]:
    """Echo back the received data"""
    return {
        'received': data,
        'timestamp': datetime.now().isoformat()
    }

# Simple GET endpoint with path parameter
@app.get('/test/{item_id}')
async def get_item(item_id: int) -> Dict[str, Any]:
    """Get item by ID"""
    if item_id < 0:
        raise HTTPException(status_code=400, detail='Item ID must be positive')
    
    return {
        'item_id': item_id,
        'message': f'Retrieved item {item_id}',
        'timestamp': datetime.now().isoformat()
    }

# Simple POST endpoint
@app.post('/test/items')
async def create_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new item"""
    if not item.get('name'):
        raise HTTPException(status_code=400, detail='Item name is required')
    
    return {
        'created': True,
        'item': item,
        'id': hash(str(item)) % 10000,  # Simple ID generation
        'timestamp': datetime.now().isoformat()
    }

# Simple error endpoint for testing error handling
@app.get('/test/error')
async def trigger_error():
    """Endpoint that triggers an error for testing"""
    raise HTTPException(status_code=500, detail='This is a test error')

# Status endpoint
@app.get('/status')
async def status() -> Dict[str, Any]:
    """Get server status"""
    return {
        'server': 'running',
        'endpoints': [
            '/',
            '/health',
            '/echo',
            '/test/{item_id}',
            '/test/items',
            '/test/error',
            '/status'
        ],
        'timestamp': datetime.now().isoformat()
    }

def main():
    """Main function to start the server"""
    try:
        logger.info('Starting Simple Test API server...')
        uvicorn.run(
            app,
            host='0.0.0.0',
            port=8000,
            log_level='info',
            access_log=True,
            reload=False  # Set to True for development
        )
    except Exception as e:
        logger.error(f'Failed to start server: {e}')
        sys.exit(1)

if __name__ == '__main__':
    main()