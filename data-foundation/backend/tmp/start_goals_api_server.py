#!/usr/bin/env python3
"""
EHS Goals API Server Starter

This script starts a FastAPI server that includes the EHS Goals API routes.
It creates a minimal server to host the goals endpoints for testing.

Usage:
    python3 start_goals_api_server.py [--port PORT]
"""

import argparse
import sys
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    
    # Import the EHS Goals API router
    from src.api.ehs_goals_api import router as goals_router
    
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install: pip install fastapi uvicorn")
    print("And ensure the EHS Goals API is properly set up")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    
    app = FastAPI(
        title="EHS Goals API Server",
        description="API server for testing EHS Goals endpoints",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include the EHS Goals API router
    app.include_router(goals_router)
    
    # Add a simple root endpoint
    @app.get("/")
    async def root():
        return {
            "message": "EHS Goals API Server",
            "version": "1.0.0",
            "docs": "/docs",
            "goals_endpoints": {
                "annual_goals": "/api/goals/annual",
                "site_goals": "/api/goals/annual/{site_id}",
                "progress": "/api/goals/progress/{site_id}",
                "summary": "/api/goals/summary",
                "health": "/api/goals/health"
            }
        }
    
    # Add health check endpoint
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "ehs-goals-api"}
    
    return app

def main():
    """Main function to start the server"""
    parser = argparse.ArgumentParser(description="Start EHS Goals API Server")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    args = parser.parse_args()
    
    # Create the FastAPI app
    app = create_app()
    
    logger.info(f"Starting EHS Goals API Server on {args.host}:{args.port}")
    logger.info("Available endpoints:")
    logger.info("  - Documentation: http://localhost:8000/docs")
    logger.info("  - Health Check: http://localhost:8000/health")
    logger.info("  - Annual Goals: http://localhost:8000/api/goals/annual")
    logger.info("  - Site Goals: http://localhost:8000/api/goals/annual/{site_id}")
    logger.info("  - Progress: http://localhost:8000/api/goals/progress/{site_id}")
    logger.info("  - Summary: http://localhost:8000/api/goals/summary")
    
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
