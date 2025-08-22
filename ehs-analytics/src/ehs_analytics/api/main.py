"""
Main FastAPI Application for EHS Analytics

This module provides the main FastAPI application with middleware configuration,
CORS setup, health checks, and router registration for the EHS Analytics platform.
"""

import asyncio
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
import structlog
import uvicorn

from .routers import analytics_router
from .dependencies import (
    startup_handler, shutdown_handler, get_settings,
    db_manager, workflow_manager
)
from .models import ErrorResponse, ErrorDetail, ErrorType, HealthCheckResponse, ServiceHealth

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
    logger.info("Starting EHS Analytics API application")
    
    try:
        # Startup
        await startup_handler()
        logger.info("Application startup completed successfully")
        yield
        
    except Exception as e:
        logger.error("Application startup failed", error=str(e))
        raise
    finally:
        # Shutdown
        logger.info("Starting application shutdown")
        await shutdown_handler()
        logger.info("Application shutdown completed")


# Create FastAPI application
app = FastAPI(
    title="EHS Analytics API",
    description="""
    **EHS Analytics API** - AI-powered environmental, health, and safety analytics platform.
    
    This API provides natural language query processing for EHS data analysis,
    compliance checking, risk assessment, and recommendation generation.
    
    ## Features
    
    * **Natural Language Processing**: Query EHS data using plain English
    * **Intent Classification**: Automatic routing to appropriate analysis workflows
    * **Multi-Modal Retrieval**: Text2Cypher, vector search, and hybrid strategies
    * **Risk Assessment**: AI-powered environmental and safety risk evaluation
    * **Compliance Monitoring**: Automated regulatory compliance checking
    * **Recommendation Engine**: Actionable insights for EHS improvements
    
    ## Query Types Supported
    
    * **Consumption Analysis**: Energy, water, and resource usage patterns
    * **Compliance Check**: Regulatory compliance status and violations
    * **Risk Assessment**: Environmental and safety risk evaluation
    * **Emission Tracking**: Carbon footprint and greenhouse gas monitoring
    * **Equipment Efficiency**: Asset performance and maintenance optimization
    * **Permit Status**: Environmental permit compliance and renewals
    * **General Inquiry**: Broad EHS information requests
    
    ## Authentication
    
    Authentication is currently in development mode. In production, all endpoints
    will require valid JWT tokens with appropriate scopes.
    """,
    version="0.1.0",
    contact={
        "name": "EHS AI Team",
        "email": "ehs-ai@company.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    servers=[
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        },
        {
            "url": "https://api.ehs-analytics.company.com",
            "description": "Production server"
        }
    ],
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# Configure middleware
settings = get_settings()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React development server
        "http://localhost:3001",  # Alternative frontend port
        "http://localhost:8080",  # Alternative frontend port
        "https://dashboard.ehs-analytics.company.com",  # Production frontend
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Response-Time"]
)

# Trusted host middleware (security)
if settings.environment == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=[
            "api.ehs-analytics.company.com",
            "localhost",
            "127.0.0.1"
        ]
    )

# GZip compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Request/Response middleware for logging and monitoring
@app.middleware("http")
async def request_response_middleware(request: Request, call_next):
    """
    Middleware for request/response logging and performance monitoring.
    """
    import time
    import uuid
    
    # Generate request ID
    request_id = str(uuid.uuid4())
    
    # Add request ID to headers
    request.state.request_id = request_id
    
    # Log request
    start_time = time.time()
    logger.info(
        "API request started",
        request_id=request_id,
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", "unknown")
    )
    
    try:
        # Process request
        response = await call_next(request)
        
        # Calculate response time
        response_time = time.time() - start_time
        response_time_ms = int(response_time * 1000)
        
        # Add response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{response_time_ms}ms"
        
        # Log response
        logger.info(
            "API request completed",
            request_id=request_id,
            status_code=response.status_code,
            response_time_ms=response_time_ms
        )
        
        return response
        
    except Exception as e:
        # Calculate response time for errors
        response_time = time.time() - start_time
        response_time_ms = int(response_time * 1000)
        
        # Log error
        logger.error(
            "API request failed",
            request_id=request_id,
            error=str(e),
            response_time_ms=response_time_ms
        )
        
        # Return error response
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error_type": ErrorType.PROCESSING_ERROR,
                "message": "Internal server error",
                "details": {
                    "request_id": request_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            },
            headers={
                "X-Request-ID": request_id,
                "X-Response-Time": f"{response_time_ms}ms"
            }
        )


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured error responses."""
    
    request_id = getattr(request.state, "request_id", "unknown")
    
    # If detail is already a dict (structured error), use it directly
    if isinstance(exc.detail, dict):
        content = exc.detail
    else:
        # Convert string detail to structured format
        content = {
            "error_type": ErrorType.PROCESSING_ERROR,
            "message": str(exc.detail),
            "details": {
                "status_code": exc.status_code,
                "request_id": request_id
            }
        }
    
    logger.warning(
        "HTTP exception occurred",
        request_id=request_id,
        status_code=exc.status_code,
        detail=content
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=content,
        headers={
            "X-Request-ID": request_id,
            **getattr(exc, "headers", {})
        }
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.warning(
        "Validation error occurred",
        request_id=request_id,
        error=str(exc)
    )
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error_type": ErrorType.VALIDATION_ERROR,
            "message": "Validation error",
            "details": {
                "error": str(exc),
                "request_id": request_id
            }
        },
        headers={"X-Request-ID": request_id}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(
        "Unexpected error occurred",
        request_id=request_id,
        error=str(exc),
        exc_info=True
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_type": ErrorType.PROCESSING_ERROR,
            "message": "Internal server error",
            "details": {
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        },
        headers={"X-Request-ID": request_id}
    )


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint providing API information and status.
    """
    return {
        "name": "EHS Analytics API",
        "version": "0.1.0",
        "description": "AI-powered environmental, health, and safety analytics platform",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
            "health": "/health",
            "analytics": "/api/v1/analytics"
        },
        "features": [
            "Natural language EHS query processing",
            "AI-powered risk assessment",
            "Regulatory compliance monitoring",
            "Equipment efficiency analysis",
            "Emission tracking and reporting"
        ]
    }


# Global health check endpoint
@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def global_health_check():
    """
    Global health check endpoint for the entire application.
    
    This endpoint provides a comprehensive health status of all application
    components including database, workflow engine, and external dependencies.
    """
    try:
        services = []
        overall_healthy = True
        
        # Check application status
        services.append(ServiceHealth(
            service_name="FastAPI Application",
            status="healthy",
            response_time_ms=5,
            metadata={
                "version": "0.1.0",
                "environment": settings.environment,
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            }
        ))
        
        # Check database connectivity
        try:
            db_healthy = await db_manager.health_check() if db_manager.is_connected else False
            services.append(ServiceHealth(
                service_name="Database Layer",
                status="healthy" if db_healthy else "unhealthy",
                response_time_ms=None,
                error_message=None if db_healthy else "Database connectivity issues",
                metadata={"neo4j_connected": db_manager.is_connected}
            ))
            if not db_healthy:
                overall_healthy = False
        except Exception as e:
            services.append(ServiceHealth(
                service_name="Database Layer",
                status="unhealthy",
                error_message=str(e)
            ))
            overall_healthy = False
        
        # Check workflow engine
        try:
            workflow_healthy = await workflow_manager.health_check() if workflow_manager.is_initialized else False
            services.append(ServiceHealth(
                service_name="Workflow Engine",
                status="healthy" if workflow_healthy else "degraded",
                response_time_ms=None,
                error_message=None if workflow_healthy else "Workflow engine issues",
                metadata={"langgraph_initialized": workflow_manager.is_initialized}
            ))
            # Workflow issues are non-critical
        except Exception as e:
            services.append(ServiceHealth(
                service_name="Workflow Engine",
                status="degraded",
                error_message=str(e)
            ))
        
        # Determine overall status
        if overall_healthy:
            if all(s.status in ["healthy", "degraded"] for s in services):
                overall_status = "healthy"
            else:
                overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        # Calculate uptime (simplified)
        import time
        uptime_seconds = int(time.time()) % 86400  # Reset daily for demo
        
        return HealthCheckResponse(
            success=True,
            message="Global health check completed",
            overall_status=overall_status,
            services=services,
            uptime_seconds=uptime_seconds,
            version="0.1.0",
            environment=settings.environment
        )
        
    except Exception as e:
        logger.error("Global health check failed", error=str(e))
        
        return HealthCheckResponse(
            success=False,
            message="Health check failed",
            overall_status="unhealthy",
            services=[ServiceHealth(
                service_name="Health Check System",
                status="unhealthy",
                error_message=str(e)
            )],
            uptime_seconds=0,
            version="0.1.0",
            environment=settings.environment
        )


# Register routers
app.include_router(analytics_router)


# Custom OpenAPI schema
def custom_openapi():
    """
    Generate custom OpenAPI schema with additional metadata.
    """
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="EHS Analytics API",
        version="0.1.0",
        description=app.description,
        routes=app.routes,
        servers=app.servers
    )
    
    # Add custom extensions
    openapi_schema["info"]["x-logo"] = {
        "url": "https://company.com/logo.png"
    }
    
    openapi_schema["info"]["x-api-features"] = [
        "Natural Language Processing",
        "AI-Powered Analytics",
        "Real-time Query Processing",
        "Multi-Modal Data Retrieval",
        "Risk Assessment",
        "Compliance Monitoring"
    ]
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token for API authentication"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Development server configuration
if __name__ == "__main__":
    # Configure logging for development
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run development server
    uvicorn.run(
        "ehs_analytics.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True,
        loop="asyncio",
        workers=1  # Single worker for development
    )