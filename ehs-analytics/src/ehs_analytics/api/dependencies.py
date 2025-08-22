"""
API Dependencies for EHS Analytics

This module provides dependency injection for FastAPI endpoints,
including workflow instances, database connections, and authentication.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog

from ..agents.query_router import QueryRouterAgent
from .models import ErrorResponse, ErrorDetail, ErrorType

# Configure structured logging
logger = structlog.get_logger(__name__)

# Security scheme for JWT tokens
security = HTTPBearer(auto_error=False)


class ConfigurationError(Exception):
    """Raised when there's a configuration issue."""
    pass


class DatabaseConnectionError(Exception):
    """Raised when database connection fails."""
    pass


class WorkflowError(Exception):
    """Raised when workflow initialization fails."""
    pass


def get_settings():
    """Get application settings with proper config import."""
    try:
        from ..config import get_settings as _get_settings
        return _get_settings()
    except ImportError:
        # Fallback configuration
        import os
        from pydantic import BaseSettings
        
        class FallbackSettings(BaseSettings):
            neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            neo4j_username: str = os.getenv("NEO4J_USERNAME", "neo4j")
            neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")
            openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
            environment: str = os.getenv("ENVIRONMENT", "development")
            log_level: str = os.getenv("LOG_LEVEL", "INFO")
            api_timeout: int = int(os.getenv("API_TIMEOUT", "300"))
            enable_auth: bool = os.getenv("ENABLE_AUTH", "false").lower() == "true"
            jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "dev-secret-key")
            jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
            
            # LLM settings
            llm_model_name: str = os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo")
            llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
            llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "2000"))
            
            # Cypher settings  
            cypher_validation: bool = os.getenv("CYPHER_VALIDATION", "true").lower() == "true"
            
        return FallbackSettings()


class DatabaseManager:
    """Database connection manager with connection pooling."""
    
    def __init__(self):
        self._neo4j_driver = None
        self._connection_pool = None
        self.is_connected = False
        
    async def initialize(self):
        """Initialize database connections."""
        try:
            settings = get_settings()
            
            logger.info(
                "Initializing database connections",
                neo4j_uri=settings.neo4j_uri,
                username=settings.neo4j_username
            )
            
            # Import neo4j driver
            from neo4j import AsyncGraphDatabase
            
            # Create Neo4j driver
            self._neo4j_driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password),
                max_connection_lifetime=3600,  # 1 hour
                max_connection_pool_size=50,
                connection_acquisition_timeout=60,
                connection_timeout=30,
                keep_alive=True
            )
            
            # Test connection
            await self.health_check()
            self.is_connected = True
            logger.info("Database connections initialized successfully")
            
        except Exception as e:
            logger.error(
                "Failed to initialize database connections",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            raise DatabaseConnectionError(f"Database initialization failed: {str(e)}")
    
    async def health_check(self) -> bool:
        """Check database health."""
        if not self._neo4j_driver:
            return False
            
        try:
            logger.debug("Performing database health check")
            async with self._neo4j_driver.session() as session:
                result = await session.run("RETURN 1 as health_check")
                record = await result.single()
                is_healthy = record and record["health_check"] == 1
                
                logger.debug("Database health check completed", is_healthy=is_healthy)
                return is_healthy
        except Exception as e:
            logger.error("Database health check failed", error=str(e), exc_info=True)
            return False
    
    async def get_neo4j_session(self):
        """Get Neo4j database session."""
        if not self._neo4j_driver or not self.is_connected:
            raise DatabaseConnectionError("Neo4j driver not initialized")
        return self._neo4j_driver.session()
    
    async def close(self):
        """Close database connections."""
        if self._neo4j_driver:
            logger.debug("Closing Neo4j driver connection")
            await self._neo4j_driver.close()
        self.is_connected = False
        logger.info("Database connections closed")


class WorkflowManager:
    """EHS Workflow manager with integrated components."""
    
    def __init__(self):
        self._workflow_graph = None
        self._query_router = None
        self._retrievers = {}
        self.is_initialized = False
    
    async def initialize(self, db_manager: DatabaseManager):
        """Initialize workflow components with real implementations."""
        try:
            settings = get_settings()
            
            logger.info(
                "Initializing workflow components",
                openai_configured=bool(settings.openai_api_key),
                model_name=settings.llm_model_name
            )
            
            # Initialize query router with proper LLM configuration
            logger.debug("Creating QueryRouterAgent")
            self._query_router = QueryRouterAgent(
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens
            )
            
            # Initialize workflow graph with real implementations
            logger.debug("Creating EHS workflow with integrated components")
            from ..workflows.ehs_workflow import create_ehs_workflow
            self._workflow_graph = await create_ehs_workflow(
                db_manager=db_manager,
                query_router=self._query_router
            )
            
            self.is_initialized = True
            
            # Log workflow status
            workflow_stats = self._workflow_graph.get_workflow_stats()
            
            logger.info(
                "Workflow components initialized successfully",
                query_router_initialized=bool(self._query_router),
                workflow_initialized=bool(self._workflow_graph),
                text2cypher_available=workflow_stats.get("retriever_usage", {}).get("text2cypher_available", False)
            )
            
        except ImportError as e:
            # Fallback when workflow is not yet implemented
            logger.warning("Workflow implementation not found, using placeholder", error=str(e))
            self._workflow_graph = None
            self.is_initialized = False
        except Exception as e:
            logger.error(
                "Failed to initialize workflow",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            # Don't raise - allow service to start with limited functionality
            logger.warning("Workflow initialization failed, continuing with limited functionality")
            self.is_initialized = False
    
    def get_workflow_graph(self):
        """Get the initialized workflow graph."""
        if not self.is_initialized or not self._workflow_graph:
            raise WorkflowError("Workflow not initialized or not available")
        return self._workflow_graph
    
    def get_query_router(self) -> QueryRouterAgent:
        """Get query router instance."""
        if not self._query_router:
            # Create fallback query router
            logger.warning("Creating fallback query router")
            settings = get_settings()
            self._query_router = QueryRouterAgent(
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens
            )
        return self._query_router
    
    async def health_check(self) -> bool:
        """Check workflow health."""
        try:
            if not self._query_router:
                logger.debug("Query router not available for health check")
                return False
            
            logger.debug("Performing workflow health check")
            
            # Test query router with a simple query
            result = self._query_router.classify_query("test health check query")
            router_healthy = result and result.confidence_score >= 0.0
            
            # Test workflow if available
            workflow_healthy = True
            if self._workflow_graph and self.is_initialized:
                try:
                    workflow_healthy = await self._workflow_graph.health_check()
                except Exception as e:
                    logger.warning("Workflow health check failed", error=str(e))
                    workflow_healthy = False
            
            overall_healthy = router_healthy and workflow_healthy
            
            logger.debug(
                "Workflow health check completed",
                router_healthy=router_healthy,
                workflow_healthy=workflow_healthy,
                overall_healthy=overall_healthy
            )
            
            return overall_healthy
            
        except Exception as e:
            logger.error("Workflow health check failed", error=str(e), exc_info=True)
            return False


# Global instances
db_manager = DatabaseManager()
workflow_manager = WorkflowManager()


@asynccontextmanager
async def get_database_session():
    """Async context manager for database sessions."""
    session = None
    try:
        session = await db_manager.get_neo4j_session()
        yield session
    except Exception as e:
        logger.error("Database session error", error=str(e))
        raise
    finally:
        if session:
            await session.close()


async def get_db_manager() -> DatabaseManager:
    """Dependency to get database manager."""
    if not db_manager.is_connected:
        try:
            logger.debug("Initializing database manager")
            await db_manager.initialize()
        except Exception as e:
            logger.error("Database manager initialization failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error_type": ErrorType.DATABASE_ERROR,
                    "message": "Database service unavailable",
                    "details": {"error": str(e)}
                }
            )
    return db_manager


async def get_workflow_manager() -> WorkflowManager:
    """Dependency to get workflow manager."""
    if not workflow_manager.is_initialized:
        try:
            logger.debug("Initializing workflow manager")
            await workflow_manager.initialize(db_manager)
        except Exception as e:
            logger.warning(
                "Workflow manager initialization failed, using fallback",
                error=str(e),
                error_type=type(e).__name__
            )
            # Continue with partial functionality
    return workflow_manager


async def get_workflow(
    workflow_mgr: WorkflowManager = Depends(get_workflow_manager)
):
    """Dependency to get the EHS workflow instance."""
    try:
        return workflow_mgr.get_workflow_graph()
    except WorkflowError as e:
        logger.warning("Workflow not available", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_type": ErrorType.SERVICE_UNAVAILABLE,
                "message": "Workflow service not available",
                "details": {"error": str(e)}
            }
        )


async def get_query_router(
    workflow_mgr: WorkflowManager = Depends(get_workflow_manager)
) -> QueryRouterAgent:
    """Dependency to get query router agent."""
    return workflow_mgr.get_query_router()


def get_current_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[str]:
    """Extract user ID from JWT token (placeholder implementation)."""
    settings = get_settings()
    
    if not settings.enable_auth:
        return "anonymous_user"
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error_type": ErrorType.AUTHORIZATION_ERROR,
                "message": "Authentication required",
                "details": {"scheme": "Bearer"}
            }
        )
    
    try:
        # Placeholder JWT validation
        # In production, implement proper JWT validation
        token = credentials.credentials
        if token == "demo-token":
            return "demo_user"
        
        # For now, return anonymous user for any token
        return "authenticated_user"
        
    except Exception as e:
        logger.error("Token validation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error_type": ErrorType.AUTHORIZATION_ERROR,
                "message": "Invalid authentication token",
                "details": {"error": str(e)}
            }
        )


def validate_request_rate_limit(request: Request) -> None:
    """Rate limiting validation (placeholder)."""
    # In production, implement proper rate limiting
    # For now, just log the request
    client_ip = request.client.host if request.client else "unknown"
    logger.info("API request received", client_ip=client_ip, path=request.url.path)


async def validate_query_request(request: Request) -> Dict[str, Any]:
    """Validate incoming query request."""
    try:
        # Check content type
        content_type = request.headers.get("content-type", "")
        if not content_type.startswith("application/json"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error_type": ErrorType.VALIDATION_ERROR,
                    "message": "Content-Type must be application/json",
                    "details": {"received_content_type": content_type}
                }
            )
        
        # Check request size (limit to 10MB)
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={
                    "error_type": ErrorType.VALIDATION_ERROR,
                    "message": "Request payload too large",
                    "details": {"max_size_bytes": 10 * 1024 * 1024}
                }
            )
        
        return {"validation": "passed"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Request validation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_type": ErrorType.VALIDATION_ERROR,
                "message": "Invalid request format",
                "details": {"error": str(e)}
            }
        )


class QuerySessionManager:
    """Manages query processing sessions."""
    
    def __init__(self):
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = 3600  # 1 hour
    
    async def create_session(self, query_id: str, user_id: str) -> str:
        """Create a new query processing session."""
        import uuid
        from datetime import datetime, timedelta
        
        session_id = str(uuid.uuid4())
        self.active_sessions[session_id] = {
            "query_id": query_id,
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(seconds=self.session_timeout),
            "status": "active"
        }
        
        logger.info("Query session created", session_id=session_id, query_id=query_id)
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information."""
        return self.active_sessions.get(session_id)
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        from datetime import datetime
        
        current_time = datetime.utcnow()
        expired_sessions = [
            session_id for session_id, session_data in self.active_sessions.items()
            if session_data["expires_at"] < current_time
        ]
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
        
        if expired_sessions:
            logger.info("Cleaned up expired sessions", count=len(expired_sessions))


# Global session manager
session_manager = QuerySessionManager()


async def get_session_manager() -> QuerySessionManager:
    """Dependency to get session manager."""
    # Periodically cleanup expired sessions
    asyncio.create_task(session_manager.cleanup_expired_sessions())
    return session_manager


# Background task for health monitoring
async def monitor_service_health():
    """Background task to monitor service health."""
    while True:
        try:
            # Check database health
            db_healthy = await db_manager.health_check() if db_manager.is_connected else False
            
            # Check workflow health
            workflow_healthy = await workflow_manager.health_check() if workflow_manager.is_initialized else False
            
            logger.info(
                "Service health check",
                database_healthy=db_healthy,
                workflow_healthy=workflow_healthy
            )
            
        except Exception as e:
            logger.error("Health monitoring error", error=str(e))
        
        # Wait 60 seconds before next check
        await asyncio.sleep(60)


# Startup and shutdown handlers
async def startup_handler():
    """Initialize services on startup."""
    logger.info("Starting EHS Analytics API services")
    
    try:
        # Initialize database
        logger.debug("Initializing database manager")
        await db_manager.initialize()
        
        # Initialize workflow
        logger.debug("Initializing workflow manager")
        await workflow_manager.initialize(db_manager)
        
        # Start background monitoring
        logger.debug("Starting background health monitoring")
        asyncio.create_task(monitor_service_health())
        
        # Log final status
        workflow_available = workflow_manager.is_initialized
        db_available = db_manager.is_connected
        
        logger.info(
            "All services initialized",
            database_available=db_available,
            workflow_available=workflow_available
        )
        
        if not workflow_available:
            logger.warning("Service started with limited functionality - workflow not available")
        
    except Exception as e:
        logger.error("Startup failed", error=str(e), exc_info=True)
        raise


async def shutdown_handler():
    """Clean up resources on shutdown."""
    logger.info("Shutting down EHS Analytics API services")
    
    try:
        # Close database connections
        await db_manager.close()
        
        # Clean up workflow resources
        if workflow_manager._workflow_graph and hasattr(workflow_manager._workflow_graph, 'cleanup'):
            try:
                await workflow_manager._workflow_graph.cleanup()
            except Exception as e:
                logger.warning("Workflow cleanup error", error=str(e))
        
        # Clean up sessions
        session_manager.active_sessions.clear()
        
        logger.info("Shutdown completed successfully")
        
    except Exception as e:
        logger.error("Shutdown error", error=str(e))