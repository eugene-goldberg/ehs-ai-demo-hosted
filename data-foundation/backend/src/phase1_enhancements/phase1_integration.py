"""
Phase 1 Enhancements Integration Module

This module serves as the single entry point for all Phase 1 features including:
- Pro-rating allocation system
- Audit trail tracking 
- Rejection workflow management

It provides unified initialization, configuration, and integration with the main FastAPI application.

Dependencies:
- All Phase 1 enhancement modules (prorating, audit_trail, rejection_tracking)
- FastAPI for web framework integration
- Neo4j graph database for data persistence
- Docker and environment configuration support
"""

import os
import sys
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.routing import APIRouter
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv

# Import Phase 1 enhancement modules
from .prorating_service import ProRatingService
from .prorating_api import AllocationResult as ProRatingAllocation, ProRatingRequest, MonthlyReportRequest as MonthlyReport

from .audit_trail_service import AuditTrailService
from .audit_trail_schema import AuditTrailEntry, DocumentInfo

from .rejection_workflow_service import RejectionWorkflowService
from .rejection_tracking_schema import RejectionEntry, ValidationRule

# Import shared utilities
from src.graph_query import get_graphDB_driver
from src.shared.common_fn import create_graph_database_connection
from src.graphDB_dataAccess import graphDBdataAccess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class Phase1Integration:
    """
    Main integration class for Phase 1 enhancements.
    
    Coordinates initialization and configuration of:
    - Pro-rating allocation system
    - Audit trail tracking
    - Rejection workflow management
    """
    
    def __init__(self, neo4j_uri: str = None, neo4j_username: str = None, 
                 neo4j_password: str = None, neo4j_database: str = None):
        """
        Initialize Phase 1 Integration with database connection parameters.
        
        Args:
            neo4j_uri: Neo4j database URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            neo4j_database: Neo4j database name
        """
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_username = neo4j_username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password")
        self.neo4j_database = neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")
        
        # Service instances
        self.prorating_service: Optional[ProRatingService] = None
        self.audit_trail_service: Optional[AuditTrailService] = None
        self.rejection_workflow_service: Optional[RejectionWorkflowService] = None
        
        # Graph connection
        self.graph: Optional[Neo4jGraph] = None
        self.driver = None
        
        # Configuration
        self.config = self._load_configuration()
        
        logger.info("Phase1Integration initialized")

    def _load_configuration(self) -> Dict[str, Any]:
        """Load configuration from environment variables and defaults."""
        return {
            # Environment detection
            "is_docker": os.path.exists('/.dockerenv'),
            "environment": os.getenv("ENVIRONMENT", "development"),
            
            # Database configuration
            "neo4j_config": {
                "uri": self.neo4j_uri,
                "username": self.neo4j_username,
                "password": self.neo4j_password,
                "database": self.neo4j_database
            },
            
            # Service-specific configurations
            "prorating_config": {
                "batch_size": int(os.getenv("PRORATING_BATCH_SIZE", "100")),
                "default_allocation_method": os.getenv("DEFAULT_ALLOCATION_METHOD", "square_footage"),
                "enable_backfill": os.getenv("ENABLE_PRORATING_BACKFILL", "true").lower() == "true"
            },
            
            "audit_trail_config": {
                "retention_days": int(os.getenv("AUDIT_RETENTION_DAYS", "365")),
                "enable_file_backup": os.getenv("ENABLE_FILE_BACKUP", "true").lower() == "true",
                "backup_location": os.getenv("BACKUP_LOCATION", "/tmp/audit_backups")
            },
            
            "rejection_config": {
                "auto_retry_enabled": os.getenv("AUTO_RETRY_ENABLED", "false").lower() == "true",
                "max_retry_attempts": int(os.getenv("MAX_RETRY_ATTEMPTS", "3")),
                "validation_strictness": os.getenv("VALIDATION_STRICTNESS", "medium")
            },
            
            # Document processing
            "document_processing": {
                "enable_utility_bill_prorating": os.getenv("ENABLE_UTILITY_PRORATING", "true").lower() == "true",
                "enable_audit_tracking": os.getenv("ENABLE_AUDIT_TRACKING", "true").lower() == "true",
                "enable_rejection_validation": os.getenv("ENABLE_REJECTION_VALIDATION", "true").lower() == "true"
            }
        }

    async def initialize_all_enhancements(self) -> bool:
        """
        Initialize all Phase 1 enhancement systems.
        
        Returns:
            bool: True if all systems initialized successfully
        """
        try:
            logger.info("Starting Phase 1 enhancements initialization...")
            
            # Create database connection
            await self._initialize_database_connection()
            
            # Initialize database schemas
            await self._initialize_schemas()
            
            # Initialize services
            await self._initialize_services()
            
            # Inject services into API modules
            self._inject_services_into_apis()
            
            # Validate initialization
            health_check_result = await self.health_check()
            
            if health_check_result["status"] == "healthy":
                logger.info("Phase 1 enhancements initialized successfully")
                return True
            else:
                logger.error(f"Phase 1 initialization failed: {health_check_result}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize Phase 1 enhancements: {str(e)}")
            return False

    async def _initialize_database_connection(self):
        """Initialize Neo4j database connection."""
        try:
            # Create graph connection - using userName parameter as required by the function
            self.graph = create_graph_database_connection(
                self.neo4j_uri,
                self.neo4j_username,  # Fixed: using userName parameter name
                self.neo4j_password,
                self.neo4j_database
            )
            
            # Create driver connection
            self.driver = get_graphDB_driver(
                self.neo4j_uri,
                self.neo4j_username,
                self.neo4j_password,
                self.neo4j_database
            )
            
            logger.info("Database connection initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {str(e)}")
            raise

    async def _initialize_schemas(self):
        """Initialize all database schemas for Phase 1 enhancements."""
        try:
            # Create ProRating schema
            prorating_service_temp = ProRatingService(self.graph)
            await prorating_service_temp.initialize_schema()
            
            # Create AuditTrail schema  
            audit_storage_path = os.getenv("AUDIT_TRAIL_STORAGE_PATH", "/app/storage/")
            audit_service_temp = AuditTrailService(audit_storage_path)
            await audit_service_temp.initialize_schema()
            
            # Create RejectionWorkflow schema
            rejection_service_temp = RejectionWorkflowService(self.graph)
            await rejection_service_temp.initialize_schema()
            
            logger.info("All Phase 1 schemas initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize schemas: {str(e)}")
            raise

    async def _initialize_services(self):
        """Initialize all Phase 1 services."""
        try:
            # Initialize ProRating service
            self.prorating_service = ProRatingService(self.graph)
            
            # Initialize AuditTrail service
            audit_storage_path = os.getenv("AUDIT_TRAIL_STORAGE_PATH", "/app/storage/")
            self.audit_trail_service = AuditTrailService(audit_storage_path)
            
            # Initialize RejectionWorkflow service
            self.rejection_workflow_service = RejectionWorkflowService(self.graph)
            
            logger.info("All Phase 1 services initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {str(e)}")
            raise

    def _inject_services_into_apis(self):
        """
        Inject initialized services into the API modules.
        This ensures the global service variables in each API module get the actual service instances.
        """
        try:
            # Import API modules to access their global service variables
            from . import prorating_api
            from . import audit_trail_api  
            from . import rejection_tracking_api
            
            # Inject services into API modules
            if self.prorating_service:
                prorating_api.prorating_service = self.prorating_service
                logger.info("ProRating service injected into prorating_api module")
            
            if self.audit_trail_service:
                audit_trail_api.audit_service = self.audit_trail_service
                logger.info("AuditTrail service injected into audit_trail_api module")
            
            if self.rejection_workflow_service:
                rejection_tracking_api.rejection_service = self.rejection_workflow_service
                logger.info("RejectionWorkflow service injected into rejection_tracking_api module")
            
            logger.info("All services successfully injected into API modules")
            
        except Exception as e:
            logger.error(f"Failed to inject services into API modules: {str(e)}")
            raise

    def integrate_with_app(self, app: FastAPI, api_prefix: str = "/api/v1") -> FastAPI:
        """
        Integrate all Phase 1 routers with the main FastAPI application.
        
        Args:
            app: FastAPI application instance
            api_prefix: API route prefix
            
        Returns:
            FastAPI: Updated application with Phase 1 routes
        """
        try:
            # Import routers after service injection to ensure services are available
            from .prorating_api import prorating_router
            from .audit_trail_api import audit_trail_router  
            from .rejection_tracking_api import rejection_tracking_router
            
            # Include Phase 1 routers
            app.include_router(
                prorating_router,
                prefix=f"{api_prefix}/prorating",
                tags=["Phase 1 - Pro-rating"]
            )
            
            app.include_router(
                audit_trail_router,
                prefix=f"{api_prefix}/audit-trail",
                tags=["Phase 1 - Audit Trail"]
            )
            
            app.include_router(
                rejection_tracking_router,
                prefix=f"{api_prefix}/rejection-tracking", 
                tags=["Phase 1 - Rejection Tracking"]
            )
            
            # Add Phase 1 health check endpoint
            @app.get(f"{api_prefix}/phase1/health")
            async def phase1_health_check():
                """Health check endpoint for Phase 1 enhancements."""
                return await self.health_check()
            
            # Add Phase 1 integration status endpoint
            @app.get(f"{api_prefix}/phase1/status")
            async def phase1_status():
                """Status endpoint for Phase 1 integration."""
                return {
                    "phase": "Phase 1",
                    "services": {
                        "prorating": self.prorating_service is not None,
                        "audit_trail": self.audit_trail_service is not None,
                        "rejection_workflow": self.rejection_workflow_service is not None
                    },
                    "database_connected": self.graph is not None,
                    "configuration": self.config,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            logger.info(f"Phase 1 routers integrated with FastAPI app at prefix: {api_prefix}")
            return app
            
        except Exception as e:
            logger.error(f"Failed to integrate Phase 1 with FastAPI app: {str(e)}")
            raise

    def configure_environment(self) -> Dict[str, Any]:
        """
        Configure environment-specific settings for Docker/local deployment.
        
        Returns:
            Dict: Environment configuration details
        """
        env_config = {
            "environment_type": "docker" if self.config["is_docker"] else "local",
            "database_host": self.neo4j_uri,
            "services_configured": []
        }
        
        try:
            # Docker-specific configurations
            if self.config["is_docker"]:
                logger.info("Configuring for Docker environment")
                
                # Adjust paths for Docker
                if self.config["audit_trail_config"]["backup_location"].startswith("/tmp"):
                    self.config["audit_trail_config"]["backup_location"] = "/app/data/audit_backups"
                
                # Docker network configuration
                env_config["database_network"] = "bridge"
                env_config["services_configured"].append("docker_paths")
                
            # Local development configurations
            else:
                logger.info("Configuring for local environment")
                
                # Ensure local directories exist
                backup_path = Path(self.config["audit_trail_config"]["backup_location"])
                backup_path.mkdir(parents=True, exist_ok=True)
                
                env_config["local_paths_created"] = str(backup_path)
                env_config["services_configured"].append("local_paths")
            
            # Common configurations
            env_config["neo4j_database"] = self.neo4j_database
            env_config["services_configured"].extend(["database", "logging"])
            
            logger.info(f"Environment configured: {env_config['environment_type']}")
            return env_config
            
        except Exception as e:
            logger.error(f"Failed to configure environment: {str(e)}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check for all Phase 1 services.
        
        Returns:
            Dict: Health status of all components
        """
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {},
            "database": {},
            "configuration": {}
        }
        
        try:
            # Check database connectivity
            if self.graph and self.driver:
                try:
                    with self.driver.session(database=self.neo4j_database) as session:
                        result = session.run("RETURN 1 as test")
                        if result.single()["test"] == 1:
                            health_status["database"]["status"] = "connected"
                        else:
                            health_status["database"]["status"] = "error"
                            health_status["status"] = "unhealthy"
                except Exception as e:
                    health_status["database"]["status"] = "disconnected"
                    health_status["database"]["error"] = str(e)
                    health_status["status"] = "unhealthy"
            else:
                health_status["database"]["status"] = "not_initialized"
                health_status["status"] = "unhealthy"
            
            # Check ProRating service
            if self.prorating_service:
                try:
                    # Test basic functionality
                    test_result = await self.prorating_service.test_connection()
                    health_status["services"]["prorating"] = {
                        "status": "healthy" if test_result else "unhealthy"
                    }
                except Exception as e:
                    health_status["services"]["prorating"] = {
                        "status": "error",
                        "error": str(e)
                    }
                    health_status["status"] = "unhealthy"
            else:
                health_status["services"]["prorating"] = {"status": "not_initialized"}
                health_status["status"] = "unhealthy"
            
            # Check AuditTrail service  
            if self.audit_trail_service:
                try:
                    test_result = await self.audit_trail_service.test_connection()
                    health_status["services"]["audit_trail"] = {
                        "status": "healthy" if test_result else "unhealthy"
                    }
                except Exception as e:
                    health_status["services"]["audit_trail"] = {
                        "status": "error", 
                        "error": str(e)
                    }
                    health_status["status"] = "unhealthy"
            else:
                health_status["services"]["audit_trail"] = {"status": "not_initialized"}
                health_status["status"] = "unhealthy"
            
            # Check RejectionWorkflow service
            if self.rejection_workflow_service:
                try:
                    test_result = await self.rejection_workflow_service.test_connection()
                    health_status["services"]["rejection_workflow"] = {
                        "status": "healthy" if test_result else "unhealthy"
                    }
                except Exception as e:
                    health_status["services"]["rejection_workflow"] = {
                        "status": "error",
                        "error": str(e)
                    }
                    health_status["status"] = "unhealthy"
            else:
                health_status["services"]["rejection_workflow"] = {"status": "not_initialized"}
                health_status["status"] = "unhealthy"
            
            # Configuration check
            health_status["configuration"] = {
                "environment": self.config["environment"],
                "is_docker": self.config["is_docker"],
                "services_enabled": {
                    "prorating": self.config["document_processing"]["enable_utility_bill_prorating"],
                    "audit_tracking": self.config["document_processing"]["enable_audit_tracking"],
                    "rejection_validation": self.config["document_processing"]["enable_rejection_validation"]
                }
            }
            
        except Exception as e:
            health_status["status"] = "error"
            health_status["error"] = str(e)
            logger.error(f"Health check failed: {str(e)}")
        
        return health_status

    async def process_document_with_phase1_features(self, document_info: Dict[str, Any], 
                                                  background_tasks: BackgroundTasks) -> Dict[str, Any]:
        """
        Process a document through all Phase 1 enhancement features.
        
        Args:
            document_info: Document information including filename, content, metadata
            background_tasks: FastAPI background tasks for async processing
            
        Returns:
            Dict: Processing results from all Phase 1 features
        """
        processing_results = {
            "document": document_info.get("filename"),
            "timestamp": datetime.utcnow().isoformat(),
            "phase1_processing": {}
        }
        
        try:
            # Create audit trail entry
            if (self.audit_trail_service and 
                self.config["document_processing"]["enable_audit_tracking"]):
                
                audit_entry = await self.audit_trail_service.create_audit_trail(
                    document_name=document_info["filename"],
                    action="document_uploaded",
                    user_id=document_info.get("user_id", "system"),
                    metadata=document_info.get("metadata", {})
                )
                processing_results["phase1_processing"]["audit_trail"] = audit_entry
            
            # Check for rejection validation
            if (self.rejection_workflow_service and 
                self.config["document_processing"]["enable_rejection_validation"]):
                
                validation_result = await self.rejection_workflow_service.validate_document(
                    document_info["filename"],
                    document_info.get("content", ""),
                    document_info.get("metadata", {})
                )
                processing_results["phase1_processing"]["validation"] = validation_result
                
                # If document is rejected, stop further processing
                if validation_result.get("status") == "rejected":
                    processing_results["status"] = "rejected"
                    processing_results["reason"] = validation_result.get("reason")
                    return processing_results
            
            # Process utility bills with pro-rating (if applicable)
            if (self.prorating_service and 
                self.config["document_processing"]["enable_utility_bill_prorating"] and
                self._is_utility_bill(document_info)):
                
                # Schedule background pro-rating processing
                background_tasks.add_task(
                    self._process_utility_bill_prorating,
                    document_info
                )
                processing_results["phase1_processing"]["prorating"] = "scheduled"
            
            processing_results["status"] = "success"
            logger.info(f"Document {document_info['filename']} processed through Phase 1 features")
            
        except Exception as e:
            processing_results["status"] = "error"
            processing_results["error"] = str(e)
            logger.error(f"Failed to process document through Phase 1 features: {str(e)}")
        
        return processing_results

    async def _process_utility_bill_prorating(self, document_info: Dict[str, Any]):
        """Background task to process utility bill pro-rating."""
        try:
            # Extract utility data from document
            utility_data = self._extract_utility_data(document_info)
            
            if utility_data:
                # Create pro-rating allocation
                allocation = await self.prorating_service.create_allocation(
                    document_name=document_info["filename"],
                    period_start=utility_data.get("period_start"),
                    period_end=utility_data.get("period_end"),
                    total_amount=utility_data.get("total_amount"),
                    utility_type=utility_data.get("utility_type", "electricity"),
                    allocation_method=self.config["prorating_config"]["default_allocation_method"]
                )
                
                logger.info(f"Pro-rating allocation created for {document_info['filename']}: {allocation}")
                
        except Exception as e:
            logger.error(f"Failed to process utility bill pro-rating: {str(e)}")

    def _is_utility_bill(self, document_info: Dict[str, Any]) -> bool:
        """Determine if document is a utility bill based on filename or content."""
        filename = document_info.get("filename", "").lower()
        utility_keywords = ["electric", "electricity", "water", "gas", "utility", "bill"]
        
        return any(keyword in filename for keyword in utility_keywords)

    def _extract_utility_data(self, document_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract utility data from document (placeholder implementation)."""
        # This would integrate with document parsing/OCR functionality
        # For now, return mock data
        return {
            "period_start": "2024-01-01",
            "period_end": "2024-01-31", 
            "total_amount": 1250.00,
            "utility_type": "electricity"
        }

    async def migrate_existing_data(self) -> Dict[str, Any]:
        """
        Migration script to update existing data with Phase 1 enhancements.
        
        Returns:
            Dict: Migration results and statistics
        """
        migration_results = {
            "started_at": datetime.utcnow().isoformat(),
            "migration_steps": [],
            "statistics": {}
        }
        
        try:
            # Step 1: Migrate existing documents to include audit trail
            logger.info("Starting audit trail migration...")
            audit_migration = await self._migrate_audit_trail_data()
            migration_results["migration_steps"].append({
                "step": "audit_trail_migration",
                "status": "completed" if audit_migration["success"] else "failed",
                "details": audit_migration
            })
            migration_results["statistics"]["documents_migrated"] = audit_migration.get("count", 0)
            
            # Step 2: Initialize rejection tracking for existing failed documents
            logger.info("Starting rejection tracking migration...")
            rejection_migration = await self._migrate_rejection_tracking_data()
            migration_results["migration_steps"].append({
                "step": "rejection_tracking_migration", 
                "status": "completed" if rejection_migration["success"] else "failed",
                "details": rejection_migration
            })
            migration_results["statistics"]["rejections_migrated"] = rejection_migration.get("count", 0)
            
            # Step 3: Create pro-rating allocations for existing utility bills
            logger.info("Starting pro-rating migration...")
            prorating_migration = await self._migrate_prorating_data()
            migration_results["migration_steps"].append({
                "step": "prorating_migration",
                "status": "completed" if prorating_migration["success"] else "failed", 
                "details": prorating_migration
            })
            migration_results["statistics"]["allocations_created"] = prorating_migration.get("count", 0)
            
            migration_results["completed_at"] = datetime.utcnow().isoformat()
            migration_results["overall_status"] = "completed"
            
            logger.info("Phase 1 data migration completed successfully")
            
        except Exception as e:
            migration_results["overall_status"] = "failed"
            migration_results["error"] = str(e)
            logger.error(f"Phase 1 data migration failed: {str(e)}")
        
        return migration_results

    async def _migrate_audit_trail_data(self) -> Dict[str, Any]:
        """Migrate existing documents to include audit trail entries."""
        try:
            # Query existing documents
            with self.driver.session(database=self.neo4j_database) as session:
                result = session.run("""
                    MATCH (doc:Document)
                    WHERE NOT EXISTS((doc)-[:HAS_AUDIT_TRAIL]->())
                    RETURN doc.file_name as filename, doc.created_at as created_at
                    LIMIT 1000
                """)
                
                documents = [record for record in result]
                
                # Create audit trail entries for existing documents
                count = 0
                for doc in documents:
                    await self.audit_trail_service.create_audit_trail(
                        document_name=doc["filename"],
                        action="migrated_to_audit_trail",
                        user_id="system",
                        metadata={"migration": True, "original_created_at": doc["created_at"]}
                    )
                    count += 1
                
                return {"success": True, "count": count}
                
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _migrate_rejection_tracking_data(self) -> Dict[str, Any]:
        """Migrate existing failed documents to rejection tracking."""
        try:
            # Query documents with failed status
            with self.driver.session(database=self.neo4j_database) as session:
                result = session.run("""
                    MATCH (doc:Document)
                    WHERE doc.status = 'Failed'
                    AND NOT EXISTS((doc)-[:HAS_REJECTION]->())
                    RETURN doc.file_name as filename, doc.created_at as failed_at
                    LIMIT 1000
                """)
                
                failed_docs = [record for record in result]
                
                # Create rejection entries for failed documents  
                count = 0
                for doc in failed_docs:
                    await self.rejection_workflow_service.create_rejection(
                        document_name=doc["filename"],
                        rejection_reason="Historical failure (migrated)",
                        details={"migration": True, "failed_at": doc["failed_at"]},
                        assigned_to="system"
                    )
                    count += 1
                
                return {"success": True, "count": count}
                
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _migrate_prorating_data(self) -> Dict[str, Any]:
        """Create pro-rating allocations for existing utility bills."""
        try:
            # Query existing utility bill documents
            with self.driver.session(database=self.neo4j_database) as session:
                result = session.run("""
                    MATCH (doc:Document)
                    WHERE doc.file_name =~ '(?i).*(electric|utility|water|gas|bill).*'
                    AND doc.status = 'Completed'
                    AND NOT EXISTS((doc)-[:HAS_ALLOCATION]->())
                    RETURN doc.file_name as filename, doc.created_at as created_at
                    LIMIT 100
                """)
                
                utility_docs = [record for record in result]
                
                # Create allocations for utility documents
                count = 0
                for doc in utility_docs:
                    # Create basic allocation (would normally extract real data)
                    await self.prorating_service.create_allocation(
                        document_name=doc["filename"],
                        period_start="2024-01-01",
                        period_end="2024-01-31",
                        total_amount=1000.0,  # Placeholder
                        utility_type="electricity",
                        allocation_method="square_footage",
                        metadata={"migration": True, "original_created_at": doc["created_at"]}
                    )
                    count += 1
                
                return {"success": True, "count": count}
                
        except Exception as e:
            return {"success": False, "error": str(e)}


# Global instance tracking
_current_phase1_integration = None

def get_current_phase1_integration():
    """Get the current Phase1Integration instance."""
    return _current_phase1_integration


def create_phase1_integration(neo4j_uri: str = None, neo4j_username: str = None,
                             neo4j_password: str = None, neo4j_database: str = None) -> Phase1Integration:
    """
    Factory function to create Phase1Integration instance.
    
    Args:
        neo4j_uri: Neo4j database URI
        neo4j_username: Neo4j username  
        neo4j_password: Neo4j password
        neo4j_database: Neo4j database name
        
    Returns:
        Phase1Integration: Configured integration instance
    """
    global _current_phase1_integration
    _current_phase1_integration = Phase1Integration(
        neo4j_uri=neo4j_uri,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password,
        neo4j_database=neo4j_database
    )
    return _current_phase1_integration


async def initialize_phase1_for_app(app: FastAPI, api_prefix: str = "/api/v1") -> Phase1Integration:
    """
    Convenience function to initialize Phase 1 integration and integrate with FastAPI app.
    
    Args:
        app: FastAPI application instance
        api_prefix: API route prefix
        
    Returns:
        Phase1Integration: Initialized integration instance
    """
    # Create integration instance
    phase1_integration = create_phase1_integration()
    
    # Initialize all enhancements
    initialization_success = await phase1_integration.initialize_all_enhancements()
    
    if not initialization_success:
        raise RuntimeError("Failed to initialize Phase 1 enhancements")
    
    # Configure environment
    env_config = phase1_integration.configure_environment()
    logger.info(f"Environment configured: {env_config}")
    
    # Integrate with FastAPI app
    updated_app = phase1_integration.integrate_with_app(app, api_prefix)
    
    logger.info("Phase 1 integration completed successfully")
    return phase1_integration


# Usage example for main application
if __name__ == "__main__":
    import asyncio
    from fastapi import FastAPI
    
    async def main():
        # Create FastAPI app
        app = FastAPI(title="EHS Data Platform with Phase 1 Enhancements")
        
        # Initialize Phase 1 integration
        phase1_integration = await initialize_phase1_for_app(app)
        
        # Run health check
        health_status = await phase1_integration.health_check()
        print(f"Health Status: {health_status}")
        
        # Run migration (if needed)
        # migration_results = await phase1_integration.migrate_existing_data()
        # print(f"Migration Results: {migration_results}")
    
    # Run the example
    asyncio.run(main())