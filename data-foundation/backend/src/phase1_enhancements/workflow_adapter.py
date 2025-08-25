"""
Workflow Adapter Module

This module provides adapter functions to integrate Phase1Integration with the enhanced workflows.
It bridges the gap between existing phase1_integration.py and the enhanced workflow systems
by providing clean interfaces, service initialization, and state conversion utilities.

Key Features:
- Service initialization and dependency injection
- State conversion between workflows and Phase 1 services
- Audit trail management during workflow execution
- Error handling and logging
- Factory methods for easy workflow creation with Phase 1 features
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from langchain_neo4j import Neo4jGraph
from neo4j import GraphDatabase

# Import Phase 1 enhancements
from .phase1_integration import Phase1Integration
from .audit_trail_service import AuditTrailService
from .audit_trail_schema import AuditTrailEntry, DocumentInfo, ProcessingStage
from .prorating_service import ProRatingService
from .prorating_schema import ProRatingAllocation, ProRatingRequest
from .rejection_workflow_service import RejectionWorkflowService, ValidationResult
from .rejection_tracking_schema import RejectionEntry, ValidationRule

# Import workflow states (assuming they exist)
try:
    from ..workflows.ingestion_workflow_enhanced import DocumentState
    from ..workflows.extraction_workflow_enhanced import ExtractionState
except ImportError:
    # Define minimal state types if imports fail
    DocumentState = Dict[str, Any]
    ExtractionState = Dict[str, Any]

logger = logging.getLogger(__name__)


class WorkflowType(Enum):
    """Supported workflow types."""
    INGESTION = "ingestion"
    EXTRACTION = "extraction"


class AdapterError(Exception):
    """Base exception for workflow adapter errors."""
    pass


class ServiceInitializationError(AdapterError):
    """Error during service initialization."""
    pass


class StateConversionError(AdapterError):
    """Error during state conversion."""
    pass


@dataclass
class WorkflowServices:
    """Container for initialized Phase 1 services."""
    prorating_service: Optional[ProRatingService] = None
    audit_trail_service: Optional[AuditTrailService] = None
    rejection_workflow_service: Optional[RejectionWorkflowService] = None
    graph: Optional[Neo4jGraph] = None
    driver: Optional[Any] = None
    
    def is_fully_initialized(self) -> bool:
        """Check if all services are properly initialized."""
        return all([
            self.prorating_service is not None,
            self.audit_trail_service is not None,
            self.rejection_workflow_service is not None,
            self.graph is not None
        ])


@dataclass
class WorkflowContext:
    """Context information for workflow execution."""
    workflow_type: WorkflowType
    workflow_id: str = field(default_factory=lambda: f"wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class WorkflowAdapter:
    """
    Main adapter class for integrating Phase 1 features with enhanced workflows.
    
    This class provides a clean interface between the Phase1Integration system
    and the LangGraph-based enhanced workflows, handling service initialization,
    state conversion, and audit trail management.
    """
    
    def __init__(self, phase1_integration: Optional[Phase1Integration] = None,
                 neo4j_uri: str = None, neo4j_username: str = None,
                 neo4j_password: str = None, neo4j_database: str = None):
        """
        Initialize the WorkflowAdapter.
        
        Args:
            phase1_integration: Existing Phase1Integration instance
            neo4j_uri: Neo4j database URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            neo4j_database: Neo4j database name
        """
        self.phase1_integration = phase1_integration
        self.services: Optional[WorkflowServices] = None
        self._initialization_lock = False
        
        # Store connection parameters
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_username = neo4j_username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password")
        self.neo4j_database = neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")
        
        logger.info("WorkflowAdapter initialized")
    
    async def initialize_services(self) -> WorkflowServices:
        """
        Initialize all Phase 1 services for workflow use.
        
        Returns:
            WorkflowServices: Container with initialized services
            
        Raises:
            ServiceInitializationError: If services cannot be initialized
        """
        if self._initialization_lock:
            logger.warning("Services initialization already in progress")
            return self.services
            
        try:
            self._initialization_lock = True
            logger.info("Initializing Phase 1 services for workflow integration...")
            
            # Create or use existing Phase1Integration
            if not self.phase1_integration:
                self.phase1_integration = Phase1Integration(
                    neo4j_uri=self.neo4j_uri,
                    neo4j_username=self.neo4j_username,
                    neo4j_password=self.neo4j_password,
                    neo4j_database=self.neo4j_database
                )
            
            # Initialize the Phase 1 system
            success = await self.phase1_integration.initialize_all_enhancements()
            if not success:
                raise ServiceInitializationError("Failed to initialize Phase 1 enhancements")
            
            # Create services container
            self.services = WorkflowServices(
                prorating_service=self.phase1_integration.prorating_service,
                audit_trail_service=self.phase1_integration.audit_trail_service,
                rejection_workflow_service=self.phase1_integration.rejection_workflow_service,
                graph=self.phase1_integration.graph,
                driver=self.phase1_integration.driver
            )
            
            # Verify initialization
            if not self.services.is_fully_initialized():
                raise ServiceInitializationError("Not all services were properly initialized")
            
            logger.info("All Phase 1 services initialized successfully")
            return self.services
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {str(e)}")
            raise ServiceInitializationError(f"Service initialization failed: {str(e)}")
        finally:
            self._initialization_lock = False
    
    def get_phase1_services(self) -> WorkflowServices:
        """
        Get initialized Phase 1 services for workflows.
        
        Returns:
            WorkflowServices: Container with services
            
        Raises:
            ServiceInitializationError: If services not initialized
        """
        if not self.services or not self.services.is_fully_initialized():
            raise ServiceInitializationError("Services not properly initialized. Call initialize_services() first.")
        
        return self.services
    
    async def adapt_ingestion_workflow(self, workflow_state: DocumentState,
                                     context: WorkflowContext) -> Dict[str, Any]:
        """
        Adapt ingestion workflow to work with Phase 1 services.
        
        Args:
            workflow_state: Current document workflow state
            context: Workflow context information
            
        Returns:
            Dict containing adapted workflow configuration
        """
        try:
            logger.info(f"Adapting ingestion workflow for document: {workflow_state.get('document_id', 'unknown')}")
            
            services = self.get_phase1_services()
            
            # Convert workflow state to DocumentInfo for audit trail
            doc_info = self._convert_document_state_to_info(workflow_state)
            
            # Start audit trail for ingestion workflow
            audit_entry = await services.audit_trail_service.log_processing_start(
                document_info=doc_info,
                stage=ProcessingStage.INGESTION,
                user_id=context.user_id,
                metadata={
                    "workflow_id": context.workflow_id,
                    "workflow_type": context.workflow_type.value,
                    **context.metadata
                }
            )
            
            # Prepare workflow configuration with Phase 1 hooks
            adapted_config = {
                "phase1_services": {
                    "prorating": services.prorating_service,
                    "audit_trail": services.audit_trail_service,
                    "rejection_workflow": services.rejection_workflow_service
                },
                "audit_entry_id": audit_entry.entry_id,
                "workflow_context": context,
                "enhanced_features": {
                    "enable_prorating": True,
                    "enable_audit_trail": True,
                    "enable_rejection_validation": True
                },
                "hooks": {
                    "pre_processing": self._create_pre_processing_hook(services, audit_entry),
                    "post_processing": self._create_post_processing_hook(services, audit_entry),
                    "error_handling": self._create_error_handling_hook(services, audit_entry)
                }
            }
            
            logger.info("Ingestion workflow adapted successfully")
            return adapted_config
            
        except Exception as e:
            logger.error(f"Failed to adapt ingestion workflow: {str(e)}")
            raise AdapterError(f"Ingestion workflow adaptation failed: {str(e)}")
    
    async def adapt_extraction_workflow(self, workflow_state: ExtractionState,
                                      context: WorkflowContext) -> Dict[str, Any]:
        """
        Adapt extraction workflow to work with Phase 1 services.
        
        Args:
            workflow_state: Current extraction workflow state
            context: Workflow context information
            
        Returns:
            Dict containing adapted workflow configuration
        """
        try:
            logger.info(f"Adapting extraction workflow for query: {context.workflow_id}")
            
            services = self.get_phase1_services()
            
            # Create DocumentInfo from extraction state
            doc_info = self._convert_extraction_state_to_info(workflow_state)
            
            # Start audit trail for extraction workflow
            audit_entry = await services.audit_trail_service.log_processing_start(
                document_info=doc_info,
                stage=ProcessingStage.EXTRACTION,
                user_id=context.user_id,
                metadata={
                    "workflow_id": context.workflow_id,
                    "workflow_type": context.workflow_type.value,
                    "query_config": workflow_state.get("query_config", {}),
                    **context.metadata
                }
            )
            
            # Prepare workflow configuration with Phase 1 hooks
            adapted_config = {
                "phase1_services": {
                    "prorating": services.prorating_service,
                    "audit_trail": services.audit_trail_service,
                    "rejection_workflow": services.rejection_workflow_service
                },
                "audit_entry_id": audit_entry.entry_id,
                "workflow_context": context,
                "enhanced_features": {
                    "enable_prorating_analysis": True,
                    "enable_audit_trail": True,
                    "enable_result_validation": True
                },
                "hooks": {
                    "pre_extraction": self._create_pre_extraction_hook(services, audit_entry),
                    "post_extraction": self._create_post_extraction_hook(services, audit_entry),
                    "error_handling": self._create_error_handling_hook(services, audit_entry)
                }
            }
            
            logger.info("Extraction workflow adapted successfully")
            return adapted_config
            
        except Exception as e:
            logger.error(f"Failed to adapt extraction workflow: {str(e)}")
            raise AdapterError(f"Extraction workflow adaptation failed: {str(e)}")
    
    def _convert_document_state_to_info(self, state: DocumentState) -> DocumentInfo:
        """Convert DocumentState to DocumentInfo for audit trail."""
        try:
            return DocumentInfo(
                document_id=state.get("document_id", "unknown"),
                filename=Path(state.get("file_path", "unknown")).name,
                file_size=state.get("file_size", 0),
                content_type=state.get("content_type", "application/octet-stream"),
                upload_timestamp=state.get("upload_timestamp", datetime.utcnow()),
                metadata=state.get("upload_metadata", {})
            )
        except Exception as e:
            logger.error(f"Failed to convert document state to info: {str(e)}")
            raise StateConversionError(f"State conversion failed: {str(e)}")
    
    def _convert_extraction_state_to_info(self, state: ExtractionState) -> DocumentInfo:
        """Convert ExtractionState to DocumentInfo for audit trail."""
        try:
            query_config = state.get("query_config", {})
            return DocumentInfo(
                document_id=f"extraction_{state.get('workflow_id', 'unknown')}",
                filename=f"extraction_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                file_size=len(str(query_config)),
                content_type="application/json",
                upload_timestamp=datetime.utcnow(),
                metadata={
                    "query_type": query_config.get("query_type", "unknown"),
                    "output_format": state.get("output_format", "json"),
                    "report_template": state.get("report_template")
                }
            )
        except Exception as e:
            logger.error(f"Failed to convert extraction state to info: {str(e)}")
            raise StateConversionError(f"State conversion failed: {str(e)}")
    
    def convert_phase1_results_to_workflow_state(self, results: Dict[str, Any],
                                               workflow_type: WorkflowType) -> Dict[str, Any]:
        """
        Convert Phase 1 service results back to workflow state format.
        
        Args:
            results: Results from Phase 1 services
            workflow_type: Type of workflow
            
        Returns:
            Dict containing workflow state updates
        """
        try:
            if workflow_type == WorkflowType.INGESTION:
                return self._convert_to_ingestion_state(results)
            elif workflow_type == WorkflowType.EXTRACTION:
                return self._convert_to_extraction_state(results)
            else:
                raise ValueError(f"Unsupported workflow type: {workflow_type}")
                
        except Exception as e:
            logger.error(f"Failed to convert Phase 1 results to workflow state: {str(e)}")
            raise StateConversionError(f"Results conversion failed: {str(e)}")
    
    def _convert_to_ingestion_state(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Phase 1 results to ingestion workflow state format."""
        return {
            "phase1_results": results,
            "audit_trail_entries": results.get("audit_entries", []),
            "prorating_allocations": results.get("prorating_results", []),
            "rejection_validations": results.get("rejection_results", []),
            "processing_metadata": {
                "phase1_processing_time": results.get("processing_time", 0),
                "services_used": list(results.keys()),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    
    def _convert_to_extraction_state(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Phase 1 results to extraction workflow state format."""
        return {
            "phase1_analysis": results,
            "audit_trail_entries": results.get("audit_entries", []),
            "prorating_analysis": results.get("prorating_analysis", {}),
            "result_validations": results.get("validation_results", []),
            "enhanced_metadata": {
                "phase1_analysis_time": results.get("analysis_time", 0),
                "services_used": list(results.keys()),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    
    def manage_audit_trail_during_workflow(self, audit_entry_id: str,
                                         stage_updates: List[Dict[str, Any]]) -> None:
        """
        Update audit trail entries during workflow execution.
        
        Args:
            audit_entry_id: ID of the audit entry to update
            stage_updates: List of stage update information
        """
        try:
            services = self.get_phase1_services()
            
            for update in stage_updates:
                asyncio.run(services.audit_trail_service.log_processing_stage(
                    entry_id=audit_entry_id,
                    stage=ProcessingStage(update["stage"]),
                    status=update["status"],
                    details=update.get("details", {}),
                    error=update.get("error")
                ))
                
        except Exception as e:
            logger.error(f"Failed to update audit trail: {str(e)}")
            # Don't raise exception to avoid breaking workflow
    
    def _create_pre_processing_hook(self, services: WorkflowServices, audit_entry: AuditTrailEntry):
        """Create pre-processing hook for ingestion workflow."""
        async def pre_processing_hook(state: Dict[str, Any]) -> Dict[str, Any]:
            try:
                # Log start of processing stage
                await services.audit_trail_service.log_processing_stage(
                    entry_id=audit_entry.entry_id,
                    stage=ProcessingStage.PARSING,
                    status="started",
                    details={"document_type": state.get("document_type")}
                )
                return state
            except Exception as e:
                logger.error(f"Pre-processing hook error: {str(e)}")
                return state
        
        return pre_processing_hook
    
    def _create_post_processing_hook(self, services: WorkflowServices, audit_entry: AuditTrailEntry):
        """Create post-processing hook for ingestion workflow."""
        async def post_processing_hook(state: Dict[str, Any]) -> Dict[str, Any]:
            try:
                # Log completion of processing
                await services.audit_trail_service.log_processing_completion(
                    entry_id=audit_entry.entry_id,
                    success=not state.get("errors"),
                    final_results=state.get("processing_results", {}),
                    metrics=state.get("processing_metrics", {})
                )
                return state
            except Exception as e:
                logger.error(f"Post-processing hook error: {str(e)}")
                return state
        
        return post_processing_hook
    
    def _create_pre_extraction_hook(self, services: WorkflowServices, audit_entry: AuditTrailEntry):
        """Create pre-extraction hook for extraction workflow."""
        async def pre_extraction_hook(state: Dict[str, Any]) -> Dict[str, Any]:
            try:
                # Log start of extraction
                await services.audit_trail_service.log_processing_stage(
                    entry_id=audit_entry.entry_id,
                    stage=ProcessingStage.EXTRACTION,
                    status="started",
                    details={"query_config": state.get("query_config", {})}
                )
                return state
            except Exception as e:
                logger.error(f"Pre-extraction hook error: {str(e)}")
                return state
        
        return pre_extraction_hook
    
    def _create_post_extraction_hook(self, services: WorkflowServices, audit_entry: AuditTrailEntry):
        """Create post-extraction hook for extraction workflow."""
        async def post_extraction_hook(state: Dict[str, Any]) -> Dict[str, Any]:
            try:
                # Log completion of extraction
                await services.audit_trail_service.log_processing_completion(
                    entry_id=audit_entry.entry_id,
                    success=not state.get("errors"),
                    final_results=state.get("report_data", {}),
                    metrics=state.get("processing_metrics", {})
                )
                return state
            except Exception as e:
                logger.error(f"Post-extraction hook error: {str(e)}")
                return state
        
        return post_extraction_hook
    
    def _create_error_handling_hook(self, services: WorkflowServices, audit_entry: AuditTrailEntry):
        """Create error handling hook for workflows."""
        async def error_handling_hook(state: Dict[str, Any], error: Exception) -> Dict[str, Any]:
            try:
                # Log error in audit trail
                await services.audit_trail_service.log_processing_stage(
                    entry_id=audit_entry.entry_id,
                    stage=ProcessingStage.ERROR_HANDLING,
                    status="failed",
                    error=str(error),
                    details={"error_type": type(error).__name__}
                )
                
                # Check if error should trigger rejection workflow
                validation_result = await services.rejection_workflow_service.validate_document(
                    document_id=state.get("document_id", "unknown"),
                    validation_data={"error": str(error), "state": state}
                )
                
                state["rejection_validation"] = validation_result
                return state
                
            except Exception as hook_error:
                logger.error(f"Error handling hook error: {str(hook_error)}")
                return state
        
        return error_handling_hook


# Factory Functions for Easy Workflow Creation

async def create_enhanced_ingestion_workflow(document_id: str, file_path: str,
                                           user_id: str = None,
                                           metadata: Dict[str, Any] = None) -> Tuple[WorkflowAdapter, Dict[str, Any]]:
    """
    Factory function to create an enhanced ingestion workflow with Phase 1 features.
    
    Args:
        document_id: Unique document identifier
        file_path: Path to the document file
        user_id: User performing the operation
        metadata: Additional metadata
        
    Returns:
        Tuple of (WorkflowAdapter, workflow_config)
    """
    try:
        # Initialize adapter
        adapter = WorkflowAdapter()
        await adapter.initialize_services()
        
        # Create workflow context
        context = WorkflowContext(
            workflow_type=WorkflowType.INGESTION,
            user_id=user_id,
            metadata=metadata or {}
        )
        
        # Create document state
        document_state = {
            "document_id": document_id,
            "file_path": file_path,
            "upload_metadata": metadata or {},
            "upload_timestamp": datetime.utcnow()
        }
        
        # Adapt workflow
        workflow_config = await adapter.adapt_ingestion_workflow(document_state, context)
        
        logger.info(f"Enhanced ingestion workflow created for document: {document_id}")
        return adapter, workflow_config
        
    except Exception as e:
        logger.error(f"Failed to create enhanced ingestion workflow: {str(e)}")
        raise


async def create_enhanced_extraction_workflow(query_config: Dict[str, Any],
                                            output_format: str = "json",
                                            user_id: str = None,
                                            metadata: Dict[str, Any] = None) -> Tuple[WorkflowAdapter, Dict[str, Any]]:
    """
    Factory function to create an enhanced extraction workflow with Phase 1 features.
    
    Args:
        query_config: Configuration for the extraction query
        output_format: Desired output format
        user_id: User performing the operation
        metadata: Additional metadata
        
    Returns:
        Tuple of (WorkflowAdapter, workflow_config)
    """
    try:
        # Initialize adapter
        adapter = WorkflowAdapter()
        await adapter.initialize_services()
        
        # Create workflow context
        context = WorkflowContext(
            workflow_type=WorkflowType.EXTRACTION,
            user_id=user_id,
            metadata=metadata or {}
        )
        
        # Create extraction state
        extraction_state = {
            "query_config": query_config,
            "output_format": output_format,
            "workflow_id": context.workflow_id
        }
        
        # Adapt workflow
        workflow_config = await adapter.adapt_extraction_workflow(extraction_state, context)
        
        logger.info(f"Enhanced extraction workflow created: {context.workflow_id}")
        return adapter, workflow_config
        
    except Exception as e:
        logger.error(f"Failed to create enhanced extraction workflow: {str(e)}")
        raise


# Utility Functions

def get_workflow_adapter_health_status() -> Dict[str, Any]:
    """
    Get health status of the workflow adapter system.
    
    Returns:
        Dict containing health status information
    """
    return {
        "service": "WorkflowAdapter",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "capabilities": {
            "ingestion_workflow_adaptation": True,
            "extraction_workflow_adaptation": True,
            "phase1_service_integration": True,
            "audit_trail_management": True,
            "state_conversion": True
        },
        "version": "1.0.0"
    }


def validate_workflow_configuration(config: Dict[str, Any]) -> List[str]:
    """
    Validate workflow configuration for completeness.
    
    Args:
        config: Workflow configuration to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    required_keys = ["phase1_services", "workflow_context", "enhanced_features"]
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required configuration key: {key}")
    
    if "phase1_services" in config:
        services = config["phase1_services"]
        required_services = ["prorating", "audit_trail", "rejection_workflow"]
        for service in required_services:
            if service not in services or services[service] is None:
                errors.append(f"Missing or invalid Phase 1 service: {service}")
    
    return errors