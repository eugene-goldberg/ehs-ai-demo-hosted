"""
Comprehensive Workflow Integration Example

This module provides a complete practical reference for developers implementing
Phase 1 enhanced workflows. It demonstrates:

1. Complete example of using Phase 1 enhanced workflows
2. How to initialize the system using WorkflowAdapter
3. Example FastAPI endpoints that use the enhanced workflows
4. Configuration from environment variables
5. Examples for document processing, reporting, rejection handling, and audit trails
6. Error handling examples
7. Docker deployment considerations

Author: EHS AI Platform Team
Created: 2025-08-23
Version: 1.0.0
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import asynccontextmanager

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, ValidationError

# LangGraph and LangChain imports
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_neo4j import Neo4jGraph

# Phase 1 enhanced workflow imports
from .workflow_adapter import WorkflowAdapter, WorkflowType, WorkflowConfig
from .phase1_integration import Phase1Integration
from .audit_trail_service import AuditTrailService
from .audit_trail_schema import AuditTrailEntry, DocumentInfo, ProcessingStage
from .prorating_service import ProRatingService
from .prorating_schema import ProRatingAllocation, ProRatingRequest, ProRatingResult
from .rejection_workflow_service import RejectionWorkflowService, ValidationResult
from .rejection_tracking_schema import RejectionEntry, ValidationRule, RejectionReason

# Import enhanced workflows
from ..workflows.ingestion_workflow_enhanced import DocumentState, create_enhanced_ingestion_workflow
from ..workflows.extraction_workflow_enhanced import ExtractionState, create_enhanced_extraction_workflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/workflow_integration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# =============================================================================
# CONFIGURATION MANAGEMENT
# =============================================================================

class AppConfig:
    """Application configuration from environment variables."""
    
    def __init__(self):
        # Database configuration
        self.neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD', 'password')
        
        # LLM configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        self.preferred_llm = os.getenv('PREFERRED_LLM', 'openai')  # 'openai' or 'anthropic'
        
        # File storage configuration
        self.upload_path = Path(os.getenv('UPLOAD_PATH', '/app/uploads'))
        self.output_path = Path(os.getenv('OUTPUT_PATH', '/app/outputs'))
        self.max_file_size = int(os.getenv('MAX_FILE_SIZE_MB', '50')) * 1024 * 1024  # Convert MB to bytes
        
        # Phase 1 feature flags
        self.enable_audit_trail = os.getenv('ENABLE_AUDIT_TRAIL', 'true').lower() == 'true'
        self.enable_pro_rating = os.getenv('ENABLE_PRO_RATING', 'true').lower() == 'true'
        self.enable_rejection_workflow = os.getenv('ENABLE_REJECTION_WORKFLOW', 'true').lower() == 'true'
        
        # Processing configuration
        self.processing_timeout = int(os.getenv('PROCESSING_TIMEOUT_SECONDS', '300'))
        self.retry_attempts = int(os.getenv('RETRY_ATTEMPTS', '3'))
        
        # Create directories
        self.upload_path.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Validate required configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate required configuration values."""
        required_config = {
            'NEO4J_URI': self.neo4j_uri,
            'NEO4J_USER': self.neo4j_user,
            'NEO4J_PASSWORD': self.neo4j_password,
        }
        
        missing_config = [key for key, value in required_config.items() if not value]
        if missing_config:
            raise ValueError(f"Missing required configuration: {', '.join(missing_config)}")
        
        if not self.openai_api_key and not self.anthropic_api_key:
            raise ValueError("At least one LLM API key (OPENAI_API_KEY or ANTHROPIC_API_KEY) must be provided")

# Global configuration instance
config = AppConfig()

# =============================================================================
# PYDANTIC MODELS FOR API
# =============================================================================

class DocumentProcessingRequest(BaseModel):
    """Request model for document processing."""
    document_type: str = Field(..., description="Type of document (incident_report, safety_inspection, etc.)")
    processing_options: Dict[str, Any] = Field(default_factory=dict, description="Processing configuration")
    enable_pro_rating: bool = Field(default=True, description="Enable pro-rating calculations")
    pro_rating_config: Optional[Dict[str, Any]] = Field(default=None, description="Pro-rating configuration")
    validation_rules: List[str] = Field(default_factory=list, description="Custom validation rules to apply")

class ExtractionRequest(BaseModel):
    """Request model for data extraction."""
    query_config: Dict[str, Any] = Field(..., description="Query configuration")
    output_format: str = Field(default="json", description="Output format (json, csv, pdf, html)")
    report_template: Optional[str] = Field(default=None, description="Report template to use")
    enable_pro_rating: bool = Field(default=True, description="Enable pro-rating in results")
    date_range: Optional[Dict[str, str]] = Field(default=None, description="Date range filter")

class ProcessingResponse(BaseModel):
    """Response model for processing operations."""
    success: bool
    processing_id: str
    message: str
    result: Optional[Dict[str, Any]] = None
    audit_trail_id: Optional[str] = None
    pro_rating_results: Optional[List[Dict[str, Any]]] = None
    rejections: Optional[List[Dict[str, Any]]] = None

class AuditTrailResponse(BaseModel):
    """Response model for audit trail queries."""
    entries: List[Dict[str, Any]]
    total_count: int
    page: int
    page_size: int

# =============================================================================
# GLOBAL SERVICES INITIALIZATION
# =============================================================================

class ServiceManager:
    """Manages all Phase 1 services and workflow adapters."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.workflow_adapter: Optional[WorkflowAdapter] = None
        self.phase1_integration: Optional[Phase1Integration] = None
        self.audit_service: Optional[AuditTrailService] = None
        self.prorating_service: Optional[ProRatingService] = None
        self.rejection_service: Optional[RejectionWorkflowService] = None
        
    async def initialize(self):
        """Initialize all services."""
        try:
            logger.info("Initializing services...")
            
            # Initialize Phase 1 integration
            self.phase1_integration = Phase1Integration()
            await self.phase1_integration.initialize()
            
            # Initialize individual services
            if self.config.enable_audit_trail:
                self.audit_service = AuditTrailService()
                await self.audit_service.initialize()
            
            if self.config.enable_pro_rating:
                self.prorating_service = ProRatingService()
                await self.prorating_service.initialize()
            
            if self.config.enable_rejection_workflow:
                self.rejection_service = RejectionWorkflowService()
                await self.rejection_service.initialize()
            
            # Initialize workflow adapter
            workflow_config = WorkflowConfig(
                neo4j_uri=self.config.neo4j_uri,
                neo4j_user=self.config.neo4j_user,
                neo4j_password=self.config.neo4j_password,
                llm_provider=self.config.preferred_llm,
                enable_audit_trail=self.config.enable_audit_trail,
                enable_pro_rating=self.config.enable_pro_rating,
                enable_rejection_workflow=self.config.enable_rejection_workflow,
                processing_timeout=self.config.processing_timeout
            )
            
            self.workflow_adapter = WorkflowAdapter(
                config=workflow_config,
                phase1_integration=self.phase1_integration,
                audit_service=self.audit_service,
                prorating_service=self.prorating_service,
                rejection_service=self.rejection_service
            )
            
            await self.workflow_adapter.initialize()
            
            logger.info("All services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown all services."""
        try:
            logger.info("Shutting down services...")
            
            if self.workflow_adapter:
                await self.workflow_adapter.shutdown()
            
            if self.phase1_integration:
                await self.phase1_integration.shutdown()
            
            logger.info("All services shut down successfully")
            
        except Exception as e:
            logger.error(f"Error during service shutdown: {e}")

# Global service manager
service_manager = ServiceManager(config)

# =============================================================================
# FASTAPI APPLICATION SETUP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    try:
        await service_manager.initialize()
        logger.info("Application started successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    finally:
        # Shutdown
        try:
            await service_manager.shutdown()
            logger.info("Application shutdown completed")
        except Exception as e:
            logger.error(f"Error during application shutdown: {e}")

# Create FastAPI application
app = FastAPI(
    title="EHS AI Platform - Enhanced Workflows API",
    description="API for Phase 1 enhanced workflows with audit trail, pro-rating, and rejection handling",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv('CORS_ORIGINS', '*').split(','),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# AUTHENTICATION AND DEPENDENCIES
# =============================================================================

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple authentication dependency."""
    # In production, implement proper JWT validation
    token = credentials.credentials
    if not token or token != os.getenv('API_TOKEN', 'demo-token'):
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return {"user_id": "demo-user", "roles": ["admin"]}

async def get_workflow_adapter() -> WorkflowAdapter:
    """Dependency to get workflow adapter."""
    if not service_manager.workflow_adapter:
        raise HTTPException(status_code=503, detail="Workflow adapter not initialized")
    return service_manager.workflow_adapter

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "workflow_adapter": service_manager.workflow_adapter is not None,
            "audit_trail": service_manager.audit_service is not None,
            "pro_rating": service_manager.prorating_service is not None,
            "rejection_workflow": service_manager.rejection_service is not None
        }
    }

@app.post("/documents/upload", response_model=ProcessingResponse)
async def upload_and_process_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    request: DocumentProcessingRequest = Depends(),
    current_user: dict = Depends(get_current_user),
    workflow_adapter: WorkflowAdapter = Depends(get_workflow_adapter)
):
    """
    Upload and process a document with all Phase 1 features.
    
    This endpoint demonstrates:
    - File upload handling
    - Document processing with enhanced workflows
    - Audit trail creation
    - Pro-rating calculations
    - Rejection handling
    - Background processing
    """
    try:
        # Validate file
        if file.size > config.max_file_size:
            raise HTTPException(
                status_code=413, 
                detail=f"File size exceeds maximum allowed size of {config.max_file_size / (1024*1024):.1f}MB"
            )
        
        # Generate processing ID
        processing_id = f"proc_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{file.filename.replace(' ', '_')}"
        
        # Save uploaded file
        file_path = config.upload_path / f"{processing_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"File uploaded: {file_path} (size: {len(content)} bytes)")
        
        # Create initial audit trail entry
        audit_trail_id = None
        if service_manager.audit_service:
            document_info = DocumentInfo(
                document_id=processing_id,
                document_type=request.document_type,
                file_path=str(file_path),
                file_size=len(content),
                mime_type=file.content_type or "application/octet-stream"
            )
            
            audit_entry = AuditTrailEntry(
                document_info=document_info,
                stage=ProcessingStage.UPLOADED,
                timestamp=datetime.utcnow(),
                user_id=current_user["user_id"],
                details={"original_filename": file.filename}
            )
            
            audit_trail_id = await service_manager.audit_service.create_entry(audit_entry)
        
        # Schedule background processing
        background_tasks.add_task(
            process_document_background,
            processing_id,
            str(file_path),
            request,
            current_user,
            audit_trail_id
        )
        
        return ProcessingResponse(
            success=True,
            processing_id=processing_id,
            message="Document uploaded successfully. Processing started in background.",
            audit_trail_id=audit_trail_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in document upload: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

async def process_document_background(
    processing_id: str,
    file_path: str,
    request: DocumentProcessingRequest,
    current_user: dict,
    audit_trail_id: Optional[str]
):
    """Background task for document processing."""
    try:
        logger.info(f"Starting background processing for {processing_id}")
        
        # Update audit trail
        if service_manager.audit_service and audit_trail_id:
            await service_manager.audit_service.update_stage(
                audit_trail_id, 
                ProcessingStage.PROCESSING,
                {"status": "started", "worker": "background_task"}
            )
        
        # Create document state
        document_state = {
            "document_id": processing_id,
            "file_path": file_path,
            "document_type": request.document_type,
            "processing_options": request.processing_options,
            "user_id": current_user["user_id"],
            "audit_trail_id": audit_trail_id,
            "enable_pro_rating": request.enable_pro_rating,
            "pro_rating_config": request.pro_rating_config or {},
            "validation_rules": request.validation_rules,
            "processing_metadata": {
                "started_at": datetime.utcnow().isoformat(),
                "processing_id": processing_id
            }
        }
        
        # Process document using workflow adapter
        result = await service_manager.workflow_adapter.process_document(
            document_state,
            WorkflowType.INGESTION
        )
        
        # Handle pro-rating if enabled and successful
        pro_rating_results = []
        if request.enable_pro_rating and result.get("success") and service_manager.prorating_service:
            try:
                # Extract financial data from processing results
                financial_data = result.get("extracted_data", {}).get("financial", {})
                if financial_data:
                    pro_rating_request = ProRatingRequest(
                        document_id=processing_id,
                        total_amount=financial_data.get("total_amount", 0),
                        allocation_rules=request.pro_rating_config.get("allocation_rules", []),
                        date_range=financial_data.get("date_range", {}),
                        cost_centers=financial_data.get("cost_centers", [])
                    )
                    
                    pro_rating_result = await service_manager.prorating_service.calculate_allocations(pro_rating_request)
                    pro_rating_results = [allocation.dict() for allocation in pro_rating_result.allocations]
                    
                    logger.info(f"Pro-rating completed for {processing_id}: {len(pro_rating_results)} allocations")
                    
            except Exception as e:
                logger.warning(f"Pro-rating failed for {processing_id}: {e}")
        
        # Update final audit trail
        if service_manager.audit_service and audit_trail_id:
            final_stage = ProcessingStage.COMPLETED if result.get("success") else ProcessingStage.FAILED
            await service_manager.audit_service.update_stage(
                audit_trail_id,
                final_stage,
                {
                    "result_summary": result.get("summary", "No summary available"),
                    "pro_rating_allocations": len(pro_rating_results),
                    "processing_time": result.get("processing_time", 0),
                    "completed_at": datetime.utcnow().isoformat()
                }
            )
        
        logger.info(f"Background processing completed for {processing_id}")
        
    except Exception as e:
        logger.error(f"Background processing failed for {processing_id}: {e}")
        
        # Update audit trail with error
        if service_manager.audit_service and audit_trail_id:
            await service_manager.audit_service.update_stage(
                audit_trail_id,
                ProcessingStage.FAILED,
                {"error": str(e), "failed_at": datetime.utcnow().isoformat()}
            )

@app.post("/reports/generate", response_model=ProcessingResponse)
async def generate_report_with_prorating(
    request: ExtractionRequest,
    current_user: dict = Depends(get_current_user),
    workflow_adapter: WorkflowAdapter = Depends(get_workflow_adapter)
):
    """
    Generate reports with pro-rating allocations.
    
    This endpoint demonstrates:
    - Data extraction workflows
    - Report generation
    - Pro-rating calculations
    - Audit trail for queries
    """
    try:
        processing_id = f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Create audit trail for report generation
        audit_trail_id = None
        if service_manager.audit_service:
            document_info = DocumentInfo(
                document_id=processing_id,
                document_type="generated_report",
                file_path="",  # Will be updated after generation
                file_size=0,   # Will be updated after generation
                mime_type="application/json"
            )
            
            audit_entry = AuditTrailEntry(
                document_info=document_info,
                stage=ProcessingStage.PROCESSING,
                timestamp=datetime.utcnow(),
                user_id=current_user["user_id"],
                details={
                    "query_config": request.query_config,
                    "output_format": request.output_format,
                    "report_template": request.report_template
                }
            )
            
            audit_trail_id = await service_manager.audit_service.create_entry(audit_entry)
        
        # Create extraction state
        extraction_state = {
            "processing_id": processing_id,
            "query_config": request.query_config,
            "output_format": request.output_format,
            "report_template": request.report_template,
            "user_id": current_user["user_id"],
            "audit_trail_id": audit_trail_id,
            "enable_pro_rating": request.enable_pro_rating,
            "date_range": request.date_range,
            "processing_metadata": {
                "started_at": datetime.utcnow().isoformat(),
                "processing_id": processing_id
            }
        }
        
        # Generate report using workflow adapter
        result = await service_manager.workflow_adapter.extract_data(
            extraction_state,
            WorkflowType.EXTRACTION
        )
        
        # Handle pro-rating for financial reports
        pro_rating_results = []
        if request.enable_pro_rating and result.get("success") and service_manager.prorating_service:
            try:
                report_data = result.get("report_data", {})
                if "financial_summary" in report_data:
                    financial_summary = report_data["financial_summary"]
                    
                    # Create pro-rating request from report data
                    pro_rating_request = ProRatingRequest(
                        document_id=processing_id,
                        total_amount=financial_summary.get("total_amount", 0),
                        allocation_rules=financial_summary.get("allocation_rules", []),
                        date_range=request.date_range or {},
                        cost_centers=financial_summary.get("cost_centers", [])
                    )
                    
                    pro_rating_result = await service_manager.prorating_service.calculate_allocations(pro_rating_request)
                    pro_rating_results = [allocation.dict() for allocation in pro_rating_result.allocations]
                    
                    # Add pro-rating results to report
                    result["report_data"]["pro_rating_allocations"] = pro_rating_results
                    
            except Exception as e:
                logger.warning(f"Pro-rating failed for report {processing_id}: {e}")
        
        # Save report file
        if result.get("success"):
            output_file = config.output_path / f"{processing_id}_report.{request.output_format}"
            with open(output_file, "w") as f:
                if request.output_format == "json":
                    json.dump(result["report_data"], f, indent=2, default=str)
                else:
                    # Handle other formats (CSV, HTML, etc.)
                    f.write(str(result["report_data"]))
            
            result["report_file_path"] = str(output_file)
            
            # Update audit trail
            if service_manager.audit_service and audit_trail_id:
                await service_manager.audit_service.update_stage(
                    audit_trail_id,
                    ProcessingStage.COMPLETED,
                    {
                        "report_file": str(output_file),
                        "pro_rating_allocations": len(pro_rating_results),
                        "completed_at": datetime.utcnow().isoformat()
                    }
                )
        
        return ProcessingResponse(
            success=result.get("success", False),
            processing_id=processing_id,
            message="Report generated successfully" if result.get("success") else "Report generation failed",
            result=result,
            audit_trail_id=audit_trail_id,
            pro_rating_results=pro_rating_results
        )
        
    except Exception as e:
        logger.error(f"Error in report generation: {e}")
        
        # Update audit trail with error
        if service_manager.audit_service and audit_trail_id:
            await service_manager.audit_service.update_stage(
                audit_trail_id,
                ProcessingStage.FAILED,
                {"error": str(e), "failed_at": datetime.utcnow().isoformat()}
            )
        
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.get("/processing/{processing_id}/status")
async def get_processing_status(
    processing_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get processing status with audit trail information."""
    try:
        # Get audit trail entries for this processing ID
        status_info = {"processing_id": processing_id, "status": "unknown"}
        
        if service_manager.audit_service:
            entries = await service_manager.audit_service.get_entries_by_document(processing_id)
            if entries:
                latest_entry = max(entries, key=lambda x: x.timestamp)
                status_info.update({
                    "status": latest_entry.stage.value,
                    "last_updated": latest_entry.timestamp.isoformat(),
                    "details": latest_entry.details,
                    "audit_entries": len(entries)
                })
            else:
                status_info["status"] = "not_found"
        
        # Check for rejection information
        if service_manager.rejection_service:
            rejections = await service_manager.rejection_service.get_rejections_by_document(processing_id)
            if rejections:
                status_info["rejections"] = [
                    {
                        "reason": rej.reason.value,
                        "message": rej.message,
                        "timestamp": rej.timestamp.isoformat()
                    } for rej in rejections
                ]
        
        # Check for pro-rating results
        if service_manager.prorating_service:
            allocations = await service_manager.prorating_service.get_allocations_by_document(processing_id)
            if allocations:
                status_info["pro_rating_allocations"] = len(allocations)
        
        return status_info
        
    except Exception as e:
        logger.error(f"Error getting processing status: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving status: {str(e)}")

@app.get("/audit-trail", response_model=AuditTrailResponse)
async def get_audit_trail(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=1000, description="Page size"),
    document_id: Optional[str] = Query(None, description="Filter by document ID"),
    stage: Optional[str] = Query(None, description="Filter by processing stage"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    current_user: dict = Depends(get_current_user)
):
    """
    Get audit trail entries with filtering and pagination.
    
    This endpoint demonstrates:
    - Audit trail querying
    - Filtering and pagination
    - Access control
    """
    try:
        if not service_manager.audit_service:
            raise HTTPException(status_code=503, detail="Audit trail service not available")
        
        # Build filters
        filters = {}
        if document_id:
            filters["document_id"] = document_id
        if stage:
            filters["stage"] = stage
        if user_id:
            filters["user_id"] = user_id
        if start_date:
            filters["start_date"] = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        if end_date:
            filters["end_date"] = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        
        # Get entries
        entries, total_count = await service_manager.audit_service.get_entries_paginated(
            page=page,
            page_size=page_size,
            filters=filters
        )
        
        # Convert to serializable format
        entries_data = [
            {
                "id": entry.id,
                "document_id": entry.document_info.document_id,
                "document_type": entry.document_info.document_type,
                "stage": entry.stage.value,
                "timestamp": entry.timestamp.isoformat(),
                "user_id": entry.user_id,
                "details": entry.details,
                "file_path": entry.document_info.file_path,
                "file_size": entry.document_info.file_size
            } for entry in entries
        ]
        
        return AuditTrailResponse(
            entries=entries_data,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving audit trail: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving audit trail: {str(e)}")

@app.get("/rejections")
async def get_rejected_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=1000),
    reason: Optional[str] = Query(None, description="Filter by rejection reason"),
    document_type: Optional[str] = Query(None, description="Filter by document type"),
    current_user: dict = Depends(get_current_user)
):
    """
    Get rejected documents with filtering.
    
    This endpoint demonstrates:
    - Rejection tracking
    - Filtering rejected documents
    - Pagination
    """
    try:
        if not service_manager.rejection_service:
            raise HTTPException(status_code=503, detail="Rejection workflow service not available")
        
        # Build filters
        filters = {}
        if reason:
            filters["reason"] = reason
        if document_type:
            filters["document_type"] = document_type
        
        # Get rejected documents
        rejections, total_count = await service_manager.rejection_service.get_rejections_paginated(
            page=page,
            page_size=page_size,
            filters=filters
        )
        
        # Convert to response format
        rejections_data = [
            {
                "id": rej.id,
                "document_id": rej.document_id,
                "document_type": rej.document_type,
                "reason": rej.reason.value,
                "message": rej.message,
                "timestamp": rej.timestamp.isoformat(),
                "user_id": rej.user_id,
                "validation_results": rej.validation_results,
                "retry_count": rej.retry_count,
                "resolved": rej.resolved
            } for rej in rejections
        ]
        
        return {
            "rejections": rejections_data,
            "total_count": total_count,
            "page": page,
            "page_size": page_size
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving rejections: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving rejections: {str(e)}")

@app.post("/rejections/{rejection_id}/retry")
async def retry_rejected_document(
    rejection_id: str,
    current_user: dict = Depends(get_current_user),
    workflow_adapter: WorkflowAdapter = Depends(get_workflow_adapter)
):
    """
    Retry processing a rejected document.
    
    This endpoint demonstrates:
    - Rejection retry workflow
    - Document reprocessing
    - Status updates
    """
    try:
        if not service_manager.rejection_service:
            raise HTTPException(status_code=503, detail="Rejection workflow service not available")
        
        # Get rejection details
        rejection = await service_manager.rejection_service.get_rejection_by_id(rejection_id)
        if not rejection:
            raise HTTPException(status_code=404, detail="Rejection not found")
        
        if rejection.resolved:
            raise HTTPException(status_code=400, detail="Rejection already resolved")
        
        # Increment retry count
        await service_manager.rejection_service.increment_retry_count(rejection_id)
        
        # Create new processing state for retry
        document_state = {
            "document_id": rejection.document_id,
            "document_type": rejection.document_type,
            "processing_options": {"retry": True, "original_rejection_id": rejection_id},
            "user_id": current_user["user_id"],
            "validation_rules": [],  # Use default rules for retry
            "processing_metadata": {
                "started_at": datetime.utcnow().isoformat(),
                "retry_attempt": rejection.retry_count + 1,
                "original_rejection": rejection_id
            }
        }
        
        # Process document using workflow adapter
        result = await service_manager.workflow_adapter.process_document(
            document_state,
            WorkflowType.INGESTION
        )
        
        # Update rejection status if successful
        if result.get("success"):
            await service_manager.rejection_service.resolve_rejection(
                rejection_id,
                current_user["user_id"],
                "Successfully reprocessed after retry"
            )
            
        return ProcessingResponse(
            success=result.get("success", False),
            processing_id=rejection.document_id,
            message="Document retry completed successfully" if result.get("success") else "Document retry failed",
            result=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrying rejected document: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrying document: {str(e)}")

@app.get("/reports/{report_id}/download")
async def download_report(
    report_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Download generated report file."""
    try:
        # Find report file
        report_files = list(config.output_path.glob(f"{report_id}_report.*"))
        if not report_files:
            raise HTTPException(status_code=404, detail="Report file not found")
        
        report_file = report_files[0]
        
        # Determine media type
        media_type = "application/octet-stream"
        if report_file.suffix == ".json":
            media_type = "application/json"
        elif report_file.suffix == ".csv":
            media_type = "text/csv"
        elif report_file.suffix == ".html":
            media_type = "text/html"
        elif report_file.suffix == ".pdf":
            media_type = "application/pdf"
        
        return FileResponse(
            path=str(report_file),
            media_type=media_type,
            filename=f"{report_id}_report{report_file.suffix}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading report: {e}")
        raise HTTPException(status_code=500, detail=f"Error downloading report: {str(e)}")

# =============================================================================
# DOCKER DEPLOYMENT CONSIDERATIONS
# =============================================================================

"""
Docker Deployment Configuration:

1. Dockerfile Example:
   ```dockerfile
   FROM python:3.11-slim
   
   WORKDIR /app
   
   # Install system dependencies
   RUN apt-get update && apt-get install -y \\
       curl \\
       && rm -rf /var/lib/apt/lists/*
   
   # Copy requirements and install Python dependencies
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   # Copy application code
   COPY src/ ./src/
   
   # Create necessary directories
   RUN mkdir -p /app/uploads /app/outputs /app/logs
   
   # Set environment variables
   ENV PYTHONPATH=/app/src
   ENV UPLOAD_PATH=/app/uploads
   ENV OUTPUT_PATH=/app/outputs
   
   # Health check
   HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
       CMD curl -f http://localhost:8000/health || exit 1
   
   # Run the application
   CMD ["uvicorn", "src.phase1_enhancements.workflow_integration_example:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. Docker Compose Example:
   ```yaml
   version: '3.8'
   services:
     ehs-ai-api:
       build: .
       ports:
         - "8000:8000"
       environment:
         - NEO4J_URI=bolt://neo4j:7687
         - NEO4J_USER=neo4j
         - NEO4J_PASSWORD=your_password
         - OPENAI_API_KEY=your_openai_key
         - PREFERRED_LLM=openai
         - ENABLE_AUDIT_TRAIL=true
         - ENABLE_PRO_RATING=true
         - ENABLE_REJECTION_WORKFLOW=true
         - API_TOKEN=your_api_token
       volumes:
         - ./uploads:/app/uploads
         - ./outputs:/app/outputs
         - ./logs:/app/logs
       depends_on:
         - neo4j
       restart: unless-stopped
     
     neo4j:
       image: neo4j:5.15
       ports:
         - "7474:7474"
         - "7687:7687"
       environment:
         - NEO4J_AUTH=neo4j/your_password
         - NEO4J_PLUGINS=["apoc"]
       volumes:
         - neo4j_data:/data
       restart: unless-stopped
   
   volumes:
     neo4j_data:
   ```

3. Environment Variables for Production:
   - NEO4J_URI: Database connection string
   - NEO4J_USER: Database username
   - NEO4J_PASSWORD: Database password
   - OPENAI_API_KEY: OpenAI API key
   - ANTHROPIC_API_KEY: Anthropic API key
   - PREFERRED_LLM: "openai" or "anthropic"
   - API_TOKEN: Authentication token
   - CORS_ORIGINS: Allowed origins for CORS
   - MAX_FILE_SIZE_MB: Maximum file size in MB
   - PROCESSING_TIMEOUT_SECONDS: Processing timeout
   - UPLOAD_PATH: Path for uploaded files
   - OUTPUT_PATH: Path for generated outputs
   - ENABLE_AUDIT_TRAIL: Enable audit trail service
   - ENABLE_PRO_RATING: Enable pro-rating service
   - ENABLE_REJECTION_WORKFLOW: Enable rejection workflow

4. Production Considerations:
   - Use secrets management for API keys
   - Implement proper authentication (JWT, OAuth2)
   - Set up monitoring and alerting
   - Configure log aggregation
   - Use persistent volumes for data
   - Implement backup strategies
   - Set resource limits and health checks
   - Use reverse proxy (nginx) for SSL termination
"""

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

"""
Usage Examples:

1. Process a Document:
   ```bash
   curl -X POST "http://localhost:8000/documents/upload" \\
        -H "Authorization: Bearer demo-token" \\
        -H "Content-Type: multipart/form-data" \\
        -F "file=@incident_report.pdf" \\
        -F "document_type=incident_report" \\
        -F "enable_pro_rating=true" \\
        -F "pro_rating_config={\"allocation_rules\": [{\"cost_center\": \"SAFETY\", \"percentage\": 60}, {\"cost_center\": \"OPERATIONS\", \"percentage\": 40}]}"
   ```

2. Generate a Report:
   ```bash
   curl -X POST "http://localhost:8000/reports/generate" \\
        -H "Authorization: Bearer demo-token" \\
        -H "Content-Type: application/json" \\
        -d '{
          "query_config": {
            "query_type": "safety_metrics",
            "parameters": {"facility": "Plant_A", "date_range": "2024-01"}
          },
          "output_format": "json",
          "enable_pro_rating": true,
          "date_range": {"start": "2024-01-01", "end": "2024-01-31"}
        }'
   ```

3. Check Processing Status:
   ```bash
   curl -H "Authorization: Bearer demo-token" \\
        "http://localhost:8000/processing/proc_20240123_143022_report.pdf/status"
   ```

4. Get Audit Trail:
   ```bash
   curl -H "Authorization: Bearer demo-token" \\
        "http://localhost:8000/audit-trail?page=1&page_size=20&stage=completed"
   ```

5. View Rejected Documents:
   ```bash
   curl -H "Authorization: Bearer demo-token" \\
        "http://localhost:8000/rejections?reason=validation_failed"
   ```

6. Retry Rejected Document:
   ```bash
   curl -X POST -H "Authorization: Bearer demo-token" \\
        "http://localhost:8000/rejections/rejection_123/retry"
   ```

7. Download Report:
   ```bash
   curl -H "Authorization: Bearer demo-token" \\
        -o report.json \\
        "http://localhost:8000/reports/report_20240123_143022/download"
   ```
"""

if __name__ == "__main__":
    import uvicorn
    
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the application
    uvicorn.run(
        "workflow_integration_example:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )