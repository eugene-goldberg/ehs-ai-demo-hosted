"""
FastAPI Router for Pro-rating API Endpoints

This module provides REST API endpoints for pro-rating allocations and processing.
It includes functionality for processing documents, batch processing, getting allocations,
monthly reports, backfill operations, and facility-specific queries.

Dependencies:
- FastAPI for REST API framework
- ProRatingService for allocation operations
- Neo4j graph database for document metadata
- Proper error handling and validation
"""

import os
import uuid
import logging
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List
from enum import Enum
from decimal import Decimal

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from starlette.status import HTTP_200_OK, HTTP_201_CREATED, HTTP_404_NOT_FOUND, HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR, HTTP_503_SERVICE_UNAVAILABLE

from .prorating_service import ProRatingService
from .prorating_calculator import BillingPeriod, FacilityInfo as CalculatorFacilityInfo, ProRatingMethod as CalculatorProRatingMethod
try:
    from graph_query import get_graphDB_driver
    from shared.common_fn import create_graph_database_connection
except ImportError:
    from src.graph_query import get_graphDB_driver
    from src.shared.common_fn import create_graph_database_connection

# Initialize logging
logger = logging.getLogger(__name__)

# Service will be initialized in startup event
prorating_service = None

# Dependencies
async def get_prorating_service():
    """Get the prorating service instance, ensuring it's initialized."""
    global prorating_service
    
    if prorating_service is None:
        # Try to get it from the phase1 integration if available
        try:
            from .phase1_integration import get_current_phase1_integration
            integration = get_current_phase1_integration()
            if integration and integration.prorating_service:
                prorating_service = integration.prorating_service
            else:
                # Initialize directly if integration not available
                from src.shared.common_fn import create_graph_database_connection
                graph = create_graph_database_connection(
                    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                    os.getenv("NEO4J_USERNAME", "neo4j"),
                    os.getenv("NEO4J_PASSWORD", "agentOS1!"),
                    os.getenv("NEO4J_DATABASE", "neo4j")
                )
                from .prorating_service import ProRatingService
                prorating_service = ProRatingService(graph)
        except Exception as e:
            logger.error(f"Failed to initialize prorating service: {e}")
            raise HTTPException(
                status_code=503,
                detail="Pro-rating service initialization failed"
            )
    
    if prorating_service is None:
        raise HTTPException(
            status_code=503,
            detail="Pro-rating service is not available"
        )
    
    return prorating_service

# Create FastAPI router without prefix (prefix added during app integration)
router = APIRouter(
    tags=["Pro-rating"],
    responses={
        404: {"description": "Resource not found"},
        500: {"description": "Internal server error"},
        503: {"description": "Service unavailable"}
    }
)

class ProRatingMethod(str, Enum):
    """Supported pro-rating methods."""
    HEADCOUNT = "headcount"
    FLOOR_AREA = "floor_area" 
    REVENUE = "revenue"
    CUSTOM = "custom"
    TIME_BASED = "time_based"
    SPACE_BASED = "space_based"
    HYBRID = "hybrid"

class FacilityInfo(BaseModel):
    """Facility information for pro-rating calculations."""
    facility_id: str = Field(..., description="Unique facility identifier")
    name: str = Field(..., description="Facility name")
    headcount: Optional[int] = Field(None, description="Number of employees")
    floor_area: Optional[float] = Field(None, description="Floor area in square feet")
    revenue: Optional[float] = Field(None, description="Annual revenue")
    custom_weight: Optional[float] = Field(None, description="Custom weighting factor")

class ProRatingRequest(BaseModel):
    """Request model for pro-rating operations."""
    document_id: str = Field(..., description="UUID of the document to process")
    method: ProRatingMethod = Field(..., description="Pro-rating method to use")
    facility_info: List[FacilityInfo] = Field(..., description="List of facilities for allocation")
    
    @validator('document_id')
    def validate_document_id(cls, v):
        try:
            uuid.UUID(v)
            return v
        except ValueError:
            raise ValueError('Invalid document ID format')

class AllocationResult(BaseModel):
    """Individual allocation result."""
    facility_id: str
    facility_name: str
    allocation_percentage: float
    allocated_amount: float
    basis_value: Optional[float] = None
    method_used: str

class AllocationSummary(BaseModel):
    """Summary of allocation results."""
    total_amount: float
    total_allocated: float
    allocation_method: str
    facility_count: int
    processing_timestamp: datetime

class AllocationResponse(BaseModel):
    """Response model for allocation operations."""
    document_id: str
    allocations: List[AllocationResult]
    summary: AllocationSummary
    status: str = "success"

class BatchProcessRequest(BaseModel):
    """Request model for batch processing operations."""
    document_ids: List[str] = Field(..., description="List of document UUIDs to process")
    method: ProRatingMethod = Field(..., description="Pro-rating method to use")
    facility_info: List[FacilityInfo] = Field(..., description="List of facilities for allocation")
    
    @validator('document_ids')
    def validate_document_ids(cls, v):
        for doc_id in v:
            try:
                uuid.UUID(doc_id)
            except ValueError:
                raise ValueError(f'Invalid document ID format: {doc_id}')
        return v

class MonthlyReportRequest(BaseModel):
    """Request model for monthly report generation."""
    year: int = Field(..., description="Report year")
    month: int = Field(..., description="Report month (1-12)")
    facility_id: Optional[str] = Field(None, description="Optional facility filter")
    
    @validator('month')
    def validate_month(cls, v):
        if not 1 <= v <= 12:
            raise ValueError('Month must be between 1 and 12')
        return v
        
    @validator('year')
    def validate_year(cls, v):
        current_year = datetime.now().year
        if not 2020 <= v <= current_year + 1:
            raise ValueError(f'Year must be between 2020 and {current_year + 1}')
        return v

class BackfillRequest(BaseModel):
    """Request model for backfill operations."""
    start_date: date = Field(..., description="Start date for backfill")
    end_date: date = Field(..., description="End date for backfill")
    method: ProRatingMethod = Field(..., description="Pro-rating method to use")
    facility_filter: Optional[List[str]] = Field(None, description="Optional facility ID filter")
    force_recalculate: bool = Field(False, description="Force recalculation of existing allocations")

def get_graph_connection():
    """
    Dependency to get Neo4j graph database connection.
    Uses environment variables for connection details.
    """
    try:
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        database = os.getenv("NEO4J_DATABASE", "neo4j")
        
        return create_graph_database_connection(uri, username, password, database)
    except Exception as e:
        logger.error(f"Failed to create graph connection: {str(e)}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection failed"
        )

def validate_document_exists(document_id: str, graph) -> Optional[Dict[str, Any]]:
    """
    Validate that a document exists in the database.
    
    Args:
        document_id: UUID of the document
        graph: Neo4j graph connection
        
    Returns:
        Document information dictionary or None if not found
    """
    try:
        query = """
        MATCH (d:Document {id: $document_id})
        RETURN d.id as id, d.fileName as file_name, d.status as status,
               d.created_at as created_at, d.total_amount as total_amount,
               d.start_date as start_date, d.end_date as end_date,
               d.total_usage as total_usage, d.total_cost as total_cost
        """
        
        result = graph.query(query, params={"document_id": document_id})
        
        if result and len(result) > 0:
            return result[0]
        return None
        
    except Exception as e:
        logger.error(f"Error validating document {document_id}: {str(e)}")
        return None

def map_api_method_to_calculator_method(api_method: ProRatingMethod) -> CalculatorProRatingMethod:
    """Map API method enum to calculator method enum."""
    mapping = {
        ProRatingMethod.TIME_BASED: CalculatorProRatingMethod.TIME_BASED,
        ProRatingMethod.SPACE_BASED: CalculatorProRatingMethod.SPACE_BASED,
        ProRatingMethod.HYBRID: CalculatorProRatingMethod.HYBRID,
        # For other methods, default to hybrid
        ProRatingMethod.HEADCOUNT: CalculatorProRatingMethod.HYBRID,
        ProRatingMethod.FLOOR_AREA: CalculatorProRatingMethod.SPACE_BASED,
        ProRatingMethod.REVENUE: CalculatorProRatingMethod.HYBRID,
        ProRatingMethod.CUSTOM: CalculatorProRatingMethod.HYBRID
    }
    return mapping.get(api_method, CalculatorProRatingMethod.HYBRID)

def convert_api_facilities_to_calculator_facilities(api_facilities: List[FacilityInfo]) -> List[CalculatorFacilityInfo]:
    """Convert API FacilityInfo objects to calculator FacilityInfo objects."""
    calculator_facilities = []
    
    for facility in api_facilities:
        # Use floor_area as square_footage, default to 1000 if not provided
        square_footage = Decimal(str(facility.floor_area)) if facility.floor_area else Decimal('1000')
        
        calculator_facility = CalculatorFacilityInfo(
            facility_id=facility.facility_id,
            square_footage=square_footage,
            occupied_percentage=Decimal('1.0'),  # Default to 100% occupied
            facility_name=facility.name
        )
        calculator_facilities.append(calculator_facility)
    
    return calculator_facilities

def create_billing_period_from_document(doc_info: Dict[str, Any]) -> BillingPeriod:
    """Create a BillingPeriod object from document information."""
    # Extract dates
    start_date = doc_info.get('start_date')
    end_date = doc_info.get('end_date')
    
    # If dates are not available, use reasonable defaults or current month
    if not start_date or not end_date:
        current_date = date.today()
        start_date = current_date.replace(day=1)
        if current_date.month == 12:
            end_date = current_date.replace(year=current_date.year + 1, month=1, day=1) - timedelta(days=1)
        else:
            end_date = current_date.replace(month=current_date.month + 1, day=1) - timedelta(days=1)
    
    # Convert string dates to date objects if necessary
    if isinstance(start_date, str):
        start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00')).date()
    if isinstance(end_date, str):
        end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00')).date()
    
    # Extract usage and cost
    total_usage = doc_info.get('total_usage', 0)
    total_cost = doc_info.get('total_cost') or doc_info.get('total_amount', 0)
    
    # Convert to Decimal
    total_usage = Decimal(str(total_usage)) if total_usage else Decimal('0')
    total_cost = Decimal(str(total_cost)) if total_cost else Decimal('0')
    
    return BillingPeriod(
        start_date=start_date,
        end_date=end_date,
        total_usage=total_usage,
        total_cost=total_cost,
        facility_id=doc_info.get('facility_id'),
        usage_type=doc_info.get('usage_type', 'utility')
    )

def convert_processed_bill_to_allocation_response(processed_bill, document_id: str) -> AllocationResponse:
    """Convert ProcessedBill result to AllocationResponse format."""
    # Create mock allocations based on the processed bill result
    # In a real implementation, you'd query the database for the created allocations
    allocations = []
    
    if processed_bill.success:
        # Create a summary allocation result
        allocation_result = AllocationResult(
            facility_id="processed",
            facility_name="Processed Allocations",
            allocation_percentage=100.0,
            allocated_amount=float(processed_bill.total_cost_allocated),
            basis_value=float(processed_bill.total_usage_allocated),
            method_used="processed"
        )
        allocations.append(allocation_result)
    
    summary = AllocationSummary(
        total_amount=float(processed_bill.total_cost_allocated),
        total_allocated=float(processed_bill.total_cost_allocated),
        allocation_method="processed",
        facility_count=processed_bill.allocations_created,
        processing_timestamp=datetime.now()
    )
    
    return AllocationResponse(
        document_id=document_id,
        allocations=allocations,
        summary=summary,
        status="success" if processed_bill.success else "error"
    )

@router.post("/process/{document_id}",
            summary="Process Single Document",
            description="Process a single document for pro-rating allocation",
            response_model=AllocationResponse)
async def process_document(
    document_id: str,
    request: ProRatingRequest,
    service: ProRatingService = Depends(get_prorating_service),
    graph=Depends(get_graph_connection)
):
    """
    Process a single document for pro-rating allocation.
    
    Args:
        document_id: UUID of the document to process
        request: Pro-rating request with method and facility information
        service: ProRatingService instance (injected dependency)
        
    Returns:
        AllocationResponse with allocation results and summary
        
    Raises:
        HTTPException: If document not found or processing fails
    """
    try:
        # Validate document exists
        doc_info = validate_document_exists(document_id, graph)
        if not doc_info:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
        
        # Ensure document_id matches between path and request
        if request.document_id != document_id:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="Document ID mismatch between path and request body"
            )
        
        # Create BillingPeriod from document data
        billing_period = create_billing_period_from_document(doc_info)
        
        # Convert API facilities to calculator facilities
        calculator_facilities = convert_api_facilities_to_calculator_facilities(request.facility_info)
        
        # Map API method to calculator method
        calculator_method = map_api_method_to_calculator_method(request.method)
        
        # Process the utility bill using the correct method
        processed_bill = service.process_utility_bill(
            document_id=document_id,
            billing_period=billing_period,
            facilities=calculator_facilities,
            method=calculator_method
        )
        
        if not processed_bill or not processed_bill.success:
            error_details = processed_bill.errors if processed_bill else ["Unknown processing error"]
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process document allocation: {'; '.join(error_details)}"
            )
        
        logger.info(f"Successfully processed utility bill for document {document_id}")
        
        # Convert ProcessedBill to AllocationResponse
        allocation_response = convert_processed_bill_to_allocation_response(processed_bill, document_id)
        
        return allocation_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process document allocation"
        )

@router.post("/batch-process",
            summary="Batch Process Documents",
            description="Process multiple documents for pro-rating allocation",
            response_model=List[AllocationResponse])
async def batch_process_documents(
    request: BatchProcessRequest,
    background_tasks: BackgroundTasks,
    service: ProRatingService = Depends(get_prorating_service),
    graph=Depends(get_graph_connection)
):
    """
    Process multiple documents for pro-rating allocation.
    
    Args:
        request: Batch processing request with document IDs and allocation parameters
        background_tasks: Background tasks for async processing
        service: ProRatingService instance (injected dependency)
        
    Returns:
        List of AllocationResponse objects
        
    Raises:
        HTTPException: If any documents not found or processing fails
    """
    try:
        # Validate all documents exist
        invalid_docs = []
        for doc_id in request.document_ids:
            doc_info = validate_document_exists(doc_id, graph)
            if not doc_info:
                invalid_docs.append(doc_id)
        
        if invalid_docs:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Documents not found: {', '.join(invalid_docs)}"
            )
        
        # Process batch allocation
        batch_results = await service.process_batch_allocation(
            document_ids=request.document_ids,
            method=request.method,
            facility_info=request.facility_info,
            graph=graph
        )
        
        logger.info(f"Successfully processed batch allocation for {len(request.document_ids)} documents")
        
        return [AllocationResponse(**result) for result in batch_results]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing batch allocation: {str(e)}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process batch allocation"
        )

@router.get("/allocations/{document_id}",
           summary="Get Document Allocations",
           description="Retrieve existing allocations for a document",
           response_model=AllocationResponse)
async def get_document_allocations(
    document_id: str,
    service: ProRatingService = Depends(get_prorating_service),
    graph=Depends(get_graph_connection)
):
    """
    Get existing allocations for a document.
    
    Args:
        document_id: UUID of the document
        service: ProRatingService instance (injected dependency)
        
    Returns:
        AllocationResponse with existing allocation data
        
    Raises:
        HTTPException: If document not found or no allocations exist
    """
    try:
        # Validate document ID format
        try:
            uuid.UUID(document_id)
        except ValueError:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="Invalid document ID format"
            )
        
        # Get allocation data
        allocation_data = await service.get_document_allocations(
            document_id=document_id,
            graph=graph
        )
        
        if not allocation_data:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"No allocations found for document {document_id}"
            )
        
        return AllocationResponse(**allocation_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving allocations for document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document allocations"
        )

@router.get("/monthly-report",
           summary="Generate Monthly Report",
           description="Generate a monthly summary report of allocations",
           response_model=Dict[str, Any])
async def get_monthly_report(
    year: int,
    month: int,
    facility_id: Optional[str] = None,
    service: ProRatingService = Depends(get_prorating_service),
    graph=Depends(get_graph_connection)
):
    """
    Generate a monthly summary report of allocations.
    
    Args:
        year: Report year
        month: Report month (1-12)
        facility_id: Optional facility filter
        service: ProRatingService instance (injected dependency)
        
    Returns:
        JSON response with monthly allocation summary
        
    Raises:
        HTTPException: If invalid parameters or report generation fails
    """
    try:
        # Validate month
        if not 1 <= month <= 12:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="Month must be between 1 and 12"
            )
        
        # Validate year
        current_year = datetime.now().year
        if not 2020 <= year <= current_year + 1:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail=f"Year must be between 2020 and {current_year + 1}"
            )
        
        # Generate monthly report
        report_data = await service.generate_monthly_report(
            year=year,
            month=month,
            facility_id=facility_id,
            graph=graph
        )
        
        if not report_data:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"No allocation data found for {year}-{month:02d}"
            )
        
        logger.info(f"Generated monthly report for {year}-{month:02d}" + 
                   (f" (facility: {facility_id})" if facility_id else ""))
        
        return JSONResponse(
            status_code=HTTP_200_OK,
            content=report_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating monthly report for {year}-{month:02d}: {str(e)}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate monthly report"
        )

@router.post("/backfill",
            summary="Trigger Backfill Process",
            description="Trigger backfill processing for existing data",
            response_model=Dict[str, Any])
async def trigger_backfill(
    request: BackfillRequest,
    background_tasks: BackgroundTasks,
    service: ProRatingService = Depends(get_prorating_service),
    graph=Depends(get_graph_connection)
):
    """
    Trigger backfill processing for existing data.
    
    Args:
        request: Backfill request with date range and processing parameters
        background_tasks: Background tasks for async processing
        service: ProRatingService instance (injected dependency)
        
    Returns:
        JSON response with backfill status and job information
        
    Raises:
        HTTPException: If invalid parameters or backfill initiation fails
    """
    try:
        # Validate date range
        if request.start_date > request.end_date:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="Start date must be before end date"
            )
        
        # Check for reasonable date range (not more than 2 years)
        date_diff = (request.end_date - request.start_date).days
        if date_diff > 730:  # 2 years
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="Date range cannot exceed 2 years"
            )
        
        # Initiate backfill process
        backfill_job_id = await service.initiate_backfill(
            start_date=request.start_date,
            end_date=request.end_date,
            method=request.method,
            facility_filter=request.facility_filter,
            force_recalculate=request.force_recalculate,
            graph=graph
        )
        
        # Add background task for processing
        background_tasks.add_task(
            service.process_backfill_job,
            backfill_job_id,
            graph
        )
        
        response_data = {
            "job_id": backfill_job_id,
            "status": "initiated",
            "start_date": request.start_date.isoformat(),
            "end_date": request.end_date.isoformat(),
            "method": request.method,
            "force_recalculate": request.force_recalculate,
            "facility_count": len(request.facility_filter) if request.facility_filter else None,
            "initiated_at": datetime.now().isoformat()
        }
        
        logger.info(f"Initiated backfill job {backfill_job_id} for {request.start_date} to {request.end_date}")
        
        return JSONResponse(
            status_code=HTTP_201_CREATED,
            content=response_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initiating backfill: {str(e)}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initiate backfill process"
        )

@router.get("/facility/{facility_id}/allocations",
           summary="Get Facility Allocations",
           description="Retrieve all allocations for a specific facility",
           response_model=Dict[str, Any])
async def get_facility_allocations(
    facility_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    service: ProRatingService = Depends(get_prorating_service),
    graph=Depends(get_graph_connection)
):
    """
    Get all allocations for a specific facility.
    
    Args:
        facility_id: Unique facility identifier
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        limit: Maximum number of results to return
        offset: Number of results to skip for pagination
        service: ProRatingService instance (injected dependency)
        
    Returns:
        JSON response with facility allocation data
        
    Raises:
        HTTPException: If facility not found or query fails
    """
    try:
        # Validate pagination parameters
        if limit > 1000:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="Limit cannot exceed 1000"
            )
        
        if offset < 0 or limit < 1:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="Offset must be non-negative and limit must be positive"
            )
        
        # Validate date filters if provided
        parsed_start_date = None
        parsed_end_date = None
        
        if start_date:
            try:
                parsed_start_date = datetime.fromisoformat(start_date).date()
            except ValueError:
                raise HTTPException(
                    status_code=HTTP_400_BAD_REQUEST,
                    detail="Invalid start_date format. Use YYYY-MM-DD"
                )
        
        if end_date:
            try:
                parsed_end_date = datetime.fromisoformat(end_date).date()
            except ValueError:
                raise HTTPException(
                    status_code=HTTP_400_BAD_REQUEST,
                    detail="Invalid end_date format. Use YYYY-MM-DD"
                )
        
        if parsed_start_date and parsed_end_date and parsed_start_date > parsed_end_date:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="Start date must be before end date"
            )
        
        # Extract year and month from date objects for service call
        start_year = parsed_start_date.year if parsed_start_date else None
        start_month = parsed_start_date.month if parsed_start_date else None
        end_year = parsed_end_date.year if parsed_end_date else None
        end_month = parsed_end_date.month if parsed_end_date else None
        
        # Get facility allocations with correct parameters
        allocation_data = await service.get_facility_allocations(
            facility_id=facility_id,
            start_year=start_year,
            start_month=start_month,
            end_year=end_year,
            end_month=end_month
        )
        
        if not allocation_data:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"No allocations found for facility {facility_id}"
            )
        
        return JSONResponse(
            status_code=HTTP_200_OK,
            content=allocation_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving allocations for facility {facility_id}: {str(e)}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve facility allocations"
        )

# Health check endpoint for the pro-rating API
@router.get("/health",
           summary="Pro-rating API Health Check",
           description="Check the health status of the pro-rating API and services",
           response_model=Dict[str, Any])
async def prorating_health_check(service: ProRatingService = Depends(get_prorating_service)):
    """
    Health check endpoint for the pro-rating API.
    
    Args:
        service: ProRatingService instance (injected dependency)
    
    Returns:
        JSON response with health status information
    """
    try:
        # Check pro-rating service health
        service_healthy = service is not None
        service_status = {}
        
        if service_healthy:
            try:
                connection_test = await service.test_connection()
                service_status = {"healthy": connection_test, "connected": connection_test}
            except Exception as e:
                service_status = {"healthy": False, "error": str(e)}
        else:
            service_status = {"healthy": False, "error": "Service not initialized"}
        
        # Check database connectivity
        try:
            graph = get_graph_connection()
            db_healthy = True
            db_error = None
        except Exception as e:
            db_healthy = False
            db_error = str(e)
        
        overall_healthy = service_healthy and db_healthy and service_status.get("healthy", False)
        
        health_data = {
            "service": "prorating_api",
            "status": "healthy" if overall_healthy else "degraded",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "components": {
                "prorating_service": {
                    "healthy": service_healthy,
                    "initialized": service is not None,
                    **service_status
                },
                "database": {
                    "healthy": db_healthy,
                    "error": db_error
                }
            }
        }
        
        status_code = HTTP_200_OK if overall_healthy else HTTP_503_SERVICE_UNAVAILABLE
        
        return JSONResponse(
            status_code=status_code,
            content=health_data
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "service": "prorating_api",
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        )
# Export router as prorating_router for external imports
prorating_router = router