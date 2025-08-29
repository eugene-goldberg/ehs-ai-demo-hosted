"""
FastAPI service for EHS data extraction using the DataExtractionWorkflow.
Provides REST API endpoints for extracting electrical consumption, water consumption, 
and waste generation data from Neo4j database.
"""

import os
import sys
import logging
import json
from datetime import datetime, date
from typing import Dict, List, Any, Optional
from pathlib import Path
import subprocess

# Add the src directory to Python path to enable imports when running from backend directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Phase 1 Enhancement Imports
from phase1_enhancements.phase1_integration import create_phase1_integration
from phase1_enhancements.prorating_api import router as prorating_router

# Simple Rejection API Import
from api.simple_rejection_api import simple_rejection_router

# Transcript API Import
from api.transcript_api import router as transcript_router

# Executive Dashboard V2 API Import
from api.executive_dashboard_v2 import router as executive_dashboard_v2_router

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
import uvicorn

from workflows.extraction_workflow import DataExtractionWorkflow, QueryType
from api_response import create_api_response
from langsmith_config import create_ingestion_session, is_langsmith_available

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app configuration
app = FastAPI(
    title="EHS Data Extraction API",
    description="API service for extracting EHS (Environmental, Health, Safety) data",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Phase 1 Integration
phase1_integration = None

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the Simple Rejection API router
app.include_router(simple_rejection_router)

# Include the Prorating API router
app.include_router(prorating_router)

# Include the Transcript API router
app.include_router(transcript_router)

# Include the Executive Dashboard V2 API router
app.include_router(executive_dashboard_v2_router, prefix="/api/v2")

# Pydantic models for request/response validation
class FacilityFilter(BaseModel):
    """Filter for facility-based queries."""
    facility_id: Optional[str] = Field(None, description="Specific facility ID")
    facility_name: Optional[str] = Field(None, description="Facility name pattern")

class DateRangeFilter(BaseModel):
    """Date range filter for temporal queries."""
    start_date: Optional[date] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[date] = Field(None, description="End date (YYYY-MM-DD)")
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        if v and 'start_date' in values and values['start_date']:
            if v < values['start_date']:
                raise ValueError('end_date must be after start_date')
        return v

class BaseExtractionRequest(BaseModel):
    """Base request model for all extraction endpoints."""
    facility_filter: Optional[FacilityFilter] = None
    date_range: Optional[DateRangeFilter] = None
    output_format: str = Field("json", description="Output format (json, txt)")
    
class ElectricalConsumptionRequest(BaseExtractionRequest):
    """Request model for electrical consumption data extraction."""
    include_emissions: bool = Field(True, description="Include emissions data")
    include_cost_analysis: bool = Field(True, description="Include cost analysis")

class WaterConsumptionRequest(BaseExtractionRequest):
    """Request model for water consumption data extraction."""
    include_meter_details: bool = Field(True, description="Include meter information")
    include_emissions: bool = Field(True, description="Include emissions data")

class WasteGenerationRequest(BaseExtractionRequest):
    """Request model for waste generation data extraction."""
    include_disposal_details: bool = Field(True, description="Include disposal facility details")
    include_transport_details: bool = Field(True, description="Include transporter details")
    include_emissions: bool = Field(True, description="Include emissions data")
    hazardous_only: bool = Field(False, description="Filter for hazardous waste only")

class ExtractionResponse(BaseModel):
    """Response model for extraction endpoints."""
    status: str = Field(description="Response status")
    message: str = Field(description="Response message")
    data: Optional[Dict[str, Any]] = Field(None, description="Extracted data")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Query metadata")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    errors: Optional[List[str]] = Field(None, description="Any errors encountered")

class BatchIngestionRequest(BaseModel):
    """Request model for batch document ingestion."""
    clear_database: bool = Field(default=True, description="Clear Neo4j database before ingestion")
    
class BatchIngestionResponse(BaseModel):
    """Response model for batch ingestion."""
    status: str = Field(..., description="Status of the ingestion process (success/failed)")
    message: str = Field(..., description="Response message")
    data: Dict[str, Any] = Field(default_factory=dict, description="Ingestion results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    processing_time: float = Field(..., description="Processing time in seconds")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = "healthy"
    timestamp: str
    neo4j_connection: bool
    version: str = "1.0.0"

# Dependency for creating workflow instance
def get_workflow() -> DataExtractionWorkflow:
    """Create and return DataExtractionWorkflow instance."""
    try:
        neo4j_uri = os.getenv('NEO4J_URI')
        neo4j_username = os.getenv('NEO4J_USERNAME') 
        neo4j_password = os.getenv('NEO4J_PASSWORD')
        llm_model = os.getenv('LLM_MODEL', 'gpt-4')
        
        if not all([neo4j_uri, neo4j_username, neo4j_password]):
            raise HTTPException(
                status_code=500,
                detail="Missing required Neo4j connection configuration"
            )
        
        return DataExtractionWorkflow(
            neo4j_uri=neo4j_uri,
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password,
            llm_model=llm_model,
            output_dir="./reports"
        )
    except Exception as e:
        logger.error(f"Failed to create workflow instance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Workflow initialization error: {str(e)}")

def build_query_parameters(facility_filter: Optional[FacilityFilter], 
                          date_range: Optional[DateRangeFilter]) -> Dict[str, Any]:
    """Build query parameters from filters."""
    params = {}
    
    if date_range and date_range.start_date:
        params['start_date'] = date_range.start_date.strftime('%Y-%m-%d')
    if date_range and date_range.end_date:
        params['end_date'] = date_range.end_date.strftime('%Y-%m-%d')
        
    if facility_filter and facility_filter.facility_id:
        params['facility_id'] = facility_filter.facility_id
    if facility_filter and facility_filter.facility_name:
        params['facility_name'] = facility_filter.facility_name
    
    return params

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Test Neo4j connection
        workflow = get_workflow()
        neo4j_connection = True
        workflow.close()
    except Exception as e:
        logger.warning(f"Neo4j connection test failed: {str(e)}")
        neo4j_connection = False
    
    return HealthResponse(
        timestamp=datetime.utcnow().isoformat(),
        neo4j_connection=neo4j_connection
    )

# Electrical consumption extraction endpoint
@app.post("/api/v1/extract/electrical-consumption", response_model=ExtractionResponse)
async def extract_electrical_consumption(request: ElectricalConsumptionRequest):
    """
    Extract electrical consumption data from utility bills.
    
    Returns electrical usage, costs, and optionally emissions data.
    """
    logger.info("Processing electrical consumption extraction request")
    
    try:
        workflow = get_workflow()
        
        # Build query parameters
        params = build_query_parameters(request.facility_filter, request.date_range)
        
        # Execute extraction workflow
        start_time = datetime.utcnow()
        result = workflow.extract_data(
            query_type=QueryType.UTILITY_CONSUMPTION,
            parameters=params,
            output_format=request.output_format
        )
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        # Close workflow connection
        workflow.close()
        
        # Prepare response
        response_data = {
            "query_type": QueryType.UTILITY_CONSUMPTION,
            "facility_filter": request.facility_filter.dict() if request.facility_filter else None,
            "date_range": request.date_range.dict() if request.date_range else None,
            "include_emissions": request.include_emissions,
            "include_cost_analysis": request.include_cost_analysis,
            "report_data": result.get("report_data", {}),
            "file_path": result.get("report_file_path")
        }
        
        metadata = {
            "total_queries": len(result.get("queries", [])),
            "successful_queries": len([q for q in result.get("query_results", []) if q.get("status") == "success"]),
            "total_records": sum(q.get("record_count", 0) for q in result.get("query_results", [])),
            "processing_status": result.get("status"),
            "generated_at": start_time.isoformat()
        }
        
        return ExtractionResponse(
            status="success" if result.get("status") == "completed" else "failed",
            message=f"Electrical consumption data extracted successfully" if result.get("status") == "completed" else "Extraction failed",
            data=response_data,
            metadata=metadata,
            processing_time=processing_time,
            errors=result.get("errors", [])
        )
        
    except Exception as e:
        logger.error(f"Failed to extract electrical consumption data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

# Water consumption extraction endpoint  
@app.post("/api/v1/extract/water-consumption", response_model=ExtractionResponse)
async def extract_water_consumption(request: WaterConsumptionRequest):
    """
    Extract water consumption data from water bills.
    
    Returns water usage, costs, meter information, and optionally emissions data.
    """
    logger.info("Processing water consumption extraction request")
    
    try:
        workflow = get_workflow()
        
        # Build query parameters
        params = build_query_parameters(request.facility_filter, request.date_range)
        
        # Execute extraction workflow
        start_time = datetime.utcnow()
        result = workflow.extract_data(
            query_type=QueryType.WATER_CONSUMPTION,
            parameters=params,
            output_format=request.output_format
        )
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        # Close workflow connection
        workflow.close()
        
        # Prepare response
        response_data = {
            "query_type": QueryType.WATER_CONSUMPTION,
            "facility_filter": request.facility_filter.dict() if request.facility_filter else None,
            "date_range": request.date_range.dict() if request.date_range else None,
            "include_meter_details": request.include_meter_details,
            "include_emissions": request.include_emissions,
            "report_data": result.get("report_data", {}),
            "file_path": result.get("report_file_path")
        }
        
        metadata = {
            "total_queries": len(result.get("queries", [])),
            "successful_queries": len([q for q in result.get("query_results", []) if q.get("status") == "success"]),
            "total_records": sum(q.get("record_count", 0) for q in result.get("query_results", [])),
            "processing_status": result.get("status"),
            "generated_at": start_time.isoformat()
        }
        
        return ExtractionResponse(
            status="success" if result.get("status") == "completed" else "failed",
            message=f"Water consumption data extracted successfully" if result.get("status") == "completed" else "Extraction failed",
            data=response_data,
            metadata=metadata,
            processing_time=processing_time,
            errors=result.get("errors", [])
        )
        
    except Exception as e:
        logger.error(f"Failed to extract water consumption data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

# Waste generation extraction endpoint
@app.post("/api/v1/extract/waste-generation", response_model=ExtractionResponse)
async def extract_waste_generation(request: WasteGenerationRequest):
    """
    Extract waste generation data from waste manifests.
    
    Returns waste quantities, disposal information, transporter details, and optionally emissions data.
    """
    logger.info("Processing waste generation extraction request")
    
    try:
        workflow = get_workflow()
        
        # Build query parameters
        params = build_query_parameters(request.facility_filter, request.date_range)
        
        # Add waste-specific parameters
        if request.hazardous_only:
            params['hazardous_only'] = True
        
        # Execute extraction workflow
        start_time = datetime.utcnow()
        result = workflow.extract_data(
            query_type=QueryType.WASTE_GENERATION,
            parameters=params,
            output_format=request.output_format
        )
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        # Close workflow connection
        workflow.close()
        
        # Prepare response
        response_data = {
            "query_type": QueryType.WASTE_GENERATION,
            "facility_filter": request.facility_filter.dict() if request.facility_filter else None,
            "date_range": request.date_range.dict() if request.date_range else None,
            "include_disposal_details": request.include_disposal_details,
            "include_transport_details": request.include_transport_details,
            "include_emissions": request.include_emissions,
            "hazardous_only": request.hazardous_only,
            "report_data": result.get("report_data", {}),
            "file_path": result.get("report_file_path")
        }
        
        metadata = {
            "total_queries": len(result.get("queries", [])),
            "successful_queries": len([q for q in result.get("query_results", []) if q.get("status") == "success"]),
            "total_records": sum(q.get("record_count", 0) for q in result.get("query_results", [])),
            "processing_status": result.get("status"),
            "generated_at": start_time.isoformat()
        }
        
        return ExtractionResponse(
            status="success" if result.get("status") == "completed" else "failed",
            message=f"Waste generation data extracted successfully" if result.get("status") == "completed" else "Extraction failed",
            data=response_data,
            metadata=metadata,
            processing_time=processing_time,
            errors=result.get("errors", [])
        )
        
    except Exception as e:
        logger.error(f"Failed to extract waste generation data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

# Batch ingestion endpoint
@app.post("/api/v1/ingest/batch", response_model=BatchIngestionResponse)
async def batch_ingest_documents(request: BatchIngestionRequest):
    """
    Run batch ingestion of all EHS documents (electric bill, water bill, waste manifest).
    
    This endpoint executes the scripts/ingest_all_documents.py script which:
    - Optionally clears the Neo4j database (default: True)
    - Ingests all three sample documents
    - Returns comprehensive results including node/relationship counts
    """
    logger.info(f"Starting batch document ingestion (clear_database={request.clear_database})")
    
    # Create LangSmith ingestion session if available
    ingestion_id = f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    langsmith_session = None
    if is_langsmith_available():
        langsmith_session = create_ingestion_session(ingestion_id)
        logger.info(f"LangSmith tracing enabled for ingestion session: {langsmith_session}")
    
    start_time = datetime.utcnow()
    
    try:
        # Get the script path relative to the API file location
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "run_batch_ingestion.py"
        )
        
        # Verify script exists
        if not os.path.exists(script_path):
            raise HTTPException(
                status_code=500,
                detail=f"Ingestion script not found at: {script_path}"
            )
        
        # Run the ingestion script
        logger.info(f"Executing ingestion script: {script_path}")
        
        # Create environment with correct PYTHONPATH
        env = {**os.environ}
        # Clear PYTHONPATH to avoid conflicts between our workflows and llama_index.workflows
        # The script will handle its own path setup
        env.pop('PYTHONPATH', None)
        
        # Pass ingestion ID to the script for LangSmith tracing
        if langsmith_session:
            env['LANGSMITH_INGESTION_ID'] = ingestion_id
        
        # Execute the script and capture output
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            env=env
        )
        
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        # Parse the output to extract results
        output_lines = result.stdout.split('\n')
        stderr_lines = result.stderr.split('\n')
        
        # Extract summary information from output
        successful_ingestions = 0
        total_nodes = 0
        total_relationships = 0
        documents_results = []
        
        # Parse output for results
        for line in output_lines:
            if "Successful Ingestions:" in line:
                try:
                    successful_ingestions = int(line.split(":")[1].strip())
                except:
                    pass
            elif "Nodes Created:" in line:
                try:
                    total_nodes = int(line.split(":")[1].strip())
                except:
                    pass
            elif "Relationships Created:" in line:
                try:
                    total_relationships = int(line.split(":")[1].strip())
                except:
                    pass
            elif "PROCESSED SUCCESSFULLY" in line or "PROCESSING FAILED" in line:
                # Extract document processing results
                doc_type = None
                if "ELECTRIC_BILL" in line:
                    doc_type = "electric_bill"
                elif "WATER_BILL" in line:
                    doc_type = "water_bill"
                elif "WASTE_MANIFEST" in line:
                    doc_type = "waste_manifest"
                
                if doc_type:
                    documents_results.append({
                        "document_type": doc_type,
                        "status": "success" if "SUCCESSFULLY" in line else "failed"
                    })
        
        # Check if script executed successfully
        if result.returncode == 0:
            # Fetch LangSmith traces if available
            langsmith_traces = None
            if langsmith_session and is_langsmith_available():
                try:
                    logger.info(f"Fetching LangSmith traces for session: {langsmith_session}")
                    
                    # Run the trace download script
                    traces_script_path = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "scripts",
                        "download_langsmith_traces.py"
                    )
                    
                    # Create traces output directory
                    traces_output_dir = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "traces_output"
                    )
                    os.makedirs(traces_output_dir, exist_ok=True)
                    
                    # Execute trace download with proper environment
                    trace_env = {**os.environ}
                    trace_env['LANGSMITH_API_KEY'] = os.getenv('LANGCHAIN_API_KEY', '')
                    
                    # Run the download script - script will generate its own filename
                    trace_result = subprocess.run(
                        [
                            sys.executable, 
                            traces_script_path,
                            "--project", "ehs-ai-demo-ingestion",
                            "--hours", "1",  # Last 1 hour
                            "--limit", "1000",
                            "--output-dir", traces_output_dir
                        ],
                        capture_output=True,
                        text=True,
                        env=trace_env
                    )
                    
                    if trace_result.returncode == 0:
                        # Look for the generated trace file in the output directory
                        trace_files = [f for f in os.listdir(traces_output_dir) if f.startswith('langsmith_traces_')]
                        if trace_files:
                            # Use the most recent trace file
                            latest_trace_file = max(trace_files, key=lambda x: os.path.getctime(os.path.join(traces_output_dir, x)))
                            trace_file_path = os.path.join(traces_output_dir, latest_trace_file)
                            
                            try:
                                with open(trace_file_path, 'r') as f:
                                    langsmith_traces = json.load(f)
                                    logger.info(f"Successfully fetched {len(langsmith_traces.get('traces', []))} LangSmith traces from {latest_trace_file}")
                            except Exception as e:
                                logger.warning(f"Failed to read trace file {trace_file_path}: {str(e)}")
                        else:
                            logger.warning("No trace files found in output directory")
                    else:
                        logger.warning(f"Failed to fetch LangSmith traces: {trace_result.stderr}")
                        
                except Exception as e:
                    logger.error(f"Error fetching LangSmith traces: {str(e)}")
            
            response_data = {
                "documents_processed": documents_results,
                "successful_ingestions": successful_ingestions,
                "total_nodes_created": total_nodes,
                "total_relationships_created": total_relationships,
                "database_cleared": request.clear_database,
                "langsmith_traces": langsmith_traces
            }
            
            metadata = {
                "script_path": script_path,
                "return_code": result.returncode,
                "python_version": sys.version,
                "generated_at": start_time.isoformat(),
                "langsmith_project": langsmith_session if langsmith_session else None,
                "ingestion_id": ingestion_id if is_langsmith_available() else None
            }
            
            return BatchIngestionResponse(
                status="success",
                message=f"Batch ingestion completed. {successful_ingestions}/3 documents processed successfully.",
                data=response_data,
                metadata=metadata,
                processing_time=processing_time,
                errors=[]
            )
        else:
            # Script failed
            errors = [line for line in stderr_lines if line.strip()]
            
            return BatchIngestionResponse(
                status="failed",
                message=f"Batch ingestion failed with return code {result.returncode}",
                data={},
                metadata={
                    "script_path": script_path,
                    "return_code": result.returncode,
                    "stdout": result.stdout[-1000:],  # Last 1000 chars
                    "stderr": result.stderr[-1000:]   # Last 1000 chars
                },
                processing_time=processing_time,
                errors=errors or ["Script execution failed"]
            )
            
    except Exception as e:
        logger.error(f"Failed to run batch ingestion: {str(e)}")
        
        return BatchIngestionResponse(
            status="failed",
            message=f"Batch ingestion failed: {str(e)}",
            data={},
            metadata={},
            processing_time=(datetime.utcnow() - start_time).total_seconds(),
            errors=[str(e)]
        )

# Generic extraction endpoint for custom queries
@app.post("/api/v1/extract/custom", response_model=ExtractionResponse)
async def extract_custom_data(
    query_type: str,
    facility_filter: Optional[FacilityFilter] = None,
    date_range: Optional[DateRangeFilter] = None,
    output_format: str = "json",
    custom_queries: Optional[List[Dict[str, Any]]] = None
):
    """
    Extract custom EHS data using provided queries or predefined query types.
    
    Supports all QueryType values from the extraction workflow.
    """
    logger.info(f"Processing custom extraction request for query type: {query_type}")
    
    try:
        # Validate query type
        if query_type not in [qt.value for qt in QueryType]:
            raise HTTPException(status_code=400, detail=f"Invalid query type: {query_type}")
        
        workflow = get_workflow()
        
        # Build query parameters
        params = build_query_parameters(facility_filter, date_range)
        
        # Execute extraction workflow
        start_time = datetime.utcnow()
        result = workflow.extract_data(
            query_type=query_type,
            queries=custom_queries,
            parameters=params,
            output_format=output_format
        )
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        # Close workflow connection
        workflow.close()
        
        # Prepare response
        response_data = {
            "query_type": query_type,
            "facility_filter": facility_filter.dict() if facility_filter else None,
            "date_range": date_range.dict() if date_range else None,
            "custom_queries": custom_queries,
            "report_data": result.get("report_data", {}),
            "file_path": result.get("report_file_path")
        }
        
        metadata = {
            "total_queries": len(result.get("queries", [])),
            "successful_queries": len([q for q in result.get("query_results", []) if q.get("status") == "success"]),
            "total_records": sum(q.get("record_count", 0) for q in result.get("query_results", [])),
            "processing_status": result.get("status"),
            "generated_at": start_time.isoformat()
        }
        
        return ExtractionResponse(
            status="success" if result.get("status") == "completed" else "failed",
            message=f"Custom data extraction completed" if result.get("status") == "completed" else "Extraction failed",
            data=response_data,
            metadata=metadata,
            processing_time=processing_time,
            errors=result.get("errors", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to extract custom data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

# Get available query types endpoint
@app.get("/api/v1/query-types")
async def get_query_types():
    """Get list of available query types for extraction."""
    return {
        "query_types": [
            {
                "value": qt.value,
                "name": qt.value.replace('_', ' ').title(),
                "description": _get_query_type_description(qt)
            }
            for qt in QueryType
        ]
    }

def _get_query_type_description(query_type: QueryType) -> str:
    """Get human-readable description for query types."""
    descriptions = {
        QueryType.FACILITY_EMISSIONS: "Extract facility-level emission data and calculations",
        QueryType.UTILITY_CONSUMPTION: "Extract electrical and utility consumption data from bills", 
        QueryType.WATER_CONSUMPTION: "Extract water usage data from water bills",
        QueryType.WASTE_GENERATION: "Extract waste generation data from manifests",
        QueryType.COMPLIANCE_STATUS: "Extract compliance and permit status information",
        QueryType.TREND_ANALYSIS: "Extract data for trend analysis over time",
        QueryType.CUSTOM: "Custom queries provided by user"
    }
    return descriptions.get(query_type, "Custom extraction query")

# Executive Dashboard endpoint
@app.get("/api/v1/executive-dashboard")
async def get_executive_dashboard(location: str, dateRange: str):
    """
    Get executive dashboard data for a specific location and date range.
    
    Args:
        location: The location identifier (required)
        dateRange: The date range identifier (required)
    
    Returns:
        JSON data from the corresponding dashboard data file
    """
    logger.info(f"Processing executive dashboard request for location: {location}, dateRange: {dateRange}")
    
    try:
        # Construct file path
        dashboard_data_dir = Path("/Users/eugene/dev/ai/agentos/ehs-ai-demo/ehs-analytics/dashboard-data")
        file_name = f"executive-dashboard-{location}-{dateRange}.json"
        file_path = dashboard_data_dir / file_name
        
        logger.debug(f"Looking for dashboard data file at: {file_path}")
        
        # Check if file exists
        if not file_path.exists():
            logger.warning(f"Dashboard data file not found: {file_path}")
            raise HTTPException(
                status_code=404,
                detail=f"Dashboard data not found for location '{location}' and date range '{dateRange}'. Expected file: {file_name}"
            )
        
        # Read and parse JSON file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                dashboard_data = json.load(f)
            
            logger.info(f"Successfully loaded dashboard data from {file_name}")
            return dashboard_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in dashboard data file {file_path}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Invalid JSON format in dashboard data file: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error reading dashboard data file {file_path}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error reading dashboard data file: {str(e)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in executive dashboard endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with consistent response format."""
    response_data = create_api_response(
        status="Failed",
        message=exc.detail,
        error=exc.detail
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=response_data
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions with consistent response format."""
    logger.error(f"Unhandled exception: {str(exc)}")
    response_data = create_api_response(
        status="Failed", 
        message="Internal server error",
        error=str(exc)
    )
    return JSONResponse(
        status_code=500,
        content=response_data
    )

# Startup/shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup tasks."""
    logger.info("EHS Extraction API starting up...")
    
    # Validate environment variables
    required_vars = ['NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        raise RuntimeError(f"Missing required environment variables: {missing_vars}")
    
    # Create reports directory if it doesn't exist
    os.makedirs("./reports", exist_ok=True)
    
    # Initialize Phase 1 enhancements
    global phase1_integration
    try:
        phase1_integration = create_phase1_integration()
        await phase1_integration.initialize_all_enhancements()
        # Integrate Phase 1 routers with the FastAPI app
        phase1_integration.integrate_with_app(app)
        logger.info("Phase 1 enhancements initialized and integrated successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Phase 1 enhancements: {str(e)}")
        # Continue running even if Phase 1 fails to initialize

    logger.info("EHS Extraction API startup completed")
    logger.info("Simple Rejection API endpoint available at: /api/v1/simple-rejected-documents")
    logger.info("Transcript API endpoints available at: /api/data/transcript")

@app.on_event("shutdown") 
async def shutdown_event():
    """Application shutdown tasks."""
    logger.info("EHS Extraction API shutting down...")

if __name__ == "__main__":
    # Get port from environment variable with fallback
    port = int(os.getenv("PORT", "8000"))
    
    # Run the application
    uvicorn.run(
        "ehs_extraction_api:app",  # Module string format for reload
        host="0.0.0.0",
        port=port,
        reload=True,
        reload_dirs=["src"]
    )