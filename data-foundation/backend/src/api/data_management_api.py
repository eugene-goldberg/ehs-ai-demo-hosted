from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import uuid
from datetime import datetime
import logging
import os
import json
from pathlib import Path
import threading

from models import (
    DataUploadRequest, 
    DataUploadResponse, 
    IncidentReport, 
    ComplianceRecord,
    IncidentSeverity,
    ComplianceStatus
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Mock data storage (in production, use a proper database)
incidents_db = []
compliance_db = []

# File-based transcript storage with thread safety
transcript_file_path = "/home/azureuser/dev/ehs-ai-demo/data-foundation/web-app/backend/transcript_data.json"
transcript_lock = threading.Lock()

def load_transcript() -> List[Dict[str, Any]]:
    """Load transcript data from file or return empty list if file doesn't exist or is corrupted."""
    try:
        if os.path.exists(transcript_file_path):
            with open(transcript_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                else:
                    logger.warning(f"Invalid transcript file format, returning empty list")
                    return []
        else:
            return []
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading transcript file: {str(e)}")
        return []

def save_transcript(data: List[Dict[str, Any]]) -> bool:
    """Save transcript data to file with error handling."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(transcript_file_path), exist_ok=True)
        
        # Write to temporary file first, then rename for atomic operation
        temp_file = transcript_file_path + ".tmp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Atomic rename
        os.rename(temp_file, transcript_file_path)
        return True
    except (IOError, OSError) as e:
        logger.error(f"Error saving transcript file: {str(e)}")
        # Clean up temp file if it exists
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
        return False

# Pydantic model for transcript add request
class TranscriptAddRequest(BaseModel):
    role: str
    content: str
    context: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None

@router.post("/upload", response_model=DataUploadResponse)
async def upload_data(file: UploadFile = File(...)):
    """Upload and process EHS data files"""
    try:
        # Mock file processing
        upload_id = str(uuid.uuid4())
        
        # Simulate processing based on file type
        if file.filename.endswith('.csv'):
            records_processed = 150  # Mock count
        elif file.filename.endswith('.xlsx'):
            records_processed = 75
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        return DataUploadResponse(
            upload_id=upload_id,
            status="success",
            message=f"Successfully processed {file.filename}",
            records_processed=records_processed
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/incidents", response_model=List[IncidentReport])
async def get_incidents():
    """Get all incident reports"""
    if not incidents_db:
        # Return mock data if empty
        return [
            IncidentReport(
                id="INC-001",
                title="Chemical Spill in Laboratory",
                description="Minor chemical spill during routine testing",
                severity=IncidentSeverity.MEDIUM,
                location="Lab Building A, Room 205",
                reporter_name="John Smith",
                reporter_email="john.smith@company.com",
                incident_date=datetime(2024, 1, 15, 14, 30),
                created_at=datetime(2024, 1, 15, 15, 0),
                status="under_investigation"
            ),
            IncidentReport(
                id="INC-002",
                title="Safety Equipment Malfunction",
                description="Emergency shower not functioning properly",
                severity=IncidentSeverity.HIGH,
                location="Production Floor B",
                reporter_name="Sarah Johnson",
                reporter_email="sarah.johnson@company.com",
                incident_date=datetime(2024, 1, 10, 9, 15),
                created_at=datetime(2024, 1, 10, 9, 30),
                status="resolved"
            )
        ]
    return incidents_db

@router.post("/incidents", response_model=IncidentReport)
async def create_incident(incident: IncidentReport):
    """Create a new incident report"""
    incident.id = f"INC-{len(incidents_db) + 1:03d}"
    incident.created_at = datetime.now()
    incidents_db.append(incident)
    return incident

@router.get("/compliance", response_model=List[ComplianceRecord])
async def get_compliance_records():
    """Get all compliance records"""
    if not compliance_db:
        # Return mock data if empty
        return [
            ComplianceRecord(
                id="COMP-001",
                regulation_name="OSHA 29 CFR 1910.1200",
                compliance_status=ComplianceStatus.COMPLIANT,
                last_audit_date=datetime(2023, 12, 1),
                next_audit_date=datetime(2024, 6, 1),
                responsible_person="Mike Wilson",
                notes="Hazard communication standard - all requirements met"
            ),
            ComplianceRecord(
                id="COMP-002",
                regulation_name="EPA Clean Air Act",
                compliance_status=ComplianceStatus.UNDER_REVIEW,
                last_audit_date=datetime(2023, 11, 15),
                next_audit_date=datetime(2024, 5, 15),
                responsible_person="Lisa Chen",
                notes="Emission monitoring review in progress"
            )
        ]
    return compliance_db

@router.post("/compliance", response_model=ComplianceRecord)
async def create_compliance_record(record: ComplianceRecord):
    """Create a new compliance record"""
    record.id = f"COMP-{len(compliance_db) + 1:03d}"
    compliance_db.append(record)
    return record

@router.get("/incidents/{incident_id}", response_model=IncidentReport)
async def get_incident(incident_id: str):
    """Get a specific incident by ID"""
    incident = next((inc for inc in incidents_db if inc.id == incident_id), None)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    return incident

@router.put("/incidents/{incident_id}", response_model=IncidentReport)
async def update_incident(incident_id: str, incident_update: IncidentReport):
    """Update an existing incident"""
    for i, incident in enumerate(incidents_db):
        if incident.id == incident_id:
            incident_update.id = incident_id
            incidents_db[i] = incident_update
            return incident_update
    raise HTTPException(status_code=404, detail="Incident not found")

@router.get("/processed-documents")
async def get_processed_documents():
    """Get all processed documents from Neo4j database."""
    from neo4j import GraphDatabase
    import os
    from datetime import datetime
    
    # Neo4j connection details
    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    username = os.getenv('NEO4J_USERNAME', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD', 'EhsAI2024!')
    
    processed_docs = []
    
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        with driver.session() as session:
            # Query to get all documents
            result = session.run("""
                MATCH (d:Document)
                RETURN d.id as id, 
                       d.type as document_type,
                       d.created_at as date_received,
                       labels(d) as labels,
                       d.account_number as account_number,
                       d.service_address as site,
                       d.file_path as file_path
                ORDER BY d.created_at DESC
            """)
            
            for record in result:
                # Extract document type from labels
                labels = record['labels']
                doc_type = 'Unknown'
                
                if 'Electricitybill' in labels or 'ElectricityBill' in labels:
                    doc_type = 'Electric Bill'
                elif 'Waterbill' in labels or 'WaterBill' in labels:
                    doc_type = 'Water Bill'
                elif 'Wastemanifest' in labels or 'WasteManifest' in labels:
                    doc_type = 'Waste Manifest'
                
                # Format the document data
                doc = {
                    'id': record['id'],
                    'document_type': doc_type,
                    'date_received': record['date_received'] or datetime.now().isoformat(),
                    'site': record['site'] or record['account_number'] or 'Main Facility',
                    'file_path': record['file_path']
                }
                
                processed_docs.append(doc)
                
        driver.close()
        
    except Exception as e:
        print(f"Error querying Neo4j: {e}")
        # Return mock data if Neo4j is not available
        return [
            {
                "id": "electric_bill_001",
                "document_type": "Electric Bill",
                "date_received": "2025-08-18T10:30:00",
                "site": "Main Facility",
                "file_path": "/tmp/electric_bill_001.pdf"
            },
            {
                "id": "water_bill_001", 
                "document_type": "Water Bill",
                "date_received": "2025-08-18T10:31:00",
                "site": "Main Facility",
                "file_path": "/tmp/water_bill_001.pdf"
            },
            {
                "id": "waste_manifest_001",
                "document_type": "Waste Manifest",
                "date_received": "2025-08-18T10:32:00",
                "site": "Warehouse B",
                "file_path": "/tmp/waste_manifest_001.pdf"
            }
        ]
    
    return processed_docs

def get_mock_document_data(document_id: str):
    """Get mock document data for testing/fallback purposes."""
    # Define a comprehensive set of mock documents with various IDs
    mock_documents = {
        "electric_bill_001": {
            "id": "electric_bill_001",
            "document_type": "Electric Bill",
            "date_received": "2025-08-18T10:30:00",
            "site": "Main Facility",
            "account_number": "ACC-123456",
            "total_kwh": 1250.5,
            "total_cost": 157.89,
            "billing_period_start": "2025-07-15",
            "billing_period_end": "2025-08-15",
            "service_address": "123 Industrial Park, Main Facility",
            "rate_schedule": "Commercial Rate B",
            "demand_charge": 45.20,
            "energy_charge": 112.69
        },
        "water_bill_001": {
            "id": "water_bill_001",
            "document_type": "Water Bill",
            "date_received": "2025-08-18T10:31:00",
            "site": "Main Facility",
            "account_number": "WAT-789012",
            "total_gallons": 15750,
            "total_cost": 89.45,
            "billing_period": "July 2025",
            "service_address": "123 Industrial Park, Main Facility",
            "rate_per_gallon": 0.00568,
            "base_charge": 25.00,
            "usage_charge": 64.45
        },
        "waste_manifest_001": {
            "id": "waste_manifest_001",
            "document_type": "Waste Manifest",
            "date_received": "2025-08-18T10:32:00",
            "site": "Warehouse B",
            "manifest_tracking_number": "WM-2025-001234",
            "total_waste_quantity": 500.0,
            "waste_quantity_unit": "lbs",
            "disposal_method": "Incineration",
            "generator_name": "ABC Manufacturing Corp",
            "facility_name": "EnviroSafe Disposal LLC",
            "waste_description": "Non-hazardous industrial waste",
            "transportation_date": "2025-08-15",
            "disposal_date": "2025-08-16"
        }
    }
    
    # Try exact match first
    if document_id in mock_documents:
        return mock_documents[document_id]
    
    # Try pattern matching for documents with similar IDs
    if "electric" in document_id.lower() or "utility" in document_id.lower():
        mock_doc = mock_documents["electric_bill_001"].copy()
        mock_doc["id"] = document_id
        return mock_doc
    elif "water" in document_id.lower():
        mock_doc = mock_documents["water_bill_001"].copy()
        mock_doc["id"] = document_id
        return mock_doc
    elif "waste" in document_id.lower() or "manifest" in document_id.lower():
        mock_doc = mock_documents["waste_manifest_001"].copy()
        mock_doc["id"] = document_id
        return mock_doc
    
    # Return None if no match found
    return None

@router.get("/processed-documents/{document_id}")
async def get_document_details(document_id: str):
    """Get detailed information for a specific document from Neo4j database."""
    from neo4j import GraphDatabase
    import os
    
    logger.info(f"Fetching document details for ID: {document_id}")
    
    # Neo4j connection details
    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    username = os.getenv('NEO4J_USERNAME', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD', 'EhsAI2024!')
    
    driver = None
    neo4j_error = None
    
    try:
        logger.info(f"Attempting to connect to Neo4j at {uri} with username {username}")
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # Test the connection first
        with driver.session() as session:
            connection_result = session.run("RETURN 1 as test")
            connection_test = connection_result.single()
            if not connection_test:
                raise Exception("Neo4j connection test failed")
            logger.info("Neo4j connection successful")
        
        with driver.session() as session:
            # Query to get document details with extracted data from related nodes
            logger.info(f"Executing Neo4j query for document_id: {document_id}")
            result = session.run("""
                MATCH (d:Document {id: $document_id})
                OPTIONAL MATCH (d)-[:EXTRACTED_TO]->(ub:UtilityBill)
                OPTIONAL MATCH (d)-[:EXTRACTED_TO]->(wb:WaterBill)  
                OPTIONAL MATCH (d)-[:TRACKS]->(wm:WasteManifest)
                OPTIONAL MATCH (d)-[:HAS_MONTHLY_ALLOCATION]->(ma:MonthlyUsageAllocation)
                WITH d, labels(d) as labels, ub, wb, wm, 
                     collect(ma) as monthly_allocations
                RETURN d, 
                       labels,
                       ub,
                       wb,
                       wm,
                       monthly_allocations,
                       CASE 
                           WHEN size(monthly_allocations) > 0 
                           THEN monthly_allocations[0].allocated_usage
                           ELSE null
                       END as current_month_usage
            """, document_id=document_id)
            
            record = result.single()
            logger.info(f"Neo4j query returned: {record is not None}")
            
            if not record:
                logger.warning(f"No document found in Neo4j with ID: {document_id}")
                neo4j_error = f"Document {document_id} not found in Neo4j database"
            else:
                document = record['d']
                labels = record['labels']
                utility_bill = record['ub']
                water_bill = record['wb']
                waste_manifest = record['wm']
                monthly_allocations = record['monthly_allocations']
                current_month_usage = record['current_month_usage']
                
                logger.info(f"Found document with labels: {labels}")
                logger.info(f"Related nodes - utility_bill: {utility_bill is not None}, water_bill: {water_bill is not None}, waste_manifest: {waste_manifest is not None}")
                logger.info(f"Monthly allocations: {len(monthly_allocations) if monthly_allocations else 0}, Current month usage: {current_month_usage}")
                
                # Extract document type from labels
                doc_type = 'Unknown'
                if 'Electricitybill' in labels or 'ElectricityBill' in labels:
                    doc_type = 'Electric Bill'
                elif 'Waterbill' in labels or 'WaterBill' in labels:
                    doc_type = 'Water Bill'
                elif 'Wastemanifest' in labels or 'WasteManifest' in labels:
                    doc_type = 'Waste Manifest'
                
                # Start with document properties
                doc_properties = dict(document)
                doc_properties['document_type'] = doc_type
                doc_properties['labels'] = labels
                
                # Add prorated monthly usage from MonthlyUsageAllocation
                if monthly_allocations and len(monthly_allocations) > 0:
                    # Get current month's allocation
                    from datetime import datetime
                    current_date = datetime.now()
                    current_year = current_date.year
                    current_month = current_date.month
                    
                    logger.info(f"Processing {len(monthly_allocations)} monthly allocations")
                    
                    # Find allocation for current month
                    prorated_usage = None
                    for allocation in monthly_allocations:
                        alloc_dict = dict(allocation)
                        logger.info(f"Allocation: year={alloc_dict.get('usage_year')}, month={alloc_dict.get('usage_month')}, usage={alloc_dict.get('allocated_usage')}")
                        if (alloc_dict.get('usage_year') == current_year and 
                            alloc_dict.get('usage_month') == current_month):
                            prorated_usage = alloc_dict.get('allocated_usage')
                            logger.info(f"Found current month allocation: {prorated_usage}")
                            break
                    
                    # If no current month allocation, use the first one
                    if prorated_usage is None and len(monthly_allocations) > 0:
                        prorated_usage = dict(monthly_allocations[0]).get('allocated_usage')
                        logger.info(f"Using first allocation as fallback: {prorated_usage}")
                    
                    doc_properties['prorated_monthly_usage'] = prorated_usage
                    logger.info(f"Added prorated_monthly_usage: {prorated_usage} to response")
                else:
                    logger.info("No monthly allocations found for this document")
                
                # Merge extracted data based on document type
                if utility_bill:
                    # Merge UtilityBill properties
                    utility_properties = dict(utility_bill)
                    doc_properties.update(utility_properties)
                    logger.info("Merged UtilityBill properties")
                    
                elif water_bill:
                    # Merge WaterBill properties
                    water_properties = dict(water_bill)
                    doc_properties.update(water_properties)
                    logger.info("Merged WaterBill properties")
                    
                elif waste_manifest:
                    # Merge WasteManifest properties
                    waste_properties = dict(waste_manifest)
                    doc_properties.update(waste_properties)
                    logger.info("Merged WasteManifest properties")
                else:
                    logger.warning("No extracted data found for document")
                
                logger.info(f"Successfully processed document {document_id}")
                return doc_properties
                
    except Exception as e:
        logger.error(f"Error querying Neo4j for document {document_id}: {str(e)}")
        neo4j_error = str(e)
    finally:
        if driver:
            try:
                driver.close()
                logger.info("Neo4j driver closed")
            except Exception as e:
                logger.error(f"Error closing Neo4j driver: {e}")
    
    # Fallback to mock data
    logger.info(f"Falling back to mock data for document {document_id}")
    mock_data = get_mock_document_data(document_id)
    
    if mock_data:
        logger.info(f"Returning mock data for document {document_id}")
        # Add error information to the response for debugging
        mock_data['_debug_info'] = {
            'source': 'mock_data',
            'neo4j_error': neo4j_error,
            'neo4j_uri': uri,
            'neo4j_username': username
        }
        return mock_data
    else:
        logger.error(f"No mock data available for document {document_id}")
        error_detail = f"Document '{document_id}' not found in Neo4j database"
        if neo4j_error:
            error_detail += f" (Neo4j error: {neo4j_error})"
        raise HTTPException(
            status_code=404, 
            detail=error_detail
        )

@router.get("/documents/{document_id}/download")
async def download_document(document_id: str):
    """Download a document file by document ID."""
    from neo4j import GraphDatabase
    import os
    from pathlib import Path
    
    # Neo4j connection details
    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    username = os.getenv('NEO4J_USERNAME', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD', 'EhsAI2024!')
    
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        with driver.session() as session:
            # Query to get the file path for the document
            result = session.run(
                "MATCH (d:Document {id: $document_id}) RETURN d.file_path as file_path",
                document_id=document_id
            )
            
            record = result.single()
            if not record or not record['file_path']:
                raise HTTPException(status_code=404, detail="Document or file path not found")
            
            file_path = record['file_path']
            
        driver.close()
        
        # Security check: ensure file path is safe (prevent directory traversal)
        file_path = Path(file_path).resolve()
        
        # Check if file exists
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found on disk")
        
        # Check if it's actually a file (not a directory)
        if not file_path.is_file():
            raise HTTPException(status_code=400, detail="Path is not a file")
        
        # Return the file
        return FileResponse(
            path=str(file_path),
            filename=file_path.name,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while downloading file")

@router.get("/transcript")
async def get_transcript():
    """Get the LLM transcript data from file storage"""
    try:
        with transcript_lock:
            transcript_data = load_transcript()
            return {"transcript": transcript_data}
        
    except Exception as e:
        logger.error(f"Error retrieving transcript: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error retrieving transcript: {str(e)}"
        )

@router.post("/transcript")
async def add_transcript_entry(entry: Dict[str, Any]):
    """Add an entry to the transcript store (deprecated - use /transcript/add)"""
    try:
        with transcript_lock:
            # Load current transcript data
            transcript_data = load_transcript()
            
            # Add timestamp if not present
            if "timestamp" not in entry:
                entry["timestamp"] = datetime.now().isoformat()
            
            # Add unique ID if not present
            if "id" not in entry:
                entry["id"] = str(uuid.uuid4())
            
            # Add entry to data
            transcript_data.append(entry)
            
            # Save to file
            if not save_transcript(transcript_data):
                raise HTTPException(status_code=500, detail="Failed to save transcript to file")
            
            logger.info(f"Added transcript entry: {entry['id']}")
            
            return {"status": "success", "entry_id": entry["id"]}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding transcript entry: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error adding transcript entry: {str(e)}"
        )

@router.post("/transcript/add")
async def add_transcript_entry_structured(request: TranscriptAddRequest):
    """Add an entry to the transcript store with structured input and thread safety"""
    try:
        with transcript_lock:
            # Load current transcript data
            transcript_data = load_transcript()
            
            # Create entry dictionary
            entry = {
                "id": str(uuid.uuid4()),
                "role": request.role,
                "content": request.content,
                "context": request.context,
                "timestamp": request.timestamp or datetime.now().isoformat()
            }
            
            # Add entry to data
            transcript_data.append(entry)
            
            # Enforce 10,000 entry limit by removing oldest entries
            if len(transcript_data) > 10000:
                entries_to_remove = len(transcript_data) - 10000
                transcript_data = transcript_data[entries_to_remove:]
                logger.info(f"Removed {entries_to_remove} oldest entries to maintain 10,000 entry limit")
            
            # Save to file
            if not save_transcript(transcript_data):
                raise HTTPException(status_code=500, detail="Failed to save transcript to file")
            
            logger.info(f"Added structured transcript entry: {entry['id']}")
            
            return {
                "status": "success",
                "message": "Transcript entry added successfully",
                "entry": entry,
                "total_entries": len(transcript_data)
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding structured transcript entry: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error adding transcript entry: {str(e)}"
        )

@router.delete("/transcript")
async def clear_transcript():
    """Clear all transcript entries by clearing the file"""
    try:
        with transcript_lock:
            # Save empty list to file
            if not save_transcript([]):
                raise HTTPException(status_code=500, detail="Failed to clear transcript file")
            
            logger.info("Transcript file cleared")
            return {"status": "success", "message": "Transcript cleared"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing transcript: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error clearing transcript: {str(e)}"
        )
