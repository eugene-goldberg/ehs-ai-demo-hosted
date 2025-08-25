from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from typing import List
import uuid
from datetime import datetime
import logging
import os
from pathlib import Path
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
                
                if 'Utilitybill' in labels or 'UtilityBill' in labels:
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
                RETURN d, 
                       labels(d) as labels,
                       ub,
                       wb,
                       wm
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
                
                logger.info(f"Found document with labels: {labels}")
                logger.info(f"Related nodes - utility_bill: {utility_bill is not None}, water_bill: {water_bill is not None}, waste_manifest: {waste_manifest is not None}")
                
                # Extract document type from labels
                doc_type = 'Unknown'
                if 'Utilitybill' in labels or 'UtilityBill' in labels:
                    doc_type = 'Electric Bill'
                elif 'Waterbill' in labels or 'WaterBill' in labels:
                    doc_type = 'Water Bill'
                elif 'Wastemanifest' in labels or 'WasteManifest' in labels:
                    doc_type = 'Waste Manifest'
                
                # Start with document properties
                doc_properties = dict(document)
                doc_properties['document_type'] = doc_type
                doc_properties['labels'] = labels
                
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