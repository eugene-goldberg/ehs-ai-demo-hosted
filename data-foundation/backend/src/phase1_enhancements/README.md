# Phase 1 Enhancements - EHS Data Platform

This module contains the Phase 1 enhancements for the EHS (Environmental, Health & Safety) Data Platform, providing advanced document processing capabilities including pro-rating allocation, audit trail tracking, and rejection workflow management.

## Overview

Phase 1 Enhancements introduces three key features to enhance the document processing pipeline:

- **Pro-rating Allocation System**: Automatically allocates utility bill costs across multiple tenants/units based on configurable allocation methods
- **Audit Trail Tracking**: Comprehensive tracking of all document processing actions with user attribution and metadata
- **Rejection Workflow Management**: Automated validation and rejection handling for non-compliant documents

## Architecture and Components

### Core Components

```
phase1_enhancements/
├── phase1_integration.py       # Main integration module and orchestrator
├── prorating_calculator.py     # Core pro-rating calculation logic
├── prorating_service.py        # Pro-rating business logic service
├── prorating_schema.py         # Pro-rating data models and schemas
├── prorating_api.py           # FastAPI router for pro-rating endpoints
├── audit_trail_service.py     # Audit trail business logic service
├── audit_trail_schema.py      # Audit trail data models and schemas
├── audit_trail_api.py         # FastAPI router for audit trail endpoints
├── audit_trail_integration.py # Integration helpers for audit tracking
├── rejection_workflow_service.py  # Rejection workflow business logic
├── rejection_tracking_schema.py   # Rejection tracking data models
├── rejection_tracking_api.py      # FastAPI router for rejection endpoints
└── integration_example.py        # Usage examples and demonstrations
```

### Database Integration

All Phase 1 components use Neo4j graph database with dedicated node types:

- **ProRatingAllocation**: Stores allocation calculations and distributions
- **AuditTrailEntry**: Records all document processing actions
- **RejectionEntry**: Tracks document rejections and validation failures
- **ValidationRule**: Defines validation criteria and rules

## Installation and Setup

### Prerequisites

- Python 3.8+
- Neo4j 4.4+ (running locally or Docker)
- FastAPI application framework
- Required Python packages (see requirements.txt)

### Environment Variables

Create or update your `.env` file with the following configuration:

```bash
# Neo4j Database Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# Phase 1 Configuration
PRORATING_BATCH_SIZE=100
DEFAULT_ALLOCATION_METHOD=square_footage
ENABLE_PRORATING_BACKFILL=true

AUDIT_RETENTION_DAYS=365
ENABLE_FILE_BACKUP=true
BACKUP_LOCATION=/tmp/audit_backups

AUTO_RETRY_ENABLED=false
MAX_RETRY_ATTEMPTS=3
VALIDATION_STRICTNESS=medium

# Document Processing Features
ENABLE_UTILITY_PRORATING=true
ENABLE_AUDIT_TRACKING=true
ENABLE_REJECTION_VALIDATION=true

# Environment Detection
ENVIRONMENT=development
```

### Installation Steps

1. **Install dependencies**:
```bash
cd data-foundation/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Initialize database schemas**:
```bash
python3 -c "
import asyncio
from src.phase1_enhancements.phase1_integration import create_phase1_integration

async def setup():
    integration = create_phase1_integration()
    await integration.initialize_all_enhancements()
    print('Phase 1 schemas initialized successfully')

asyncio.run(setup())
"
```

3. **Verify installation**:
```bash
python3 -c "
import asyncio
from src.phase1_enhancements.phase1_integration import create_phase1_integration

async def health_check():
    integration = create_phase1_integration()
    await integration.initialize_all_enhancements()
    health = await integration.health_check()
    print(f'Health Status: {health[\"status\"]}')

asyncio.run(health_check())
"
```

## Integration Guide with main.py

### Adding Phase 1 to Existing FastAPI Application

1. **Import the integration module** in your main FastAPI application:

```python
from src.phase1_enhancements.phase1_integration import initialize_phase1_for_app
```

2. **Integrate with your FastAPI app**:

```python
from fastapi import FastAPI
import asyncio

app = FastAPI(title="EHS Data Platform")

# Initialize Phase 1 integration
@app.on_event("startup")
async def startup_event():
    global phase1_integration
    phase1_integration = await initialize_phase1_for_app(app, api_prefix="/api/v1")
    print("Phase 1 enhancements initialized")
```

3. **Process documents with Phase 1 features**:

```python
@app.post("/documents/upload")
async def upload_document(file: UploadFile, background_tasks: BackgroundTasks):
    # Your existing document processing logic...
    
    # Add Phase 1 processing
    document_info = {
        "filename": file.filename,
        "content": await file.read(),
        "user_id": "current_user",
        "metadata": {"upload_time": datetime.utcnow()}
    }
    
    phase1_results = await phase1_integration.process_document_with_phase1_features(
        document_info, background_tasks
    )
    
    return {
        "message": "Document processed",
        "phase1_enhancements": phase1_results
    }
```

### Integration Points

The Phase 1 system integrates at several key points in the document processing pipeline:

- **Document Upload**: Automatic audit trail creation
- **Document Validation**: Rejection workflow validation
- **Utility Bill Processing**: Automatic pro-rating allocation
- **Background Processing**: Asynchronous enhancement processing

## API Endpoint Documentation Summary

### Pro-rating Endpoints

- `POST /api/v1/prorating/allocations` - Create new pro-rating allocation
- `GET /api/v1/prorating/allocations/{allocation_id}` - Get allocation details
- `GET /api/v1/prorating/allocations` - List all allocations
- `PUT /api/v1/prorating/allocations/{allocation_id}` - Update allocation
- `POST /api/v1/prorating/calculate` - Calculate pro-rating for given parameters
- `GET /api/v1/prorating/reports/monthly` - Generate monthly pro-rating reports

### Audit Trail Endpoints

- `POST /api/v1/audit-trail/entries` - Create audit trail entry
- `GET /api/v1/audit-trail/entries` - List audit trail entries with filtering
- `GET /api/v1/audit-trail/entries/{entry_id}` - Get specific audit entry
- `GET /api/v1/audit-trail/document/{document_name}` - Get document's audit history
- `GET /api/v1/audit-trail/reports` - Generate audit reports

### Rejection Tracking Endpoints

- `POST /api/v1/rejection-tracking/rejections` - Create rejection entry
- `GET /api/v1/rejection-tracking/rejections` - List rejections
- `GET /api/v1/rejection-tracking/rejections/{rejection_id}` - Get rejection details
- `PUT /api/v1/rejection-tracking/rejections/{rejection_id}/resolve` - Resolve rejection
- `POST /api/v1/rejection-tracking/validate` - Validate document against rules
- `GET /api/v1/rejection-tracking/validation-rules` - List validation rules

### Health and Status Endpoints

- `GET /api/v1/phase1/health` - Comprehensive health check for all Phase 1 services
- `GET /api/v1/phase1/status` - Current status and configuration of Phase 1 integration

## Configuration Requirements

### Core Configuration

All Phase 1 services require proper Neo4j database connectivity and the following minimum configuration:

```python
{
    "neo4j_config": {
        "uri": "bolt://localhost:7687",
        "username": "neo4j", 
        "password": "your_password",
        "database": "neo4j"
    }
}
```

### Service-Specific Configuration

**Pro-rating Service**:
- `PRORATING_BATCH_SIZE`: Number of allocations to process in batch (default: 100)
- `DEFAULT_ALLOCATION_METHOD`: Default method for allocations (options: square_footage, equal_split, custom)
- `ENABLE_PRORATING_BACKFILL`: Enable backfill for historical data

**Audit Trail Service**:
- `AUDIT_RETENTION_DAYS`: Number of days to retain audit records (default: 365)
- `ENABLE_FILE_BACKUP`: Enable backup of audit trail files
- `BACKUP_LOCATION`: Directory for audit trail backups

**Rejection Workflow Service**:
- `AUTO_RETRY_ENABLED`: Enable automatic retry of failed documents
- `MAX_RETRY_ATTEMPTS`: Maximum number of retry attempts
- `VALIDATION_STRICTNESS`: Validation level (low, medium, high)

## Migration Instructions for Existing Data

### Automatic Migration

Run the built-in migration script to update existing data:

```python
import asyncio
from src.phase1_enhancements.phase1_integration import create_phase1_integration

async def run_migration():
    integration = create_phase1_integration()
    await integration.initialize_all_enhancements()
    
    print("Starting Phase 1 data migration...")
    results = await integration.migrate_existing_data()
    
    print(f"Migration completed with status: {results['overall_status']}")
    print(f"Documents migrated: {results['statistics'].get('documents_migrated', 0)}")
    print(f"Rejections migrated: {results['statistics'].get('rejections_migrated', 0)}")
    print(f"Allocations created: {results['statistics'].get('allocations_created', 0)}")

asyncio.run(run_migration())
```

### Manual Migration Steps

If you prefer manual migration or need custom migration logic:

1. **Audit Trail Migration**:
```cypher
MATCH (doc:Document)
WHERE NOT EXISTS((doc)-[:HAS_AUDIT_TRAIL]->())
CREATE (audit:AuditTrailEntry {
    id: randomUUID(),
    document_name: doc.file_name,
    action: "migrated_to_audit_trail",
    user_id: "system",
    timestamp: datetime(),
    metadata: "migration=true"
})
CREATE (doc)-[:HAS_AUDIT_TRAIL]->(audit)
```

2. **Rejection Tracking Migration**:
```cypher
MATCH (doc:Document)
WHERE doc.status = 'Failed' 
AND NOT EXISTS((doc)-[:HAS_REJECTION]->())
CREATE (rejection:RejectionEntry {
    id: randomUUID(),
    document_name: doc.file_name,
    rejection_reason: "Historical failure (migrated)",
    status: "rejected",
    created_at: datetime(),
    metadata: "migration=true"
})
CREATE (doc)-[:HAS_REJECTION]->(rejection)
```

3. **Pro-rating Migration**:
```cypher
MATCH (doc:Document)
WHERE doc.file_name =~ '(?i).*(electric|utility|water|gas|bill).*'
AND doc.status = 'Completed'
AND NOT EXISTS((doc)-[:HAS_ALLOCATION]->())
CREATE (allocation:ProRatingAllocation {
    id: randomUUID(),
    document_name: doc.file_name,
    period_start: date("2024-01-01"),
    period_end: date("2024-01-31"),
    total_amount: 1000.0,
    utility_type: "electricity",
    allocation_method: "square_footage",
    created_at: datetime(),
    metadata: "migration=true"
})
CREATE (doc)-[:HAS_ALLOCATION]->(allocation)
```

## Testing Recommendations

### Unit Testing

Run individual service tests:

```bash
# Test pro-rating functionality
python3 -m pytest tests/test_prorating_service.py -v

# Test audit trail functionality  
python3 -m pytest tests/test_audit_trail_service.py -v

# Test rejection workflow functionality
python3 -m pytest tests/test_rejection_workflow_service.py -v
```

### Integration Testing

Test the complete Phase 1 integration:

```bash
python3 src/phase1_enhancements/integration_example.py
```

### API Testing

Use the provided test script to validate API endpoints:

```bash
# Test all Phase 1 endpoints
curl -X GET http://localhost:8000/api/v1/phase1/health
curl -X GET http://localhost:8000/api/v1/phase1/status

# Test pro-rating endpoints
curl -X POST http://localhost:8000/api/v1/prorating/calculate \
  -H "Content-Type: application/json" \
  -d '{"total_amount": 1000, "allocation_method": "square_footage"}'

# Test audit trail endpoints
curl -X GET http://localhost:8000/api/v1/audit-trail/entries?limit=10
```

### Performance Testing

Monitor performance with sample data:

```python
import asyncio
import time
from src.phase1_enhancements.phase1_integration import create_phase1_integration

async def performance_test():
    integration = create_phase1_integration()
    await integration.initialize_all_enhancements()
    
    # Test processing 100 documents
    start_time = time.time()
    for i in range(100):
        await integration.process_document_with_phase1_features({
            "filename": f"test_document_{i}.pdf",
            "content": "Test content",
            "user_id": "test_user"
        }, None)
    
    end_time = time.time()
    print(f"Processed 100 documents in {end_time - start_time:.2f} seconds")

asyncio.run(performance_test())
```

## Troubleshooting Guide

### Common Issues and Solutions

#### Database Connection Issues

**Problem**: "Failed to connect to Neo4j database"
**Solution**: 
1. Verify Neo4j is running: `docker ps` or `systemctl status neo4j`
2. Check connection parameters in `.env` file
3. Test connection manually:
```python
from neo4j import GraphDatabase
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
with driver.session() as session:
    result = session.run("RETURN 1 as test")
    print(result.single()["test"])
```

#### Schema Initialization Failures

**Problem**: "Schema initialization failed"
**Solution**:
1. Ensure Neo4j database is empty or compatible
2. Check database permissions for user
3. Run schema initialization manually:
```bash
python3 -c "
import asyncio
from src.phase1_enhancements.prorating_service import ProRatingService
from src.shared.common_fn import create_graph_database_connection

async def init_schema():
    graph = create_graph_database_connection('bolt://localhost:7687', 'neo4j', 'password', 'neo4j')
    service = ProRatingService(graph)
    await service.initialize_schema()

asyncio.run(init_schema())
"
```

#### Service Health Check Failures

**Problem**: Services showing as "unhealthy" in health check
**Solution**:
1. Check individual service logs for specific errors
2. Verify all required environment variables are set
3. Test services individually:
```python
from src.phase1_enhancements.prorating_service import ProRatingService
# Test each service independently
```

#### Memory/Performance Issues

**Problem**: High memory usage or slow performance
**Solution**:
1. Reduce batch sizes in configuration
2. Enable connection pooling for Neo4j
3. Monitor and optimize Cypher queries
4. Implement proper cleanup in background tasks

### Debugging Tools

#### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("phase1_enhancements")
logger.setLevel(logging.DEBUG)
```

#### Database Query Monitoring

```cypher
// Monitor slow queries in Neo4j
CALL dbms.listQueries() YIELD queryId, query, elapsedTimeMillis
WHERE elapsedTimeMillis > 1000
RETURN queryId, query, elapsedTimeMillis
```

#### Service Monitoring

```python
# Monitor service health continuously
import asyncio
from src.phase1_enhancements.phase1_integration import create_phase1_integration

async def monitor_health():
    integration = create_phase1_integration()
    await integration.initialize_all_enhancements()
    
    while True:
        health = await integration.health_check()
        print(f"Health Status: {health['status']} at {health['timestamp']}")
        await asyncio.sleep(30)

asyncio.run(monitor_health())
```

## Next Steps and Future Enhancements

### Phase 2 Planning

The following enhancements are planned for Phase 2:

1. **Advanced Analytics Dashboard**
   - Real-time pro-rating analytics
   - Audit trail visualization
   - Rejection pattern analysis

2. **Machine Learning Integration**
   - Automated utility bill data extraction
   - Predictive rejection detection
   - Intelligent allocation method selection

3. **Multi-tenant Support**
   - Tenant-specific configuration
   - Isolated data processing
   - Role-based access control

### Immediate Improvements

Consider implementing these enhancements:

1. **Enhanced Validation Rules**
   - Custom validation rule engine
   - Dynamic rule configuration
   - A/B testing for validation strictness

2. **Improved Error Handling**
   - Circuit breaker patterns
   - Automatic retry with exponential backoff
   - Dead letter queue for failed processing

3. **Performance Optimizations**
   - Connection pooling for Neo4j
   - Caching layer for frequent queries
   - Asynchronous batch processing

### Integration Opportunities

- **Document Management Systems**: Direct integration with SharePoint, Box, or Google Drive
- **Accounting Software**: Export pro-rating allocations to QuickBooks or SAP
- **Notification Systems**: Slack/Teams integration for rejection notifications
- **Reporting Tools**: Integration with Tableau or Power BI for advanced reporting

### Contributing

To contribute to Phase 1 enhancements:

1. Follow the existing code structure and patterns
2. Add comprehensive tests for new features
3. Update documentation for any new endpoints or configuration options
4. Ensure backward compatibility with existing implementations

For questions or support, refer to the main project documentation or contact the development team.