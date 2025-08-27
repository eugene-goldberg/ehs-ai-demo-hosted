# Quick Fix Guide: Prorating API Registration

## Problem
The prorating API endpoints return 404 Not Found even though the service is initialized.

## Root Cause
The prorating router is not registered with the main FastAPI application in `ehs_extraction_api.py`.

## Solution Steps

### Step 1: Check Current Registration
Look in `/data-foundation/backend/src/ehs_extraction_api.py` for:
```python
# Check if phase1 features are imported
from phase1_enhancements import ...

# Check if prorating router is included
app.include_router(...)
```

### Step 2: Add Prorating Router Import
Add to imports section:
```python
from phase1_enhancements.prorating_api import prorating_router
# OR
from phase1_enhancements.phase1_integration import setup_phase1_features
```

### Step 3: Register Router with App
Option A - Direct registration:
```python
# Add after other routers
app.include_router(
    prorating_router,
    prefix="/api/v1",
    tags=["prorating"]
)
```

Option B - Use phase1 integration:
```python
# After app creation and graph initialization
setup_phase1_features(app, graph)
```

### Step 4: Fix Router Prefix
In `/data-foundation/backend/src/phase1_enhancements/prorating_api.py`:
```python
# Current (might have double prefix):
router = APIRouter(
    prefix="/prorating",  # This might result in /api/v1/prorating
    ...
)

# If main app adds /api/v1, just use:
router = APIRouter(
    prefix="/prorating",
    ...
)
```

### Step 5: Verify Service Initialization
Ensure prorating service is initialized before first request:
```python
@app.on_event("startup")
async def startup_event():
    # Initialize graph connection
    # Initialize prorating service
    global prorating_service
    from phase1_enhancements.prorating_service import ProRatingService
    prorating_service = ProRatingService(graph)
```

### Step 6: Test
After making changes:
1. Restart the API server
2. Run: `curl http://localhost:8000/api/v1/prorating/health`
3. Should return 200 OK with health status

### Common Issues

1. **Import Error**: Make sure PYTHONPATH includes the src directory
2. **Service Not Initialized**: Check startup logs for initialization
3. **Wrong URL**: Try both `/api/v1/prorating/health` and `/prorating/health`
4. **Graph Connection**: Ensure Neo4j is running and accessible

### Verification Script
```bash
# Test health endpoint
curl -v http://localhost:8000/api/v1/prorating/health

# Check registered routes
curl http://localhost:8000/openapi.json | jq '.paths | keys[] | select(contains("prorating"))'
```

## Expected Result
```json
{
  "status": "healthy",
  "service": "prorating",
  "version": "1.0.0",
  "timestamp": "2025-08-25T..."
}