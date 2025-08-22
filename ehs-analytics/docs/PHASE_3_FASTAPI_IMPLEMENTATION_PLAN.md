# Phase 3 FastAPI Implementation Plan - Integration Approach

**Date:** 2025-08-21  
**Status:** Ready for Implementation  
**Dependencies:** Phase 3 Core Components (85% Complete)  
**Estimated Duration:** 1-2 weeks  
**Priority:** High - Integrate Phase 3 capabilities into existing API

---

## 1. Executive Summary

This document provides a comprehensive implementation plan for integrating Phase 3 risk assessment capabilities into the existing FastAPI structure. Rather than creating separate endpoints, this plan focuses on enhancing the existing `/api/v1/analytics/query` endpoint and adding minimal specialized endpoints only where absolutely necessary.

### Key Deliverables
- Enhanced existing `/api/v1/analytics/query` endpoint with risk assessment capabilities
- Integration of Phase 3 analyzers into the existing LangGraph workflow
- Enhanced existing Pydantic models to support risk assessment responses
- 3-5 specialized endpoints for risk-specific operations (monitoring, configuration)
- Seamless backward compatibility with existing API consumers

### Success Metrics
- **Response Time**: All enhanced endpoints respond within 3 seconds (95th percentile)
- **Backward Compatibility**: 100% compatibility with existing API consumers
- **Risk Integration**: Natural language queries automatically include risk context
- **Error Rate**: <1% error rate under normal load

---

## 2. Integration Architecture Overview

### 2.1 Enhanced Existing Structure
```
Enhanced Existing Endpoints:
├── /api/v1/analytics/query          # Enhanced with automatic risk assessment
├── /api/v1/analytics/classify       # Enhanced with risk context detection
├── /api/v1/analytics/health         # Enhanced with risk component health

Minimal New Endpoints (Only Where Necessary):
├── /api/v1/analytics/risk/monitor   # Real-time risk monitoring
├── /api/v1/analytics/risk/configure # Risk thresholds and alerts
└── /api/v1/analytics/risk/alerts    # Active alerts management
```

### 2.2 Integration Points in Existing Workflow
- **Query Router Enhancement**: Detect risk-related queries automatically
- **EHS Workflow Enhancement**: Add risk assessment steps to existing LangGraph workflow
- **Response Enhancement**: Include risk context in existing response models
- **Context Builder Enhancement**: Add risk factors to context building

### 2.3 Backward Compatibility
All enhancements maintain complete backward compatibility:
```json
{
  "success": true,
  "message": "Query processed successfully",
  "query_id": "uuid",
  "classification": { /* existing classification */ },
  "results": [ /* existing results */ ],
  "analysis": { /* existing analysis */ },
  "recommendations": [ /* existing recommendations */ ],
  "risk_assessment": { /* NEW: optional risk context */ },
  "processing_time_ms": 150
}
```

---

## 3. Enhanced Pydantic Models

### 3.1 Enhanced Existing Models

```python
# File: src/ehs_analytics/api/models.py (Enhanced)

from enum import Enum
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator

# NEW: Risk-related enums and models to add to existing models.py
class RiskSeverity(str, Enum):
    """Risk severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskCategory(str, Enum):
    """Risk assessment categories."""
    COMPLIANCE = "compliance"
    CONSUMPTION = "consumption"
    EQUIPMENT = "equipment"
    ENVIRONMENTAL = "environmental"
    FINANCIAL = "financial"

class RiskFactor(BaseModel):
    """Individual risk factor."""
    name: str = Field(..., description="Risk factor name")
    value: float = Field(..., ge=0.0, le=1.0, description="Risk factor value (0-1)")
    weight: float = Field(..., ge=0.0, le=1.0, description="Factor weight (0-1)")
    description: str = Field(..., description="Human-readable description")

class RiskAssessment(BaseModel):
    """Risk assessment context for queries."""
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall risk score")
    severity: RiskSeverity = Field(..., description="Risk severity level")
    categories: List[RiskCategory] = Field(..., description="Assessed risk categories")
    factors: List[RiskFactor] = Field(..., description="Contributing risk factors")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Assessment confidence")
    recommendations: List[str] = Field(default_factory=list, description="Risk mitigation recommendations")
    assessed_at: datetime = Field(default_factory=datetime.utcnow)

# ENHANCED: Add risk_assessment field to existing QueryResponse
class QueryResponse(BaseModel):  # This extends the existing model
    """Enhanced query response with optional risk assessment."""
    # ... existing fields remain unchanged ...
    query_id: str
    success: bool
    message: str
    classification: Optional[ClassificationResponse]
    results: List[QueryResult]
    analysis: Optional[str]
    recommendations: List[str]
    total_results: int
    processing_time_ms: int
    timestamp: datetime
    
    # NEW: Optional risk assessment context
    risk_assessment: Optional[RiskAssessment] = Field(
        None, 
        description="Risk assessment context when query involves risk analysis"
    )

# NEW: Risk monitoring models for minimal specialized endpoints
class RiskMonitoringConfig(BaseModel):
    """Configuration for risk monitoring."""
    facility_id: str = Field(..., description="Facility identifier")
    metrics: List[str] = Field(..., description="Metrics to monitor")
    thresholds: Dict[str, float] = Field(..., description="Risk thresholds")
    alert_channels: List[str] = Field(..., description="Alert notification channels")
    monitoring_frequency_minutes: int = Field(default=15, ge=5, le=1440)

class RiskAlert(BaseModel):
    """Active risk alert."""
    alert_id: str = Field(..., description="Unique alert identifier")
    facility_id: str = Field(..., description="Facility identifier")
    metric_name: str = Field(..., description="Affected metric")
    severity: RiskSeverity = Field(..., description="Alert severity")
    message: str = Field(..., description="Alert message")
    detected_at: datetime = Field(..., description="Detection timestamp")
    acknowledged: bool = Field(default=False, description="Acknowledgment status")
    recommended_actions: List[str] = Field(default_factory=list)
```

### 3.2 Enhanced Request Models

```python
# Enhanced existing QueryRequest with optional risk parameters
class QueryRequest(BaseModel):  # This extends the existing model
    """Enhanced query request with optional risk assessment parameters."""
    # ... existing fields remain unchanged ...
    query: str = Field(..., description="Natural language query")
    include_recommendations: bool = Field(default=True)
    context: Optional[Dict[str, Any]] = Field(None)
    preferences: Optional[Dict[str, Any]] = Field(None)
    
    # NEW: Optional risk assessment parameters
    include_risk_assessment: bool = Field(
        default=True, 
        description="Whether to include risk assessment in response"
    )
    risk_categories: Optional[List[RiskCategory]] = Field(
        None, 
        description="Specific risk categories to assess (auto-detected if None)"
    )
    risk_time_period_days: int = Field(
        default=30, 
        ge=1, 
        le=365, 
        description="Time period for risk analysis"
    )

# NEW: Request models for specialized risk endpoints
class RiskMonitoringRequest(BaseModel):
    """Request to configure risk monitoring."""
    config: RiskMonitoringConfig = Field(..., description="Monitoring configuration")

class RiskAlertRequest(BaseModel):
    """Request for risk alerts."""
    facility_id: Optional[str] = Field(None, description="Filter by facility")
    severity: Optional[RiskSeverity] = Field(None, description="Filter by severity")
    unacknowledged_only: bool = Field(default=True, description="Only unacknowledged alerts")
```

---

## 4. Integration Implementation Strategy

### 4.1 Enhanced Query Endpoint Integration

#### 4.1.1 Enhanced `/api/v1/analytics/query` Endpoint
```python
# File: src/ehs_analytics/api/routers/analytics.py (Enhanced)

@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def process_ehs_query(
    request: Request,
    query_request: QueryRequest,  # Enhanced with risk parameters
    user_id: str = Depends(get_current_user_id),
    workflow: EHSWorkflow = Depends(get_workflow),  # Enhanced with risk assessment
    session_manager: QuerySessionManager = Depends(get_session_manager)
) -> QueryResponse:
    """
    Enhanced EHS query processing with automatic risk assessment integration.
    
    New capabilities:
    - Automatic risk context detection in natural language queries
    - Risk assessment for facility-related queries
    - Risk-aware recommendations
    - Time-series risk analysis integration
    """
```

**Integration Points:**
- **File**: `/src/ehs_analytics/api/routers/analytics.py`
- **Enhancement**: Add risk assessment logic to existing query processing
- **Risk Detection**: Integrate with `RiskAwareQueryProcessor` 
- **Implementation Effort**: 2-3 days

#### 4.1.2 LangGraph Workflow Integration
```python
# File: src/ehs_analytics/workflows/ehs_workflow.py (Enhanced)

class EHSWorkflow:
    """Enhanced EHS workflow with integrated risk assessment."""
    
    def __init__(self):
        # Existing components
        self.query_router = QueryRouterAgent()
        self.context_builder = ContextBuilderAgent() 
        self.retriever = EHSRetrieverOrchestrator()
        self.response_generator = ResponseGeneratorAgent()
        
        # NEW: Risk assessment components
        self.risk_processor = RiskAwareQueryProcessor()
        self.water_analyzer = WaterConsumptionRiskAnalyzer()
        self.electricity_analyzer = ElectricityRiskAnalyzer() 
        self.waste_analyzer = WasteGenerationRiskAnalyzer()
        self.anomaly_detector = AnomalyDetectionSystem()
    
    async def process_query_with_risk_assessment(self, query_request: QueryRequest) -> QueryResponse:
        """Enhanced query processing with automatic risk assessment."""
        # Existing workflow steps + NEW risk assessment step
        pass
```

**Integration Points:**
- **File**: `/src/ehs_analytics/workflows/ehs_workflow.py`
- **Enhancement**: Add risk assessment nodes to existing LangGraph workflow
- **Components**: Integrate existing risk analyzers from `/src/ehs_analytics/risk_assessment/`
- **Implementation Effort**: 2-3 days

### 4.2 Minimal Specialized Endpoints

Only create specialized endpoints where the existing natural language interface is insufficient:

#### 4.2.1 Risk Monitoring Configuration
```python
# File: src/ehs_analytics/api/routers/analytics.py (New endpoints)

@router.post("/risk/monitor", response_model=BaseResponse)
async def configure_risk_monitoring(
    request: RiskMonitoringRequest,
    user_id: str = Depends(get_current_user_id)
) -> BaseResponse:
    """Configure real-time risk monitoring for facilities."""
```

- **Purpose**: Configure thresholds and alerts (not easily done via natural language)
- **Integration**: Uses existing `RiskMonitoringSystem` from `/src/ehs_analytics/risk_assessment/monitoring.py`
- **Implementation Effort**: 1 day

#### 4.2.2 Active Risk Alerts
```python
@router.get("/risk/alerts", response_model=List[RiskAlert])
async def get_active_risk_alerts(
    request: RiskAlertRequest,
    user_id: str = Depends(get_current_user_id)
) -> List[RiskAlert]:
    """Retrieve active risk alerts across facilities."""
```

- **Purpose**: Real-time alert management (operational necessity)
- **Integration**: Uses existing `RiskMonitoringSystem` and `AnomalyDetectionSystem`
- **Implementation Effort**: 1 day

#### 4.2.3 Risk Configuration Management
```python
@router.post("/risk/configure", response_model=BaseResponse)
async def configure_risk_settings(
    config: Dict[str, Any],
    user_id: str = Depends(get_current_user_id)
) -> BaseResponse:
    """Configure risk assessment parameters and thresholds."""
```

- **Purpose**: Administrative configuration (not suitable for natural language)
- **Integration**: Configure existing risk analyzers and monitoring systems
- **Implementation Effort**: 1 day

---

## 5. Integration Implementation Timeline

### Week 1: Model Enhancement and Core Integration (Days 1-5)
- **Day 1**: Enhance existing Pydantic models in `/src/ehs_analytics/api/models.py`
  - Add `RiskAssessment`, `RiskFactor`, `RiskSeverity` models
  - Enhance `QueryResponse` with optional `risk_assessment` field
  - Enhance `QueryRequest` with risk parameters
- **Day 2-3**: Integrate risk components into LangGraph workflow (`/src/ehs_analytics/workflows/ehs_workflow.py`)
  - Add risk assessment nodes to existing workflow
  - Integrate `RiskAwareQueryProcessor` into query routing
  - Connect Phase 3 analyzers to workflow graph
- **Day 4-5**: Enhance main query endpoint (`/src/ehs_analytics/api/routers/analytics.py`)
  - Modify existing `process_ehs_query` function
  - Add automatic risk detection and assessment logic
  - Maintain complete backward compatibility

### Week 2: Specialized Endpoints and Testing (Days 6-10)
- **Day 6-7**: Implement minimal specialized endpoints
  - Add `/api/v1/analytics/risk/monitor` endpoint
  - Add `/api/v1/analytics/risk/alerts` endpoint  
  - Add `/api/v1/analytics/risk/configure` endpoint
- **Day 8**: Enhance dependencies and services (`/src/ehs_analytics/api/dependencies.py`, `/src/ehs_analytics/api/services.py`)
  - Add risk component dependency injection
  - Enhance existing services with risk capabilities
- **Day 9**: Integration testing and validation
  - Test enhanced query endpoint with risk-related queries
  - Verify backward compatibility with existing API consumers
  - Test specialized risk endpoints
- **Day 10**: Documentation updates and deployment preparation
  - Update OpenAPI documentation
  - Create integration examples and usage guides

---

## 6. Enhanced Dependencies and Services

### 6.1 Enhanced Dependency Injection
```python
# File: src/ehs_analytics/api/dependencies.py (Enhanced)

# Existing dependencies remain unchanged
# NEW: Add risk assessment dependencies

from ..risk_assessment import (
    WaterConsumptionRiskAnalyzer, ElectricityRiskAnalyzer,
    WasteGenerationRiskAnalyzer, TimeSeriesAnalyzer,
    ForecastingEngine, AnomalyDetectionSystem,
    RiskAwareQueryProcessor, RiskMonitoringSystem
)

# NEW: Risk component dependencies
async def get_risk_processor(llm=Depends(get_llm)) -> RiskAwareQueryProcessor:
    """Get risk-aware query processor with LLM dependency."""
    return RiskAwareQueryProcessor(llm=llm)

async def get_water_risk_analyzer() -> WaterConsumptionRiskAnalyzer:
    """Get water risk analyzer instance."""
    return WaterConsumptionRiskAnalyzer()

async def get_electricity_risk_analyzer() -> ElectricityRiskAnalyzer:
    """Get electricity risk analyzer instance."""
    return ElectricityRiskAnalyzer()

async def get_waste_risk_analyzer() -> WasteGenerationRiskAnalyzer:
    """Get waste risk analyzer instance."""
    return WasteGenerationRiskAnalyzer()

async def get_anomaly_detection_system() -> AnomalyDetectionSystem:
    """Get anomaly detection system instance."""
    return AnomalyDetectionSystem()

async def get_risk_monitoring_system() -> RiskMonitoringSystem:
    """Get risk monitoring system instance."""
    return RiskMonitoringSystem()

# ENHANCED: Existing workflow dependency with risk integration
async def get_workflow(
    db_manager: DatabaseManager = Depends(get_db_manager),
    risk_processor: RiskAwareQueryProcessor = Depends(get_risk_processor)
) -> EHSWorkflow:
    """Get enhanced EHS workflow with risk assessment capabilities."""
    return EHSWorkflow(
        db_manager=db_manager,
        risk_processor=risk_processor  # NEW: Risk integration
    )
```

### 6.2 Enhanced Error Handling
```python
# File: src/ehs_analytics/api/services.py (Enhanced)

# ENHANCED: Existing ErrorHandlingService with risk assessment error support
class ErrorHandlingService:
    """Enhanced error handling service with risk assessment support."""
    
    def handle_workflow_error(self, error: Exception, query_id: str) -> HTTPException:
        """Enhanced workflow error handling including risk assessment errors."""
        # Existing error handling logic remains unchanged
        
        # NEW: Handle risk assessment specific errors
        if isinstance(error, RiskAssessmentError):
            return self._handle_risk_assessment_error(error, query_id)
        
        # Existing error handling for other types
        return self._handle_generic_error(error, query_id)
    
    def _handle_risk_assessment_error(self, error: RiskAssessmentError, query_id: str) -> HTTPException:
        """Handle risk assessment specific errors."""
        if "data not found" in str(error).lower():
            return HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error_type": ErrorType.DATA_ERROR,
                    "message": "Risk assessment data not available",
                    "query_id": query_id,
                    "details": {"error": str(error)}
                }
            )
        else:
            return HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error_type": ErrorType.PROCESSING_ERROR,
                    "message": "Risk assessment processing failed",
                    "query_id": query_id,
                    "details": {"error": str(error)}
                }
            )
```

---

## 7. Integration Examples and Workflows

### 7.1 Enhanced Natural Language Query Examples

#### Natural Language Queries with Automatic Risk Assessment
```python
# Example: Enhanced query processing with automatic risk assessment

# Request to enhanced endpoint
query_request = QueryRequest(
    query="What are the water consumption risks for Facility A over the past month?",
    include_risk_assessment=True,  # NEW: Enable risk assessment
    risk_time_period_days=30       # NEW: Risk analysis period
)

# Response includes both regular results AND risk assessment
response = await process_ehs_query(query_request)

print(f"Query Results: {len(response.results)} items")
print(f"Risk Assessment: {response.risk_assessment.severity}")
print(f"Risk Score: {response.risk_assessment.overall_score}")
print(f"Recommendations: {response.risk_assessment.recommendations}")
```

#### Example Queries That Trigger Risk Assessment
```python
# These natural language queries automatically include risk assessment:

queries_with_automatic_risk_assessment = [
    "Show me water consumption risks for all facilities",
    "What are the compliance violations at Facility B?", 
    "Analyze electricity usage trends and predict future risks",
    "Are there any anomalies in waste generation this month?",
    "What facilities have the highest environmental risks?",
    "Predict water permit compliance issues for next quarter"
]

# All of these would return QueryResponse with populated risk_assessment field
```

### 7.2 Specialized Risk Endpoint Usage

#### Risk Monitoring Configuration
```python
# Example: Configure risk monitoring for a facility
config_request = RiskMonitoringRequest(
    config=RiskMonitoringConfig(
        facility_id="FACILITY_001",
        metrics=["water_usage", "electricity_usage", "waste_generation"],
        thresholds={
            "water_usage_high": 15000,
            "water_usage_critical": 18000,
            "electricity_usage_high": 50000,
            "risk_score_critical": 0.9
        },
        alert_channels=["email", "webhook"],
        monitoring_frequency_minutes=15
    )
)

response = await configure_risk_monitoring(config_request)
```

#### Active Risk Alerts Management
```python
# Example: Get and manage active risk alerts
alert_request = RiskAlertRequest(
    facility_id="FACILITY_001",
    severity=RiskSeverity.HIGH,
    unacknowledged_only=True
)

active_alerts = await get_active_risk_alerts(alert_request)

for alert in active_alerts:
    print(f"Alert: {alert.message}")
    print(f"Severity: {alert.severity}")
    print(f"Actions: {alert.recommended_actions}")
```

---

## 8. Integration Testing Strategy

### 8.1 Enhanced Query Endpoint Tests
```python
# File: tests/test_enhanced_analytics.py

import pytest
from fastapi.testclient import TestClient

class TestEnhancedQueryEndpoint:
    """Test enhanced query endpoint with risk assessment integration."""
    
    def test_query_with_risk_assessment_enabled(self, test_client, auth_headers):
        """Test query with risk assessment enabled."""
        request_data = {
            "query": "What are the water consumption risks for Facility A?",
            "include_risk_assessment": True,
            "risk_time_period_days": 30
        }
        
        response = test_client.post(
            "/api/v1/analytics/query",
            json=request_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify existing response structure is maintained
        assert "query_id" in data
        assert "results" in data
        assert "analysis" in data
        assert "recommendations" in data
        
        # Verify new risk assessment is included
        assert "risk_assessment" in data
        assert data["risk_assessment"] is not None
        assert "overall_score" in data["risk_assessment"]
        assert "severity" in data["risk_assessment"]
        assert "factors" in data["risk_assessment"]
    
    def test_query_with_risk_assessment_disabled(self, test_client, auth_headers):
        """Test backward compatibility with risk assessment disabled."""
        request_data = {
            "query": "Show me utility bills for Facility A",
            "include_risk_assessment": False
        }
        
        response = test_client.post(
            "/api/v1/analytics/query",
            json=request_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify existing response structure
        assert "results" in data
        
        # Verify risk assessment is not included
        assert data.get("risk_assessment") is None
    
    def test_automatic_risk_detection(self, test_client, auth_headers):
        """Test automatic risk detection in natural language queries."""
        risk_related_queries = [
            "Are there any compliance violations at Facility B?",
            "What are the electricity consumption risks?",
            "Predict water usage anomalies for next month",
            "Show me facilities with high environmental risks"
        ]
        
        for query in risk_related_queries:
            request_data = {
                "query": query,
                "include_risk_assessment": True  # Default: True
            }
            
            response = test_client.post(
                "/api/v1/analytics/query",
                json=request_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Risk-related queries should include risk assessment
            assert data.get("risk_assessment") is not None

class TestSpecializedRiskEndpoints:
    """Test new specialized risk endpoints."""
    
    def test_risk_monitoring_configuration(self, test_client, auth_headers):
        """Test risk monitoring configuration endpoint."""
        config_data = {
            "config": {
                "facility_id": "TEST_FACILITY",
                "metrics": ["water_usage", "electricity_usage"],
                "thresholds": {
                    "water_high": 15000,
                    "electricity_critical": 60000
                },
                "alert_channels": ["email"],
                "monitoring_frequency_minutes": 15
            }
        }
        
        response = test_client.post(
            "/api/v1/analytics/risk/monitor",
            json=config_data,
            headers=auth_headers
        )
        
        assert response.status_code == 200
    
    def test_active_risk_alerts(self, test_client, auth_headers):
        """Test active risk alerts endpoint."""
        response = test_client.get(
            "/api/v1/analytics/risk/alerts?facility_id=TEST_FACILITY&unacknowledged_only=true",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
```

### 10.2 Integration Tests
```python
# File: tests/test_risk_integration.py

import pytest
import asyncio
from datetime import datetime, timedelta

class TestRiskAssessmentIntegration:
    """Integration tests for risk assessment workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_facility_assessment(self, test_client, auth_headers):
        """Test complete facility assessment workflow."""
        facility_id = "INTEGRATION_TEST_FACILITY"
        
        # 1. Assess water risk
        water_response = test_client.post(
            "/api/v1/risk-assessment/water/assess",
            json={
                "facility_id": facility_id,
                "assessment_types": ["compliance", "consumption"],
                "time_period_days": 30
            },
            headers=auth_headers
        )
        assert water_response.status_code == 200
        
        # 2. Assess electricity risk
        electricity_response = test_client.post(
            "/api/v1/risk-assessment/electricity/assess",
            json={
                "facility_id": facility_id,
                "assessment_types": ["consumption", "financial"],
                "time_period_days": 30
            },
            headers=auth_headers
        )
        assert electricity_response.status_code == 200
        
        # 3. Get facility dashboard
        dashboard_response = test_client.get(
            f"/api/v1/risk-assessment/composite/facility-dashboard/{facility_id}",
            headers=auth_headers
        )
        assert dashboard_response.status_code == 200
        
        # Verify dashboard includes data from assessments
        dashboard_data = dashboard_response.json()["data"]
        assert "water_risk" in dashboard_data
        assert "electricity_risk" in dashboard_data
        assert "overall_risk_score" in dashboard_data
```

### 10.3 Performance Tests
```python
# File: tests/test_risk_performance.py

import pytest
import time
from concurrent.futures import ThreadPoolExecutor

class TestRiskAssessmentPerformance:
    """Performance tests for risk assessment endpoints."""
    
    def test_risk_assessment_response_time(self, test_client, auth_headers):
        """Test risk assessment response time meets SLA."""
        request_data = {
            "facility_id": "PERF_TEST_FACILITY",
            "assessment_types": ["compliance"],
            "time_period_days": 30
        }
        
        start_time = time.time()
        response = test_client.post(
            "/api/v1/risk-assessment/water/assess",
            json=request_data,
            headers=auth_headers
        )
        end_time = time.time()
        
        response_time_ms = (end_time - start_time) * 1000
        
        assert response.status_code == 200
        assert response_time_ms < 3000  # 3 second SLA
    
    def test_concurrent_requests(self, test_client, auth_headers):
        """Test handling of concurrent requests."""
        def make_request():
            return test_client.post(
                "/api/v1/risk-assessment/water/assess",
                json={
                    "facility_id": "CONCURRENT_TEST_FACILITY",
                    "assessment_types": ["compliance"],
                    "time_period_days": 30
                },
                headers=auth_headers
            )
        
        # Test 10 concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in futures]
        
        # All requests should succeed
        assert all(response.status_code == 200 for response in responses)
```

---

## 11. Documentation Requirements

### 11.1 OpenAPI Schema Enhancements
```python
# File: src/ehs_analytics/api/routers/risk_assessment.py

from fastapi import APIRouter, Depends
from fastapi.openapi.models import Example

router = APIRouter(
    prefix="/api/v1/risk-assessment",
    tags=["Risk Assessment"],
    responses={
        404: {"description": "Facility or resource not found"},
        422: {"description": "Validation error"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    }
)

@router.post(
    "/water/assess",
    response_model=RiskAssessmentResult,
    summary="Assess Water Consumption Risk",
    description="""
    Perform comprehensive water consumption risk assessment for a facility.
    
    This endpoint analyzes:
    - Permit compliance status
    - Consumption trends and patterns
    - Seasonal variations
    - Equipment efficiency impacts
    - Predictive risk indicators
    
    **Use Cases:**
    - Monthly compliance reporting
    - Early warning system for permit violations
    - Operational efficiency optimization
    - Budget planning and forecasting
    """,
    responses={
        200: {
            "description": "Risk assessment completed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "message": "Water risk assessment completed",
                        "data": {
                            "facility_id": "FACILITY_001",
                            "assessment_type": "consumption",
                            "overall_score": 0.75,
                            "severity": "high",
                            "factors": [
                                {
                                    "name": "permit_compliance",
                                    "value": 0.85,
                                    "weight": 0.4,
                                    "description": "15% over permit limit"
                                }
                            ],
                            "recommendations": [
                                "Review water recycling opportunities",
                                "Schedule equipment efficiency audit"
                            ]
                        }
                    }
                }
            }
        }
    }
)
async def assess_water_risk(...):
    pass
```

### 11.2 API Documentation Examples
```markdown
## Risk Assessment API Examples

### Water Risk Assessment
```bash
curl -X POST "http://localhost:8000/api/v1/risk-assessment/water/assess" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "facility_id": "FACILITY_001",
    "assessment_types": ["compliance", "consumption"],
    "time_period_days": 30,
    "include_recommendations": true
  }'
```

### Anomaly Detection
```bash
curl -X POST "http://localhost:8000/api/v1/risk-assessment/anomalies/detect" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "facility_ids": ["FACILITY_001", "FACILITY_002"],
    "metrics": ["water_usage", "electricity_usage"],
    "detection_methods": ["statistical", "ml"],
    "sensitivity": 0.95,
    "time_window_hours": 24
  }'
```

### Facility Dashboard
```bash
curl -X GET "http://localhost:8000/api/v1/risk-assessment/composite/facility-dashboard/FACILITY_001?time_period=30" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```
```

---

## 12. Deployment Configuration

### 12.1 Environment Variables
```bash
# File: .env.risk-assessment

# Risk Assessment Configuration
RISK_ASSESSMENT_CACHE_TTL_MINUTES=15
RISK_ASSESSMENT_MAX_CONCURRENT_REQUESTS=100
RISK_ASSESSMENT_RATE_LIMIT_REQUESTS_PER_MINUTE=100

# Monitoring Configuration
RISK_MONITORING_ENABLED=true
RISK_MONITORING_FREQUENCY_MINUTES=15
RISK_ALERTING_ENABLED=true

# Performance Configuration
RISK_BACKGROUND_PROCESSING_ENABLED=true
RISK_CACHE_SIZE_MB=512
RISK_DATABASE_POOL_SIZE=20

# Security Configuration
RISK_REQUIRE_AUTHENTICATION=true
RISK_AUDIT_LOGGING_ENABLED=true
RISK_SENSITIVE_DATA_MASKING=true
```

### 12.2 Docker Configuration
```dockerfile
# File: docker/risk-assessment.Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements/risk-assessment.txt .
RUN pip install --no-cache-dir -r risk-assessment.txt

# Copy application code
COPY src/ src/
COPY docs/ docs/

# Set environment variables
ENV PYTHONPATH=/app/src
ENV RISK_ASSESSMENT_MODULE=enabled

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/risk-assessment/health || exit 1

# Run application
CMD ["uvicorn", "ehs_analytics.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 12.3 Kubernetes Deployment
```yaml
# File: k8s/risk-assessment-deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: ehs-risk-assessment-api
  labels:
    app: ehs-risk-assessment
    component: api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ehs-risk-assessment
      component: api
  template:
    metadata:
      labels:
        app: ehs-risk-assessment
        component: api
    spec:
      containers:
      - name: risk-assessment-api
        image: ehs-analytics/risk-assessment:latest
        ports:
        - containerPort: 8000
        env:
        - name: RISK_ASSESSMENT_CACHE_TTL_MINUTES
          value: "15"
        - name: RISK_MONITORING_ENABLED
          value: "true"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /api/v1/risk-assessment/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/risk-assessment/health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

---

## 13. Success Metrics and KPIs

### 13.1 Performance Metrics
- **Response Time**: 95th percentile < 3 seconds for all endpoints
- **Throughput**: Support 100+ concurrent requests per endpoint
- **Availability**: 99.9% uptime for critical endpoints
- **Error Rate**: <1% error rate under normal load

### 13.2 Business Metrics
- **Risk Assessment Accuracy**: >85% correlation with actual incidents
- **False Positive Rate**: <10% for anomaly detection
- **User Adoption**: >90% of facilities using risk assessment endpoints
- **Cost Savings**: Measurable reduction in utility costs from recommendations

### 13.3 Technical Metrics
- **Test Coverage**: >90% code coverage for all endpoints
- **API Documentation**: 100% endpoint documentation coverage
- **Security Compliance**: 0 critical security vulnerabilities
- **Performance Regression**: 0 endpoints failing SLA requirements

---

## 14. Post-Implementation Considerations

### 14.1 Monitoring and Alerting
- **Endpoint Performance Monitoring**: Track response times and error rates
- **Business Logic Monitoring**: Monitor risk calculation accuracy
- **Security Monitoring**: Track authentication failures and suspicious activity
- **Resource Usage Monitoring**: Monitor CPU, memory, and database usage

### 8.2 Workflow Integration Tests
```python
# File: tests/test_workflow_integration.py

import pytest

class TestWorkflowIntegration:
    """Test integration of risk assessment into existing LangGraph workflow."""
    
    @pytest.mark.asyncio
    async def test_workflow_risk_node_integration(self):
        """Test that risk assessment nodes are properly integrated into workflow."""
        # Test that workflow includes risk assessment steps
        workflow = EHSWorkflow()
        
        # Verify risk components are available
        assert hasattr(workflow, 'risk_processor')
        assert hasattr(workflow, 'water_analyzer')
        assert hasattr(workflow, 'electricity_analyzer')
        assert hasattr(workflow, 'waste_analyzer')
        
    @pytest.mark.asyncio
    async def test_end_to_end_query_with_risk(self):
        """Test complete query processing with risk assessment."""
        query_request = QueryRequest(
            query="What are the compliance risks for Facility A?",
            include_risk_assessment=True
        )
        
        workflow = EHSWorkflow()
        response = await workflow.process_query_with_risk_assessment(query_request)
        
        # Verify complete response structure
        assert response.success is True
        assert response.risk_assessment is not None
        assert len(response.results) > 0
        assert len(response.recommendations) > 0
```

---

## 9. Success Metrics and KPIs

### 9.1 Integration Success Metrics
- **Backward Compatibility**: 100% compatibility with existing API consumers
- **Risk Detection Accuracy**: >90% accurate detection of risk-related queries
- **Response Time**: Enhanced endpoints respond within 3 seconds (95th percentile)  
- **Adoption Rate**: >80% of risk-related queries automatically include risk context

### 9.2 Technical Integration Metrics
- **Code Reuse**: >95% reuse of existing Phase 3 risk assessment components
- **Test Coverage**: >90% code coverage for enhanced endpoints
- **Error Rate**: <1% error rate for enhanced functionality
- **Implementation Speed**: 50% faster than separate endpoint approach

---

## 10. Conclusion and Next Steps

This integration-focused implementation plan provides a streamlined approach to adding Phase 3 risk assessment capabilities to the existing EHS Analytics API. By enhancing the existing `/api/v1/analytics/query` endpoint and adding only minimal specialized endpoints, we achieve:

### Key Benefits
1. **Seamless Integration**: Risk assessment becomes a natural part of the existing workflow
2. **Backward Compatibility**: No breaking changes for existing API consumers  
3. **Reduced Complexity**: 3-5 specialized endpoints vs 25+ separate endpoints
4. **Enhanced User Experience**: Natural language queries automatically include risk context
5. **Faster Implementation**: 1-2 weeks vs 3-4 weeks for separate endpoint approach

### Critical Implementation Points
1. **File Locations**: All enhancements target specific existing files
   - `/src/ehs_analytics/api/models.py` - Enhanced models
   - `/src/ehs_analytics/api/routers/analytics.py` - Enhanced endpoints  
   - `/src/ehs_analytics/workflows/ehs_workflow.py` - Risk integration
   - `/src/ehs_analytics/api/dependencies.py` - Risk dependencies

2. **Phase 3 Component Integration**: Direct use of existing risk assessment components
   - `WaterConsumptionRiskAnalyzer` from `/src/ehs_analytics/risk_assessment/water_risk.py`
   - `ElectricityRiskAnalyzer` from `/src/ehs_analytics/risk_assessment/electricity_risk.py`
   - `WasteGenerationRiskAnalyzer` from `/src/ehs_analytics/risk_assessment/waste_risk.py`
   - `RiskAwareQueryProcessor` from `/src/ehs_analytics/risk_assessment/risk_query_processor.py`
   - `AnomalyDetectionSystem` from `/src/ehs_analytics/risk_assessment/anomaly_detection.py`

3. **LangGraph Workflow Enhancement**: Add risk assessment nodes to existing workflow graph

### Implementation Summary
- **Enhanced Endpoints**: 1 main endpoint + 3 specialized = 4 total (vs 25+ in original plan)
- **Timeline**: 1-2 weeks (vs 3-4 weeks)
- **Risk**: Lower (building on existing vs creating new)
- **Maintainability**: Higher (fewer endpoints, integrated workflow)
- **User Experience**: Superior (natural language + automatic risk context)

---

**Last Updated**: 2025-08-21  
**Status**: Ready for Implementation - Integration Approach  
**Next Review**: Weekly during implementation phase  
**Estimated Completion**: 1-2 weeks from start date

This comprehensive integration plan provides a complete roadmap for seamlessly integrating Phase 3 risk assessment capabilities into the existing FastAPI structure. The plan emphasizes enhancing existing endpoints rather than creating new ones, ensuring backward compatibility while delivering powerful risk assessment capabilities through the natural language query interface.