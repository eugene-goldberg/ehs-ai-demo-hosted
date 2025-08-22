"""
Pytest configuration and fixtures for EHS Analytics tests.

This module provides comprehensive fixtures for testing all core components
of the EHS Analytics system, including mocked dependencies, sample data,
and test configurations.
"""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Generator, AsyncGenerator, Dict, Any, List
import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ============================================================================
# Event Loop Configuration
# ============================================================================

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Database and Connection Mocks
# ============================================================================

@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver for testing."""
    driver = Mock()
    
    # Mock session and results
    session = Mock()
    result = Mock()
    record = Mock()
    
    # Configure record mock
    record.get.return_value = 1
    record.__getitem__ = Mock(return_value=1)
    result.single.return_value = record
    result.consume.return_value = Mock()
    
    # Configure session mock  
    session.run.return_value = result
    session.__enter__.return_value = session
    session.__exit__.return_value = None
    session.close = Mock()
    
    # Configure driver mock
    driver.session.return_value = session
    driver.close = Mock()
    driver.verify_connectivity = AsyncMock()
    
    return driver


@pytest.fixture
def mock_neo4j_graph():
    """Mock LangChain Neo4j graph for testing."""
    graph = Mock()
    
    # Mock schema information
    graph.schema = {
        "node_props": {
            "Facility": {
                "name": "STRING",
                "location": "STRING", 
                "facility_type": "STRING"
            },
            "Equipment": {
                "name": "STRING",
                "type": "STRING",
                "model": "STRING",
                "installation_date": "DATE"
            },
            "UtilityBill": {
                "amount": "FLOAT",
                "billing_period_start": "DATE",
                "billing_period_end": "DATE",
                "utility_type": "STRING"
            },
            "Permit": {
                "permit_number": "STRING",
                "type": "STRING",
                "expiry_date": "DATE",
                "status": "STRING"
            }
        },
        "rel_props": {
            "CONTAINS": {},
            "RECORDED_AT": {},
            "REQUIRES": {}
        },
        "relationships": [
            {"start": "Facility", "type": "CONTAINS", "end": "Equipment"},
            {"start": "UtilityBill", "type": "RECORDED_AT", "end": "Facility"},
            {"start": "Equipment", "type": "REQUIRES", "end": "Permit"}
        ]
    }
    
    graph.refresh_schema = Mock()
    return graph


@pytest.fixture
def mock_database_manager():
    """Mock DatabaseManager for testing."""
    from ehs_analytics.api.dependencies import DatabaseManager
    
    db_manager = Mock(spec=DatabaseManager)
    db_manager.is_connected = True
    db_manager.health_check = AsyncMock(return_value=True)
    db_manager.get_connection = Mock()
    db_manager.close_connection = AsyncMock()
    
    return db_manager


# ============================================================================
# LLM and AI Component Mocks
# ============================================================================

@pytest.fixture
def mock_openai_llm():
    """Mock OpenAI LLM for testing."""
    llm = Mock()
    
    # Mock response structure
    mock_response = Mock()
    mock_response.content = '{"intent": "consumption_analysis", "confidence": 0.85, "reasoning": "Query focuses on consumption patterns"}'
    
    llm.invoke.return_value = mock_response
    llm.agenerate = AsyncMock(return_value="Mock LLM response")
    llm.model_name = "gpt-3.5-turbo"
    llm.temperature = 0.1
    llm.max_tokens = 1000
    
    return llm


@pytest.fixture
def mock_cypher_chain():
    """Mock GraphCypherQAChain for testing."""
    chain = Mock()
    
    # Mock successful Cypher execution result
    chain.invoke = AsyncMock(return_value={
        "result": [
            {"facility_name": "Plant A", "total_consumption": 1500.0},
            {"facility_name": "Plant B", "total_consumption": 1200.0}
        ],
        "intermediate_steps": [{
            "query": "MATCH (f:Facility)-[:RECORDED_AT]-(u:UtilityBill) WHERE u.utility_type = 'electricity' RETURN f.name as facility_name, SUM(u.amount) as total_consumption"
        }]
    })
    
    return chain


# ============================================================================
# Query Router and Classification Fixtures
# ============================================================================

@pytest.fixture
def sample_entity_extraction():
    """Sample EntityExtraction for testing."""
    try:
        from ehs_analytics.agents.query_router import EntityExtraction
        return EntityExtraction(
            facilities=["Plant A", "Manufacturing Site B"],
            date_ranges=["last quarter", "January 2024"],
            equipment=["Boiler B-1", "HVAC Unit 2"],
            pollutants=["CO2", "NOx"],
            regulations=["EPA", "OSHA"],
            departments=["EHS Department"],
            metrics=["electricity", "efficiency"]
        )
    except ImportError:
        # Fallback if module not available
        return Mock()


@pytest.fixture  
def sample_query_classification(sample_entity_extraction):
    """Sample QueryClassification for testing."""
    try:
        from ehs_analytics.agents.query_router import QueryClassification, IntentType, RetrieverType
        return QueryClassification(
            intent_type=IntentType.CONSUMPTION_ANALYSIS,
            confidence_score=0.85,
            entities_identified=sample_entity_extraction,
            suggested_retriever=RetrieverType.CONSUMPTION_RETRIEVER,
            reasoning="Query focuses on electricity consumption patterns over time",
            query_rewrite="Analyze consumption patterns and usage data for facilities: Plant A during: last quarter"
        )
    except ImportError:
        # Fallback if module not available
        return Mock()


@pytest.fixture
def mock_query_router(sample_query_classification):
    """Mock QueryRouterAgent for testing."""
    try:
        from ehs_analytics.agents.query_router import QueryRouterAgent
        router = Mock(spec=QueryRouterAgent)
    except ImportError:
        router = Mock()
    
    router.classify_query.return_value = sample_query_classification
    
    # Mock intent examples
    router.get_intent_examples.return_value = {
        "consumption_analysis": [
            "What is the electricity usage for Plant A?",
            "Show me water consumption trends",
        ],
        "compliance_check": [
            "Are we compliant with EPA standards?",
            "Check OSHA violations"
        ]
    }
    
    router.get_classification_stats.return_value = {
        "total_classifications": 100,
        "intent_distribution": {"consumption_analysis": 45, "compliance_check": 30},
        "average_confidence": 0.82,
        "average_processing_time_ms": 150.0
    }
    
    return router


# ============================================================================
# Retrieval and Workflow Components
# ============================================================================

@pytest.fixture
def mock_retrieval_result():
    """Mock RetrievalResult for testing."""
    try:
        from ehs_analytics.retrieval.base import RetrievalResult, RetrievalMetadata, RetrievalStrategy, QueryType
        
        metadata = RetrievalMetadata(
            strategy=RetrievalStrategy.TEXT2CYPHER,
            query_type=QueryType.CONSUMPTION,
            confidence_score=0.88,
            execution_time_ms=250.0,
            cypher_query="MATCH (f:Facility) RETURN f",
            nodes_retrieved=5,
            relationships_retrieved=3
        )
        
        return RetrievalResult(
            data=[
                {"facility": "Plant A", "consumption": 1500.0},
                {"facility": "Plant B", "consumption": 1200.0}
            ],
            metadata=metadata,
            success=True,
            message="Retrieved 2 facility consumption records"
        )
    except ImportError:
        # Fallback if module not available
        mock_result = Mock()
        mock_result.success = True
        mock_result.data = [{"facility": "Plant A", "consumption": 1500.0}]
        return mock_result


@pytest.fixture
async def mock_text2cypher_retriever(mock_retrieval_result):
    """Mock Text2CypherRetriever for testing."""
    retriever = Mock()
    retriever._initialized = True
    
    retriever.initialize = AsyncMock()
    retriever.retrieve = AsyncMock(return_value=mock_retrieval_result)
    retriever.validate_query = AsyncMock(return_value=True)
    retriever.cleanup = AsyncMock()
    
    try:
        from ehs_analytics.retrieval.base import RetrievalStrategy
        retriever.get_strategy.return_value = RetrievalStrategy.TEXT2CYPHER
    except ImportError:
        retriever.get_strategy.return_value = "text2cypher"
    
    return retriever


@pytest.fixture
def mock_workflow_state():
    """Mock EHSWorkflowState for testing."""
    try:
        from ehs_analytics.workflows.ehs_workflow import EHSWorkflowState
        return EHSWorkflowState(
            query_id="test-workflow-123",
            original_query="What is the electricity consumption trend?",
            user_id="user-456"
        )
    except ImportError:
        # Fallback if module not available
        state = Mock()
        state.query_id = "test-workflow-123"
        state.original_query = "What is the electricity consumption trend?"
        state.user_id = "user-456"
        state.classification = None
        state.retrieval_results = None
        state.analysis_results = None
        state.recommendations = None
        state.error = None
        state.metadata = {}
        state.workflow_trace = []
        state.step_durations = {}
        state.total_duration_ms = None
        state.created_at = datetime.utcnow()
        state.updated_at = datetime.utcnow()
        return state


@pytest.fixture
def mock_workflow_manager(mock_query_router):
    """Mock WorkflowManager for testing."""
    try:
        from ehs_analytics.api.dependencies import WorkflowManager
        manager = Mock(spec=WorkflowManager)
    except ImportError:
        manager = Mock()
    
    manager.is_initialized = True
    manager.health_check = AsyncMock(return_value=True)
    manager.get_query_router.return_value = mock_query_router
    
    return manager


# ============================================================================
# API and Web Component Fixtures
# ============================================================================

@pytest.fixture
def sample_query_request():
    """Sample QueryRequest for API testing."""
    try:
        from ehs_analytics.api.models import QueryRequest
        return QueryRequest(
            query="What is the electricity consumption trend for Plant A over the last quarter?",
            context={"facility": "Plant A", "time_period": "last_quarter"},
            preferences={"include_charts": True, "detail_level": "high"}
        )
    except ImportError:
        # Fallback if module not available
        return {
            "query": "What is the electricity consumption trend for Plant A over the last quarter?",
            "context": {"facility": "Plant A"},
            "preferences": {"include_charts": True}
        }


@pytest.fixture
def sample_processing_options():
    """Sample QueryProcessingOptions for API testing."""
    try:
        from ehs_analytics.api.models import QueryProcessingOptions
        return QueryProcessingOptions(
            timeout_seconds=300,
            include_recommendations=True,
            max_results=20
        )
    except ImportError:
        # Fallback if module not available
        return {
            "timeout_seconds": 300,
            "include_recommendations": True,
            "max_results": 20
        }


@pytest.fixture
def mock_session_manager():
    """Mock QuerySessionManager for testing."""
    try:
        from ehs_analytics.api.dependencies import QuerySessionManager
        manager = Mock(spec=QuerySessionManager)
    except ImportError:
        manager = Mock()
    
    manager.create_session = AsyncMock(return_value="session-" + str(uuid.uuid4()))
    manager.get_session = AsyncMock(return_value={"session_id": "test-session"})
    manager.close_session = AsyncMock()
    
    return manager


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_ehs_document():
    """Sample EHS document for testing."""
    return {
        "id": "doc-001",
        "content": "Sample EHS document content about facility operations and environmental compliance",
        "document_type": "environmental_report",
        "metadata": {
            "file_path": "/documents/ehs/facility_report_2024.pdf",
            "uploaded_at": "2024-01-15T10:30:00Z",
            "facility": "Plant A",
            "document_date": "2024-01-01",
            "categories": ["environmental", "compliance"]
        }
    }


@pytest.fixture
def sample_utility_bill():
    """Sample utility bill data for testing."""
    return {
        "id": "bill-001",
        "billing_period_start": "2024-01-01",
        "billing_period_end": "2024-01-31", 
        "total_kwh": 1500.0,
        "total_cost": 250.75,
        "peak_demand_kw": 45.2,
        "facility_name": "Plant A",
        "utility_type": "electricity",
        "rate_schedule": "Commercial",
        "meter_readings": [
            {"date": "2024-01-01", "reading": 12500},
            {"date": "2024-01-31", "reading": 14000}
        ]
    }


@pytest.fixture
def sample_equipment_data():
    """Sample equipment data for testing."""
    return {
        "id": "equip-001",
        "name": "Boiler Unit B-1",
        "type": "Steam Boiler",
        "model": "Cleaver-Brooks CB-150",
        "manufacturer": "Cleaver-Brooks",
        "installation_date": "2020-03-15",
        "facility": "Plant A",
        "specifications": {
            "capacity": "150 HP",
            "fuel_type": "Natural Gas",
            "efficiency_rating": 82.5
        },
        "maintenance_schedule": "Monthly",
        "last_inspection": "2024-01-10",
        "status": "Operational"
    }


@pytest.fixture
def sample_permit_data():
    """Sample permit data for testing."""
    return {
        "id": "permit-001",
        "permit_number": "EPA-001-2024",
        "type": "Air Quality Permit",
        "issuing_authority": "EPA",
        "facility": "Plant A",
        "issue_date": "2024-01-01",
        "expiry_date": "2026-12-31",
        "status": "Active",
        "conditions": [
            "Maximum emission rate: 50 tons/year NOx",
            "Quarterly monitoring reports required",
            "Annual stack testing required"
        ],
        "compliance_status": "Compliant"
    }


@pytest.fixture
def sample_waste_manifest():
    """Sample waste manifest data for testing."""
    return {
        "id": "manifest-001",
        "manifest_tracking_number": "WM-2024-001",
        "generator_name": "Plant A Manufacturing",
        "issue_date": "2024-01-15",
        "total_quantity": 500.0,
        "weight_unit": "pounds",
        "waste_streams": [
            {
                "waste_code": "D001",
                "description": "Ignitable Waste",
                "quantity": 200.0,
                "container_type": "Drum"
            },
            {
                "waste_code": "F003",
                "description": "Spent Solvents",
                "quantity": 300.0,
                "container_type": "Tank"
            }
        ],
        "transporter": "SafeDisposal Inc",
        "disposal_facility": "ToxicWaste Treatment Center",
        "disposal_method": "Incineration",
        "status": "Disposed"
    }


# ============================================================================
# Configuration and Settings
# ============================================================================

@pytest.fixture
def test_config():
    """Comprehensive test configuration dictionary."""
    return {
        "neo4j": {
            "uri": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "test_password",
            "database": "test_ehs"
        },
        "openai": {
            "api_key": "test_api_key",
            "model": "gpt-3.5-turbo",
            "temperature": 0.1,
            "max_tokens": 2000
        },
        "retrieval": {
            "strategies": {
                "text2cypher": {"enabled": True, "timeout": 30},
                "vector": {"enabled": True, "top_k": 5, "threshold": 0.7},
                "hybrid": {"enabled": True, "top_k": 10, "alpha": 0.5}
            }
        },
        "workflow": {
            "default_timeout": 300,
            "max_retries": 3,
            "enable_monitoring": True
        },
        "api": {
            "rate_limit": {"requests_per_minute": 60},
            "pagination": {"default_limit": 20, "max_limit": 100},
            "authentication": {"enabled": True}
        }
    }


@pytest.fixture
def retrieval_config():
    """Configuration specifically for retrieval components."""
    return {
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_user": "neo4j", 
        "neo4j_password": "test_password",
        "openai_api_key": "test_api_key",
        "model_name": "gpt-3.5-turbo",
        "temperature": 0.0,
        "max_tokens": 2000,
        "cypher_validation": True
    }


# ============================================================================
# Monitoring and Utilities
# ============================================================================

@pytest.fixture
def mock_monitor():
    """Mock monitoring system for testing."""
    monitor = Mock()
    monitor.record_query = Mock()
    monitor.record_retrieval = Mock()
    monitor.record_analysis = Mock()
    monitor.get_metrics = Mock(return_value={
        "total_queries": 100,
        "success_rate": 0.95,
        "average_response_time": 150.0
    })
    return monitor


@pytest.fixture
def mock_profiler():
    """Mock profiler for testing."""
    profiler = Mock()
    profiler.profile_operation = Mock()
    profiler.profile_operation.return_value.__enter__ = Mock()
    profiler.profile_operation.return_value.__exit__ = Mock()
    return profiler


@pytest.fixture
def mock_logger():
    """Mock structured logger for testing."""
    logger = Mock()
    logger.info = Mock()
    logger.debug = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.query_start = Mock()
    logger.query_end = Mock()
    logger.retrieval_operation = Mock()
    logger.recommendation_generated = Mock()
    return logger


# ============================================================================
# Test Data Generators
# ============================================================================

@pytest.fixture
def query_samples():
    """Various sample queries categorized by intent type."""
    return {
        "consumption_analysis": [
            "What is the electricity consumption trend for Plant A over the last quarter?",
            "Show me water usage patterns for all facilities in 2024",
            "Compare energy consumption between Plant A and Plant B",
            "Analyze peak demand patterns during summer months"
        ],
        "compliance_check": [
            "Are we compliant with EPA air quality standards?",
            "Check OSHA compliance status for safety violations",
            "Show me any regulatory non-compliance issues this month",
            "Which permits are expiring in the next 60 days?"
        ],
        "risk_assessment": [
            "What are the environmental risks at our chemical facility?",
            "Assess safety risks for equipment maintenance operations", 
            "Evaluate operational risks from our emission sources",
            "Identify high-risk areas requiring immediate attention"
        ],
        "emission_tracking": [
            "Track our carbon footprint for Scope 1 emissions",
            "Show CO2 emissions from our manufacturing processes",
            "What are our greenhouse gas emission trends over the past year?",
            "Compare emission levels before and after equipment upgrades"
        ],
        "equipment_efficiency": [
            "How efficient is our HVAC system performing?",
            "Show equipment utilization rates for our boilers",
            "Analyze maintenance schedules and downtime patterns",
            "Compare efficiency metrics across similar equipment"
        ],
        "permit_status": [
            "When do our environmental permits expire?",
            "Check the status of our air quality permits",
            "Are there any permits requiring renewal this quarter?",
            "Show compliance history for Permit EPA-001-2024"
        ],
        "general_inquiry": [
            "What EHS data do we have available?",
            "Explain our environmental management system",
            "What are the key EHS metrics we track?",
            "Provide an overview of our sustainability initiatives"
        ]
    }


@pytest.fixture
def performance_benchmarks():
    """Performance benchmarks for testing."""
    return {
        "classification": {
            "max_duration_ms": 500,
            "target_confidence": 0.8
        },
        "retrieval": {
            "max_duration_ms": 2000,
            "min_results": 1,
            "target_confidence": 0.7
        },
        "workflow": {
            "max_total_duration_ms": 10000,
            "max_step_duration_ms": 3000
        },
        "api": {
            "max_response_time_ms": 5000,
            "rate_limit_requests_per_minute": 60
        }
    }


# ============================================================================
# Cleanup and Teardown
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_stores():
    """Automatically cleanup in-memory stores before and after each test."""
    # Clear stores before test
    try:
        from ehs_analytics.api.routers.analytics import query_results_store, query_status_store
        query_results_store.clear()
        query_status_store.clear()
    except ImportError:
        pass
    
    yield
    
    # Clear stores after test
    try:
        from ehs_analytics.api.routers.analytics import query_results_store, query_status_store
        query_results_store.clear()
        query_status_store.clear()
    except ImportError:
        pass


# ============================================================================
# Integration Test Fixtures
# ============================================================================

@pytest.fixture
def integration_test_data():
    """Comprehensive test data for integration testing."""
    return {
        "facilities": [
            {"id": "fac-001", "name": "Plant A", "location": "City A", "type": "Manufacturing"},
            {"id": "fac-002", "name": "Plant B", "location": "City B", "type": "Processing"}
        ],
        "equipment": [
            {"id": "eq-001", "name": "Boiler B-1", "type": "Steam Boiler", "facility_id": "fac-001"},
            {"id": "eq-002", "name": "HVAC Unit 2", "type": "HVAC", "facility_id": "fac-001"}
        ],
        "utility_bills": [
            {"id": "bill-001", "facility_id": "fac-001", "amount": 1500.0, "type": "electricity"},
            {"id": "bill-002", "facility_id": "fac-002", "amount": 1200.0, "type": "electricity"}
        ],
        "permits": [
            {"id": "perm-001", "facility_id": "fac-001", "type": "Air Quality", "status": "Active"},
            {"id": "perm-002", "facility_id": "fac-002", "type": "Water Discharge", "status": "Active"}
        ]
    }