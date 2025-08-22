"""
Unit tests for Analytics API.

These tests cover all API endpoints, request/response validation,
authentication, authorization, and error responses.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import json
import uuid
from typing import Dict, Any, List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from fastapi.testclient import TestClient
from fastapi import HTTPException, status
from httpx import AsyncClient

from ehs_analytics.api.routers.analytics import (
    router, process_ehs_query, get_query_result, get_query_status,
    cancel_query, analytics_health_check, list_user_queries,
    process_query_workflow, query_results_store, query_status_store
)
from ehs_analytics.api.models import (
    QueryRequest, QueryResponse, QueryResultResponse, QueryStatusResponse,
    QueryProcessingOptions, HealthCheckResponse, ServiceHealth,
    QueryStatus, ErrorType, BaseResponse
)
from ehs_analytics.api.dependencies import (
    DatabaseManager, WorkflowManager, QuerySessionManager
)
from ehs_analytics.agents.query_router import QueryRouterAgent


# Mock FastAPI app for testing
from fastapi import FastAPI
app = FastAPI()
app.include_router(router)

client = TestClient(app)


class TestAnalyticsAPIEndpoints:
    """Test suite for Analytics API endpoints."""

    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        """Setup and cleanup for each test."""
        # Clear stores before each test
        query_results_store.clear()
        query_status_store.clear()
        yield
        # Clear stores after each test
        query_results_store.clear()
        query_status_store.clear()

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all API dependencies."""
        mocks = {
            'get_current_user_id': Mock(return_value="test-user-123"),
            'get_db_manager': Mock(),
            'get_workflow_manager': Mock(),
            'get_session_manager': Mock(),
            'validate_request_rate_limit': Mock(),
            'validate_query_request': AsyncMock()
        }
        
        # Configure mock managers
        db_manager = Mock(spec=DatabaseManager)
        db_manager.is_connected = True
        db_manager.health_check = AsyncMock(return_value=True)
        mocks['get_db_manager'].return_value = db_manager
        
        workflow_manager = Mock(spec=WorkflowManager)
        workflow_manager.is_initialized = True
        workflow_manager.health_check = AsyncMock(return_value=True)
        
        # Mock query router
        query_router = Mock(spec=QueryRouterAgent)
        from ehs_analytics.agents.query_router import QueryClassification, EntityExtraction, IntentType, RetrieverType
        mock_classification = QueryClassification(
            intent_type=IntentType.CONSUMPTION_ANALYSIS,
            confidence_score=0.85,
            entities_identified=EntityExtraction([], [], [], [], [], [], []),
            suggested_retriever=RetrieverType.CONSUMPTION_RETRIEVER,
            reasoning="Test classification"
        )
        query_router.classify_query.return_value = mock_classification
        workflow_manager.get_query_router.return_value = query_router
        workflow_manager.health_check = AsyncMock(return_value=True)
        mocks['get_workflow_manager'].return_value = workflow_manager
        
        session_manager = Mock(spec=QuerySessionManager)
        session_manager.create_session = AsyncMock(return_value="session-123")
        mocks['get_session_manager'].return_value = session_manager
        
        return mocks

    @pytest.fixture
    def sample_query_request(self):
        """Sample query request for testing."""
        return QueryRequest(
            query="What is the electricity consumption trend for Plant A?",
            context={"facility": "Plant A"},
            preferences={"include_charts": True}
        )

    @pytest.fixture
    def sample_processing_options(self):
        """Sample processing options."""
        return QueryProcessingOptions(
            timeout_seconds=300,
            include_recommendations=True,
            max_results=20
        )

    def test_process_ehs_query_success(self, mock_dependencies, sample_query_request, sample_processing_options):
        """Test successful EHS query processing."""
        with patch.multiple('ehs_analytics.api.routers.analytics', **mock_dependencies):
            response = client.post(
                "/api/v1/analytics/query",
                json=sample_query_request.dict(),
                params=sample_processing_options.dict()
            )
            
            assert response.status_code == status.HTTP_202_ACCEPTED
            
            data = response.json()
            assert data["success"] is True
            assert "query_id" in data
            assert data["status"] == QueryStatus.PENDING
            assert data["message"] == "Query processing initiated successfully"
            assert data["estimated_completion_time"] == sample_processing_options.timeout_seconds

    def test_process_ehs_query_validation_error(self, mock_dependencies):
        """Test query processing with validation error."""
        with patch.multiple('ehs_analytics.api.routers.analytics', **mock_dependencies):
            # Send invalid request (empty query)
            response = client.post(
                "/api/v1/analytics/query",
                json={"query": ""}  # Empty query should fail validation
            )
            
            # Should return validation error
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_process_ehs_query_rate_limit_error(self, mock_dependencies, sample_query_request):
        """Test query processing with rate limit exceeded."""
        # Mock rate limit validation to raise HTTPException
        mock_dependencies['validate_request_rate_limit'].side_effect = HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
        
        with patch.multiple('ehs_analytics.api.routers.analytics', **mock_dependencies):
            response = client.post(
                "/api/v1/analytics/query",
                json=sample_query_request.dict()
            )
            
            assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS

    def test_process_ehs_query_internal_error(self, mock_dependencies, sample_query_request):
        """Test query processing with internal server error."""
        # Mock session manager to raise exception
        mock_dependencies['get_session_manager'].return_value.create_session.side_effect = Exception("Internal error")
        
        with patch.multiple('ehs_analytics.api.routers.analytics', **mock_dependencies):
            response = client.post(
                "/api/v1/analytics/query",
                json=sample_query_request.dict()
            )
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            
            data = response.json()
            assert "error_type" in data["detail"]
            assert data["detail"]["error_type"] == ErrorType.PROCESSING_ERROR

    def test_get_query_result_success(self, mock_dependencies):
        """Test successful query result retrieval."""
        # Setup query result in store
        query_id = str(uuid.uuid4())
        query_results_store[query_id] = {
            "query_id": query_id,
            "original_query": "Test query",
            "user_id": "test-user-123",
            "status": QueryStatus.COMPLETED,
            "created_at": datetime.utcnow(),
            "classification": {"intent_type": "consumption_analysis"},
            "retrieval_results": {"documents": [], "total_count": 0},
            "analysis_results": [{"analysis_type": "general"}],
            "recommendations": {"recommendations": [], "recommendations_count": 0},
            "confidence_score": 0.85,
            "workflow_trace": ["Step 1", "Step 2"]
        }
        query_status_store[query_id] = QueryStatus.COMPLETED
        
        with patch.multiple('ehs_analytics.api.routers.analytics', **mock_dependencies):
            response = client.get(f"/api/v1/analytics/query/{query_id}")
            
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert data["success"] is True
            assert data["query_id"] == query_id
            assert data["status"] == QueryStatus.COMPLETED
            assert data["original_query"] == "Test query"
            assert data["confidence_score"] == 0.85

    def test_get_query_result_with_trace(self, mock_dependencies):
        """Test query result retrieval with workflow trace."""
        query_id = str(uuid.uuid4())
        query_results_store[query_id] = {
            "query_id": query_id,
            "original_query": "Test query",
            "user_id": "test-user-123",
            "status": QueryStatus.COMPLETED,
            "created_at": datetime.utcnow(),
            "workflow_trace": ["Step 1: Classification", "Step 2: Retrieval"]
        }
        query_status_store[query_id] = QueryStatus.COMPLETED
        
        with patch.multiple('ehs_analytics.api.routers.analytics', **mock_dependencies):
            response = client.get(f"/api/v1/analytics/query/{query_id}?include_trace=true")
            
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert "workflow_trace" in data
            assert len(data["workflow_trace"]) == 2

    def test_get_query_result_not_found(self, mock_dependencies):
        """Test query result retrieval for non-existent query."""
        query_id = str(uuid.uuid4())
        
        with patch.multiple('ehs_analytics.api.routers.analytics', **mock_dependencies):
            response = client.get(f"/api/v1/analytics/query/{query_id}")
            
            assert response.status_code == status.HTTP_404_NOT_FOUND
            
            data = response.json()
            assert data["detail"]["error_type"] == ErrorType.NOT_FOUND_ERROR

    def test_get_query_result_unauthorized(self, mock_dependencies):
        """Test query result retrieval with unauthorized access."""
        query_id = str(uuid.uuid4())
        query_results_store[query_id] = {
            "query_id": query_id,
            "user_id": "other-user-456",  # Different user
            "status": QueryStatus.COMPLETED,
            "created_at": datetime.utcnow()
        }
        
        with patch.multiple('ehs_analytics.api.routers.analytics', **mock_dependencies):
            response = client.get(f"/api/v1/analytics/query/{query_id}")
            
            assert response.status_code == status.HTTP_403_FORBIDDEN
            
            data = response.json()
            assert data["detail"]["error_type"] == ErrorType.AUTHORIZATION_ERROR

    def test_get_query_status_success(self, mock_dependencies):
        """Test successful query status retrieval."""
        query_id = str(uuid.uuid4())
        query_results_store[query_id] = {
            "query_id": query_id,
            "user_id": "test-user-123",
            "status": QueryStatus.IN_PROGRESS,
            "current_step": "analysis",
            "progress_percentage": 70,
            "processing_options": {"timeout_seconds": 300}
        }
        query_status_store[query_id] = QueryStatus.IN_PROGRESS
        
        with patch.multiple('ehs_analytics.api.routers.analytics', **mock_dependencies):
            response = client.get(f"/api/v1/analytics/query/{query_id}/status")
            
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert data["success"] is True
            assert data["query_id"] == query_id
            assert data["status"] == QueryStatus.IN_PROGRESS
            assert data["progress_percentage"] == 70
            assert data["current_step"] == "analysis"
            assert data["estimated_remaining_time"] is not None

    def test_get_query_status_not_found(self, mock_dependencies):
        """Test query status retrieval for non-existent query."""
        query_id = str(uuid.uuid4())
        
        with patch.multiple('ehs_analytics.api.routers.analytics', **mock_dependencies):
            response = client.get(f"/api/v1/analytics/query/{query_id}/status")
            
            assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_cancel_query_success(self, mock_dependencies):
        """Test successful query cancellation."""
        query_id = str(uuid.uuid4())
        query_results_store[query_id] = {
            "query_id": query_id,
            "user_id": "test-user-123",
            "status": QueryStatus.IN_PROGRESS
        }
        query_status_store[query_id] = QueryStatus.IN_PROGRESS
        
        with patch.multiple('ehs_analytics.api.routers.analytics', **mock_dependencies):
            response = client.delete(f"/api/v1/analytics/query/{query_id}")
            
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert data["success"] is True
            assert f"Query {query_id} has been cancelled successfully" in data["message"]
            
            # Verify query was marked as cancelled
            assert query_status_store[query_id] == QueryStatus.CANCELLED
            assert query_results_store[query_id]["status"] == QueryStatus.CANCELLED

    def test_cancel_query_not_found(self, mock_dependencies):
        """Test query cancellation for non-existent query."""
        query_id = str(uuid.uuid4())
        
        with patch.multiple('ehs_analytics.api.routers.analytics', **mock_dependencies):
            response = client.delete(f"/api/v1/analytics/query/{query_id}")
            
            assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_cancel_query_already_completed(self, mock_dependencies):
        """Test query cancellation for already completed query."""
        query_id = str(uuid.uuid4())
        query_results_store[query_id] = {
            "query_id": query_id,
            "user_id": "test-user-123",
            "status": QueryStatus.COMPLETED
        }
        query_status_store[query_id] = QueryStatus.COMPLETED
        
        with patch.multiple('ehs_analytics.api.routers.analytics', **mock_dependencies):
            response = client.delete(f"/api/v1/analytics/query/{query_id}")
            
            assert response.status_code == status.HTTP_400_BAD_REQUEST
            
            data = response.json()
            assert data["detail"]["error_type"] == ErrorType.VALIDATION_ERROR

    def test_analytics_health_check_healthy(self, mock_dependencies):
        """Test analytics health check when all services are healthy."""
        with patch.multiple('ehs_analytics.api.routers.analytics', **mock_dependencies):
            response = client.get("/api/v1/analytics/health")
            
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert data["success"] is True
            assert data["overall_status"] in ["healthy", "degraded"]  # degraded acceptable for some services
            assert "services" in data
            assert len(data["services"]) > 0
            assert "uptime_seconds" in data
            assert data["version"] == "0.1.0"

    def test_analytics_health_check_unhealthy_db(self, mock_dependencies):
        """Test analytics health check when database is unhealthy."""
        # Make database unhealthy
        mock_dependencies['get_db_manager'].return_value.health_check = AsyncMock(return_value=False)
        mock_dependencies['get_db_manager'].return_value.is_connected = False
        
        with patch.multiple('ehs_analytics.api.routers.analytics', **mock_dependencies):
            response = client.get("/api/v1/analytics/health")
            
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert data["overall_status"] == "unhealthy"
            
            # Find database service status
            db_service = next((s for s in data["services"] if "Database" in s["service_name"]), None)
            assert db_service is not None
            assert db_service["status"] == "unhealthy"

    def test_analytics_health_check_error(self, mock_dependencies):
        """Test analytics health check when health check itself fails."""
        # Make health check raise exception
        mock_dependencies['get_db_manager'].side_effect = Exception("Health check failed")
        
        with patch.multiple('ehs_analytics.api.routers.analytics', **mock_dependencies):
            response = client.get("/api/v1/analytics/health")
            
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert data["success"] is False
            assert data["overall_status"] == "unhealthy"
            assert len(data["services"]) == 1  # Error service

    def test_list_user_queries_success(self, mock_dependencies):
        """Test successful user queries listing."""
        # Add multiple queries for the user
        user_id = "test-user-123"
        queries = []
        for i in range(5):
            query_id = str(uuid.uuid4())
            query_data = {
                "query_id": query_id,
                "original_query": f"Test query {i}",
                "user_id": user_id,
                "status": QueryStatus.COMPLETED if i % 2 == 0 else QueryStatus.IN_PROGRESS,
                "created_at": datetime.utcnow() - timedelta(hours=i),
                "updated_at": datetime.utcnow() - timedelta(hours=i),
                "classification": {"intent_type": "consumption_analysis"}
            }
            query_results_store[query_id] = query_data
            queries.append(query_data)
        
        # Add query from different user (should not appear)
        other_query_id = str(uuid.uuid4())
        query_results_store[other_query_id] = {
            "query_id": other_query_id,
            "user_id": "other-user-456",
            "status": QueryStatus.COMPLETED
        }
        
        with patch.multiple('ehs_analytics.api.routers.analytics', **mock_dependencies):
            response = client.get("/api/v1/analytics/queries")
            
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 5  # Only user's queries
            
            # Should be sorted by creation time (newest first)
            assert data[0]["original_query"] == "Test query 0"
            
            # Check query structure
            for query in data:
                assert "query_id" in query
                assert "original_query" in query
                assert "status" in query
                assert "created_at" in query

    def test_list_user_queries_with_pagination(self, mock_dependencies):
        """Test user queries listing with pagination."""
        # Add multiple queries
        user_id = "test-user-123"
        for i in range(15):
            query_id = str(uuid.uuid4())
            query_results_store[query_id] = {
                "query_id": query_id,
                "original_query": f"Test query {i}",
                "user_id": user_id,
                "status": QueryStatus.COMPLETED,
                "created_at": datetime.utcnow() - timedelta(hours=i),
                "updated_at": datetime.utcnow() - timedelta(hours=i)
            }
        
        with patch.multiple('ehs_analytics.api.routers.analytics', **mock_dependencies):
            # First page
            response = client.get("/api/v1/analytics/queries?limit=5&offset=0")
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data) == 5
            
            # Second page
            response = client.get("/api/v1/analytics/queries?limit=5&offset=5")
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data) == 5
            
            # Third page
            response = client.get("/api/v1/analytics/queries?limit=5&offset=10")
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data) == 5

    def test_list_user_queries_with_status_filter(self, mock_dependencies):
        """Test user queries listing with status filter."""
        user_id = "test-user-123"
        
        # Add queries with different statuses
        for i, status in enumerate([QueryStatus.COMPLETED, QueryStatus.IN_PROGRESS, QueryStatus.FAILED]):
            query_id = str(uuid.uuid4())
            query_results_store[query_id] = {
                "query_id": query_id,
                "original_query": f"Query {i}",
                "user_id": user_id,
                "status": status,
                "created_at": datetime.utcnow()
            }
        
        with patch.multiple('ehs_analytics.api.routers.analytics', **mock_dependencies):
            # Filter for completed queries only
            response = client.get(f"/api/v1/analytics/queries?status_filter={QueryStatus.COMPLETED}")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data) == 1
            assert data[0]["status"] == QueryStatus.COMPLETED

    def test_list_user_queries_empty(self, mock_dependencies):
        """Test user queries listing when user has no queries."""
        with patch.multiple('ehs_analytics.api.routers.analytics', **mock_dependencies):
            response = client.get("/api/v1/analytics/queries")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 0

    def test_list_user_queries_truncated_content(self, mock_dependencies):
        """Test user queries listing truncates long query content."""
        user_id = "test-user-123"
        long_query = "What is the electricity consumption " * 10  # Long query
        
        query_id = str(uuid.uuid4())
        query_results_store[query_id] = {
            "query_id": query_id,
            "original_query": long_query,
            "user_id": user_id,
            "status": QueryStatus.COMPLETED,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        with patch.multiple('ehs_analytics.api.routers.analytics', **mock_dependencies):
            response = client.get("/api/v1/analytics/queries")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data) == 1
            
            returned_query = data[0]["original_query"]
            assert len(returned_query) <= 103  # 100 + "..."
            assert returned_query.endswith("...")


class TestProcessQueryWorkflow:
    """Test suite for process_query_workflow background task."""

    @pytest.fixture
    def mock_workflow_manager(self):
        """Mock workflow manager with query router."""
        workflow_manager = Mock()
        
        # Mock query router
        query_router = Mock()
        from ehs_analytics.agents.query_router import QueryClassification, EntityExtraction, IntentType, RetrieverType
        mock_classification = QueryClassification(
            intent_type=IntentType.CONSUMPTION_ANALYSIS,
            confidence_score=0.85,
            entities_identified=EntityExtraction([], [], [], [], [], [], []),
            suggested_retriever=RetrieverType.CONSUMPTION_RETRIEVER,
            reasoning="Test classification"
        )
        query_router.classify_query.return_value = mock_classification
        workflow_manager.get_query_router.return_value = query_router
        
        return workflow_manager

    @pytest.fixture
    def sample_query_request(self):
        """Sample query request."""
        return QueryRequest(
            query="Test query for workflow",
            context={},
            preferences={}
        )

    @pytest.fixture
    def sample_options(self):
        """Sample processing options."""
        return QueryProcessingOptions(
            timeout_seconds=300,
            include_recommendations=True
        )

    @pytest.mark.asyncio
    async def test_process_query_workflow_success(self, mock_workflow_manager, sample_query_request, sample_options):
        """Test successful query workflow processing."""
        query_id = str(uuid.uuid4())
        user_id = "test-user"
        db_manager = Mock()
        
        # Clear and initialize stores
        query_results_store.clear()
        query_status_store.clear()
        query_status_store[query_id] = QueryStatus.PENDING
        query_results_store[query_id] = {
            "query_id": query_id,
            "user_id": user_id,
            "status": QueryStatus.PENDING
        }
        
        await process_query_workflow(
            query_id=query_id,
            query_request=sample_query_request,
            options=sample_options,
            user_id=user_id,
            db_manager=db_manager,
            workflow_manager=mock_workflow_manager
        )
        
        # Verify final status
        assert query_status_store[query_id] == QueryStatus.COMPLETED
        assert query_results_store[query_id]["status"] == QueryStatus.COMPLETED
        
        # Verify workflow data was populated
        assert "classification" in query_results_store[query_id]
        assert "retrieval_results" in query_results_store[query_id]
        assert "analysis_results" in query_results_store[query_id]
        assert "recommendations" in query_results_store[query_id]  # Should be included due to options
        assert query_results_store[query_id]["progress_percentage"] == 100

    @pytest.mark.asyncio
    async def test_process_query_workflow_no_recommendations(self, mock_workflow_manager, sample_query_request):
        """Test query workflow processing without recommendations."""
        query_id = str(uuid.uuid4())
        user_id = "test-user"
        db_manager = Mock()
        
        options = QueryProcessingOptions(include_recommendations=False)
        
        # Initialize stores
        query_results_store.clear()
        query_status_store.clear()
        query_status_store[query_id] = QueryStatus.PENDING
        query_results_store[query_id] = {
            "query_id": query_id,
            "user_id": user_id,
            "status": QueryStatus.PENDING
        }
        
        await process_query_workflow(
            query_id=query_id,
            query_request=sample_query_request,
            options=options,
            user_id=user_id,
            db_manager=db_manager,
            workflow_manager=mock_workflow_manager
        )
        
        # Should not have recommendations
        assert "recommendations" not in query_results_store[query_id]
        assert query_status_store[query_id] == QueryStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_process_query_workflow_cancellation(self, mock_workflow_manager, sample_query_request, sample_options):
        """Test query workflow processing cancellation."""
        query_id = str(uuid.uuid4())
        user_id = "test-user"
        db_manager = Mock()
        
        # Initialize stores
        query_results_store.clear()
        query_status_store.clear()
        query_status_store[query_id] = QueryStatus.PENDING
        query_results_store[query_id] = {
            "query_id": query_id,
            "user_id": user_id,
            "status": QueryStatus.PENDING
        }
        
        # Create a task that we can cancel
        task = asyncio.create_task(
            process_query_workflow(
                query_id=query_id,
                query_request=sample_query_request,
                options=sample_options,
                user_id=user_id,
                db_manager=db_manager,
                workflow_manager=mock_workflow_manager
            )
        )
        
        # Cancel the task quickly
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        # Should be marked as cancelled
        assert query_status_store[query_id] == QueryStatus.CANCELLED
        assert query_results_store[query_id]["status"] == QueryStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_process_query_workflow_error(self, mock_workflow_manager, sample_query_request, sample_options):
        """Test query workflow processing with error."""
        query_id = str(uuid.uuid4())
        user_id = "test-user"
        db_manager = Mock()
        
        # Make query router raise exception
        mock_workflow_manager.get_query_router.return_value.classify_query.side_effect = Exception("Classification failed")
        
        # Initialize stores
        query_results_store.clear()
        query_status_store.clear()
        query_status_store[query_id] = QueryStatus.PENDING
        query_results_store[query_id] = {
            "query_id": query_id,
            "user_id": user_id,
            "status": QueryStatus.PENDING
        }
        
        await process_query_workflow(
            query_id=query_id,
            query_request=sample_query_request,
            options=sample_options,
            user_id=user_id,
            db_manager=db_manager,
            workflow_manager=mock_workflow_manager
        )
        
        # Should be marked as failed
        assert query_status_store[query_id] == QueryStatus.FAILED
        assert query_results_store[query_id]["status"] == QueryStatus.FAILED
        assert "error" in query_results_store[query_id]


class TestAPIValidation:
    """Test suite for API request/response validation."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock dependencies for validation tests."""
        return {
            'get_current_user_id': Mock(return_value="test-user"),
            'validate_request_rate_limit': Mock(),
            'validate_query_request': AsyncMock()
        }

    def test_query_request_validation_valid(self):
        """Test valid query request passes validation."""
        valid_request = {
            "query": "What is the electricity consumption?",
            "context": {"facility": "Plant A"},
            "preferences": {"include_charts": True}
        }
        
        # Should not raise validation error
        query_request = QueryRequest(**valid_request)
        assert query_request.query == "What is the electricity consumption?"
        assert query_request.context == {"facility": "Plant A"}
        assert query_request.preferences == {"include_charts": True}

    def test_query_request_validation_minimal(self):
        """Test minimal valid query request."""
        minimal_request = {"query": "Test query"}
        
        query_request = QueryRequest(**minimal_request)
        assert query_request.query == "Test query"
        assert query_request.context == {}  # Should default to empty dict
        assert query_request.preferences == {}  # Should default to empty dict

    def test_query_processing_options_defaults(self):
        """Test query processing options with default values."""
        options = QueryProcessingOptions()
        
        assert options.timeout_seconds == 300  # Default
        assert options.include_recommendations is True  # Default
        assert options.max_results == 50  # Default

    def test_query_processing_options_custom(self):
        """Test query processing options with custom values."""
        options = QueryProcessingOptions(
            timeout_seconds=600,
            include_recommendations=False,
            max_results=100
        )
        
        assert options.timeout_seconds == 600
        assert options.include_recommendations is False
        assert options.max_results == 100

    def test_health_check_response_structure(self):
        """Test health check response structure."""
        services = [
            ServiceHealth(
                service_name="Test Service",
                status="healthy",
                response_time_ms=50
            )
        ]
        
        response = HealthCheckResponse(
            success=True,
            message="All systems operational",
            overall_status="healthy",
            services=services,
            uptime_seconds=3600,
            version="1.0.0",
            environment="test"
        )
        
        assert response.success is True
        assert response.overall_status == "healthy"
        assert len(response.services) == 1
        assert response.services[0].service_name == "Test Service"


class TestErrorHandling:
    """Test suite for API error handling."""

    def test_error_response_structure(self):
        """Test error response follows expected structure."""
        # This would typically be tested through actual API calls
        # but we can test the error models directly
        from ehs_analytics.api.models import ErrorDetail, ErrorType
        
        error = ErrorDetail(
            error_type=ErrorType.VALIDATION_ERROR,
            error_code="INVALID_INPUT",
            message="Invalid query format",
            details={"field": "query", "issue": "cannot be empty"}
        )
        
        assert error.error_type == ErrorType.VALIDATION_ERROR
        assert error.error_code == "INVALID_INPUT"
        assert error.message == "Invalid query format"
        assert error.details["field"] == "query"

    def test_query_status_enum_values(self):
        """Test QueryStatus enum has expected values."""
        expected_statuses = [
            QueryStatus.PENDING,
            QueryStatus.IN_PROGRESS,
            QueryStatus.COMPLETED,
            QueryStatus.FAILED,
            QueryStatus.CANCELLED
        ]
        
        for status in expected_statuses:
            assert isinstance(status, QueryStatus)
            assert isinstance(status.value, str)

    def test_error_type_enum_values(self):
        """Test ErrorType enum has expected values."""
        expected_types = [
            ErrorType.VALIDATION_ERROR,
            ErrorType.PROCESSING_ERROR,
            ErrorType.NOT_FOUND_ERROR,
            ErrorType.AUTHORIZATION_ERROR,
            ErrorType.RATE_LIMIT_ERROR
        ]
        
        for error_type in expected_types:
            assert isinstance(error_type, ErrorType)
            assert isinstance(error_type.value, str)


class TestConcurrency:
    """Test suite for API concurrency handling."""

    @pytest.mark.asyncio
    async def test_concurrent_query_submissions(self):
        """Test handling of concurrent query submissions."""
        # This test would verify that the API can handle multiple
        # concurrent query submissions without race conditions
        # Implementation would depend on actual concurrency controls
        pass

    @pytest.mark.asyncio 
    async def test_concurrent_result_access(self):
        """Test concurrent access to query results."""
        # This test would verify that multiple clients can
        # access query results concurrently without issues
        pass