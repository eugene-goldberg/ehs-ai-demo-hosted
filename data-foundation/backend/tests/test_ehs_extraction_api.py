"""
Comprehensive test suite for the EHS Data Extraction API.

This test suite covers all endpoints, error handling, edge cases, and response validation
for the EHS extraction API using pytest and FastAPI test client.
"""

import os
import sys
import json
import pytest
import time
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

# Add the src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastapi.testclient import TestClient
from fastapi import HTTPException

# Import the API components
from ehs_extraction_api import app, get_workflow, build_query_parameters
from workflows.extraction_workflow import DataExtractionWorkflow, QueryType, ExtractionState


class TestClient:
    """Test client wrapper for the FastAPI application."""
    
    def __init__(self):
        self.client = TestClient(app)


# Test fixtures
@pytest.fixture
def test_client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture
def mock_workflow():
    """Mock DataExtractionWorkflow for testing without Neo4j."""
    with patch('ehs_extraction_api.DataExtractionWorkflow') as mock_workflow_class:
        mock_instance = Mock(spec=DataExtractionWorkflow)
        mock_workflow_class.return_value = mock_instance
        
        # Mock successful extraction result
        mock_instance.extract_data.return_value = {
            "status": "completed",
            "report_data": {
                "test_data": "sample_data",
                "records_processed": 10
            },
            "report_file_path": "/path/to/report.json",
            "queries": [
                {"query": "MATCH (n) RETURN n", "parameters": {}}
            ],
            "query_results": [
                {
                    "status": "success",
                    "record_count": 10,
                    "results": [{"test": "data"}]
                }
            ],
            "errors": []
        }
        
        mock_instance.close.return_value = None
        yield mock_instance


@pytest.fixture
def mock_neo4j_env():
    """Mock Neo4j environment variables."""
    with patch.dict(os.environ, {
        'NEO4J_URI': 'bolt://localhost:7687',
        'NEO4J_USERNAME': 'neo4j',
        'NEO4J_PASSWORD': 'password',
        'LLM_MODEL': 'gpt-4'
    }):
        yield


@pytest.fixture
def sample_electrical_request():
    """Sample electrical consumption request data."""
    return {
        "facility_filter": {
            "facility_id": "FAC-001",
            "facility_name": "Main Campus"
        },
        "date_range": {
            "start_date": "2023-01-01",
            "end_date": "2023-12-31"
        },
        "output_format": "json",
        "include_emissions": True,
        "include_cost_analysis": True
    }


@pytest.fixture
def sample_water_request():
    """Sample water consumption request data."""
    return {
        "date_range": {
            "start_date": "2023-01-01", 
            "end_date": "2023-12-31"
        },
        "output_format": "json",
        "include_meter_details": True,
        "include_emissions": True
    }


@pytest.fixture
def sample_waste_request():
    """Sample waste generation request data."""
    return {
        "facility_filter": {
            "facility_id": "FAC-001"
        },
        "date_range": {
            "start_date": "2023-01-01",
            "end_date": "2023-12-31"
        },
        "output_format": "json",
        "include_disposal_details": True,
        "include_transport_details": True,
        "include_emissions": True,
        "hazardous_only": False
    }


# Health check endpoint tests
class TestHealthEndpoint:
    """Test the health check endpoint."""
    
    def test_health_check_success(self, test_client, mock_neo4j_env):
        """Test successful health check."""
        with patch('ehs_extraction_api.get_workflow') as mock_get_workflow:
            mock_workflow = Mock()
            mock_workflow.close.return_value = None
            mock_get_workflow.return_value = mock_workflow
            
            response = test_client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "timestamp" in data
            assert data["neo4j_connection"] is True
            assert data["version"] == "1.0.0"
    
    def test_health_check_neo4j_failure(self, test_client, mock_neo4j_env):
        """Test health check with Neo4j connection failure."""
        with patch('ehs_extraction_api.get_workflow', side_effect=Exception("Connection failed")):
            response = test_client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["neo4j_connection"] is False


# Electrical consumption endpoint tests
class TestElectricalConsumptionEndpoint:
    """Test the electrical consumption extraction endpoint."""
    
    def test_electrical_consumption_success(self, test_client, mock_neo4j_env, mock_workflow, sample_electrical_request):
        """Test successful electrical consumption extraction."""
        with patch('ehs_extraction_api.get_workflow', return_value=mock_workflow):
            response = test_client.post("/api/v1/extract/electrical-consumption", json=sample_electrical_request)
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["message"] == "Electrical consumption data extracted successfully"
            assert "data" in data
            assert "metadata" in data
            assert "processing_time" in data
            
            # Verify data structure
            assert data["data"]["query_type"] == QueryType.UTILITY_CONSUMPTION
            assert data["data"]["include_emissions"] is True
            assert data["data"]["include_cost_analysis"] is True
    
    def test_electrical_consumption_with_minimal_data(self, test_client, mock_neo4j_env, mock_workflow):
        """Test electrical consumption with minimal request data."""
        minimal_request = {"output_format": "json"}
        
        with patch('ehs_extraction_api.get_workflow', return_value=mock_workflow):
            response = test_client.post("/api/v1/extract/electrical-consumption", json=minimal_request)
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
    
    def test_electrical_consumption_workflow_failure(self, test_client, mock_neo4j_env):
        """Test electrical consumption with workflow failure."""
        mock_workflow = Mock()
        mock_workflow.extract_data.side_effect = Exception("Workflow failed")
        
        with patch('ehs_extraction_api.get_workflow', return_value=mock_workflow):
            response = test_client.post("/api/v1/extract/electrical-consumption", json={"output_format": "json"})
            
            assert response.status_code == 500
            assert "Extraction failed" in response.json()["detail"]
    
    def test_electrical_consumption_failed_status(self, test_client, mock_neo4j_env, sample_electrical_request):
        """Test electrical consumption with failed extraction status."""
        mock_workflow = Mock()
        mock_workflow.extract_data.return_value = {
            "status": "failed",
            "report_data": {},
            "query_results": [],
            "errors": ["Query execution failed"]
        }
        mock_workflow.close.return_value = None
        
        with patch('ehs_extraction_api.get_workflow', return_value=mock_workflow):
            response = test_client.post("/api/v1/extract/electrical-consumption", json=sample_electrical_request)
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "failed"
            assert data["message"] == "Extraction failed"
            assert len(data["errors"]) > 0


# Water consumption endpoint tests
class TestWaterConsumptionEndpoint:
    """Test the water consumption extraction endpoint."""
    
    def test_water_consumption_success(self, test_client, mock_neo4j_env, mock_workflow, sample_water_request):
        """Test successful water consumption extraction."""
        with patch('ehs_extraction_api.get_workflow', return_value=mock_workflow):
            response = test_client.post("/api/v1/extract/water-consumption", json=sample_water_request)
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["message"] == "Water consumption data extracted successfully"
            assert data["data"]["query_type"] == QueryType.WATER_CONSUMPTION
    
    def test_water_consumption_date_range_only(self, test_client, mock_neo4j_env, mock_workflow):
        """Test water consumption with only date range."""
        request_data = {
            "date_range": {
                "start_date": "2023-01-01",
                "end_date": "2023-12-31"
            },
            "output_format": "json"
        }
        
        with patch('ehs_extraction_api.get_workflow', return_value=mock_workflow):
            response = test_client.post("/api/v1/extract/water-consumption", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["data"]["facility_filter"] is None
    
    def test_water_consumption_with_text_output(self, test_client, mock_neo4j_env, mock_workflow):
        """Test water consumption with text output format."""
        request_data = {
            "output_format": "txt",
            "include_meter_details": False,
            "include_emissions": False
        }
        
        with patch('ehs_extraction_api.get_workflow', return_value=mock_workflow):
            response = test_client.post("/api/v1/extract/water-consumption", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["data"]["include_meter_details"] is False
            assert data["data"]["include_emissions"] is False


# Waste generation endpoint tests  
class TestWasteGenerationEndpoint:
    """Test the waste generation extraction endpoint."""
    
    def test_waste_generation_success(self, test_client, mock_neo4j_env, mock_workflow, sample_waste_request):
        """Test successful waste generation extraction."""
        with patch('ehs_extraction_api.get_workflow', return_value=mock_workflow):
            response = test_client.post("/api/v1/extract/waste-generation", json=sample_waste_request)
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["message"] == "Waste generation data extracted successfully"
            assert data["data"]["query_type"] == QueryType.WASTE_GENERATION
    
    def test_waste_generation_hazardous_only(self, test_client, mock_neo4j_env, mock_workflow):
        """Test waste generation with hazardous waste filter."""
        request_data = {
            "date_range": {
                "start_date": "2023-01-01",
                "end_date": "2023-12-31"
            },
            "output_format": "json",
            "hazardous_only": True,
            "include_disposal_details": False,
            "include_transport_details": False,
            "include_emissions": False
        }
        
        # Mock workflow to return parameters including hazardous_only
        mock_workflow.extract_data.return_value = {
            "status": "completed",
            "report_data": {"test": "data"},
            "queries": [],
            "query_results": []
        }
        
        with patch('ehs_extraction_api.get_workflow', return_value=mock_workflow):
            response = test_client.post("/api/v1/extract/waste-generation", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["data"]["hazardous_only"] is True
            
            # Verify hazardous_only parameter was passed to workflow
            mock_workflow.extract_data.assert_called_once()
            args, kwargs = mock_workflow.extract_data.call_args
            assert kwargs["parameters"]["hazardous_only"] is True
    
    def test_waste_generation_all_parameters(self, test_client, mock_neo4j_env, mock_workflow, sample_waste_request):
        """Test waste generation with all parameters enabled."""
        with patch('ehs_extraction_api.get_workflow', return_value=mock_workflow):
            response = test_client.post("/api/v1/extract/waste-generation", json=sample_waste_request)
            
            assert response.status_code == 200
            data = response.json()
            assert data["data"]["include_disposal_details"] is True
            assert data["data"]["include_transport_details"] is True
            assert data["data"]["include_emissions"] is True
            assert data["data"]["hazardous_only"] is False


# Custom extraction endpoint tests
class TestCustomExtractionEndpoint:
    """Test the custom extraction endpoint."""
    
    def test_custom_extraction_success(self, test_client, mock_neo4j_env, mock_workflow):
        """Test successful custom extraction."""
        request_params = {
            "query_type": QueryType.FACILITY_EMISSIONS,
            "facility_filter": {"facility_id": "FAC-001"},
            "date_range": {"start_date": "2023-01-01", "end_date": "2023-12-31"},
            "output_format": "json"
        }
        
        with patch('ehs_extraction_api.get_workflow', return_value=mock_workflow):
            response = test_client.post("/api/v1/extract/custom", params=request_params)
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["message"] == "Custom data extraction completed"
    
    def test_custom_extraction_with_custom_queries(self, test_client, mock_neo4j_env, mock_workflow):
        """Test custom extraction with custom queries."""
        request_params = {
            "query_type": QueryType.CUSTOM,
            "output_format": "json",
            "custom_queries": [
                {"query": "MATCH (n:Facility) RETURN n", "parameters": {}}
            ]
        }
        
        with patch('ehs_extraction_api.get_workflow', return_value=mock_workflow):
            response = test_client.post("/api/v1/extract/custom", params=request_params)
            
            assert response.status_code == 200
            data = response.json()
            assert data["data"]["custom_queries"] is not None
    
    def test_custom_extraction_invalid_query_type(self, test_client, mock_neo4j_env):
        """Test custom extraction with invalid query type."""
        request_params = {
            "query_type": "invalid_query_type",
            "output_format": "json"
        }
        
        response = test_client.post("/api/v1/extract/custom", params=request_params)
        
        assert response.status_code == 400
        assert "Invalid query type" in response.json()["detail"]
    
    def test_custom_extraction_all_query_types(self, test_client, mock_neo4j_env, mock_workflow):
        """Test custom extraction with all valid query types."""
        valid_query_types = [qt.value for qt in QueryType]
        
        for query_type in valid_query_types:
            request_params = {
                "query_type": query_type,
                "output_format": "json"
            }
            
            with patch('ehs_extraction_api.get_workflow', return_value=mock_workflow):
                response = test_client.post("/api/v1/extract/custom", params=request_params)
                
                assert response.status_code == 200, f"Failed for query type: {query_type}"


# Query types endpoint tests
class TestQueryTypesEndpoint:
    """Test the query types information endpoint."""
    
    def test_get_query_types(self, test_client):
        """Test getting available query types."""
        response = test_client.get("/api/v1/query-types")
        
        assert response.status_code == 200
        data = response.json()
        assert "query_types" in data
        assert len(data["query_types"]) == len(QueryType)
        
        # Verify structure of query type entries
        for qt in data["query_types"]:
            assert "value" in qt
            assert "name" in qt
            assert "description" in qt
    
    def test_query_types_content(self, test_client):
        """Test query types content and descriptions."""
        response = test_client.get("/api/v1/query-types")
        data = response.json()
        
        query_type_values = [qt["value"] for qt in data["query_types"]]
        expected_values = [qt.value for qt in QueryType]
        
        assert set(query_type_values) == set(expected_values)
        
        # Check that descriptions are meaningful
        for qt in data["query_types"]:
            assert len(qt["description"]) > 10  # Should have meaningful descriptions


# Edge case and validation tests
class TestEdgeCasesAndValidation:
    """Test edge cases and input validation."""
    
    def test_invalid_date_range_end_before_start(self, test_client, mock_neo4j_env):
        """Test validation for end date before start date."""
        invalid_request = {
            "date_range": {
                "start_date": "2023-12-31",
                "end_date": "2023-01-01"  # End before start
            },
            "output_format": "json"
        }
        
        response = test_client.post("/api/v1/extract/electrical-consumption", json=invalid_request)
        
        assert response.status_code == 422  # Validation error
    
    def test_missing_required_parameters(self, test_client, mock_neo4j_env):
        """Test handling of missing required parameters."""
        # Empty request should still work with defaults
        response = test_client.post("/api/v1/extract/electrical-consumption", json={})
        
        # Should not fail validation, but may fail at workflow level
        assert response.status_code in [200, 500]
    
    def test_invalid_facility_filters(self, test_client, mock_neo4j_env, mock_workflow):
        """Test handling of invalid facility filters."""
        request_with_empty_facility = {
            "facility_filter": {},  # Empty facility filter
            "output_format": "json"
        }
        
        with patch('ehs_extraction_api.get_workflow', return_value=mock_workflow):
            response = test_client.post("/api/v1/extract/electrical-consumption", json=request_with_empty_facility)
            
            assert response.status_code == 200
    
    def test_empty_results_handling(self, test_client, mock_neo4j_env):
        """Test handling of empty query results."""
        mock_workflow = Mock()
        mock_workflow.extract_data.return_value = {
            "status": "completed",
            "report_data": {},
            "query_results": [],  # Empty results
            "queries": [],
            "errors": []
        }
        mock_workflow.close.return_value = None
        
        with patch('ehs_extraction_api.get_workflow', return_value=mock_workflow):
            response = test_client.post("/api/v1/extract/electrical-consumption", json={"output_format": "json"})
            
            assert response.status_code == 200
            data = response.json()
            assert data["metadata"]["total_records"] == 0
    
    def test_date_validation_edge_cases(self, test_client, mock_neo4j_env, mock_workflow):
        """Test various date validation edge cases."""
        # Test with same start and end date
        same_date_request = {
            "date_range": {
                "start_date": "2023-06-15",
                "end_date": "2023-06-15"
            },
            "output_format": "json"
        }
        
        with patch('ehs_extraction_api.get_workflow', return_value=mock_workflow):
            response = test_client.post("/api/v1/extract/water-consumption", json=same_date_request)
            assert response.status_code == 200
    
    def test_invalid_json_payload(self, test_client):
        """Test handling of invalid JSON payload."""
        response = test_client.post(
            "/api/v1/extract/electrical-consumption",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422


# Error handling tests
class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_neo4j_connection_failure(self, test_client):
        """Test handling of Neo4j connection failures."""
        with patch.dict(os.environ, {}, clear=True):  # Clear env vars
            response = test_client.post("/api/v1/extract/electrical-consumption", json={"output_format": "json"})
            
            assert response.status_code == 500
            assert "Missing required Neo4j connection configuration" in response.json()["detail"]
    
    def test_workflow_initialization_error(self, test_client, mock_neo4j_env):
        """Test workflow initialization errors."""
        with patch('ehs_extraction_api.DataExtractionWorkflow', side_effect=Exception("Init failed")):
            response = test_client.post("/api/v1/extract/electrical-consumption", json={"output_format": "json"})
            
            assert response.status_code == 500
            assert "Workflow initialization error" in response.json()["detail"]
    
    def test_workflow_processing_errors(self, test_client, mock_neo4j_env):
        """Test workflow processing errors."""
        mock_workflow = Mock()
        mock_workflow.extract_data.return_value = {
            "status": "failed",
            "errors": ["Database connection lost", "Query timeout"],
            "query_results": [],
            "report_data": {}
        }
        mock_workflow.close.return_value = None
        
        with patch('ehs_extraction_api.get_workflow', return_value=mock_workflow):
            response = test_client.post("/api/v1/extract/waste-generation", json={"output_format": "json"})
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "failed"
            assert len(data["errors"]) > 0
    
    def test_partial_query_failures(self, test_client, mock_neo4j_env):
        """Test handling of partial query failures."""
        mock_workflow = Mock()
        mock_workflow.extract_data.return_value = {
            "status": "completed",
            "report_data": {"partial_data": True},
            "query_results": [
                {"status": "success", "record_count": 5},
                {"status": "failed", "error": "Timeout"},
                {"status": "success", "record_count": 3}
            ],
            "queries": [{}, {}, {}],
            "errors": ["One query failed"]
        }
        mock_workflow.close.return_value = None
        
        with patch('ehs_extraction_api.get_workflow', return_value=mock_workflow):
            response = test_client.post("/api/v1/extract/electrical-consumption", json={"output_format": "json"})
            
            assert response.status_code == 200
            data = response.json()
            assert data["metadata"]["total_queries"] == 3
            assert data["metadata"]["successful_queries"] == 2
            assert data["metadata"]["total_records"] == 8


# Response validation tests
class TestResponseValidation:
    """Test response structure and content validation."""
    
    def test_response_structure_electrical(self, test_client, mock_neo4j_env, mock_workflow, sample_electrical_request):
        """Test electrical consumption response structure."""
        with patch('ehs_extraction_api.get_workflow', return_value=mock_workflow):
            response = test_client.post("/api/v1/extract/electrical-consumption", json=sample_electrical_request)
            
            assert response.status_code == 200
            data = response.json()
            
            # Validate required fields
            required_fields = ["status", "message", "data", "metadata", "processing_time"]
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"
            
            # Validate data structure
            data_fields = ["query_type", "facility_filter", "date_range", "report_data"]
            for field in data_fields:
                assert field in data["data"], f"Missing data field: {field}"
            
            # Validate metadata structure  
            metadata_fields = ["total_queries", "successful_queries", "total_records", "processing_status"]
            for field in metadata_fields:
                assert field in data["metadata"], f"Missing metadata field: {field}"
    
    def test_metadata_fields_presence(self, test_client, mock_neo4j_env, mock_workflow):
        """Test presence of all metadata fields."""
        with patch('ehs_extraction_api.get_workflow', return_value=mock_workflow):
            response = test_client.post("/api/v1/extract/water-consumption", json={"output_format": "json"})
            
            data = response.json()
            metadata = data["metadata"]
            
            assert isinstance(metadata["total_queries"], int)
            assert isinstance(metadata["successful_queries"], int)
            assert isinstance(metadata["total_records"], int)
            assert "generated_at" in metadata
            assert "processing_status" in metadata
    
    def test_error_response_format(self, test_client):
        """Test error response format consistency."""
        # Test with missing env vars
        with patch.dict(os.environ, {}, clear=True):
            response = test_client.post("/api/v1/extract/electrical-consumption", json={"output_format": "json"})
            
            assert response.status_code == 500
            error_data = response.json()
            assert "detail" in error_data
    
    def test_processing_time_validation(self, test_client, mock_neo4j_env, mock_workflow):
        """Test processing time is properly calculated."""
        with patch('ehs_extraction_api.get_workflow', return_value=mock_workflow):
            start_time = time.time()
            response = test_client.post("/api/v1/extract/electrical-consumption", json={"output_format": "json"})
            end_time = time.time()
            
            assert response.status_code == 200
            data = response.json()
            
            processing_time = data["processing_time"]
            assert isinstance(processing_time, (int, float))
            assert 0 <= processing_time <= (end_time - start_time) + 1  # Allow some overhead


# Performance tests (optional)
class TestPerformance:
    """Optional performance tests for the API."""
    
    def test_response_time_validation(self, test_client, mock_neo4j_env, mock_workflow):
        """Test that API responses are returned within reasonable time."""
        with patch('ehs_extraction_api.get_workflow', return_value=mock_workflow):
            start_time = time.time()
            response = test_client.post("/api/v1/extract/electrical-consumption", json={"output_format": "json"})
            response_time = time.time() - start_time
            
            assert response.status_code == 200
            assert response_time < 5.0  # Should respond within 5 seconds
    
    def test_concurrent_request_handling(self, test_client, mock_neo4j_env, mock_workflow):
        """Test handling of concurrent requests."""
        def make_request():
            with patch('ehs_extraction_api.get_workflow', return_value=mock_workflow):
                response = test_client.post("/api/v1/extract/electrical-consumption", json={"output_format": "json"})
                return response.status_code
        
        # Test with multiple concurrent requests
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request) for _ in range(3)]
            results = [future.result() for future in futures]
        
        # All requests should succeed
        assert all(status == 200 for status in results)
    
    @pytest.mark.slow
    def test_large_date_range_handling(self, test_client, mock_neo4j_env, mock_workflow):
        """Test handling of large date ranges."""
        large_range_request = {
            "date_range": {
                "start_date": "2020-01-01",
                "end_date": "2023-12-31"  # 4 year range
            },
            "output_format": "json"
        }
        
        with patch('ehs_extraction_api.get_workflow', return_value=mock_workflow):
            response = test_client.post("/api/v1/extract/waste-generation", json=large_range_request)
            
            assert response.status_code == 200


# Utility function tests
class TestUtilityFunctions:
    """Test utility functions used by the API."""
    
    def test_build_query_parameters_with_all_filters(self):
        """Test build_query_parameters with all filter types."""
        from ehs_extraction_api import FacilityFilter, DateRangeFilter
        
        facility_filter = FacilityFilter(facility_id="FAC-001", facility_name="Test Facility")
        date_range = DateRangeFilter(start_date=date(2023, 1, 1), end_date=date(2023, 12, 31))
        
        params = build_query_parameters(facility_filter, date_range)
        
        expected_params = {
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "facility_id": "FAC-001",
            "facility_name": "Test Facility"
        }
        
        assert params == expected_params
    
    def test_build_query_parameters_with_partial_filters(self):
        """Test build_query_parameters with partial filters."""
        from ehs_extraction_api import DateRangeFilter
        
        date_range = DateRangeFilter(start_date=date(2023, 1, 1))  # No end date
        
        params = build_query_parameters(None, date_range)
        
        assert params == {"start_date": "2023-01-01"}
    
    def test_build_query_parameters_with_no_filters(self):
        """Test build_query_parameters with no filters."""
        params = build_query_parameters(None, None)
        assert params == {}


# Integration-style tests
class TestWorkflowIntegration:
    """Test integration with the workflow component."""
    
    def test_workflow_parameter_passing(self, test_client, mock_neo4j_env):
        """Test that parameters are correctly passed to workflow."""
        mock_workflow = Mock()
        mock_workflow.extract_data.return_value = {
            "status": "completed",
            "report_data": {},
            "query_results": [],
            "queries": [],
            "errors": []
        }
        mock_workflow.close.return_value = None
        
        request_data = {
            "facility_filter": {"facility_id": "FAC-123"},
            "date_range": {"start_date": "2023-01-01", "end_date": "2023-12-31"},
            "output_format": "txt"
        }
        
        with patch('ehs_extraction_api.get_workflow', return_value=mock_workflow):
            response = test_client.post("/api/v1/extract/electrical-consumption", json=request_data)
            
            assert response.status_code == 200
            
            # Verify workflow was called with correct parameters
            mock_workflow.extract_data.assert_called_once()
            args, kwargs = mock_workflow.extract_data.call_args
            
            assert kwargs["query_type"] == QueryType.UTILITY_CONSUMPTION
            assert kwargs["output_format"] == "txt"
            assert "facility_id" in kwargs["parameters"]
            assert kwargs["parameters"]["facility_id"] == "FAC-123"
    
    def test_workflow_cleanup(self, test_client, mock_neo4j_env, mock_workflow):
        """Test that workflow connections are properly closed."""
        with patch('ehs_extraction_api.get_workflow', return_value=mock_workflow):
            response = test_client.post("/api/v1/extract/electrical-consumption", json={"output_format": "json"})
            
            assert response.status_code == 200
            mock_workflow.close.assert_called_once()


# Parametrized tests for all endpoints
class TestAllEndpoints:
    """Parametrized tests covering all extraction endpoints."""
    
    @pytest.mark.parametrize("endpoint,request_data", [
        ("/api/v1/extract/electrical-consumption", {"output_format": "json", "include_emissions": True}),
        ("/api/v1/extract/water-consumption", {"output_format": "json", "include_meter_details": True}),
        ("/api/v1/extract/waste-generation", {"output_format": "json", "hazardous_only": False}),
    ])
    def test_all_endpoints_basic_functionality(self, test_client, mock_neo4j_env, mock_workflow, endpoint, request_data):
        """Test basic functionality of all extraction endpoints."""
        with patch('ehs_extraction_api.get_workflow', return_value=mock_workflow):
            response = test_client.post(endpoint, json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "data" in data
            assert "metadata" in data
    
    @pytest.mark.parametrize("output_format", ["json", "txt"])
    def test_output_formats(self, test_client, mock_neo4j_env, mock_workflow, output_format):
        """Test different output formats for all endpoints."""
        request_data = {"output_format": output_format}
        
        endpoints = [
            "/api/v1/extract/electrical-consumption",
            "/api/v1/extract/water-consumption", 
            "/api/v1/extract/waste-generation"
        ]
        
        for endpoint in endpoints:
            with patch('ehs_extraction_api.get_workflow', return_value=mock_workflow):
                response = test_client.post(endpoint, json=request_data)
                
                assert response.status_code == 200
                data = response.json()
                assert data["data"]["file_path"] is not None  # File should be generated


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])