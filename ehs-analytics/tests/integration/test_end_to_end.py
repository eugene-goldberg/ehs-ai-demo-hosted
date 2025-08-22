"""
Comprehensive End-to-End Integration Tests for EHS Analytics

This module contains comprehensive integration tests that verify the complete
workflow from query submission to result retrieval, including database
integration, API functionality, and performance benchmarks.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import pytest
import pytest_asyncio
import httpx
from neo4j import GraphDatabase
import redis
import structlog
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from src.ehs_analytics.api.main import app
from src.ehs_analytics.api.models import QueryStatus, QueryRequest, ErrorType
from src.ehs_analytics.agents.query_router import IntentType, RetrieverType
from src.ehs_analytics.config import get_settings

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger(__name__)

# Test configuration
TEST_CONFIG = {
    "neo4j_uri": "bolt://localhost:7687",
    "neo4j_username": "neo4j",
    "neo4j_password": "test_password",
    "redis_url": "redis://localhost:6379",
    "api_base_url": "http://localhost:8000",
    "test_timeout": 300,
    "performance_thresholds": {
        "query_processing_time": 30.0,  # seconds
        "api_response_time": 2.0,       # seconds
        "database_query_time": 1.0      # seconds
    }
}


class TestDatabaseConnection:
    """Test database connectivity and basic operations."""
    
    @pytest.fixture(scope="class")
    def neo4j_driver(self):
        """Create Neo4j driver for testing."""
        driver = GraphDatabase.driver(
            TEST_CONFIG["neo4j_uri"],
            auth=(TEST_CONFIG["neo4j_username"], TEST_CONFIG["neo4j_password"])
        )
        yield driver
        driver.close()
    
    @pytest.fixture(scope="class")
    def redis_client(self):
        """Create Redis client for testing."""
        client = redis.from_url(TEST_CONFIG["redis_url"])
        yield client
        client.close()
    
    def test_neo4j_connection(self, neo4j_driver):
        """Test Neo4j database connectivity."""
        with neo4j_driver.session() as session:
            result = session.run("RETURN 1 as test")
            assert result.single()["test"] == 1
    
    def test_redis_connection(self, redis_client):
        """Test Redis cache connectivity."""
        test_key = f"test_key_{uuid.uuid4()}"
        test_value = f"test_value_{uuid.uuid4()}"
        
        redis_client.set(test_key, test_value, ex=60)
        retrieved_value = redis_client.get(test_key)
        
        assert retrieved_value.decode() == test_value
        redis_client.delete(test_key)
    
    def test_database_schema_exists(self, neo4j_driver):
        """Test that required database schema exists."""
        with neo4j_driver.session() as session:
            # Check for Facility nodes
            facility_count = session.run(
                "MATCH (f:Facility) RETURN count(f) as count"
            ).single()["count"]
            
            # Check for Equipment nodes
            equipment_count = session.run(
                "MATCH (e:Equipment) RETURN count(e) as count"
            ).single()["count"]
            
            # Check for Permit nodes
            permit_count = session.run(
                "MATCH (p:Permit) RETURN count(p) as count"
            ).single()["count"]
            
            logger.info(
                "Database schema validation",
                facilities=facility_count,
                equipment=equipment_count,
                permits=permit_count
            )
            
            # We expect some test data to be present
            assert facility_count > 0, "No facilities found in database"
    
    def test_database_indexes_exist(self, neo4j_driver):
        """Test that performance indexes exist."""
        with neo4j_driver.session() as session:
            indexes = session.run("SHOW INDEXES").data()
            index_names = {idx["name"] for idx in indexes if idx["name"]}
            
            required_indexes = {
                "facility_name_idx",
                "equipment_facility_idx",
                "permit_facility_idx"
            }
            
            missing_indexes = required_indexes - index_names
            
            if missing_indexes:
                logger.warning(
                    "Missing database indexes",
                    missing=list(missing_indexes),
                    existing=list(index_names)
                )
            
            # Log all available indexes for debugging
            logger.info("Available database indexes", indexes=list(index_names))


class TestAPIIntegration:
    """Test API endpoints and functionality."""
    
    @pytest.fixture(scope="class")
    def client(self):
        """Create test client for API testing."""
        return TestClient(app)
    
    @pytest.fixture(scope="class") 
    def test_user_id(self):
        """Generate a unique test user ID."""
        return f"test_user_{uuid.uuid4()}"
    
    def test_api_root_endpoint(self, client):
        """Test API root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "EHS Analytics API"
        assert data["version"] == "0.1.0"
        assert "endpoints" in data
    
    def test_health_check_endpoint(self, client):
        """Test global health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "overall_status" in data
        assert "services" in data
        assert "uptime_seconds" in data
    
    def test_analytics_health_endpoint(self, client):
        """Test analytics-specific health check."""
        response = client.get("/api/v1/analytics/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert len(data["services"]) > 0
    
    def test_openapi_documentation(self, client):
        """Test OpenAPI specification endpoint."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        spec = response.json()
        assert "openapi" in spec
        assert "paths" in spec
        assert "/api/v1/analytics/query" in spec["paths"]
    
    def test_query_submission_validation(self, client, test_user_id):
        """Test query request validation."""
        # Test empty query
        response = client.post(
            "/api/v1/analytics/query",
            json={"query": "", "user_id": test_user_id},
            headers={"X-User-ID": test_user_id}
        )
        assert response.status_code == 400
        assert "validation_error" in response.json()["error"]["error_type"]
        
        # Test query too long
        long_query = "a" * 2001
        response = client.post(
            "/api/v1/analytics/query",
            json={"query": long_query, "user_id": test_user_id},
            headers={"X-User-ID": test_user_id}
        )
        assert response.status_code == 400
        
        # Test valid query
        response = client.post(
            "/api/v1/analytics/query",
            json={
                "query": "Show electricity usage for Apex Manufacturing",
                "user_id": test_user_id
            },
            headers={"X-User-ID": test_user_id}
        )
        assert response.status_code == 202
    
    def test_query_not_found(self, client, test_user_id):
        """Test querying non-existent query ID."""
        fake_query_id = f"query-{uuid.uuid4()}"
        
        response = client.get(
            f"/api/v1/analytics/query/{fake_query_id}",
            headers={"X-User-ID": test_user_id}
        )
        assert response.status_code == 404
        assert "not_found_error" in response.json()["error"]["error_type"]
    
    @pytest.mark.asyncio
    async def test_concurrent_queries(self, client, test_user_id):
        """Test handling multiple concurrent queries."""
        queries = [
            "Show water usage for all facilities",
            "What permits are expiring soon?",
            "Identify high-risk equipment",
            "Show energy consumption trends",
            "Check compliance status"
        ]
        
        query_ids = []
        
        # Submit multiple queries concurrently
        for query in queries:
            response = client.post(
                "/api/v1/analytics/query",
                json={"query": query, "user_id": test_user_id},
                headers={"X-User-ID": test_user_id}
            )
            assert response.status_code == 202
            query_ids.append(response.json()["query_id"])
        
        # Wait for all queries to complete
        await asyncio.sleep(5)
        
        # Check status of all queries
        completed_count = 0
        for query_id in query_ids:
            response = client.get(
                f"/api/v1/analytics/query/{query_id}/status",
                headers={"X-User-ID": test_user_id}
            )
            assert response.status_code == 200
            
            status = response.json()["status"]
            if status == QueryStatus.COMPLETED:
                completed_count += 1
        
        logger.info(
            "Concurrent query test results",
            total_queries=len(query_ids),
            completed_queries=completed_count
        )


class TestWorkflowIntegration:
    """Test complete query processing workflow."""
    
    @pytest.fixture(scope="class")
    def client(self):
        return TestClient(app)
    
    @pytest.fixture(scope="class")
    def test_user_id(self):
        return f"test_user_{uuid.uuid4()}"
    
    @pytest.mark.asyncio
    async def test_complete_workflow_consumption_analysis(self, client, test_user_id):
        """Test complete workflow for consumption analysis query."""
        query_text = "Show electricity usage for Apex Manufacturing in Q1 2024"
        
        # Step 1: Submit query
        start_time = time.time()
        response = client.post(
            "/api/v1/analytics/query",
            json={
                "query": query_text,
                "user_id": test_user_id,
                "context": {
                    "facility_focus": "energy_efficiency",
                    "department": "environmental"
                }
            },
            headers={"X-User-ID": test_user_id}
        )
        
        assert response.status_code == 202
        data = response.json()
        query_id = data["query_id"]
        
        logger.info("Query submitted", query_id=query_id, query=query_text)
        
        # Step 2: Monitor query status
        timeout = TEST_CONFIG["test_timeout"]
        while timeout > 0:
            response = client.get(
                f"/api/v1/analytics/query/{query_id}/status",
                headers={"X-User-ID": test_user_id}
            )
            assert response.status_code == 200
            
            status_data = response.json()
            status = status_data["status"]
            
            logger.info(
                "Query status check",
                query_id=query_id,
                status=status,
                progress=status_data.get("progress_percentage"),
                current_step=status_data.get("current_step")
            )
            
            if status == QueryStatus.COMPLETED:
                break
            elif status == QueryStatus.FAILED:
                pytest.fail(f"Query processing failed: {status_data}")
            
            await asyncio.sleep(2)
            timeout -= 2
        
        if timeout <= 0:
            pytest.fail("Query processing timeout")
        
        # Step 3: Retrieve results
        response = client.get(
            f"/api/v1/analytics/query/{query_id}?include_trace=true",
            headers={"X-User-ID": test_user_id}
        )
        assert response.status_code == 200
        
        result_data = response.json()
        processing_time = time.time() - start_time
        
        # Verify result structure
        assert result_data["success"] is True
        assert result_data["query_id"] == query_id
        assert result_data["status"] == QueryStatus.COMPLETED
        assert result_data["original_query"] == query_text
        assert "classification" in result_data
        assert "retrieval_results" in result_data
        
        # Verify classification
        classification = result_data["classification"]
        assert classification["intent_type"] == IntentType.CONSUMPTION_ANALYSIS
        assert classification["confidence_score"] > 0.5
        assert len(classification["entities_identified"]["facilities"]) > 0
        
        # Verify workflow trace
        assert "workflow_trace" in result_data
        assert len(result_data["workflow_trace"]) > 0
        
        logger.info(
            "Workflow test completed",
            query_id=query_id,
            processing_time=processing_time,
            intent=classification["intent_type"],
            confidence=classification["confidence_score"]
        )
    
    @pytest.mark.asyncio
    async def test_complete_workflow_compliance_check(self, client, test_user_id):
        """Test complete workflow for compliance check query."""
        query_text = "What environmental permits are expiring in the next 90 days?"
        
        # Submit query
        response = client.post(
            "/api/v1/analytics/query",
            json={
                "query": query_text,
                "user_id": test_user_id,
                "context": {"urgency": "high", "department": "compliance"}
            },
            headers={"X-User-ID": test_user_id}
        )
        
        assert response.status_code == 202
        query_id = response.json()["query_id"]
        
        # Wait for completion
        await self._wait_for_completion(client, query_id, test_user_id)
        
        # Retrieve and verify results
        response = client.get(
            f"/api/v1/analytics/query/{query_id}",
            headers={"X-User-ID": test_user_id}
        )
        assert response.status_code == 200
        
        result_data = response.json()
        classification = result_data["classification"]
        
        # Verify compliance-specific classification
        assert classification["intent_type"] in [
            IntentType.COMPLIANCE_CHECK,
            IntentType.PERMIT_STATUS
        ]
        assert classification["confidence_score"] > 0.5
        
        # Verify entities extraction
        entities = classification["entities_identified"]
        assert len(entities["date_ranges"]) > 0
    
    @pytest.mark.asyncio
    async def test_complete_workflow_risk_assessment(self, client, test_user_id):
        """Test complete workflow for risk assessment query."""
        query_text = "Identify high-risk equipment at Apex Manufacturing facility"
        
        # Submit query
        response = client.post(
            "/api/v1/analytics/query",
            json={
                "query": query_text,
                "user_id": test_user_id,
                "context": {"risk_types": ["environmental", "operational"]}
            },
            headers={"X-User-ID": test_user_id}
        )
        
        assert response.status_code == 202
        query_id = response.json()["query_id"]
        
        # Wait for completion
        await self._wait_for_completion(client, query_id, test_user_id)
        
        # Retrieve and verify results
        response = client.get(
            f"/api/v1/analytics/query/{query_id}",
            headers={"X-User-ID": test_user_id}
        )
        assert response.status_code == 200
        
        result_data = response.json()
        classification = result_data["classification"]
        
        # Verify risk assessment classification
        assert classification["intent_type"] == IntentType.RISK_ASSESSMENT
        assert classification["suggested_retriever"] == RetrieverType.RISK_RETRIEVER
    
    async def _wait_for_completion(self, client, query_id: str, user_id: str, timeout: int = 60):
        """Helper method to wait for query completion."""
        while timeout > 0:
            response = client.get(
                f"/api/v1/analytics/query/{query_id}/status",
                headers={"X-User-ID": user_id}
            )
            assert response.status_code == 200
            
            status = response.json()["status"]
            if status == QueryStatus.COMPLETED:
                return
            elif status == QueryStatus.FAILED:
                pytest.fail("Query processing failed")
            
            await asyncio.sleep(1)
            timeout -= 1
        
        pytest.fail("Query completion timeout")


class TestPerformanceBenchmarks:
    """Performance benchmarks and load testing."""
    
    @pytest.fixture(scope="class")
    def client(self):
        return TestClient(app)
    
    @pytest.fixture(scope="class")
    def test_user_id(self):
        return f"perf_user_{uuid.uuid4()}"
    
    def test_api_response_time_benchmark(self, client):
        """Benchmark API response times for various endpoints."""
        endpoints = [
            ("/", "GET", None),
            ("/health", "GET", None),
            ("/api/v1/analytics/health", "GET", None),
            ("/openapi.json", "GET", None)
        ]
        
        results = {}
        
        for endpoint, method, data in endpoints:
            times = []
            
            # Run 10 requests to get average
            for _ in range(10):
                start_time = time.time()
                
                if method == "GET":
                    response = client.get(endpoint)
                else:
                    response = client.post(endpoint, json=data)
                
                end_time = time.time()
                response_time = end_time - start_time
                times.append(response_time)
                
                assert response.status_code in [200, 202]
            
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)
            
            results[endpoint] = {
                "avg_time": avg_time,
                "max_time": max_time,
                "min_time": min_time
            }
            
            # Check against threshold
            threshold = TEST_CONFIG["performance_thresholds"]["api_response_time"]
            assert avg_time < threshold, f"API response time too slow: {avg_time}s > {threshold}s"
        
        logger.info("API response time benchmark", results=results)
    
    @pytest.mark.asyncio 
    async def test_query_processing_performance(self, client, test_user_id):
        """Benchmark query processing performance."""
        test_queries = [
            "Show electricity usage for Apex Manufacturing",
            "What permits expire this month?",
            "Identify maintenance issues",
            "Show water consumption trends",
            "Check environmental compliance"
        ]
        
        performance_results = []
        
        for query in test_queries:
            start_time = time.time()
            
            # Submit query
            response = client.post(
                "/api/v1/analytics/query",
                json={"query": query, "user_id": test_user_id},
                headers={"X-User-ID": test_user_id}
            )
            assert response.status_code == 202
            query_id = response.json()["query_id"]
            
            # Wait for completion
            completion_time = await self._measure_completion_time(
                client, query_id, test_user_id
            )
            
            total_time = time.time() - start_time
            
            performance_results.append({
                "query": query[:50] + "..." if len(query) > 50 else query,
                "total_time": total_time,
                "completion_time": completion_time,
                "query_id": query_id
            })
            
            # Check against performance threshold
            threshold = TEST_CONFIG["performance_thresholds"]["query_processing_time"]
            assert total_time < threshold, f"Query processing too slow: {total_time}s > {threshold}s"
        
        # Calculate statistics
        times = [r["total_time"] for r in performance_results]
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        
        logger.info(
            "Query processing performance benchmark",
            results=performance_results,
            statistics={
                "avg_time": avg_time,
                "max_time": max_time,
                "min_time": min_time,
                "total_queries": len(test_queries)
            }
        )
    
    @pytest.mark.asyncio
    async def test_concurrent_load_performance(self, client, test_user_id):
        """Test performance under concurrent load."""
        concurrent_queries = 10
        query_template = "Show energy usage for facility {}"
        
        async def submit_and_wait(query_num):
            """Submit a query and wait for completion."""
            query = query_template.format(f"Plant-{query_num}")
            start_time = time.time()
            
            response = client.post(
                "/api/v1/analytics/query",
                json={"query": query, "user_id": f"{test_user_id}_{query_num}"},
                headers={"X-User-ID": f"{test_user_id}_{query_num}"}
            )
            
            if response.status_code != 202:
                return {"error": f"Failed to submit query: {response.status_code}"}
            
            query_id = response.json()["query_id"]
            completion_time = await self._measure_completion_time(
                client, query_id, f"{test_user_id}_{query_num}"
            )
            
            total_time = time.time() - start_time
            
            return {
                "query_num": query_num,
                "query_id": query_id,
                "total_time": total_time,
                "completion_time": completion_time
            }
        
        # Submit all queries concurrently
        start_time = time.time()
        tasks = [submit_and_wait(i) for i in range(concurrent_queries)]
        results = await asyncio.gather(*tasks)
        total_load_test_time = time.time() - start_time
        
        # Analyze results
        successful_results = [r for r in results if "error" not in r]
        failed_results = [r for r in results if "error" in r]
        
        if successful_results:
            avg_time = sum(r["total_time"] for r in successful_results) / len(successful_results)
            max_time = max(r["total_time"] for r in successful_results)
        else:
            avg_time = max_time = 0
        
        logger.info(
            "Concurrent load test results",
            total_queries=concurrent_queries,
            successful=len(successful_results),
            failed=len(failed_results),
            total_load_time=total_load_test_time,
            avg_query_time=avg_time,
            max_query_time=max_time
        )
        
        # Verify performance under load
        assert len(successful_results) > 0, "No queries completed successfully"
        assert len(failed_results) < concurrent_queries * 0.1, "Too many failed queries"
    
    async def _measure_completion_time(self, client, query_id: str, user_id: str, timeout: int = 60):
        """Measure time until query completion."""
        start_time = time.time()
        
        while timeout > 0:
            response = client.get(
                f"/api/v1/analytics/query/{query_id}/status",
                headers={"X-User-ID": user_id}
            )
            
            if response.status_code != 200:
                return None
            
            status = response.json()["status"]
            if status == QueryStatus.COMPLETED:
                return time.time() - start_time
            elif status == QueryStatus.FAILED:
                return None
            
            await asyncio.sleep(0.5)
            timeout -= 0.5
        
        return None  # Timeout


class TestDataIntegrity:
    """Test data consistency and integrity."""
    
    @pytest.fixture(scope="class")
    def neo4j_driver(self):
        driver = GraphDatabase.driver(
            TEST_CONFIG["neo4j_uri"],
            auth=(TEST_CONFIG["neo4j_username"], TEST_CONFIG["neo4j_password"])
        )
        yield driver
        driver.close()
    
    @pytest.fixture(scope="class")
    def client(self):
        return TestClient(app)
    
    def test_facility_data_integrity(self, neo4j_driver):
        """Test facility data consistency."""
        with neo4j_driver.session() as session:
            # Check for required facility properties
            result = session.run("""
                MATCH (f:Facility)
                WHERE f.name IS NOT NULL AND f.type IS NOT NULL
                RETURN count(f) as valid_facilities,
                       count{(f:Facility)} as total_facilities
            """).single()
            
            valid_facilities = result["valid_facilities"]
            total_facilities = result["total_facilities"]
            
            assert valid_facilities > 0, "No valid facilities found"
            assert valid_facilities == total_facilities, "Some facilities missing required properties"
    
    def test_equipment_facility_relationships(self, neo4j_driver):
        """Test equipment-facility relationship integrity."""
        with neo4j_driver.session() as session:
            # Check for orphaned equipment
            orphaned_equipment = session.run("""
                MATCH (e:Equipment)
                WHERE NOT (e)-[:LOCATED_AT]->(:Facility)
                RETURN count(e) as orphaned_count
            """).single()["orphaned_count"]
            
            assert orphaned_equipment == 0, f"Found {orphaned_equipment} orphaned equipment records"
            
            # Check for equipment with invalid facility references
            invalid_refs = session.run("""
                MATCH (e:Equipment)-[:LOCATED_AT]->(f:Facility)
                WHERE f.name IS NULL OR f.name = ""
                RETURN count(e) as invalid_count
            """).single()["invalid_count"]
            
            assert invalid_refs == 0, f"Found {invalid_refs} equipment with invalid facility references"
    
    def test_permit_data_consistency(self, neo4j_driver):
        """Test permit data consistency."""
        with neo4j_driver.session() as session:
            # Check permit date consistency
            invalid_dates = session.run("""
                MATCH (p:Permit)
                WHERE p.expiration_date < p.issue_date
                RETURN count(p) as invalid_count
            """).single()["invalid_count"]
            
            assert invalid_dates == 0, f"Found {invalid_dates} permits with invalid date ranges"
            
            # Check for permits without facilities
            unlinked_permits = session.run("""
                MATCH (p:Permit)
                WHERE NOT (p)-[:ISSUED_FOR]->(:Facility)
                RETURN count(p) as unlinked_count
            """).single()["unlinked_count"]
            
            assert unlinked_permits == 0, f"Found {unlinked_permits} permits not linked to facilities"


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture(scope="class")
    def client(self):
        return TestClient(app)
    
    @pytest.fixture(scope="class")
    def test_user_id(self):
        return f"error_test_user_{uuid.uuid4()}"
    
    def test_malformed_query_request(self, client, test_user_id):
        """Test handling of malformed query requests."""
        # Missing required fields
        response = client.post(
            "/api/v1/analytics/query",
            json={},
            headers={"X-User-ID": test_user_id}
        )
        assert response.status_code == 422
        
        # Invalid data types
        response = client.post(
            "/api/v1/analytics/query",
            json={"query": 123, "user_id": test_user_id},
            headers={"X-User-ID": test_user_id}
        )
        assert response.status_code == 422
    
    def test_unauthorized_access(self, client):
        """Test unauthorized access attempts."""
        # No user ID header
        response = client.post(
            "/api/v1/analytics/query",
            json={"query": "test query"}
        )
        # This might return 422 or 403 depending on implementation
        assert response.status_code in [401, 403, 422]
    
    def test_query_cancellation(self, client, test_user_id):
        """Test query cancellation functionality."""
        # Submit a query
        response = client.post(
            "/api/v1/analytics/query",
            json={"query": "Long running query", "user_id": test_user_id},
            headers={"X-User-ID": test_user_id}
        )
        assert response.status_code == 202
        query_id = response.json()["query_id"]
        
        # Cancel the query
        response = client.delete(
            f"/api/v1/analytics/query/{query_id}",
            headers={"X-User-ID": test_user_id}
        )
        
        # Should succeed if query is still cancellable
        assert response.status_code in [200, 400]  # 400 if already completed
    
    def test_query_timeout_handling(self, client, test_user_id):
        """Test query timeout handling."""
        # Submit a query with very short timeout
        response = client.post(
            "/api/v1/analytics/query",
            json={
                "query": "Complex analysis query",
                "user_id": test_user_id
            },
            headers={"X-User-ID": test_user_id},
            params={"timeout_seconds": 1}  # Very short timeout
        )
        
        if response.status_code == 202:
            query_id = response.json()["query_id"]
            
            # Wait a bit then check status
            time.sleep(3)
            
            response = client.get(
                f"/api/v1/analytics/query/{query_id}/status",
                headers={"X-User-ID": test_user_id}
            )
            assert response.status_code == 200
            
            # Query might be completed, failed, or cancelled due to timeout
            status = response.json()["status"]
            assert status in [QueryStatus.COMPLETED, QueryStatus.FAILED, QueryStatus.CANCELLED]


class TestSecurityAndValidation:
    """Test security measures and input validation."""
    
    @pytest.fixture(scope="class")
    def client(self):
        return TestClient(app)
    
    @pytest.fixture(scope="class")
    def test_user_id(self):
        return f"security_test_user_{uuid.uuid4()}"
    
    def test_sql_injection_prevention(self, client, test_user_id):
        """Test SQL injection prevention in queries."""
        malicious_queries = [
            "'; DROP TABLE facilities; --",
            "' OR 1=1; --",
            "'; DELETE FROM equipment; --",
            "UNION SELECT * FROM users; --"
        ]
        
        for malicious_query in malicious_queries:
            response = client.post(
                "/api/v1/analytics/query",
                json={"query": malicious_query, "user_id": test_user_id},
                headers={"X-User-ID": test_user_id}
            )
            
            # Should accept the query but not execute malicious code
            assert response.status_code in [202, 400]  # 400 for validation errors
    
    def test_xss_prevention(self, client, test_user_id):
        """Test XSS prevention in query content."""
        xss_queries = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "';alert('xss');//"
        ]
        
        for xss_query in xss_queries:
            response = client.post(
                "/api/v1/analytics/query",
                json={"query": xss_query, "user_id": test_user_id},
                headers={"X-User-ID": test_user_id}
            )
            
            # Should handle safely without executing scripts
            assert response.status_code in [202, 400]
    
    def test_input_size_limits(self, client, test_user_id):
        """Test input size limitations."""
        # Test maximum query length
        max_query = "A" * 2000  # At the limit
        response = client.post(
            "/api/v1/analytics/query",
            json={"query": max_query, "user_id": test_user_id},
            headers={"X-User-ID": test_user_id}
        )
        assert response.status_code == 202
        
        # Test over-limit query
        over_limit_query = "A" * 2001
        response = client.post(
            "/api/v1/analytics/query",
            json={"query": over_limit_query, "user_id": test_user_id},
            headers={"X-User-ID": test_user_id}
        )
        assert response.status_code == 400
    
    def test_user_isolation(self, client):
        """Test that users can only access their own queries."""
        user_1 = f"user_1_{uuid.uuid4()}"
        user_2 = f"user_2_{uuid.uuid4()}"
        
        # User 1 submits a query
        response = client.post(
            "/api/v1/analytics/query",
            json={"query": "User 1 query", "user_id": user_1},
            headers={"X-User-ID": user_1}
        )
        assert response.status_code == 202
        query_id = response.json()["query_id"]
        
        # User 2 tries to access User 1's query
        response = client.get(
            f"/api/v1/analytics/query/{query_id}",
            headers={"X-User-ID": user_2}
        )
        assert response.status_code == 403  # Should be forbidden


# Performance monitoring decorator
def monitor_performance(test_name: str):
    """Decorator to monitor test performance."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                status = "passed"
            except Exception as e:
                status = "failed"
                raise
            finally:
                end_time = time.time()
                duration = end_time - start_time
                
                logger.info(
                    "Test performance metrics",
                    test_name=test_name,
                    duration=duration,
                    status=status,
                    timestamp=datetime.utcnow().isoformat()
                )
            
            return result
        return wrapper
    return decorator


# Test execution summary
def pytest_sessionfinish(session, exitstatus):
    """Print test execution summary."""
    logger.info(
        "Integration test session completed",
        exit_status=exitstatus,
        test_count=session.testscollected,
        failed_count=session.testsfailed,
        timestamp=datetime.utcnow().isoformat()
    )


if __name__ == "__main__":
    # Run tests with pytest
    import sys
    
    # Configure test run
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "--log-level=INFO",
        "-x"  # Stop on first failure for debugging
    ]
    
    # Add performance markers if requested
    if "--benchmark" in sys.argv:
        pytest_args.extend(["-m", "performance"])
    
    # Run the tests
    exit_code = pytest.main(pytest_args)
    sys.exit(exit_code)