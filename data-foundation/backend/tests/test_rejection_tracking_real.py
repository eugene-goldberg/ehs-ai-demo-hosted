"""
Real Database Tests for Document Rejection Tracking API

This test suite validates the rejection tracking API endpoints using real Neo4j database
operations. It tests complete workflows including document rejection, unreejection,
bulk operations, and statistics generation with actual database state changes.

Key Features:
- Real Neo4j database operations (no mocks)
- FastAPI test client for HTTP requests
- Complete data isolation between tests
- Database state verification
- End-to-end workflow testing

Test Coverage:
1. Document rejection with database updates
2. Document unreejection with state changes
3. Rejected documents retrieval from real data
4. Rejection statistics from actual database
5. Bulk rejection operations
6. Error handling with real database constraints

Dependencies:
- pytest for test framework
- FastAPI TestClient for HTTP testing
- Neo4j database with test isolation
- test_database.py configuration
"""

import os
import sys
import json
import pytest
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import uuid
import asyncio

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastapi.testclient import TestClient
from fastapi import status
import httpx

# Import test database configuration
from test_database import (
    Neo4jTestClient, TestDataFactory, neo4j_test_client, test_data_factory,
    assert_node_exists, get_node_properties, assert_relationship_exists
)

# Import the FastAPI application
from ehs_extraction_api import app

# Import rejection tracking components
from phase1_enhancements.rejection_tracking_schema import DocumentStatus
from phase1_enhancements.rejection_workflow_service import RejectionReason, RejectionStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create test client
client = TestClient(app)

class RejectionTrackingTestSuite:
    """Test suite for rejection tracking API with real database operations."""
    
    def __init__(self, neo4j_client: Neo4jTestClient, data_factory: TestDataFactory):
        self.neo4j_client = neo4j_client
        self.data_factory = data_factory
        self.created_documents = []
        self.rejection_ids = []
        
    def setup_test_documents(self, count: int = 3) -> List[str]:
        """Create test documents for rejection tracking tests."""
        documents = []
        
        for i in range(count):
            doc_id = self.data_factory.create_test_document(
                doc_type=f"test_document_{i+1}",
                content=f"Test document content {i+1} for rejection tracking tests",
                metadata={
                    "filename": f"test_doc_{i+1}.pdf",
                    "source": "test_suite",
                    "status": DocumentStatus.PROCESSING.value,
                    "created_at": datetime.utcnow().isoformat(),
                    "owner": f"test_user_{i+1}",
                    "size": 1024 * (i+1),
                    "checksum": f"test_checksum_{i+1}"
                }
            )
            documents.append(doc_id)
            self.created_documents.append(doc_id)
            
        logger.info(f"Created {len(documents)} test documents for rejection tracking tests")
        return documents
    
    def verify_document_status(self, document_id: str, expected_status: str) -> bool:
        """Verify the current status of a document in the database."""
        query = """
        MATCH (d:Document)
        WHERE elementId(d) = $doc_id
        RETURN d.status as status, d as document
        """
        
        result = self.neo4j_client.execute_query(query, {"doc_id": document_id})
        if not result:
            return False
            
        current_status = result[0]["status"]
        logger.info(f"Document {document_id} has status: {current_status} (expected: {expected_status})")
        return current_status == expected_status
    
    def verify_rejection_record_exists(self, document_id: str) -> Optional[Dict]:
        """Verify that a rejection record exists for a document."""
        query = """
        MATCH (d:Document)-[r:REJECTED]->(rejection:RejectionRecord)
        WHERE elementId(d) = $doc_id
        RETURN rejection, r, rejection.rejection_id as rejection_id
        """
        
        result = self.neo4j_client.execute_query(query, {"doc_id": document_id})
        if result:
            rejection_data = result[0]["rejection"]
            logger.info(f"Found rejection record for document {document_id}: {rejection_data}")
            return rejection_data
        return None
    
    def count_rejected_documents(self) -> int:
        """Count total rejected documents in test session."""
        query = """
        MATCH (d:Document)
        WHERE d.status = 'REJECTED' 
        AND d.test_session_id = $session_id
        RETURN count(d) as count
        """
        
        result = self.neo4j_client.execute_query(query, {"session_id": self.neo4j_client.test_session_id})
        return result[0]["count"] if result else 0
    
    def cleanup_test_data(self):
        """Clean up test-specific data."""
        deleted = self.neo4j_client.cleanup_test_data()
        logger.info(f"Cleaned up {deleted} test records")
        return deleted


@pytest.fixture
def rejection_test_suite(neo4j_test_client, test_data_factory):
    """Create a rejection tracking test suite instance."""
    suite = RejectionTrackingTestSuite(neo4j_test_client, test_data_factory)
    yield suite
    suite.cleanup_test_data()


@pytest.fixture
def test_documents(rejection_test_suite):
    """Create test documents for rejection tracking tests."""
    return rejection_test_suite.setup_test_documents(count=5)


# Test 1: Basic Document Rejection with Real Database Update
def test_reject_document_real_database(rejection_test_suite, test_documents):
    """Test document rejection with real database state changes."""
    logger.info("Testing document rejection with real database operations")
    
    document_id = test_documents[0]
    
    # Verify document exists and has initial status
    assert rejection_test_suite.verify_document_status(document_id, DocumentStatus.PROCESSING.value)
    
    # Prepare rejection request
    rejection_request = {
        "reason": RejectionReason.DATA_QUALITY_ISSUES.value,
        "notes": "Test rejection - data quality issues found in test document",
        "notify_owner": False  # Skip notifications for testing
    }
    
    # Make API request to reject document
    response = client.post(
        f"/api/v1/documents/{document_id}/reject",
        json=rejection_request
    )
    
    # Verify API response
    assert response.status_code == status.HTTP_201_CREATED
    response_data = response.json()
    
    assert response_data["status"] == "rejected"
    assert "rejection_id" in response_data
    assert "timestamp" in response_data
    assert response_data["message"].find(document_id) != -1
    
    # Verify database state changes
    assert rejection_test_suite.verify_document_status(document_id, DocumentStatus.REJECTED.value)
    
    # Verify rejection record was created
    rejection_record = rejection_test_suite.verify_rejection_record_exists(document_id)
    assert rejection_record is not None
    assert rejection_record["reason"] == RejectionReason.DATA_QUALITY_ISSUES.value
    assert rejection_record["notes"] == rejection_request["notes"]
    
    logger.info("✓ Document rejection test passed with real database verification")


# Test 2: Document Unreejection with Database State Change
def test_unreject_document_real_database(rejection_test_suite, test_documents):
    """Test document unreejection with real database state restoration."""
    logger.info("Testing document unreejection with real database state changes")
    
    document_id = test_documents[1]
    
    # First reject the document
    rejection_request = {
        "reason": RejectionReason.INCOMPLETE_DATA.value,
        "notes": "Test rejection for unreejection test",
        "notify_owner": False
    }
    
    reject_response = client.post(
        f"/api/v1/documents/{document_id}/reject",
        json=rejection_request
    )
    assert reject_response.status_code == status.HTTP_201_CREATED
    assert rejection_test_suite.verify_document_status(document_id, DocumentStatus.REJECTED.value)
    
    # Now unreject the document
    unreject_request = {
        "reason": "Test unreejection - data issues resolved",
        "notes": "Unrejecting for test verification"
    }
    
    response = client.post(
        f"/api/v1/documents/{document_id}/unreject",
        json=unreject_request
    )
    
    # Verify API response
    assert response.status_code == status.HTTP_200_OK
    response_data = response.json()
    
    assert response_data["status"] == "success"
    assert response_data["document_id"] == document_id
    assert "timestamp" in response_data
    assert "new_status" in response_data
    
    # Verify database state has been restored
    # Note: The exact status depends on the unreject implementation
    # It might be PROCESSING, PENDING, or another valid status
    current_props = get_node_properties(rejection_test_suite.neo4j_client, document_id)
    assert current_props["status"] != DocumentStatus.REJECTED.value
    
    logger.info("✓ Document unreejection test passed with real database verification")


# Test 3: Get Rejected Documents from Real Database
def test_get_rejected_documents_real_data(rejection_test_suite, test_documents):
    """Test retrieving rejected documents from real database data."""
    logger.info("Testing rejected documents retrieval from real database")
    
    # Reject multiple documents with different reasons
    rejection_scenarios = [
        {
            "doc_id": test_documents[0],
            "reason": RejectionReason.DATA_QUALITY_ISSUES.value,
            "notes": "Test rejection 1"
        },
        {
            "doc_id": test_documents[1], 
            "reason": RejectionReason.INCOMPLETE_DATA.value,
            "notes": "Test rejection 2"
        },
        {
            "doc_id": test_documents[2],
            "reason": RejectionReason.DATA_QUALITY_ISSUES.value,
            "notes": "Test rejection 3"
        }
    ]
    
    # Reject all test documents
    for scenario in rejection_scenarios:
        response = client.post(
            f"/api/v1/documents/{scenario['doc_id']}/reject",
            json={
                "reason": scenario["reason"],
                "notes": scenario["notes"],
                "notify_owner": False
            }
        )
        assert response.status_code == status.HTTP_201_CREATED
    
    # Verify documents are rejected in database
    rejected_count = rejection_test_suite.count_rejected_documents()
    assert rejected_count >= 3
    
    # Test API endpoint to get rejected documents
    response = client.get("/api/v1/documents/rejected?limit=10&offset=0")
    assert response.status_code == status.HTTP_200_OK
    
    response_data = response.json()
    
    # Verify response structure
    assert "documents" in response_data
    assert "pagination" in response_data
    assert response_data["pagination"]["total"] >= 3
    
    # Verify that rejected documents are returned
    returned_docs = response_data["documents"]
    assert len(returned_docs) >= 3
    
    # Verify each document has required rejection information
    for doc in returned_docs:
        assert "document_id" in doc
        assert "status" in doc
        assert doc["status"] == DocumentStatus.REJECTED.value
        assert "rejection_reason" in doc
        assert "rejected_at" in doc
    
    # Test filtering by rejection reason
    response = client.get(
        f"/api/v1/documents/rejected?reason={RejectionReason.DATA_QUALITY_ISSUES.value}&limit=10"
    )
    assert response.status_code == status.HTTP_200_OK
    
    filtered_data = response.json()
    filtered_docs = filtered_data["documents"]
    
    # All returned documents should have the specified rejection reason
    for doc in filtered_docs:
        assert doc["rejection_reason"] == RejectionReason.DATA_QUALITY_ISSUES.value
    
    logger.info("✓ Rejected documents retrieval test passed with real database data")


# Test 4: Rejection Statistics from Real Database
def test_rejection_statistics_real_data(rejection_test_suite, test_documents):
    """Test rejection statistics generation from real database data."""
    logger.info("Testing rejection statistics from real database data")
    
    # Create a diverse set of rejections for statistics
    rejection_data = [
        {"doc": test_documents[0], "reason": RejectionReason.DATA_QUALITY_ISSUES.value},
        {"doc": test_documents[1], "reason": RejectionReason.DATA_QUALITY_ISSUES.value},
        {"doc": test_documents[2], "reason": RejectionReason.INCOMPLETE_DATA.value},
        {"doc": test_documents[3], "reason": RejectionReason.PROCESSING_ERROR.value},
    ]
    
    # Reject documents
    for item in rejection_data:
        response = client.post(
            f"/api/v1/documents/{item['doc']}/reject",
            json={
                "reason": item["reason"],
                "notes": f"Test rejection for statistics - {item['reason']}",
                "notify_owner": False
            }
        )
        assert response.status_code == status.HTTP_201_CREATED
    
    # Get rejection statistics
    response = client.get("/api/v1/documents/rejection-statistics")
    assert response.status_code == status.HTTP_200_OK
    
    stats_data = response.json()
    
    # Verify statistics structure
    required_fields = [
        "total_documents", "total_rejected", "rejection_rate",
        "rejection_by_reason", "rejection_trends", 
        "top_rejection_reasons", "recent_rejections"
    ]
    
    for field in required_fields:
        assert field in stats_data, f"Missing required field: {field}"
    
    # Verify statistics accuracy
    assert stats_data["total_rejected"] >= 4
    assert stats_data["rejection_rate"] >= 0.0
    assert stats_data["rejection_rate"] <= 100.0
    
    # Verify rejection by reason breakdown
    reason_counts = stats_data["rejection_by_reason"]
    assert reason_counts[RejectionReason.DATA_QUALITY_ISSUES.value] >= 2
    assert reason_counts[RejectionReason.INCOMPLETE_DATA.value] >= 1
    assert reason_counts[RejectionReason.PROCESSING_ERROR.value] >= 1
    
    # Verify recent rejections
    recent_rejections = stats_data["recent_rejections"]
    assert len(recent_rejections) >= 4
    
    for rejection in recent_rejections:
        assert "document_id" in rejection
        assert "rejection_reason" in rejection
        assert "rejected_at" in rejection
    
    logger.info("✓ Rejection statistics test passed with real database data")


# Test 5: Bulk Rejection Operations with Real Database
def test_bulk_rejection_real_database(rejection_test_suite, test_documents):
    """Test bulk rejection operations with real database updates."""
    logger.info("Testing bulk rejection with real database operations")
    
    # Use first 3 documents for bulk rejection
    bulk_doc_ids = test_documents[:3]
    
    # Verify all documents are in processing state initially
    for doc_id in bulk_doc_ids:
        assert rejection_test_suite.verify_document_status(doc_id, DocumentStatus.PROCESSING.value)
    
    # Prepare bulk rejection request
    bulk_request = {
        "document_ids": bulk_doc_ids,
        "reason": RejectionReason.INCOMPLETE_DATA.value,
        "notes": "Bulk rejection test - multiple documents with incomplete data",
        "notify_owners": False
    }
    
    # Execute bulk rejection
    response = client.post(
        "/api/v1/documents/bulk-reject",
        json=bulk_request
    )
    
    # Verify API response
    assert response.status_code == status.HTTP_200_OK
    response_data = response.json()
    
    assert response_data["status"] == "completed"
    assert response_data["total_requested"] == 3
    assert response_data["successful_rejections"] >= 3
    assert response_data["failed_rejections"] == 0
    assert len(response_data["successful_ids"]) >= 3
    
    # Verify database state changes for all documents
    for doc_id in bulk_doc_ids:
        assert rejection_test_suite.verify_document_status(doc_id, DocumentStatus.REJECTED.value)
        
        # Verify rejection record exists
        rejection_record = rejection_test_suite.verify_rejection_record_exists(doc_id)
        assert rejection_record is not None
        assert rejection_record["reason"] == RejectionReason.INCOMPLETE_DATA.value
    
    logger.info("✓ Bulk rejection test passed with real database verification")


# Test 6: Error Handling with Real Database Constraints
def test_error_handling_real_database(rejection_test_suite, test_documents):
    """Test error handling scenarios with real database constraints."""
    logger.info("Testing error handling with real database constraints")
    
    document_id = test_documents[0]
    
    # Test 1: Reject non-existent document
    fake_doc_id = "fake_document_id_12345"
    response = client.post(
        f"/api/v1/documents/{fake_doc_id}/reject",
        json={
            "reason": RejectionReason.DATA_QUALITY_ISSUES.value,
            "notes": "This should fail",
            "notify_owner": False
        }
    )
    
    assert response.status_code == status.HTTP_404_NOT_FOUND
    response_data = response.json()
    assert "not found" in response_data["detail"].lower()
    
    # Test 2: Reject already rejected document
    # First reject the document
    response = client.post(
        f"/api/v1/documents/{document_id}/reject",
        json={
            "reason": RejectionReason.DATA_QUALITY_ISSUES.value,
            "notes": "First rejection",
            "notify_owner": False
        }
    )
    assert response.status_code == status.HTTP_201_CREATED
    
    # Try to reject again
    response = client.post(
        f"/api/v1/documents/{document_id}/reject",
        json={
            "reason": RejectionReason.INCOMPLETE_DATA.value,
            "notes": "Second rejection attempt",
            "notify_owner": False
        }
    )
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    response_data = response.json()
    assert "already rejected" in response_data["detail"].lower()
    
    # Test 3: Unreject non-rejected document
    non_rejected_doc = test_documents[1]
    response = client.post(
        f"/api/v1/documents/{non_rejected_doc}/unreject",
        json={
            "reason": "This should fail",
            "notes": "Cannot unreject non-rejected document"
        }
    )
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    response_data = response.json()
    assert "not currently rejected" in response_data["detail"].lower()
    
    # Test 4: Invalid rejection reason
    response = client.post(
        f"/api/v1/documents/{test_documents[2]}/reject",
        json={
            "reason": "INVALID_REJECTION_REASON",
            "notes": "This should fail with invalid reason",
            "notify_owner": False
        }
    )
    
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    logger.info("✓ Error handling test passed with real database constraints")


# Test 7: End-to-End Rejection Workflow
def test_end_to_end_rejection_workflow(rejection_test_suite, test_documents):
    """Test complete end-to-end rejection workflow with real database."""
    logger.info("Testing end-to-end rejection workflow")
    
    document_id = test_documents[0]
    
    # Step 1: Verify initial state
    assert rejection_test_suite.verify_document_status(document_id, DocumentStatus.PROCESSING.value)
    
    # Step 2: Reject document
    reject_response = client.post(
        f"/api/v1/documents/{document_id}/reject",
        json={
            "reason": RejectionReason.DATA_QUALITY_ISSUES.value,
            "notes": "End-to-end test rejection",
            "notify_owner": False
        }
    )
    
    assert reject_response.status_code == status.HTTP_201_CREATED
    rejection_data = reject_response.json()
    rejection_id = rejection_data["rejection_id"]
    
    # Step 3: Verify rejection in database
    assert rejection_test_suite.verify_document_status(document_id, DocumentStatus.REJECTED.value)
    
    # Step 4: Get rejection history
    history_response = client.get(f"/api/v1/documents/{document_id}/rejection-history")
    assert history_response.status_code == status.HTTP_200_OK
    
    history_data = history_response.json()
    assert history_data["document_id"] == document_id
    assert len(history_data["history"]) >= 1
    assert history_data["current_status"] == DocumentStatus.REJECTED.value
    
    # Step 5: Get rejected documents list
    rejected_response = client.get("/api/v1/documents/rejected")
    assert rejected_response.status_code == status.HTTP_200_OK
    
    rejected_data = rejected_response.json()
    doc_found = False
    for doc in rejected_data["documents"]:
        if doc["document_id"] == document_id:
            doc_found = True
            assert doc["rejection_reason"] == RejectionReason.DATA_QUALITY_ISSUES.value
            break
    
    assert doc_found, "Rejected document not found in rejected documents list"
    
    # Step 6: Unreject document
    unreject_response = client.post(
        f"/api/v1/documents/{document_id}/unreject",
        json={
            "reason": "End-to-end test unreject",
            "notes": "Data quality issues resolved"
        }
    )
    
    assert unreject_response.status_code == status.HTTP_200_OK
    
    # Step 7: Verify final state
    final_props = get_node_properties(rejection_test_suite.neo4j_client, document_id)
    assert final_props["status"] != DocumentStatus.REJECTED.value
    
    # Step 8: Verify updated history
    final_history_response = client.get(f"/api/v1/documents/{document_id}/rejection-history")
    assert final_history_response.status_code == status.HTTP_200_OK
    
    final_history_data = final_history_response.json()
    assert len(final_history_data["history"]) >= 2  # Rejection + Unrerejection
    
    logger.info("✓ End-to-end rejection workflow test passed")


# Test 8: Database Performance with Multiple Operations
def test_database_performance_multiple_operations(rejection_test_suite):
    """Test database performance with multiple concurrent rejection operations."""
    logger.info("Testing database performance with multiple rejection operations")
    
    # Create more documents for performance testing
    performance_docs = rejection_test_suite.setup_test_documents(count=10)
    
    start_time = datetime.utcnow()
    
    # Perform multiple rejection operations
    successful_operations = 0
    
    for i, doc_id in enumerate(performance_docs):
        try:
            # Vary rejection reasons
            reasons = list(RejectionReason)
            reason = reasons[i % len(reasons)]
            
            response = client.post(
                f"/api/v1/documents/{doc_id}/reject",
                json={
                    "reason": reason.value,
                    "notes": f"Performance test rejection {i+1}",
                    "notify_owner": False
                }
            )
            
            if response.status_code == status.HTTP_201_CREATED:
                successful_operations += 1
                
        except Exception as e:
            logger.warning(f"Operation {i+1} failed: {e}")
    
    # Get statistics after operations
    stats_response = client.get("/api/v1/documents/rejection-statistics")
    assert stats_response.status_code == status.HTTP_200_OK
    
    end_time = datetime.utcnow()
    total_time = (end_time - start_time).total_seconds()
    
    # Performance assertions
    assert successful_operations >= 8  # At least 80% success rate
    assert total_time < 30.0  # Should complete within 30 seconds
    
    # Verify database consistency
    final_rejected_count = rejection_test_suite.count_rejected_documents()
    assert final_rejected_count >= successful_operations
    
    logger.info(f"✓ Performance test passed: {successful_operations}/{len(performance_docs)} operations completed in {total_time:.2f}s")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])