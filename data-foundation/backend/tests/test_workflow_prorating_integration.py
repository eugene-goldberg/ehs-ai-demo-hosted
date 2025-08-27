import pytest
import logging
import asyncio
from datetime import datetime, date
from decimal import Decimal
import uuid
from typing import Dict, List, Any

from neo4j import GraphDatabase
from unittest.mock import patch
import requests
import json

# Configure logging for test tracking
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestWorkflowProratingIntegration:
    """
    Comprehensive integration test for prorating functionality when electric bills 
    are processed through the workflow.
    
    Tests the complete flow from document creation through prorating calculations
    and verification of results in Neo4j.
    """
    
    @pytest.fixture(scope="class")
    def neo4j_driver(self):
        """Create Neo4j driver for test database operations."""
        driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "EhsAI2024!")
        )
        yield driver
        driver.close()
    
    @pytest.fixture(scope="function")
    def test_document_id(self):
        """Generate unique document ID for each test."""
        return str(uuid.uuid4())
    
    @pytest.fixture(scope="function")
    def cleanup_test_data(self, neo4j_driver, test_document_id):
        """Cleanup test data after each test."""
        yield  # Run the test first
        
        logger.info(f"Cleaning up test data for document {test_document_id}")
        with neo4j_driver.session() as session:
            # Clean up MonthlyUsageAllocation nodes - check both id formats and both relationship types
            session.run(
                """
                MATCH (d:Document)
                WHERE d.id = $doc_id OR d.document_id = $doc_id OR d.documentId = $doc_id
                MATCH (d)-[r:HAS_ALLOCATION|HAS_MONTHLY_ALLOCATION]->(a:MonthlyUsageAllocation)
                DETACH DELETE a
                """,
                doc_id=test_document_id
            )
            
            # Clean up Document node - check both id formats
            session.run(
                """
                MATCH (d:Document)
                WHERE d.id = $doc_id OR d.document_id = $doc_id OR d.documentId = $doc_id
                DETACH DELETE d
                """,
                doc_id=test_document_id
            )
            
            # Clean up test facilities (with test prefix)
            session.run(
                """
                MATCH (f:Facility)
                WHERE f.facility_name STARTS WITH 'TEST_'
                DETACH DELETE f
                """
            )
        logger.info("Test data cleanup completed")
    
    def create_test_electric_bill(self, neo4j_driver, document_id: str) -> Dict[str, Any]:
        """Create a test electric bill document in Neo4j with API-expected properties."""
        logger.info(f"Creating test electric bill document: {document_id}")
        
        bill_data = {
            'id': document_id,  # For API validation
            'documentId': document_id,  # For service processing
            'fileName': f'test_electric_bill_{document_id[:8]}.pdf',  # Added fileName
            'document_type': 'electric_bill',
            'total_cost': 1500.00,  # Changed from total_amount to total_cost
            'total_usage': 3000.0,  # Changed from total_kwh to total_usage
            'start_date': '2024-01-01',  # Changed from billing_period_start
            'end_date': '2024-01-31',  # Changed from billing_period_end
            'status': 'processed',  # Added status property
            'account_number': 'TEST_ACCT_001',
            'service_address': '123 Test Street, Test City, TS 12345',
            'created_at': datetime.now().isoformat()
        }
        
        with neo4j_driver.session() as session:
            # Create document with ProcessedDocument and ElectricBill labels
            session.run(
                """
                CREATE (d:Document:ProcessedDocument:ElectricBill {
                    id: $id,
                    documentId: $documentId,
                    fileName: $fileName,
                    document_type: $document_type,
                    total_cost: $total_cost,
                    total_usage: $total_usage,
                    start_date: $start_date,
                    end_date: $end_date,
                    status: $status,
                    account_number: $account_number,
                    service_address: $service_address,
                    created_at: $created_at
                })
                """,
                **bill_data
            )
        
        logger.info(f"Test electric bill created successfully: {document_id}")
        return bill_data
    
    def create_test_facilities(self, neo4j_driver) -> List[Dict[str, Any]]:
        """Create test facility data required for prorating."""
        logger.info("Creating test facility data")
        
        facilities = [
            {
                'facility_id': 'TEST_FAC_001',
                'facility_name': 'TEST_Building A',
                'headcount': 50,
                'floor_area': 10000.0,
                'revenue': 500000.00
            },
            {
                'facility_id': 'TEST_FAC_002', 
                'facility_name': 'TEST_Building B',
                'headcount': 30,
                'floor_area': 8000.0,
                'revenue': 300000.00
            },
            {
                'facility_id': 'TEST_FAC_003',
                'facility_name': 'TEST_Building C',
                'headcount': 20,
                'floor_area': 5000.0,
                'revenue': 200000.00
            }
        ]
        
        with neo4j_driver.session() as session:
            for facility in facilities:
                session.run(
                    """
                    CREATE (f:Facility {
                        facility_id: $facility_id,
                        facility_name: $facility_name,
                        headcount: $headcount,
                        floor_area: $floor_area,
                        revenue: $revenue
                    })
                    """,
                    **facility
                )
        
        logger.info(f"Created {len(facilities)} test facilities")
        return facilities
    
    def call_prorating_api(self, document_id: str, allocation_method: str, facilities: List[Dict[str, Any]]) -> requests.Response:
        """Call the prorating API directly."""
        logger.info(f"Calling prorating API for document {document_id} with method {allocation_method}")
        
        # Convert facilities to the expected API format
        facility_info = []
        for facility in facilities:
            facility_info.append({
                'facility_id': facility['facility_id'],
                'name': facility['facility_name'],
                'headcount': facility['headcount'],
                'floor_area': facility['floor_area'],
                'revenue': facility['revenue']
            })
        
        payload = {
            'document_id': document_id,
            'method': allocation_method,
            'facility_info': facility_info
        }
        
        response = requests.post(
            f'http://localhost:8000/api/v1/prorating/process/{document_id}',
            json=payload,
            timeout=30
        )
        
        logger.info(f"Prorating API response: {response.status_code} - {response.text[:200]}")
        return response
    
    def verify_monthly_usage_allocations(self, neo4j_driver, document_id: str, 
                                       expected_count: int, allocation_method: str) -> List[Dict]:
        """Verify that MonthlyUsageAllocation nodes are created correctly."""
        logger.info(f"Verifying MonthlyUsageAllocation nodes for document {document_id}")
        
        with neo4j_driver.session() as session:
            # Check all possible id formats for compatibility and use the correct relationship name
            result = session.run(
                """
                MATCH (d:Document)
                WHERE d.id = $doc_id OR d.document_id = $doc_id OR d.documentId = $doc_id
                MATCH (d)-[:HAS_MONTHLY_ALLOCATION]->(a:MonthlyUsageAllocation)
                RETURN a.facility_id as facility_id,
                       a.allocated_cost as allocated_cost,
                       a.allocated_usage as allocated_usage,
                       a.allocation_percentage as allocation_percentage,
                       a.allocation_method as allocation_method
                ORDER BY a.facility_id
                """,
                doc_id=document_id
            )
            
            allocations = [dict(record) for record in result]
            
        logger.info(f"Found {len(allocations)} allocation records")
        
        # Verify count
        assert len(allocations) == expected_count, f"Expected {expected_count} allocations, got {len(allocations)}"
        
        # Verify allocation method
        for allocation in allocations:
            assert allocation['allocation_method'] == allocation_method, f"Expected method {allocation_method}, got {allocation['allocation_method']}"
        
        # Verify total percentages sum to 100%
        total_percentage = sum(float(a['allocation_percentage']) for a in allocations)
        assert abs(total_percentage - 100.0) < 0.01, f"Total percentage should be 100%, got {total_percentage}"
        
        # Verify total amounts sum to original bill amount - use updated property name
        total_amount = sum(float(a['allocated_cost']) for a in allocations)
        assert abs(total_amount - 1500.00) < 0.01, f"Total amount should be 1500.00, got {total_amount}"
        
        logger.info("MonthlyUsageAllocation verification completed successfully")
        return allocations
    
    def calculate_expected_allocations(self, facilities: List[Dict], allocation_method: str, 
                                     total_amount: float, total_kwh: float) -> Dict[str, Dict]:
        """Calculate expected allocation values for verification."""
        logger.info(f"Calculating expected allocations using method: {allocation_method}")
        
        if allocation_method == 'headcount':
            total_metric = sum(f['headcount'] for f in facilities)
            metric_key = 'headcount'
        elif allocation_method == 'floor_area':
            total_metric = sum(f['floor_area'] for f in facilities)
            metric_key = 'floor_area'
        elif allocation_method == 'revenue':
            total_metric = sum(f['revenue'] for f in facilities)
            metric_key = 'revenue'
        else:
            raise ValueError(f"Unknown allocation method: {allocation_method}")
        
        expected = {}
        for facility in facilities:
            percentage = (facility[metric_key] / total_metric) * 100
            allocated_amount = (facility[metric_key] / total_metric) * total_amount
            allocated_kwh = (facility[metric_key] / total_metric) * total_kwh
            
            expected[facility['facility_id']] = {
                'percentage': percentage,
                'allocated_cost': allocated_amount,  # Updated property name
                'allocated_usage': allocated_kwh    # Updated property name
            }
        
        logger.info(f"Expected allocations calculated for {len(expected)} facilities")
        return expected
    
    def test_headcount_allocation(self, neo4j_driver, test_document_id, cleanup_test_data):
        """Test prorating using headcount allocation method."""
        logger.info("=== Testing Headcount Allocation ===")
        
        # Setup test data
        bill_data = self.create_test_electric_bill(neo4j_driver, test_document_id)
        facilities = self.create_test_facilities(neo4j_driver)
        
        # Call prorating API
        response = self.call_prorating_api(test_document_id, 'headcount', facilities)
        assert response.status_code == 200, f"API call failed: {response.text}"
        
        # Verify allocations were created
        allocations = self.verify_monthly_usage_allocations(
            neo4j_driver, test_document_id, len(facilities), 'headcount'
        )
        
        # Calculate expected values and verify accuracy - use updated property names
        expected = self.calculate_expected_allocations(
            facilities, 'headcount', bill_data['total_cost'], bill_data['total_usage']
        )
        
        for allocation in allocations:
            facility_id = allocation['facility_id']
            expected_values = expected[facility_id]
            
            assert abs(float(allocation['allocation_percentage']) - expected_values['percentage']) < 0.01
            assert abs(float(allocation['allocated_cost']) - expected_values['allocated_cost']) < 0.01
            assert abs(float(allocation['allocated_usage']) - expected_values['allocated_usage']) < 0.01
        
        logger.info("Headcount allocation test completed successfully")
    
    def test_floor_area_allocation(self, neo4j_driver, test_document_id, cleanup_test_data):
        """Test prorating using floor area allocation method."""
        logger.info("=== Testing Floor Area Allocation ===")
        
        # Setup test data
        bill_data = self.create_test_electric_bill(neo4j_driver, test_document_id)
        facilities = self.create_test_facilities(neo4j_driver)
        
        # Call prorating API
        response = self.call_prorating_api(test_document_id, 'floor_area', facilities)
        assert response.status_code == 200, f"API call failed: {response.text}"
        
        # Verify allocations were created
        allocations = self.verify_monthly_usage_allocations(
            neo4j_driver, test_document_id, len(facilities), 'floor_area'
        )
        
        # Calculate expected values and verify accuracy - use updated property names
        expected = self.calculate_expected_allocations(
            facilities, 'floor_area', bill_data['total_cost'], bill_data['total_usage']
        )
        
        for allocation in allocations:
            facility_id = allocation['facility_id']
            expected_values = expected[facility_id]
            
            assert abs(float(allocation['allocation_percentage']) - expected_values['percentage']) < 0.01
            assert abs(float(allocation['allocated_cost']) - expected_values['allocated_cost']) < 0.01
            assert abs(float(allocation['allocated_usage']) - expected_values['allocated_usage']) < 0.01
        
        logger.info("Floor area allocation test completed successfully")
    
    def test_revenue_allocation(self, neo4j_driver, test_document_id, cleanup_test_data):
        """Test prorating using revenue allocation method."""
        logger.info("=== Testing Revenue Allocation ===")
        
        # Setup test data
        bill_data = self.create_test_electric_bill(neo4j_driver, test_document_id)
        facilities = self.create_test_facilities(neo4j_driver)
        
        # Call prorating API
        response = self.call_prorating_api(test_document_id, 'revenue', facilities)
        assert response.status_code == 200, f"API call failed: {response.text}"
        
        # Verify allocations were created
        allocations = self.verify_monthly_usage_allocations(
            neo4j_driver, test_document_id, len(facilities), 'revenue'
        )
        
        # Calculate expected values and verify accuracy - use updated property names
        expected = self.calculate_expected_allocations(
            facilities, 'revenue', bill_data['total_cost'], bill_data['total_usage']
        )
        
        for allocation in allocations:
            facility_id = allocation['facility_id']
            expected_values = expected[facility_id]
            
            assert abs(float(allocation['allocation_percentage']) - expected_values['percentage']) < 0.01
            assert abs(float(allocation['allocated_cost']) - expected_values['allocated_cost']) < 0.01
            assert abs(float(allocation['allocated_usage']) - expected_values['allocated_usage']) < 0.01
        
        logger.info("Revenue allocation test completed successfully")
    
    def test_nonexistent_document_error(self, neo4j_driver, cleanup_test_data):
        """Test error handling for nonexistent document."""
        logger.info("=== Testing Nonexistent Document Error Handling ===")
        
        fake_document_id = str(uuid.uuid4())
        facilities = [
            {
                'facility_id': 'TEST_FAC_001',
                'facility_name': 'TEST_Building A',
                'headcount': 50,
                'floor_area': 10000.0,
                'revenue': 500000.00
            }
        ]
        
        # Try to prorate nonexistent document
        response = self.call_prorating_api(fake_document_id, 'headcount', facilities)
        
        # Should return error status
        assert response.status_code in [404, 400], f"Expected error status, got {response.status_code}"
        
        logger.info("Nonexistent document error handling test completed successfully")
    
    def test_invalid_allocation_method_error(self, neo4j_driver, test_document_id, cleanup_test_data):
        """Test error handling for invalid allocation method."""
        logger.info("=== Testing Invalid Allocation Method Error Handling ===")
        
        # Setup test data
        self.create_test_electric_bill(neo4j_driver, test_document_id)
        facilities = self.create_test_facilities(neo4j_driver)
        
        # Try invalid allocation method
        response = self.call_prorating_api(test_document_id, 'invalid_method', facilities)
        
        # Should return error status (422 for validation errors)
        assert response.status_code in [400, 422], f"Expected error status, got {response.status_code}"
        
        logger.info("Invalid allocation method error handling test completed successfully")
    
    def test_no_facilities_error(self, neo4j_driver, test_document_id, cleanup_test_data):
        """Test error handling when no facilities are provided."""
        logger.info("=== Testing No Facilities Error Handling ===")
        
        # Create document but no facilities
        self.create_test_electric_bill(neo4j_driver, test_document_id)
        
        # Try to prorate with empty facilities list
        response = self.call_prorating_api(test_document_id, 'headcount', [])
        
        # Should return error status or handle gracefully
        # The exact behavior depends on implementation - could be 400, 422, or 200 with empty results
        assert response.status_code in [200, 400, 422], f"Unexpected status code: {response.status_code}"
        
        if response.status_code == 200:
            # If successful, verify no allocations were created
            allocations = []
            with neo4j_driver.session() as session:
                result = session.run(
                    """
                    MATCH (d:Document)
                    WHERE d.id = $doc_id OR d.document_id = $doc_id OR d.documentId = $doc_id
                    MATCH (d)-[:HAS_MONTHLY_ALLOCATION]->(a:MonthlyUsageAllocation)
                    RETURN count(a) as count
                    """,
                    doc_id=test_document_id
                )
                count = result.single()['count']
                assert count == 0, f"Expected 0 allocations, got {count}"
        
        logger.info("No facilities error handling test completed successfully")
    
    def test_duplicate_prorating_handling(self, neo4j_driver, test_document_id, cleanup_test_data):
        """Test handling of duplicate prorating requests."""
        logger.info("=== Testing Duplicate Prorating Handling ===")
        
        # Setup test data
        bill_data = self.create_test_electric_bill(neo4j_driver, test_document_id)
        facilities = self.create_test_facilities(neo4j_driver)
        
        # First prorating call
        response1 = self.call_prorating_api(test_document_id, 'headcount', facilities)
        assert response1.status_code == 200, f"First API call failed: {response1.text}"
        
        # Verify first set of allocations
        allocations1 = self.verify_monthly_usage_allocations(
            neo4j_driver, test_document_id, len(facilities), 'headcount'
        )
        
        # Second prorating call (duplicate)
        response2 = self.call_prorating_api(test_document_id, 'headcount', facilities)
        
        # Should handle gracefully - either update existing or return error
        assert response2.status_code in [200, 400, 409], f"Unexpected duplicate handling: {response2.status_code}"
        
        # Verify no duplicate allocations were created
        with neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (d:Document)
                WHERE d.id = $doc_id OR d.document_id = $doc_id OR d.documentId = $doc_id
                MATCH (d)-[:HAS_MONTHLY_ALLOCATION]->(a:MonthlyUsageAllocation)
                RETURN count(a) as count
                """,
                doc_id=test_document_id
            )
            count = result.single()['count']
            assert count == len(facilities), f"Expected {len(facilities)} allocations after duplicate, got {count}"
        
        logger.info("Duplicate prorating handling test completed successfully")

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main(["-v", "-s", __file__])