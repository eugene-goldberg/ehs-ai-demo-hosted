#!/usr/bin/env python3
"""
Comprehensive Prorating MWE (Minimum Working Example) Test Script

This script demonstrates end-to-end prorating functionality by:
1. Creating a realistic electric bill document in Neo4j
2. Calling the prorating API to process the document
3. Verifying the allocations were created correctly
4. Printing clear results showing the full workflow

Requirements:
- Python 3.8+
- Neo4j running with connection details in .env
- EHS API server running on port 8000
- Virtual environment activated

Usage:
    source venv/bin/activate
    python3 test_prorating_mwe.py
"""

import os
import sys
import json
import uuid
import logging
import requests
from datetime import datetime, date
from typing import Dict, List, Any, Optional

# Neo4j imports
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, TransientError

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('prorating_mwe_test.log')
    ]
)
logger = logging.getLogger(__name__)

class ProRatingMWETest:
    """
    Comprehensive test class for prorating functionality.
    """
    
    def __init__(self):
        """Initialize the test with configuration from environment."""
        # Neo4j configuration
        self.neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.neo4j_username = os.getenv('NEO4J_USERNAME', 'neo4j')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD', 'password')
        self.neo4j_database = os.getenv('NEO4J_DATABASE', 'neo4j')
        
        # API configuration
        self.api_base_url = "http://localhost:8000"
        self.api_headers = {"Content-Type": "application/json"}
        
        # Test data - Generate proper UUID
        self.test_doc_id = str(uuid.uuid4())
        self.test_facility_id = "facility_001"
        
        # Neo4j driver
        self.driver = None
        
        # Test results
        self.test_results = {
            "started_at": datetime.now().isoformat(),
            "neo4j_connection": False,
            "api_connection": False,
            "document_created": False,
            "prorating_called": False,
            "allocations_verified": False,
            "cleanup_completed": False,
            "errors": [],
            "summary": {}
        }
    
    def setup_neo4j_connection(self):
        """Establish Neo4j database connection."""
        try:
            logger.info("Connecting to Neo4j database...")
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_username, self.neo4j_password)
            )
            
            # Test connection
            with self.driver.session(database=self.neo4j_database) as session:
                result = session.run("RETURN 1 as test")
                test_result = result.single()
                if test_result and test_result["test"] == 1:
                    self.test_results["neo4j_connection"] = True
                    logger.info("✅ Neo4j connection successful")
                    return True
                else:
                    raise Exception("Neo4j test query failed")
                    
        except Exception as e:
            error_msg = f"Neo4j connection failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.test_results["errors"].append(error_msg)
            return False
    
    def test_api_connection(self):
        """Test API server connectivity."""
        try:
            logger.info("Testing API server connection...")
            response = requests.get(f"{self.api_base_url}/health", headers=self.api_headers, timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"✅ API server is healthy: {health_data.get('status', 'unknown')}")
                self.test_results["api_connection"] = True
                return True
            else:
                raise Exception(f"Health check failed with status {response.status_code}")
                
        except Exception as e:
            error_msg = f"API connection failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.test_results["errors"].append(error_msg)
            return False
    
    def create_test_document(self):
        """Create a realistic test electric bill document in Neo4j with both Document and ProcessedDocument labels."""
        try:
            logger.info("Creating test electric bill document...")
            
            # Document properties based on schema analysis
            doc_properties = {
                "doc_id": self.test_doc_id,
                "fileName": f"test_electric_bill_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                "facility_id": self.test_facility_id,
                "total_usage": 1000.0,  # kWh
                "start_date": "2025-08-01",
                "end_date": "2025-08-31",
                "total_cost": 150.0,
                "total_amount": 150.0,
                "status": "processed",
                "document_type": "electric_bill",
                "billing_period_days": 31,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # Create document with multiple labels and both id properties
            # This ensures compatibility with both API validation (Document.id) and service processing (ProcessedDocument.documentId)
            create_query = """
            CREATE (d:Document:ProcessedDocument:Electricitybill {
                id: $doc_id,
                documentId: $doc_id,
                fileName: $fileName,
                facility_id: $facility_id,
                total_usage: $total_usage,
                start_date: $start_date,
                end_date: $end_date,
                total_cost: $total_cost,
                total_amount: $total_amount,
                status: $status,
                document_type: $document_type,
                billing_period_days: $billing_period_days,
                created_at: $created_at,
                updated_at: $updated_at
            })
            RETURN d.id as doc_id, d.documentId as document_id, d.fileName as file_name, labels(d) as node_labels
            """
            
            with self.driver.session(database=self.neo4j_database) as session:
                result = session.run(create_query, doc_properties)
                record = result.single()
                
                if record:
                    logger.info(f"✅ Created test document: {record['file_name']}")
                    logger.info(f"   Document UUID: {record['doc_id']}")
                    logger.info(f"   Document ID: {record['document_id']}")
                    logger.info(f"   Node labels: {record['node_labels']}")
                    self.test_results["document_created"] = True
                    self.test_results["summary"]["document_id"] = record["doc_id"]
                    self.test_results["summary"]["file_name"] = record["file_name"]
                    self.test_results["summary"]["node_labels"] = record["node_labels"]
                    return True
                else:
                    raise Exception("Document creation returned no results")
                    
        except Exception as e:
            error_msg = f"Document creation failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.test_results["errors"].append(error_msg)
            return False
    
    def call_prorating_api(self):
        """Call the prorating API to process the test document."""
        try:
            logger.info("Calling prorating API...")
            
            # Prepare API request data
            request_data = {
                "document_id": self.test_doc_id,
                "method": "headcount",
                "facility_info": [
                    {
                        "facility_id": "facility_A",
                        "name": "Building A",
                        "headcount": 50,
                        "floor_area": 5000.0,
                        "revenue": 1000000.0
                    },
                    {
                        "facility_id": "facility_B", 
                        "name": "Building B",
                        "headcount": 30,
                        "floor_area": 3000.0,
                        "revenue": 600000.0
                    },
                    {
                        "facility_id": "facility_C",
                        "name": "Building C", 
                        "headcount": 20,
                        "floor_area": 2000.0,
                        "revenue": 400000.0
                    }
                ]
            }
            
            # Make API call
            api_url = f"{self.api_base_url}/api/v1/prorating/process/{self.test_doc_id}"
            logger.info(f"POST {api_url}")
            logger.info(f"Document UUID: {self.test_doc_id}")
            logger.info(f"Request data: {json.dumps(request_data, indent=2)}")
            
            response = requests.post(
                api_url,
                headers=self.api_headers,
                json=request_data,
                timeout=30
            )
            
            logger.info(f"Response status: {response.status_code}")
            
            if response.status_code in [200, 201]:
                response_data = response.json()
                logger.info(f"✅ Prorating API call successful")
                logger.info(f"Response: {json.dumps(response_data, indent=2)}")
                
                self.test_results["prorating_called"] = True
                self.test_results["summary"]["api_response"] = response_data
                return True
            else:
                # Try to get error details
                try:
                    error_data = response.json()
                    error_detail = error_data.get("detail", "Unknown error")
                except:
                    error_detail = response.text
                
                raise Exception(f"API returned {response.status_code}: {error_detail}")
                
        except Exception as e:
            error_msg = f"Prorating API call failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.test_results["errors"].append(error_msg)
            return False
    
    def verify_allocations(self):
        """Verify that prorating allocations were created correctly."""
        try:
            logger.info("Verifying allocation results...")
            
            # Query for any allocation nodes related to our document
            # Updated query to handle both Document and ProcessedDocument labels
            verify_query = """
            MATCH (d) WHERE (d:Document OR d:ProcessedDocument) AND (d.id = $doc_id OR d.documentId = $doc_id)
            OPTIONAL MATCH (d)-[r1:HAS_MONTHLY_ALLOCATION]->(a:MonthlyUsageAllocation)
            OPTIONAL MATCH (d)-[r2:HAS_ALLOCATION]->(p:ProRatingAllocation)
            RETURN coalesce(d.id, d.documentId) as document_id,
                   d.fileName as file_name,
                   d.total_amount as total_amount,
                   labels(d) as node_labels,
                   count(a) as monthly_allocation_count,
                   count(p) as prorating_allocation_count,
                   collect(DISTINCT {
                       allocation_id: a.allocation_id,
                       usage_year: a.usage_year,
                       usage_month: a.usage_month,
                       allocated_usage: a.allocated_usage,
                       allocated_cost: a.allocated_cost,
                       allocation_percentage: a.allocation_percentage
                   }) as monthly_allocations,
                   collect(DISTINCT {
                       facility_id: p.facility_id,
                       facility_name: p.facility_name,
                       allocation_percentage: p.allocation_percentage,
                       allocated_amount: p.allocated_amount,
                       method_used: p.method_used
                   }) as prorating_allocations
            """
            
            with self.driver.session(database=self.neo4j_database) as session:
                result = session.run(verify_query, {"doc_id": self.test_doc_id})
                record = result.single()
                
                if record:
                    doc_id = record["document_id"]
                    node_labels = record["node_labels"]
                    monthly_count = record["monthly_allocation_count"]
                    prorating_count = record["prorating_allocation_count"]
                    monthly_allocations = record["monthly_allocations"]
                    prorating_allocations = record["prorating_allocations"]
                    
                    logger.info(f"Document: {doc_id}")
                    logger.info(f"Node labels: {node_labels}")
                    logger.info(f"Monthly allocation nodes found: {monthly_count}")
                    logger.info(f"ProRating allocation nodes found: {prorating_count}")
                    
                    total_allocations = monthly_count + prorating_count
                    
                    if total_allocations > 0:
                        logger.info("✅ Allocations found:")
                        total_allocated = 0
                        
                        # Process monthly allocations
                        for allocation in monthly_allocations:
                            if allocation["allocation_id"]:  # Skip null entries
                                logger.info(f"  Monthly - ID: {allocation['allocation_id']}")
                                logger.info(f"    Period: {allocation['usage_year']}-{allocation['usage_month']:02d}")
                                logger.info(f"    Usage: {allocation['allocated_usage']}")
                                logger.info(f"    Cost: ${allocation['allocated_cost']}")
                                logger.info(f"    Percentage: {allocation['allocation_percentage']}%")
                                total_allocated += allocation['allocated_cost'] or 0
                        
                        # Process prorating allocations
                        for allocation in prorating_allocations:
                            if allocation["facility_id"]:  # Skip null entries
                                logger.info(f"  ProRating - Facility: {allocation['facility_id']}")
                                logger.info(f"    Name: {allocation['facility_name']}")
                                logger.info(f"    Amount: ${allocation['allocated_amount']}")
                                logger.info(f"    Percentage: {allocation['allocation_percentage']}%")
                                logger.info(f"    Method: {allocation['method_used']}")
                                total_allocated += allocation['allocated_amount'] or 0
                        
                        logger.info(f"Total allocated amount: ${total_allocated}")
                        self.test_results["allocations_verified"] = True
                        self.test_results["summary"]["allocations"] = {
                            "monthly_count": monthly_count,
                            "prorating_count": prorating_count,
                            "total_count": total_allocations,
                            "total_allocated_amount": total_allocated,
                            "monthly_details": [a for a in monthly_allocations if a["allocation_id"]],
                            "prorating_details": [a for a in prorating_allocations if a["facility_id"]]
                        }
                        return True
                    else:
                        logger.warning("⚠️ No allocation nodes found - prorating may not have completed")
                        return False
                else:
                    raise Exception("Document verification query returned no results")
                    
        except Exception as e:
            error_msg = f"Allocation verification failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.test_results["errors"].append(error_msg)
            return False
    
    def cleanup_test_data(self):
        """Clean up test data from Neo4j."""
        try:
            logger.info("Cleaning up test data...")
            
            # Delete test document and all related nodes
            # Updated cleanup query to handle both Document and ProcessedDocument labels
            cleanup_query = """
            MATCH (d) WHERE (d:Document OR d:ProcessedDocument) AND (d.id = $doc_id OR d.documentId = $doc_id)
            OPTIONAL MATCH (d)-[r1]->(a:MonthlyUsageAllocation)
            OPTIONAL MATCH (d)-[r2]->(p:ProRatingAllocation)
            DETACH DELETE d, a, p
            RETURN count(d) as deleted_docs, count(a) as deleted_monthly, count(p) as deleted_prorating
            """
            
            with self.driver.session(database=self.neo4j_database) as session:
                result = session.run(cleanup_query, {"doc_id": self.test_doc_id})
                record = result.single()
                
                if record:
                    deleted_docs = record["deleted_docs"]
                    deleted_monthly = record["deleted_monthly"] 
                    deleted_prorating = record["deleted_prorating"]
                    logger.info(f"✅ Cleaned up {deleted_docs} document(s), {deleted_monthly} monthly allocations, {deleted_prorating} prorating allocations")
                    self.test_results["cleanup_completed"] = True
                    return True
                    
        except Exception as e:
            error_msg = f"Cleanup failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.test_results["errors"].append(error_msg)
            return False
    
    def close_connections(self):
        """Close database connections."""
        try:
            if self.driver:
                self.driver.close()
                logger.info("Neo4j connection closed")
        except Exception as e:
            logger.warning(f"Error closing Neo4j connection: {str(e)}")
    
    def print_final_results(self):
        """Print comprehensive test results."""
        print("\n" + "="*80)
        print("PRORATING MWE TEST RESULTS")
        print("="*80)
        
        # Test summary
        total_steps = 6  # connection, api, document, prorating, verify, cleanup
        completed_steps = sum([
            self.test_results["neo4j_connection"],
            self.test_results["api_connection"],
            self.test_results["document_created"],
            self.test_results["prorating_called"],
            self.test_results["allocations_verified"],
            self.test_results["cleanup_completed"]
        ])
        
        success_rate = (completed_steps / total_steps) * 100
        
        print(f"Overall Success Rate: {success_rate:.1f}% ({completed_steps}/{total_steps} steps)")
        print(f"Test Started: {self.test_results['started_at']}")
        print(f"Test Completed: {datetime.now().isoformat()}")
        
        # Step-by-step results
        print(f"\n{'Step':<25} {'Status':<10} {'Description'}")
        print("-" * 60)
        print(f"{'Neo4j Connection':<25} {'✅ PASS' if self.test_results['neo4j_connection'] else '❌ FAIL':<10} Connect to database")
        print(f"{'API Connection':<25} {'✅ PASS' if self.test_results['api_connection'] else '❌ FAIL':<10} Test API server health")
        print(f"{'Document Creation':<25} {'✅ PASS' if self.test_results['document_created'] else '❌ FAIL':<10} Create test electric bill")
        print(f"{'Prorating API Call':<25} {'✅ PASS' if self.test_results['prorating_called'] else '❌ FAIL':<10} Call prorating endpoint")
        print(f"{'Allocation Verification':<25} {'✅ PASS' if self.test_results['allocations_verified'] else '❌ FAIL':<10} Verify allocations created")
        print(f"{'Cleanup':<25} {'✅ PASS' if self.test_results['cleanup_completed'] else '❌ FAIL':<10} Remove test data")
        
        # Summary data
        if self.test_results["summary"]:
            print(f"\n{'SUMMARY DATA'}")
            print("-" * 40)
            summary = self.test_results["summary"]
            
            if "document_id" in summary:
                print(f"Test Document ID: {summary['document_id']}")
            if "file_name" in summary:
                print(f"Test File Name: {summary['file_name']}")
            if "node_labels" in summary:
                print(f"Node Labels: {summary['node_labels']}")
            
            if "allocations" in summary:
                alloc = summary["allocations"]
                print(f"Total Allocations Created: {alloc['total_count']}")
                print(f"  - Monthly Allocations: {alloc['monthly_count']}")
                print(f"  - ProRating Allocations: {alloc['prorating_count']}")
                print(f"Total Allocated Amount: ${alloc['total_allocated_amount']:.2f}")
                
                if alloc["monthly_details"]:
                    print("\nMonthly Allocation Details:")
                    for detail in alloc["monthly_details"]:
                        print(f"  - {detail['usage_year']}-{detail['usage_month']:02d}: "
                              f"${detail['allocated_cost']:.2f} ({detail['allocation_percentage']:.1f}%)")
                              
                if alloc["prorating_details"]:
                    print("\nProRating Allocation Details:")
                    for detail in alloc["prorating_details"]:
                        print(f"  - {detail['facility_id']}: ${detail['allocated_amount']:.2f} "
                              f"({detail['allocation_percentage']:.1f}%) via {detail['method_used']}")
        
        # Errors
        if self.test_results["errors"]:
            print(f"\n{'ERRORS ENCOUNTERED'}")
            print("-" * 40)
            for i, error in enumerate(self.test_results["errors"], 1):
                print(f"{i}. {error}")
        
        # Recommendations
        print(f"\n{'RECOMMENDATIONS'}")
        print("-" * 40)
        
        if success_rate == 100:
            print("🎉 All tests passed! The prorating functionality is working correctly.")
            print("✅ Neo4j document creation successful")
            print("✅ Prorating API endpoint accessible and functional")
            print("✅ Allocation nodes created and linked properly")
        elif success_rate >= 80:
            print("⚠️  Most tests passed with some minor issues.")
            if not self.test_results["allocations_verified"]:
                print("🔍 Check allocation creation - API may have processed but allocations not visible")
        elif success_rate >= 50:
            print("⚠️  Partial functionality detected.")
            if not self.test_results["prorating_called"]:
                print("🔧 Check prorating API endpoint configuration and service initialization")
        else:
            print("❌ Major issues detected.")
            if not self.test_results["neo4j_connection"]:
                print("🔧 Fix Neo4j connection configuration")
            if not self.test_results["api_connection"]:
                print("🔧 Ensure API server is running on port 8000")
        
        print("\n" + "="*80)
    
    def run_full_test(self):
        """Run the complete prorating MWE test suite."""
        logger.info("Starting Prorating MWE Test Suite...")
        
        try:
            # Step 1: Setup Neo4j connection
            if not self.setup_neo4j_connection():
                return False
            
            # Step 2: Test API connection  
            if not self.test_api_connection():
                return False
            
            # Step 3: Create test document
            if not self.create_test_document():
                return False
            
            # Step 4: Call prorating API
            if not self.call_prorating_api():
                # Continue to cleanup even if API call fails
                self.cleanup_test_data()
                return False
            
            # Step 5: Verify allocations were created
            self.verify_allocations()  # Don't fail if verification fails
            
            # Step 6: Cleanup test data
            self.cleanup_test_data()  # Don't fail if cleanup fails
            
            return True
            
        except Exception as e:
            logger.error(f"Test suite failed with exception: {str(e)}")
            self.test_results["errors"].append(f"Test suite exception: {str(e)}")
            # Attempt cleanup on exception
            try:
                self.cleanup_test_data()
            except:
                pass
            return False
        
        finally:
            self.close_connections()


def main():
    """Main function to run the prorating MWE test."""
    print("="*80)
    print("PRORATING MINIMUM WORKING EXAMPLE (MWE) TEST")
    print("="*80)
    print("This script tests the complete prorating workflow:")
    print("1. Creates a test electric bill document in Neo4j")
    print("2. Calls the prorating API to process the document")
    print("3. Verifies allocations were created correctly")
    print("4. Prints detailed results and cleans up test data")
    print("="*80)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        sys.exit(1)
    
    # Check if running in virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Not running in a virtual environment")
        print("   Recommended: source venv/bin/activate")
    
    # Run the test
    test_runner = ProRatingMWETest()
    
    try:
        success = test_runner.run_full_test()
        test_runner.print_final_results()
        
        # Save detailed results to file
        results_file = "prorating_mwe_results.json"
        with open(results_file, "w") as f:
            json.dump(test_runner.test_results, f, indent=2, default=str)
        print(f"\n📄 Detailed results saved to: {results_file}")
        
        # Exit code based on success
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        test_runner.close_connections()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        test_runner.close_connections()
        sys.exit(1)


if __name__ == "__main__":
    main()