#!/usr/bin/env python3
"""
Test script for the EHS data extraction workflow.
Demonstrates querying existing Neo4j data and generating reports.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.src.workflows.extraction_workflow import DataExtractionWorkflow, QueryType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_facility_emissions_report():
    """Test facility emissions extraction and reporting."""
    logger.info("\n=== Testing Facility Emissions Report ===")
    
    # Neo4j connection details
    neo4j_uri = "bolt://localhost:7687"
    neo4j_username = "neo4j"
    neo4j_password = "EhsAI2024!"
    
    try:
        # Initialize extraction workflow
        workflow = DataExtractionWorkflow(
            neo4j_uri=neo4j_uri,
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password,
            llm_model="gpt-4",
            output_dir="./reports"
        )
        
        # Extract facility emissions data
        result = workflow.extract_data(
            query_type=QueryType.FACILITY_EMISSIONS,
            output_format="txt"
        )
        
        logger.info(f"Extraction Status: {result['status']}")
        logger.info(f"Report saved to: {result.get('report_file_path', 'Not saved')}")
        
        if result['errors']:
            logger.error(f"Errors: {result['errors']}")
        
        # Close connection
        workflow.close()
        
        return result
        
    except Exception as e:
        logger.error(f"Error in facility emissions test: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_utility_consumption_report():
    """Test utility consumption extraction and reporting."""
    logger.info("\n=== Testing Utility Consumption Report ===")
    
    # Neo4j connection details
    neo4j_uri = "bolt://localhost:7687"
    neo4j_username = "neo4j"
    neo4j_password = "EhsAI2024!"
    
    try:
        # Initialize extraction workflow
        workflow = DataExtractionWorkflow(
            neo4j_uri=neo4j_uri,
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password,
            llm_model="gpt-4",
            output_dir="./reports"
        )
        
        # Extract utility consumption data
        result = workflow.extract_data(
            query_type=QueryType.UTILITY_CONSUMPTION,
            parameters={
                "start_date": "2025-06-01",
                "end_date": "2025-06-30"
            },
            output_format="json"
        )
        
        logger.info(f"Extraction Status: {result['status']}")
        logger.info(f"Report saved to: {result.get('report_file_path', 'Not saved')}")
        
        if result['errors']:
            logger.error(f"Errors: {result['errors']}")
        
        # Close connection
        workflow.close()
        
        return result
        
    except Exception as e:
        logger.error(f"Error in utility consumption test: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_custom_queries():
    """Test custom query extraction."""
    logger.info("\n=== Testing Custom Query Extraction ===")
    
    # Neo4j connection details
    neo4j_uri = "bolt://localhost:7687"
    neo4j_username = "neo4j"
    neo4j_password = "EhsAI2024!"
    
    try:
        # Initialize extraction workflow
        workflow = DataExtractionWorkflow(
            neo4j_uri=neo4j_uri,
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password,
            llm_model="gpt-4",
            output_dir="./reports"
        )
        
        # Define custom queries
        custom_queries = [
            {
                "query": """
                MATCH (d:Document:UtilityBill)-[:EXTRACTED_TO]->(b:UtilityBill)-[:BILLED_TO]->(f:Facility)
                OPTIONAL MATCH (b)-[:RESULTED_IN]->(e:Emission)
                OPTIONAL MATCH (m:Meter)-[:MONITORS]->(f)
                RETURN d, b, f, e, m
                """,
                "parameters": {}
            },
            {
                "query": """
                MATCH (b:UtilityBill)
                RETURN b.billing_period_end as period,
                       b.total_kwh as consumption,
                       b.total_cost as cost
                ORDER BY b.billing_period_end DESC
                LIMIT 10
                """,
                "parameters": {}
            }
        ]
        
        # Extract with custom queries
        result = workflow.extract_data(
            query_type=QueryType.CUSTOM,
            queries=custom_queries,
            output_format="txt"
        )
        
        logger.info(f"Extraction Status: {result['status']}")
        logger.info(f"Report saved to: {result.get('report_file_path', 'Not saved')}")
        
        if result['errors']:
            logger.error(f"Errors: {result['errors']}")
        
        # Log summary
        if result['report_data']:
            summary = result['report_data']['summary']
            logger.info(f"\nQuery Summary:")
            logger.info(f"  Total Queries: {summary['total_queries']}")
            logger.info(f"  Successful: {summary['successful_queries']}")
            logger.info(f"  Total Records: {summary['total_records']}")
        
        # Close connection
        workflow.close()
        
        return result
        
    except Exception as e:
        logger.error(f"Error in custom queries test: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all extraction workflow tests."""
    logger.info("Starting EHS Data Extraction Workflow Tests")
    logger.info(f"Test started at: {datetime.now()}")
    logger.info("=" * 60)
    
    # Create reports directory
    Path("./reports").mkdir(exist_ok=True)
    
    # Run tests
    test_results = []
    
    # Test 1: Facility Emissions
    result1 = test_facility_emissions_report()
    test_results.append(("Facility Emissions", result1))
    
    # Test 2: Utility Consumption
    result2 = test_utility_consumption_report()
    test_results.append(("Utility Consumption", result2))
    
    # Test 3: Custom Queries
    result3 = test_custom_queries()
    test_results.append(("Custom Queries", result3))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    for test_name, result in test_results:
        if result and result['status'] == 'completed':
            logger.info(f"✓ {test_name}: SUCCESS")
            if result.get('report_file_path'):
                logger.info(f"  Report: {result['report_file_path']}")
        else:
            logger.info(f"✗ {test_name}: FAILED")
    
    logger.info("\nAll tests completed!")
    logger.info(f"Reports saved to: ./reports/")


if __name__ == "__main__":
    main()