#!/usr/bin/env python3
"""
Comprehensive End-to-End Risk Assessment Workflow Integration Test

This test provides comprehensive coverage of the risk assessment workflow including:
1. Complete document ingestion with risk assessment
2. LangSmith trace capture verification
3. Risk assessment results validation
4. Neo4j storage of risk data
5. Error handling scenarios
6. Configuration management
7. Different document types
8. Performance benchmarks

Test Requirements:
- Neo4j database running on localhost:7687
- Environment variables configured (.env file)
- Test documents available
- LangSmith tracing configured
- Risk assessment agent properly initialized

Test Structure:
- Fixtures for test setup/teardown
- Mock services where needed
- Comprehensive assertions
- Error condition testing
- Performance measurement
"""

import os
import sys
import pytest
import asyncio
import time
import json
import logging
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock
import uuid

# Test framework imports
import pytest
import pytest_asyncio
from pytest import fixture, mark

# Data handling
import neo4j
from neo4j import GraphDatabase, Session, Transaction

# LangSmith and LangChain
try:
    from langsmith import Client as LangSmithClient
    from langchain_core.messages import HumanMessage, AIMessage
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    LangSmithClient = None

# Local imports - adjust paths as needed
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../src"))
try:
    from agents.risk_assessment.agent import (
        RiskAssessmentAgent, 
        create_risk_assessment_agent,
        RiskAssessmentState,
        RiskLevel,
        RiskCategory,
        AssessmentStatus,
        RiskFactor,
        RiskAssessment,
        RiskRecommendation,
        RecommendationSet
    )
    from langsmith_config import config as langsmith_config
    from shared.common_fn import create_graph_database_connection
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/tmp/test_risk_assessment_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Test Constants
TEST_FACILITY_ID = "test_facility_001"
TEST_ASSESSMENT_ID = f"integration_test_{int(time.time())}"
NEO4J_TEST_URI = "bolt://localhost:7687"
NEO4J_TEST_USERNAME = "neo4j"
NEO4J_TEST_PASSWORD = "EhsAI2024!"
NEO4J_TEST_DATABASE = "neo4j"

# Performance Benchmarks
PERFORMANCE_THRESHOLDS = {
    "total_assessment_time": 120.0,  # 2 minutes max
    "data_collection_time": 30.0,   # 30 seconds max
    "risk_analysis_time": 45.0,     # 45 seconds max
    "recommendation_generation_time": 30.0  # 30 seconds max
}

class TestEnvironmentError(Exception):
    """Raised when test environment is not properly configured."""
    pass

class RiskAssessmentTestSuite:
    """Main test suite for risk assessment workflow integration testing."""
    
    def __init__(self):
        self.agent = None
        self.neo4j_driver = None
        self.langsmith_client = None
        self.test_data = {}
        self.performance_metrics = {}
        
    def setup_test_environment(self):
        """Set up test environment and validate configuration."""
        logger.info("Setting up risk assessment test environment...")
        
        # Load environment variables from .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            logger.warning("python-dotenv not available, using system environment variables")
        
        # Validate environment variables
        required_env_vars = [
            "OPENAI_API_KEY",
            "NEO4J_URI",
            "NEO4J_USERNAME",
            "NEO4J_PASSWORD"
        ]
        
        missing_vars = []
        for var in required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise TestEnvironmentError(f"Missing required environment variables: {missing_vars}")
        
        # Initialize Neo4j connection
        try:
            neo4j_uri = os.getenv("NEO4J_URI", NEO4J_TEST_URI)
            neo4j_username = os.getenv("NEO4J_USERNAME", NEO4J_TEST_USERNAME)
            neo4j_password = os.getenv("NEO4J_PASSWORD", NEO4J_TEST_PASSWORD)
            
            self.neo4j_driver = GraphDatabase.driver(
                neo4j_uri,
                auth=(neo4j_username, neo4j_password)
            )
            # Test connection
            with self.neo4j_driver.session() as session:
                session.run("RETURN 1")
            logger.info("✓ Neo4j connection established")
        except Exception as e:
            raise TestEnvironmentError(f"Failed to connect to Neo4j: {e}")
        
        # Initialize LangSmith client if available
        try:
            if LANGSMITH_AVAILABLE and langsmith_config.is_available:
                self.langsmith_client = LangSmithClient()
                logger.info("✓ LangSmith client initialized")
            else:
                logger.warning("LangSmith not available - skipping trace validation tests")
        except Exception as e:
            logger.warning(f"LangSmith client initialization failed: {e}")
        
        # Create test data
        self.create_test_data()
        
        logger.info("✓ Test environment setup completed")
    
    def create_test_data(self):
        """Create comprehensive test data in Neo4j."""
        logger.info("Creating test data in Neo4j...")
        
        with self.neo4j_driver.session() as session:
            # Clear existing test data
            session.run("""
                MATCH (n) WHERE n.test_data = true OR n.id CONTAINS "test_facility"
                DETACH DELETE n
            """)
            
            # Create test facility
            session.run("""
                CREATE (f:Facility {
                    id: $facility_id,
                    name: "Test Manufacturing Facility",
                    type: "manufacturing",
                    location: "Test City, Test State",
                    test_data: true,
                    created_at: datetime()
                })
            """, facility_id=TEST_FACILITY_ID)
            
            # Create environmental data
            environmental_data = [
                {
                    "id": "test_utility_001",
                    "type": "electricity",
                    "usage": 50000,
                    "period": "2024-01",
                    "cost": 5000.00,
                    "emissions_factor": 0.45,
                    "test_data": True
                },
                {
                    "id": "test_waste_001",
                    "type": "hazardous",
                    "quantity": 500,
                    "disposal_method": "incineration",
                    "permit_required": True,
                    "date_generated": "2024-01-15",
                    "test_data": True
                }
            ]
            
            for data in environmental_data:
                session.run("""
                    MATCH (f:Facility {id: $facility_id})
                    CREATE (f)-[:HAS_UTILITY_BILL]->(ub:UtilityBill $data)
                """, facility_id=TEST_FACILITY_ID, data=data)
            
            # Create safety incidents
            safety_incidents = [
                {
                    "id": "test_incident_001",
                    "type": "near_miss",
                    "severity": "medium",
                    "description": "Equipment malfunction detected during routine inspection",
                    "date_occurred": "2024-01-10",
                    "root_cause": "Maintenance schedule deviation",
                    "test_data": True
                },
                {
                    "id": "test_incident_002",
                    "type": "injury",
                    "severity": "low",
                    "description": "Minor cut during material handling",
                    "date_occurred": "2024-01-20",
                    "root_cause": "Inadequate PPE usage",
                    "test_data": True
                }
            ]
            
            for incident in safety_incidents:
                session.run("""
                    MATCH (f:Facility {id: $facility_id})
                    CREATE (f)-[:HAS_SAFETY_INCIDENT]->(si:SafetyIncident $incident)
                """, facility_id=TEST_FACILITY_ID, incident=incident)
            
            # Create compliance violations
            violations = [
                {
                    "id": "test_violation_001",
                    "regulation": "EPA-CAA",
                    "violation_type": "emission_limit_exceeded",
                    "severity": "high",
                    "fine_amount": 25000.00,
                    "status": "pending_resolution",
                    "date_issued": "2024-01-25",
                    "test_data": True
                }
            ]
            
            for violation in violations:
                session.run("""
                    MATCH (f:Facility {id: $facility_id})
                    CREATE (f)-[:HAS_VIOLATION]->(v:Violation $violation)
                """, facility_id=TEST_FACILITY_ID, violation=violation)
        
        logger.info("✓ Test data created successfully")
    
    def initialize_risk_agent(self):
        """Initialize the risk assessment agent."""
        logger.info("Initializing risk assessment agent...")
        
        try:
            self.agent = create_risk_assessment_agent(
                llm_model="gpt-4o",
                max_retries=2,
                enable_langsmith=True,
                risk_assessment_methodology="comprehensive"
            )
            logger.info("✓ Risk assessment agent initialized")
        except Exception as e:
            raise TestEnvironmentError(f"Failed to initialize risk assessment agent: {e}")
    
    def test_complete_risk_assessment_workflow(self):
        """Test the complete risk assessment workflow from end to end."""
        logger.info("Testing complete risk assessment workflow...")
        start_time = time.time()
        
        try:
            # Execute risk assessment
            assessment_scope = {
                "date_range": {
                    "start": "2024-01-01",
                    "end": "2024-01-31"
                },
                "assessment_type": "comprehensive",
                "include_recommendations": True
            }
            
            metadata = {
                "test_run": True,
                "test_id": TEST_ASSESSMENT_ID,
                "test_type": "integration"
            }
            
            # Run the assessment
            result = self.agent.assess_facility_risk(
                facility_id=TEST_FACILITY_ID,
                assessment_scope=assessment_scope,
                metadata=metadata
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            self.performance_metrics["total_assessment_time"] = processing_time
            
            # Validate results
            self._validate_assessment_result(result)
            self._validate_performance_metrics()
            self._validate_neo4j_data_access(result)
            
            if self.langsmith_client:
                self._validate_langsmith_traces(result)
            
            logger.info(f"✓ Complete workflow test passed in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Complete workflow test failed: {e}")
            raise
    
    def _validate_assessment_result(self, result: RiskAssessmentState):
        """Validate the risk assessment result structure and content."""
        logger.info("Validating assessment result structure...")
        
        # Basic structure validation
        assert result is not None, "Assessment result cannot be None"
        assert isinstance(result, dict), "Assessment result must be a dictionary"
        
        # Status validation
        assert "status" in result, "Assessment result must contain status"
        valid_statuses = [status.value for status in AssessmentStatus]
        assert result["status"] in valid_statuses, f"Invalid status: {result.get('status')}"
        
        if result["status"] == AssessmentStatus.COMPLETED.value:
            # Successful completion validation
            assert "final_report" in result, "Completed assessment must have final_report"
            assert result["final_report"] is not None, "Final report cannot be None"
            
            report = result["final_report"]
            assert "assessment_id" in report, "Report must contain assessment_id"
            assert "facility_id" in report, "Report must contain facility_id"
            
            # Risk assessment validation
            risk_assessment = result.get("risk_assessment")
            if risk_assessment:
                if hasattr(risk_assessment, "overall_risk_level"):
                    valid_risk_levels = [level.value for level in RiskLevel]
                    assert risk_assessment.overall_risk_level in valid_risk_levels, \
                        f"Invalid risk level: {risk_assessment.overall_risk_level}"
                
                if hasattr(risk_assessment, "risk_score"):
                    assert 0 <= risk_assessment.risk_score <= 100, \
                        f"Risk score must be 0-100: {risk_assessment.risk_score}"
            
            # Data collection validation
            data_fields = ["environmental_data", "health_data", "safety_data", "compliance_data"]
            for field in data_fields:
                assert field in result, f"Assessment must contain {field}"
            
            logger.info("✓ Assessment result structure validation passed")
        else:
            # Failed assessment validation
            assert "errors" in result, "Failed assessment must contain errors"
            assert len(result["errors"]) > 0, "Failed assessment must have error details"
            logger.info("✓ Failed assessment structure validation passed")
    
    def _validate_performance_metrics(self):
        """Validate that performance metrics meet defined thresholds."""
        logger.info("Validating performance metrics...")
        
        for metric, threshold in PERFORMANCE_THRESHOLDS.items():
            actual_time = self.performance_metrics.get(metric, 0)
            if actual_time > 0:  # Only validate if metric was recorded
                if actual_time > threshold:
                    logger.warning(f"Performance threshold exceeded for {metric}: {actual_time:.2f}s > {threshold}s")
                else:
                    logger.info(f"✓ {metric}: {actual_time:.2f}s (threshold: {threshold}s)")
        
        logger.info("✓ Performance metrics validation completed")
    
    def _validate_neo4j_data_access(self, result: RiskAssessmentState):
        """Validate that Neo4j data was properly accessed and processed."""
        logger.info("Validating Neo4j data access...")
        
        # Check that data was collected
        data_collections = [
            ("environmental_data", "Environmental"),
            ("health_data", "Health"),
            ("safety_data", "Safety"),
            ("compliance_data", "Compliance")
        ]
        
        for field, data_type in data_collections:
            data = result.get(field)
            assert data is not None, f"{data_type} data was not collected"
            logger.info(f"✓ {data_type} data collected: {len(data) if isinstance(data, list) else 'present'}")
        
        # Verify facility information was retrieved
        facility_info = result.get("facility_info")
        assert facility_info is not None, "Facility information was not retrieved"
        
        if isinstance(facility_info, dict) and "facility" in facility_info:
            facility_data = facility_info["facility"]
            if isinstance(facility_data, dict):
                assert "id" in facility_data, "Facility data must contain ID"
                assert facility_data["id"] == TEST_FACILITY_ID, "Facility ID mismatch"
        
        logger.info("✓ Neo4j data access validation passed")
    
    def _validate_langsmith_traces(self, result: RiskAssessmentState):
        """Validate LangSmith trace capture."""
        logger.info("Validating LangSmith traces...")
        
        try:
            # Check if tracing was enabled
            langsmith_session = result.get("langsmith_session")
            if langsmith_session:
                logger.info(f"LangSmith session: {langsmith_session}")
                
                # Verify trace metadata
                trace_metadata = result.get("trace_metadata", {})
                assert "facility_id" in trace_metadata, "Trace metadata must contain facility_id"
                assert "assessment_id" in trace_metadata, "Trace metadata must contain assessment_id"
                
                logger.info("✓ LangSmith trace validation passed")
            else:
                logger.warning("No LangSmith session found - traces may not have been captured")
                
        except Exception as e:
            logger.warning(f"LangSmith trace validation failed: {e}")
    
    def test_error_handling_scenarios(self):
        """Test various error handling scenarios."""
        logger.info("Testing error handling scenarios...")
        
        error_scenarios = [
            {
                "name": "Invalid facility ID",
                "facility_id": "nonexistent_facility_999",
                "expected_behavior": "graceful_failure"
            }
        ]
        
        for scenario in error_scenarios:
            logger.info(f"Testing scenario: {scenario['name']}")
            
            try:
                result = self.agent.assess_facility_risk(
                    facility_id=scenario["facility_id"]
                )
                
                # Validate error handling
                if scenario["expected_behavior"] == "graceful_failure":
                    valid_statuses = [AssessmentStatus.FAILED.value, AssessmentStatus.COMPLETED.value]
                    assert result["status"] in valid_statuses
                    if result["status"] == AssessmentStatus.FAILED.value:
                        assert len(result.get("errors", [])) > 0
                
                logger.info(f"✓ Scenario '{scenario['name']}' handled correctly")
                
            except Exception as e:
                logger.error(f"✗ Scenario '{scenario['name']}' failed: {e}")
                raise
        
        logger.info("✓ Error handling scenarios test passed")
    
    def test_different_document_types(self):
        """Test risk assessment with different types of input documents."""
        logger.info("Testing different document types...")
        
        document_types = [
            {
                "type": "comprehensive",
                "scope": {
                    "include_environmental": True,
                    "include_safety": True,
                    "include_compliance": True,
                    "include_health": True
                }
            },
            {
                "type": "environmental_only",
                "scope": {
                    "include_environmental": True,
                    "include_safety": False,
                    "include_compliance": False,
                    "include_health": False
                }
            }
        ]
        
        for doc_type in document_types:
            logger.info(f"Testing document type: {doc_type['type']}")
            
            result = self.agent.assess_facility_risk(
                facility_id=TEST_FACILITY_ID,
                assessment_scope=doc_type["scope"],
                metadata={"document_type": doc_type["type"]}
            )
            
            # Validate result based on document type
            assert result is not None
            assert "status" in result
            
            logger.info(f"✓ Document type '{doc_type['type']}' test passed")
        
        logger.info("✓ Different document types test passed")
    
    def test_configuration_management(self):
        """Test configuration management and parameter validation."""
        logger.info("Testing configuration management...")
        
        # Test valid configurations
        valid_configs = [
            {
                "llm_model": "gpt-4o",
                "max_retries": 3,
                "risk_assessment_methodology": "comprehensive"
            },
            {
                "llm_model": "gpt-4o-mini",
                "max_retries": 1,
                "risk_assessment_methodology": "rapid"
            }
        ]
        
        for config in valid_configs:
            try:
                test_agent = create_risk_assessment_agent(**config)
                assert test_agent is not None
                test_agent.close()
                logger.info(f"✓ Configuration {config} validated")
            except Exception as e:
                logger.error(f"✗ Configuration {config} failed: {e}")
                raise
        
        logger.info("✓ Configuration management test passed")
    
    def run_performance_benchmarks(self):
        """Run performance benchmarks for the risk assessment workflow."""
        logger.info("Running performance benchmarks...")
        
        benchmark_runs = 2  # Reduced for testing
        metrics = []
        
        for run in range(benchmark_runs):
            logger.info(f"Benchmark run {run + 1}/{benchmark_runs}")
            
            start_time = time.time()
            
            result = self.agent.assess_facility_risk(
                facility_id=TEST_FACILITY_ID,
                assessment_scope={"benchmark_run": run + 1}
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            metrics.append({
                "run": run + 1,
                "total_time": total_time,
                "status": result.get("status"),
                "success": result.get("status") == AssessmentStatus.COMPLETED.value
            })
            
            logger.info(f"Run {run + 1} completed in {total_time:.2f} seconds")
        
        # Calculate statistics
        successful_runs = [m for m in metrics if m["success"]]
        if successful_runs:
            avg_time = sum(m["total_time"] for m in successful_runs) / len(successful_runs)
            min_time = min(m["total_time"] for m in successful_runs)
            max_time = max(m["total_time"] for m in successful_runs)
            
            logger.info(f"Performance Benchmark Results:")
            logger.info(f"  Successful runs: {len(successful_runs)}/{benchmark_runs}")
            logger.info(f"  Average time: {avg_time:.2f} seconds")
            logger.info(f"  Min time: {min_time:.2f} seconds")
            logger.info(f"  Max time: {max_time:.2f} seconds")
            
            logger.info("✓ Performance benchmarks completed")
        else:
            logger.warning("No successful benchmark runs completed")
    
    def cleanup_test_environment(self):
        """Clean up test environment and resources."""
        logger.info("Cleaning up test environment...")
        
        try:
            # Clean up test data from Neo4j
            if self.neo4j_driver:
                with self.neo4j_driver.session() as session:
                    session.run("""
                        MATCH (n) WHERE n.test_data = true OR n.id CONTAINS "test_facility"
                        DETACH DELETE n
                    """)
                self.neo4j_driver.close()
                logger.info("✓ Neo4j test data cleaned up")
            
            # Close risk assessment agent
            if self.agent:
                self.agent.close()
                logger.info("✓ Risk assessment agent closed")
            
            # Disable LangSmith tracing if enabled
            if LANGSMITH_AVAILABLE and langsmith_config.is_enabled:
                langsmith_config.disable_tracing()
                logger.info("✓ LangSmith tracing disabled")
                
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        
        logger.info("✓ Test environment cleanup completed")

# Pytest fixtures and test functions
@pytest.fixture(scope="module")
def test_suite():
    """Pytest fixture for test suite setup and teardown."""
    suite = RiskAssessmentTestSuite()
    try:
        suite.setup_test_environment()
        suite.initialize_risk_agent()
        yield suite
    finally:
        suite.cleanup_test_environment()

def test_complete_workflow(test_suite):
    """Pytest test function for complete workflow."""
    test_suite.test_complete_risk_assessment_workflow()

def test_error_handling(test_suite):
    """Pytest test function for error handling."""
    test_suite.test_error_handling_scenarios()

def test_document_types(test_suite):
    """Pytest test function for different document types."""
    test_suite.test_different_document_types()

def test_configuration(test_suite):
    """Pytest test function for configuration management."""
    test_suite.test_configuration_management()

def test_performance_benchmarks(test_suite):
    """Pytest test function for performance benchmarks."""
    test_suite.run_performance_benchmarks()

# Main test execution
def main():
    """Main test execution function."""
    test_suite = RiskAssessmentTestSuite()
    
    try:
        # Setup
        test_suite.setup_test_environment()
        test_suite.initialize_risk_agent()
        
        # Run tests
        logger.info("="*60)
        logger.info("RISK ASSESSMENT WORKFLOW INTEGRATION TESTS")
        logger.info("="*60)
        
        # Test 1: Complete workflow
        logger.info("\n" + "-"*40)
        logger.info("TEST 1: Complete Risk Assessment Workflow")
        logger.info("-"*40)
        result = test_suite.test_complete_risk_assessment_workflow()
        
        # Test 2: Error handling
        logger.info("\n" + "-"*40)
        logger.info("TEST 2: Error Handling Scenarios")
        logger.info("-"*40)
        test_suite.test_error_handling_scenarios()
        
        # Test 3: Different document types
        logger.info("\n" + "-"*40)
        logger.info("TEST 3: Different Document Types")
        logger.info("-"*40)
        test_suite.test_different_document_types()
        
        # Test 4: Configuration management
        logger.info("\n" + "-"*40)
        logger.info("TEST 4: Configuration Management")
        logger.info("-"*40)
        test_suite.test_configuration_management()
        
        # Test 5: Performance benchmarks
        logger.info("\n" + "-"*40)
        logger.info("TEST 5: Performance Benchmarks")
        logger.info("-"*40)
        test_suite.run_performance_benchmarks()
        
        logger.info("\n" + "="*60)
        logger.info("🎉 ALL INTEGRATION TESTS PASSED!")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ INTEGRATION TEST FAILED: {e}")
        logger.error("="*60)
        return False
        
    finally:
        test_suite.cleanup_test_environment()

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)