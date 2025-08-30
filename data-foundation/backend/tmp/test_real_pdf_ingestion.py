#!/usr/bin/env python3
"""
Real PDF Ingestion Test Script

This script tests the complete document ingestion workflow including:
1. PDF parsing with LlamaParse
2. Data extraction and processing
3. Risk assessment (if enabled)
4. Neo4j database storage

Test Configuration:
- Uses document-1.pdf (46KB) as test file
- Enables detailed logging with low recursion limit
- Tests with risk assessment enabled
- Captures exact recursion errors if they occur
- Simulates actual API ingestion workflow

Usage:
    cd /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend
    python3 tmp/test_real_pdf_ingestion.py
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import asyncio

# Increase recursion limit to handle deep import chains
sys.setrecursionlimit(5000)

# Set up the Python path to include the backend src directory
backend_dir = Path(__file__).parent.parent.absolute()
src_dir = backend_dir / 'src'
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(src_dir))

# Set environment variables before imports
os.environ.setdefault('PYTHONPATH', str(src_dir))
os.environ.setdefault('AUDIT_TRAIL_STORAGE_PATH', '/tmp/audit_trail_storage')

# Load environment variables first
from dotenv import load_dotenv
load_dotenv(backend_dir / '.env')

# Now import after path and environment setup
try:
    from ehs_workflows.ingestion_workflow_with_risk_assessment import (
        RiskAssessmentIntegratedWorkflow,
        DocumentStateWithRisk
    )
    from shared.common_fn import create_graph_database_connection
    from langsmith_config import config as langsmith_config
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running this from the backend directory and all dependencies are installed.")
    sys.exit(1)

# Test configuration
TEST_CONFIG = {
    "pdf_file": "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/data/document-1.pdf",
    "enable_risk_assessment": True,
    "enable_detailed_logging": True,
    "log_file": "/tmp/real_pdf_ingestion_test.log",
    "results_file": "/tmp/real_pdf_ingestion_results.json",
    "timeout_seconds": 300,  # 5 minute timeout
}

def setup_comprehensive_logging():
    """Set up comprehensive logging with detailed output."""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Set up file handler
    file_handler = logging.FileHandler(TEST_CONFIG["log_file"], mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Set specific logger levels
    logging.getLogger('neo4j').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.INFO)
    logging.getLogger('llamaparse').setLevel(logging.DEBUG)
    
    return logging.getLogger(__name__)

def check_environment_variables() -> bool:
    """Check if all required environment variables are set."""
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API for LLM operations',
        'LLAMA_PARSE_API_KEY': 'LlamaParse API for PDF parsing',
        'NEO4J_URI': 'Neo4j database connection',
        'NEO4J_USERNAME': 'Neo4j username',
        'NEO4J_PASSWORD': 'Neo4j password',
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value:
            missing_vars.append(f"{var} ({description})")
        else:
            logger.info(f"✅ {var}: {'*' * min(len(value), 8)}...")
    
    if missing_vars:
        logger.error(f"❌ Missing required environment variables:")
        for var in missing_vars:
            logger.error(f"   - {var}")
        return False
    
    return True

def check_test_file() -> bool:
    """Check if the test PDF file exists and is readable."""
    test_file_path = Path(TEST_CONFIG["pdf_file"])
    
    if not test_file_path.exists():
        logger.error(f"❌ Test file not found: {test_file_path}")
        return False
    
    if not test_file_path.is_file():
        logger.error(f"❌ Test path is not a file: {test_file_path}")
        return False
    
    try:
        file_size = test_file_path.stat().st_size
        logger.info(f"✅ Test file found: {test_file_path.name} ({file_size:,} bytes)")
        return True
    except Exception as e:
        logger.error(f"❌ Cannot access test file: {e}")
        return False

def test_neo4j_connection() -> Optional[Any]:
    """Test Neo4j database connection."""
    try:
        uri = os.getenv('NEO4J_URI')
        username = os.getenv('NEO4J_USERNAME')
        password = os.getenv('NEO4J_PASSWORD')
        database = os.getenv('NEO4J_DATABASE', 'neo4j')
        
        logger.info(f"🔗 Testing Neo4j connection to {uri}")
        
        graph = create_graph_database_connection(uri, username, password, database)
        
        # Test with a simple query
        result = graph.query("RETURN 'Connection successful' as status, datetime() as timestamp")
        
        if result:
            logger.info("✅ Neo4j connection successful")
            timestamp = result[0].get('timestamp') if result else 'unknown'
            logger.info(f"   Database timestamp: {timestamp}")
            return graph
        else:
            logger.error("❌ Neo4j query returned no results")
            return None
            
    except Exception as e:
        logger.error(f"❌ Neo4j connection failed: {e}")
        logger.debug(traceback.format_exc())
        return None

def create_test_facility(graph, facility_id: str = "TEST_FACILITY_PDF_001") -> bool:
    """Create a test facility for document ingestion."""
    try:
        logger.info(f"🏢 Creating test facility: {facility_id}")
        
        create_query = """
        MERGE (f:Facility {facility_id: $facility_id})
        SET f.name = $facility_name,
            f.type = 'Test Manufacturing',
            f.location = 'Test Location for PDF Ingestion',
            f.created_date = datetime(),
            f.test_facility = true
        RETURN f.facility_id as id, f.name as name
        """
        
        result = graph.query(create_query, {
            "facility_id": facility_id,
            "facility_name": f"Test Facility {facility_id}"
        })
        
        if result:
            logger.info(f"✅ Test facility created: {result[0]['id']}")
            return True
        else:
            logger.error("❌ Failed to create test facility")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error creating test facility: {e}")
        logger.debug(traceback.format_exc())
        return False

def run_pdf_ingestion_test() -> Dict[str, Any]:
    """Run the complete PDF ingestion test with risk assessment."""
    
    test_results = {
        "test_started": datetime.utcnow().isoformat(),
        "test_config": TEST_CONFIG.copy(),
        "status": "starting",
        "steps_completed": [],
        "errors": [],
        "warnings": [],
        "processing_times": {},
        "workflow_results": {},
        "recursion_error_detected": False
    }
    
    try:
        logger.info("=" * 80)
        logger.info("🚀 STARTING REAL PDF INGESTION TEST")
        logger.info("=" * 80)
        
        step_start_time = datetime.utcnow()
        
        # Step 1: Environment validation
        logger.info("🔧 Step 1: Validating environment...")
        if not check_environment_variables():
            test_results["status"] = "failed"
            test_results["errors"].append("Environment validation failed")
            return test_results
        
        test_results["steps_completed"].append("environment_validation")
        test_results["processing_times"]["environment_validation"] = (datetime.utcnow() - step_start_time).total_seconds()
        
        # Step 2: Test file validation
        step_start_time = datetime.utcnow()
        logger.info("📄 Step 2: Validating test file...")
        if not check_test_file():
            test_results["status"] = "failed"
            test_results["errors"].append("Test file validation failed")
            return test_results
        
        test_results["steps_completed"].append("file_validation")
        test_results["processing_times"]["file_validation"] = (datetime.utcnow() - step_start_time).total_seconds()
        
        # Step 3: Neo4j connection test
        step_start_time = datetime.utcnow()
        logger.info("🔗 Step 3: Testing Neo4j connection...")
        graph = test_neo4j_connection()
        if not graph:
            test_results["status"] = "failed"
            test_results["errors"].append("Neo4j connection failed")
            return test_results
        
        test_results["steps_completed"].append("neo4j_connection")
        test_results["processing_times"]["neo4j_connection"] = (datetime.utcnow() - step_start_time).total_seconds()
        
        # Step 4: Create test facility
        step_start_time = datetime.utcnow()
        logger.info("🏢 Step 4: Setting up test facility...")
        facility_id = "TEST_FACILITY_PDF_001"
        if not create_test_facility(graph, facility_id):
            test_results["warnings"].append("Test facility creation failed - continuing anyway")
        
        test_results["steps_completed"].append("test_facility_setup")
        test_results["processing_times"]["test_facility_setup"] = (datetime.utcnow() - step_start_time).total_seconds()
        
        # Step 5: Initialize workflow
        step_start_time = datetime.utcnow()
        logger.info("🤖 Step 5: Initializing ingestion workflow...")
        try:
            workflow = RiskAssessmentIntegratedWorkflow(
                llama_parse_api_key=os.getenv("LLAMA_PARSE_API_KEY"),
                neo4j_uri=os.getenv("NEO4J_URI"),
                neo4j_username=os.getenv("NEO4J_USERNAME"),
                neo4j_password=os.getenv("NEO4J_PASSWORD"),
                neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
                llm_model="gpt-4o",
                enable_risk_assessment=TEST_CONFIG["enable_risk_assessment"],
                max_retries=2
            )
            logger.info("✅ Workflow initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Workflow initialization failed: {e}")
            logger.debug(traceback.format_exc())
            test_results["status"] = "failed"
            test_results["errors"].append(f"Workflow initialization failed: {str(e)}")
            return test_results
        
        test_results["steps_completed"].append("workflow_initialization")
        test_results["processing_times"]["workflow_initialization"] = (datetime.utcnow() - step_start_time).total_seconds()
        
        # Step 6: Run document ingestion
        step_start_time = datetime.utcnow()
        logger.info("📊 Step 6: Running document ingestion with risk assessment...")
        
        # Prepare document metadata
        document_metadata = {
            "facility_id": facility_id,
            "document_type": "test_document",
            "source": "pdf_ingestion_test",
            "original_filename": Path(TEST_CONFIG["pdf_file"]).name,
            "test_run": True,
            "enable_risk_assessment": TEST_CONFIG["enable_risk_assessment"]
        }
        
        try:
            # Generate unique document ID
            document_id = f"test_doc_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"Processing document: {document_id}")
            logger.info(f"Source file: {TEST_CONFIG['pdf_file']}")
            logger.info(f"Metadata: {json.dumps(document_metadata, indent=2)}")
            
            # Run the ingestion workflow
            ingestion_start_time = datetime.utcnow()
            
            # Create initial state
            initial_state = DocumentStateWithRisk(
                document_id=document_id,
                file_path=TEST_CONFIG["pdf_file"],
                metadata=document_metadata,
                status="initialized",
                risk_assessment_enabled=TEST_CONFIG["enable_risk_assessment"],
                risk_errors=[]
            )
            
            logger.info("🔄 Starting workflow execution...")
            
            # Execute the workflow
            final_state = workflow.process_document(
                file_path=str(TEST_CONFIG["pdf_file"]),
                document_id=document_id,
                metadata=document_metadata
            )
            
            ingestion_time = (datetime.utcnow() - ingestion_start_time).total_seconds()
            
            logger.info(f"✅ Workflow execution completed in {ingestion_time:.2f} seconds")
            
            # Extract results
            test_results["workflow_results"] = {
                "document_id": document_id,
                "final_status": final_state.status if final_state else "unknown",
                "processing_time": ingestion_time,
                "chunks_created": getattr(final_state, 'chunks_processed', 0) if final_state else 0,
                "entities_extracted": getattr(final_state, 'entities_count', 0) if final_state else 0,
                "risk_assessment_completed": getattr(final_state, 'risk_assessment_status', 'unknown') if final_state else 'unknown'
            }
            
            # Risk assessment results
            if final_state and TEST_CONFIG["enable_risk_assessment"]:
                risk_results = getattr(final_state, 'risk_assessment_results', {})
                if risk_results:
                    test_results["workflow_results"]["risk_level"] = risk_results.get('risk_level', 'unknown')
                    test_results["workflow_results"]["risk_score"] = risk_results.get('risk_score', 0.0)
                    test_results["workflow_results"]["risk_factors_count"] = len(risk_results.get('risk_factors', []))
                
                risk_errors = getattr(final_state, 'risk_errors', [])
                if risk_errors:
                    test_results["errors"].extend([f"Risk assessment error: {err}" for err in risk_errors])
            
            # Check for any errors in the final state
            if final_state:
                state_errors = getattr(final_state, 'errors', [])
                if state_errors:
                    test_results["errors"].extend([f"Workflow error: {err}" for err in state_errors])
                    
                state_warnings = getattr(final_state, 'warnings', [])
                if state_warnings:
                    test_results["warnings"].extend([f"Workflow warning: {warn}" for warn in state_warnings])
            
            logger.info(f"📈 Document processing completed:")
            logger.info(f"   Document ID: {document_id}")
            logger.info(f"   Status: {final_state.status if final_state else 'unknown'}")
            if TEST_CONFIG["enable_risk_assessment"]:
                risk_level = test_results["workflow_results"].get("risk_level", "unknown")
                risk_score = test_results["workflow_results"].get("risk_score", 0.0)
                logger.info(f"   Risk Level: {risk_level}")
                logger.info(f"   Risk Score: {risk_score}")
            
        except RecursionError as e:
            logger.error(f"🔄 RECURSION ERROR DETECTED: {e}")
            test_results["recursion_error_detected"] = True
            test_results["status"] = "failed"
            test_results["errors"].append(f"Recursion error: {str(e)}")
            test_results["errors"].append(f"Recursion limit was set to: {sys.getrecursionlimit()}")
            logger.error("Stack trace for recursion error:")
            logger.error(traceback.format_exc())
            return test_results
            
        except Exception as e:
            logger.error(f"❌ Document ingestion failed: {e}")
            logger.debug(traceback.format_exc())
            test_results["status"] = "failed"
            test_results["errors"].append(f"Document ingestion failed: {str(e)}")
            return test_results
        
        test_results["steps_completed"].append("document_ingestion")
        test_results["processing_times"]["document_ingestion"] = (datetime.utcnow() - step_start_time).total_seconds()
        
        # Step 7: Validate results in Neo4j
        step_start_time = datetime.utcnow()
        logger.info("🔍 Step 7: Validating results in Neo4j...")
        try:
            validation_query = """
            MATCH (d:Document {document_id: $document_id})
            OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
            OPTIONAL MATCH (d)-[:BELONGS_TO]->(f:Facility)
            RETURN 
                d.document_id as document_id,
                d.status as document_status,
                count(DISTINCT c) as chunk_count,
                f.facility_id as facility_id
            """
            
            validation_results = graph.query(validation_query, {"document_id": document_id})
            
            if validation_results:
                result = validation_results[0]
                logger.info(f"✅ Document found in Neo4j:")
                logger.info(f"   Document ID: {result['document_id']}")
                logger.info(f"   Status: {result['document_status']}")
                logger.info(f"   Chunks: {result['chunk_count']}")
                logger.info(f"   Facility: {result['facility_id']}")
                
                test_results["workflow_results"]["neo4j_validation"] = {
                    "document_found": True,
                    "document_status": result["document_status"],
                    "chunk_count": result["chunk_count"],
                    "facility_id": result["facility_id"]
                }
            else:
                logger.warning("⚠️ Document not found in Neo4j - this may indicate ingestion issues")
                test_results["warnings"].append("Document not found in Neo4j after ingestion")
                test_results["workflow_results"]["neo4j_validation"] = {
                    "document_found": False
                }
                
        except Exception as e:
            logger.error(f"❌ Neo4j validation failed: {e}")
            test_results["errors"].append(f"Neo4j validation failed: {str(e)}")
        
        test_results["steps_completed"].append("neo4j_validation")
        test_results["processing_times"]["neo4j_validation"] = (datetime.utcnow() - step_start_time).total_seconds()
        
        # Calculate total test time
        total_time = sum(test_results["processing_times"].values())
        test_results["total_processing_time"] = total_time
        
        # Determine final status
        if not test_results["errors"]:
            test_results["status"] = "completed"
            logger.info("✅ PDF ingestion test completed successfully!")
        else:
            test_results["status"] = "completed_with_errors"
            logger.warning(f"⚠️ PDF ingestion test completed with {len(test_results['errors'])} errors")
        
    except Exception as e:
        logger.error(f"❌ Unexpected error during PDF ingestion test: {e}")
        logger.debug(traceback.format_exc())
        test_results["status"] = "failed"
        test_results["errors"].append(f"Unexpected test error: {str(e)}")
    
    finally:
        test_results["test_completed"] = datetime.utcnow().isoformat()
        test_results["test_duration_seconds"] = test_results.get("total_processing_time", 0)
    
    return test_results

def save_test_results(results: Dict[str, Any]):
    """Save detailed test results to JSON file."""
    try:
        results_file = TEST_CONFIG["results_file"]
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"💾 Test results saved to: {results_file}")
    except Exception as e:
        logger.error(f"⚠️ Could not save test results: {e}")

def print_test_summary(results: Dict[str, Any]):
    """Print a comprehensive test summary."""
    print("\n" + "=" * 80)
    print("📋 REAL PDF INGESTION TEST SUMMARY")
    print("=" * 80)
    
    print(f"🎯 Test Status: {results['status'].upper()}")
    print(f"⏱️  Total Duration: {results.get('test_duration_seconds', 0):.2f} seconds")
    print(f"📝 Steps Completed: {len(results['steps_completed'])}")
    
    if results.get('steps_completed'):
        print("   Steps:")
        for step in results['steps_completed']:
            time_taken = results['processing_times'].get(step, 0)
            print(f"   ✅ {step.replace('_', ' ').title()}: {time_taken:.2f}s")
    
    # Workflow results
    if results.get('workflow_results'):
        wr = results['workflow_results']
        print(f"\n📊 WORKFLOW RESULTS:")
        print(f"   Document ID: {wr.get('document_id', 'N/A')}")
        print(f"   Final Status: {wr.get('final_status', 'unknown')}")
        print(f"   Processing Time: {wr.get('processing_time', 0):.2f}s")
        print(f"   Chunks Created: {wr.get('chunks_created', 0)}")
        print(f"   Entities Extracted: {wr.get('entities_extracted', 0)}")
        
        if TEST_CONFIG["enable_risk_assessment"]:
            print(f"   Risk Assessment: {wr.get('risk_assessment_completed', 'unknown')}")
            print(f"   Risk Level: {wr.get('risk_level', 'unknown')}")
            print(f"   Risk Score: {wr.get('risk_score', 0.0)}")
            print(f"   Risk Factors: {wr.get('risk_factors_count', 0)}")
    
    # Errors and warnings
    if results.get('errors'):
        print(f"\n❌ ERRORS ({len(results['errors'])}):")
        for i, error in enumerate(results['errors'], 1):
            print(f"   {i}. {error}")
    
    if results.get('warnings'):
        print(f"\n⚠️  WARNINGS ({len(results['warnings'])}):")
        for i, warning in enumerate(results['warnings'], 1):
            print(f"   {i}. {warning}")
    
    # Recursion error detection
    if results.get('recursion_error_detected'):
        print(f"\n🔄 RECURSION ERROR DETECTED!")
        print(f"   This indicates an infinite recursion bug in the workflow.")
        print(f"   Recursion limit was set to: {sys.getrecursionlimit()}")
        print(f"   Check the log file for detailed stack trace: {TEST_CONFIG['log_file']}")
    
    print("\n" + "=" * 80)
    print(f"📄 Detailed logs: {TEST_CONFIG['log_file']}")
    print(f"💾 Results file: {TEST_CONFIG['results_file']}")
    print("=" * 80)

def main():
    """Main function to run the real PDF ingestion test."""
    global logger
    
    # Set up logging
    logger = setup_comprehensive_logging()
    
    logger.info("🚀 Real PDF Ingestion Test Starting...")
    logger.info(f"📋 Test Configuration: {json.dumps(TEST_CONFIG, indent=2)}")
    logger.info(f"🔄 Python recursion limit: {sys.getrecursionlimit()}")
    
    # Run the test
    test_results = run_pdf_ingestion_test()
    
    # Save and print results
    save_test_results(test_results)
    print_test_summary(test_results)
    
    # Return appropriate exit code
    if test_results['status'] in ['completed', 'completed_with_errors']:
        return 0 if test_results['status'] == 'completed' else 1
    else:
        return 2

if __name__ == '__main__':
    sys.exit(main())