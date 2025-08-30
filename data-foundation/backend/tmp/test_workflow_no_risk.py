#!/usr/bin/env python3
"""
Test script for base EnhancedIngestionWorkflow without risk assessment integration.

This script tests the EnhancedIngestionWorkflow (not RiskAssessmentIntegratedWorkflow)
to confirm that the recursion issue is specifically related to risk assessment integration
and not the base workflow functionality.
"""

import os
import sys
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import json
from dotenv import load_dotenv

# Fix import path conflicts - be very specific about the order
backend_dir = Path(__file__).parent.parent
src_dir = backend_dir / "src"

# Only add the backend directory to the path, not the src directory
# This helps avoid conflicts with the workflows package
sys.path.insert(0, str(backend_dir))

# Import the base enhanced workflow (NOT the risk assessment one)
try:
    from src.workflows.ingestion_workflow_enhanced import EnhancedIngestionWorkflow
except ImportError as e:
    print(f"Import error: {e}")
    print("This is likely due to the workflows package conflict.")
    print("Trying alternative import approach...")
    
    # Try running without LlamaIndex workflow imports
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "enhanced_workflow", 
        backend_dir / "src/workflows/ingestion_workflow_enhanced.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load enhanced workflow module")
    
    enhanced_workflow_module = importlib.util.module_from_spec(spec)
    sys.modules["enhanced_workflow"] = enhanced_workflow_module
    spec.loader.exec_module(enhanced_workflow_module)
    EnhancedIngestionWorkflow = enhanced_workflow_module.EnhancedIngestionWorkflow

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/tmp/test_workflow_no_risk.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_environment() -> Dict[str, str]:
    """Load and validate environment variables."""
    logger.info("Loading environment variables...")
    
    # Load from .env file
    env_file = backend_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        logger.info(f"Loaded environment from: {env_file}")
    else:
        logger.warning(f"No .env file found at: {env_file}")
    
    # Required environment variables
    required_vars = {
        'LLAMA_PARSE_API_KEY': os.getenv('LLAMA_PARSE_API_KEY'),
        'NEO4J_URI': os.getenv('NEO4J_URI'),
        'NEO4J_USERNAME': os.getenv('NEO4J_USERNAME'),
        'NEO4J_PASSWORD': os.getenv('NEO4J_PASSWORD'),
        'NEO4J_DATABASE': os.getenv('NEO4J_DATABASE', 'neo4j'),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY')
    }
    
    # Validate required variables
    missing_vars = [key for key, value in required_vars.items() if not value]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    logger.info("Environment variables validated successfully")
    return required_vars


def get_test_pdf_file() -> str:
    """Get a test PDF file path."""
    test_files = [
        backend_dir / "test/test_documents/electricity_bills/electric_bill.pdf",
        backend_dir / "electric_bill.pdf",
        backend_dir / "waste_manifest.pdf",
        backend_dir / "test/test_documents/water_bills/water_bill.pdf",
        backend_dir / "test/test_documents/waste_manifests/waste_manifest.pdf"
    ]
    
    for test_file in test_files:
        if test_file.exists():
            logger.info(f"Using test file: {test_file}")
            return str(test_file)
    
    raise FileNotFoundError(f"No test PDF files found. Checked: {[str(f) for f in test_files]}")


def create_test_workflow(env_vars: Dict[str, str], enable_risk_assessment: bool = False) -> Any:
    """Create a test workflow instance with explicit risk assessment control."""
    logger.info(f"Creating EnhancedIngestionWorkflow with enable_risk_assessment={enable_risk_assessment}")
    
    try:
        # Create workflow with Phase 1 features enabled but NO risk assessment
        # The enable_risk_assessment parameter should be False to test base workflow
        workflow = EnhancedIngestionWorkflow(
            llama_parse_api_key=env_vars['LLAMA_PARSE_API_KEY'],
            neo4j_uri=env_vars['NEO4J_URI'],
            neo4j_username=env_vars['NEO4J_USERNAME'],
            neo4j_password=env_vars['NEO4J_PASSWORD'],
            neo4j_database=env_vars['NEO4J_DATABASE'],
            llm_model="gpt-4o-mini",  # Use a lighter model for testing
            max_retries=2,  # Reduced for testing
            enable_phase1_features=True,  # Keep Phase 1 features enabled
            storage_path="/tmp/audit_trail_storage"
        )
        
        # Log workflow initialization
        logger.info(f"Workflow created successfully")
        logger.info(f"Phase 1 features enabled: {workflow.enable_phase1}")
        logger.info(f"Max retries: {workflow.max_retries}")
        
        return workflow
        
    except Exception as e:
        logger.error(f"Failed to create workflow: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def run_workflow_test(workflow: Any, test_file: str) -> Dict[str, Any]:
    """Run the workflow test with comprehensive logging and error handling."""
    logger.info(f"Starting workflow test with file: {test_file}")
    
    # Generate unique document ID
    document_id = f"test_doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Test metadata
    metadata = {
        "source": "test_script",
        "test_run": True,
        "uploaded_by": "test_user",
        "upload_timestamp": datetime.now().isoformat()
    }
    
    try:
        logger.info(f"Processing document ID: {document_id}")
        logger.info(f"Metadata: {json.dumps(metadata, indent=2)}")
        
        # Process the document through the workflow
        start_time = datetime.now()
        logger.info("Starting document processing...")
        
        result = workflow.process_document(
            file_path=test_file,
            document_id=document_id,
            metadata=metadata
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        logger.info(f"Document processing completed in {processing_time:.2f} seconds")
        
        # Log detailed results
        logger.info("=== WORKFLOW RESULTS ===")
        logger.info(f"Status: {result.get('status')}")
        logger.info(f"Document Type: {result.get('document_type')}")
        logger.info(f"Errors: {result.get('errors', [])}")
        logger.info(f"Retry Count: {result.get('retry_count', 0)}")
        logger.info(f"Processing Time: {result.get('processing_time')}")
        
        # Phase 1 processing details
        if result.get('phase1_processing'):
            logger.info("=== PHASE 1 PROCESSING ===")
            phase1_data = result['phase1_processing']
            for key, value in phase1_data.items():
                logger.info(f"{key}: {value}")
        
        # Additional state information
        logger.info("=== ADDITIONAL STATE INFO ===")
        logger.info(f"Is Duplicate: {result.get('is_duplicate', False)}")
        logger.info(f"Validation Score: {result.get('validation_score')}")
        logger.info(f"Rejection ID: {result.get('rejection_id')}")
        logger.info(f"Rejection Reason: {result.get('rejection_reason')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Workflow test failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "document_id": document_id
        }


def analyze_results(result: Dict[str, Any]) -> None:
    """Analyze and report on test results."""
    logger.info("=== RESULT ANALYSIS ===")
    
    status = result.get('status', 'unknown')
    errors = result.get('errors', [])
    
    if status == 'completed':
        logger.info("✅ SUCCESS: Workflow completed successfully without risk assessment")
        logger.info("This confirms the base workflow functions correctly")
        
    elif status == 'failed':
        logger.error("❌ FAILURE: Base workflow failed")
        logger.error("This indicates an issue with the base workflow, not risk assessment")
        
    elif status == 'rejected':
        logger.info("⚠️  REJECTED: Document was rejected by Phase 1 validation")
        logger.info("This is expected behavior for certain document types")
        
    else:
        logger.warning(f"⚠️  UNEXPECTED STATUS: {status}")
    
    if errors:
        logger.error("Errors encountered:")
        for i, error in enumerate(errors, 1):
            logger.error(f"  {i}. {error}")
    
    # Check for any signs of recursion issues
    if result.get('error') and 'recursion' in str(result.get('error')).lower():
        logger.error("🚨 RECURSION DETECTED: Base workflow has recursion issues!")
    elif result.get('traceback') and 'recursion' in str(result.get('traceback')).lower():
        logger.error("🚨 RECURSION DETECTED: Base workflow has recursion issues!")
    else:
        logger.info("✅ NO RECURSION: No recursion issues detected in base workflow")


def main():
    """Main test function."""
    logger.info("=" * 60)
    logger.info("STARTING BASE WORKFLOW TEST (WITHOUT RISK ASSESSMENT)")
    logger.info("=" * 60)
    
    try:
        # Step 1: Load environment
        logger.info("Step 1: Loading environment...")
        env_vars = load_environment()
        logger.info("Environment loaded successfully")
        
        # Step 2: Get test file
        logger.info("Step 2: Finding test PDF file...")
        test_file = get_test_pdf_file()
        logger.info(f"Test file selected: {test_file}")
        
        # Step 3: Create workflow (explicitly disable risk assessment)
        logger.info("Step 3: Creating base workflow...")
        workflow = create_test_workflow(env_vars, enable_risk_assessment=False)
        logger.info("Base workflow created successfully")
        
        # Step 4: Run test
        logger.info("Step 4: Running workflow test...")
        result = run_workflow_test(workflow, test_file)
        
        # Step 5: Analyze results
        logger.info("Step 5: Analyzing results...")
        analyze_results(result)
        
        # Step 6: Save results to file
        logger.info("Step 6: Saving results...")
        results_file = "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/tmp/test_workflow_no_risk_results.json"
        with open(results_file, 'w') as f:
            # Convert result to JSON-serializable format
            json_result = {}
            for key, value in result.items():
                try:
                    json.dumps(value)  # Test if JSON serializable
                    json_result[key] = value
                except (TypeError, ValueError):
                    json_result[key] = str(value)  # Convert to string if not serializable
            
            json.dump(json_result, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {results_file}")
        
        # Final summary
        logger.info("=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Test file: {test_file}")
        logger.info(f"Final status: {result.get('status', 'unknown')}")
        logger.info(f"Risk assessment enabled: False")
        logger.info(f"Workflow type: EnhancedIngestionWorkflow (base)")
        
        if result.get('status') == 'completed':
            logger.info("🎉 SUCCESS: Base workflow works without risk assessment!")
        else:
            logger.warning("⚠️  Base workflow did not complete successfully")
        
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Test script failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())