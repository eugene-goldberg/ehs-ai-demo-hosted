#!/usr/bin/env python3
"""
Debug script for EHS AI Ingestion Workflow with Risk Assessment Integration
Modified to handle Python 3.13 compatibility issues with pydantic/typing
"""

import os
import sys
import logging
import traceback
import tempfile
from datetime import datetime
from typing import Dict, Any
import json

# IMPORTANT: Handle Python 3.13 typing compatibility issues
# Set recursion limit BEFORE any imports that might trigger the pydantic issue
original_recursion_limit = sys.getrecursionlimit()
sys.setrecursionlimit(3000)  # Increase limit to handle pydantic/typing issues

print(f"Python version: {sys.version}")
print(f"Original recursion limit: {original_recursion_limit}")
print(f"Set recursion limit to: {sys.getrecursionlimit()}")

# Try to import typing modules with error handling
try:
    import typing
    from typing import Optional, Union, List
    print("✓ Successfully imported typing modules")
except Exception as e:
    print(f"✗ Failed to import typing modules: {e}")
    sys.exit(1)

# Set up the Python path to include the project source
project_backend_path = "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend"
project_src_path = "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/src"
sys.path.insert(0, project_src_path)
sys.path.insert(0, project_backend_path)

# Set environment variables
os.environ.setdefault("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
os.environ.setdefault("LLAMA_PARSE_API_KEY", "YOUR_LLAMA_PARSE_API_KEY")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USERNAME", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "EhsAI2024!")
os.environ.setdefault("NEO4J_DATABASE", "neo4j")
os.environ.setdefault("LANGCHAIN_API_KEY", "YOUR_LANGCHAIN_API_KEY")
os.environ.setdefault("LANGCHAIN_PROJECT", "ehs-ai-demo-ingestion-debug")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
os.environ.setdefault("AUDIT_TRAIL_STORAGE_PATH", "/tmp/audit_trail_storage")

# Ensure audit trail storage directory exists
os.makedirs("/tmp/audit_trail_storage", exist_ok=True)

# Set up comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/ingestion_workflow_debug_fixed.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def test_imports_step_by_step():
    """Test imports step by step to identify where the recursion issue occurs"""
    
    logger.info("="*60)
    logger.info("TESTING IMPORTS STEP BY STEP")
    logger.info("="*60)
    
    try:
        logger.info("Step 1: Testing pydantic imports...")
        import pydantic
        logger.info("✓ pydantic imported successfully")
        
        logger.info("Step 2: Testing pydantic.v1 imports...")
        import pydantic.v1
        logger.info("✓ pydantic.v1 imported successfully")
        
        logger.info("Step 3: Testing langchain_core imports...")
        import langchain_core
        logger.info("✓ langchain_core imported successfully")
        
        logger.info("Step 4: Testing langchain_core.runnables imports...")
        from langchain_core.runnables import Runnable, RunnableConfig
        logger.info("✓ langchain_core.runnables imported successfully")
        
        logger.info("Step 5: Testing langgraph imports...")
        import langgraph
        logger.info("✓ langgraph imported successfully")
        
        logger.info("Step 6: Testing langgraph.graph imports...")
        from langgraph.graph import StateGraph, END
        logger.info("✓ langgraph.graph imported successfully")
        
        logger.info("Step 7: Testing workflow imports...")
        try:
            from ehs_workflows.ingestion_workflow_with_risk_assessment import RiskAssessmentIntegratedWorkflow
            logger.info("✓ RiskAssessmentIntegratedWorkflow imported successfully")
            return RiskAssessmentIntegratedWorkflow
        except Exception as workflow_error:
            logger.error(f"✗ Failed to import workflow: {workflow_error}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Try importing the base enhanced workflow instead
            logger.info("Step 7b: Trying base enhanced workflow...")
            try:
                from ehs_workflows.ingestion_workflow_enhanced import EnhancedIngestionWorkflow
                logger.info("✓ EnhancedIngestionWorkflow imported successfully")
                return EnhancedIngestionWorkflow
            except Exception as base_error:
                logger.error(f"✗ Failed to import base workflow: {base_error}")
                raise workflow_error
        
    except RecursionError as recursion_error:
        logger.error("="*60)
        logger.error("RECURSION ERROR DURING IMPORTS!")
        logger.error("="*60)
        logger.error(f"Recursion error: {recursion_error}")
        logger.error(f"Current recursion limit: {sys.getrecursionlimit()}")
        
        # Try increasing recursion limit even more
        if sys.getrecursionlimit() < 5000:
            new_limit = 5000
            sys.setrecursionlimit(new_limit)
            logger.info(f"Increased recursion limit to {new_limit}, retrying imports...")
            return test_imports_step_by_step()  # Recursive call with higher limit
        else:
            raise
    
    except Exception as general_error:
        logger.error(f"General import error: {general_error}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def create_debug_document():
    """Create a simple test document for processing"""
    test_content = """
# EHS Safety Report

## Facility Information
- Facility ID: TEST-FACILITY-001
- Facility Name: Test Manufacturing Plant
- Location: Test City, Test State

## Safety Incident Report
Date: 2024-01-15
Type: Near Miss
Description: Worker observed unsafe condition in machinery area.
Severity: Low
Status: Resolved

## Recommendations
- Increase safety training frequency
- Install additional warning signs
- Schedule equipment maintenance
"""
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False)
    temp_file.write(test_content)
    temp_file.flush()
    temp_file.close()
    
    logger.info(f"Created test document: {temp_file.name}")
    return temp_file.name

class WorkflowExecutionTracker:
    """Track workflow node executions to identify loops"""
    
    def __init__(self):
        self.node_executions = []
        self.node_counts = {}
        self.max_node_executions = 30  # Increased threshold
        
    def track_node_execution(self, node_name: str, state: Dict[str, Any]):
        """Track a node execution"""
        execution_info = {
            'timestamp': datetime.utcnow().isoformat(),
            'node_name': node_name,
            'document_id': state.get('document_id'),
            'status': state.get('status'),
            'retry_count': state.get('retry_count', 0),
            'errors': len(state.get('errors', [])),
            'risk_errors': len(state.get('risk_errors', []))
        }
        
        self.node_executions.append(execution_info)
        
        # Track node counts
        if node_name not in self.node_counts:
            self.node_counts[node_name] = 0
        self.node_counts[node_name] += 1
        
        logger.info(f"WORKFLOW TRACKER: Node '{node_name}' executed {self.node_counts[node_name]} time(s)")
        logger.info(f"WORKFLOW TRACKER: Current state - Document: {execution_info['document_id']}, Status: {execution_info['status']}, Retry: {execution_info['retry_count']}")
        
        # Check for potential loops (more lenient threshold)
        if self.node_counts[node_name] > 8:
            logger.warning(f"POTENTIAL LOOP DETECTED: Node '{node_name}' has been executed {self.node_counts[node_name]} times!")
            self.log_recent_executions()
            
        if len(self.node_executions) > self.max_node_executions:
            logger.error(f"RECURSION LOOP DETECTED: Total executions ({len(self.node_executions)}) exceeded threshold!")
            self.log_execution_pattern()
            raise RecursionError(f"Workflow execution loop detected after {len(self.node_executions)} node executions")
    
    def log_recent_executions(self, count: int = 10):
        """Log recent node executions to identify patterns"""
        logger.info("=== RECENT WORKFLOW EXECUTIONS ===")
        recent = self.node_executions[-count:]
        for i, exec_info in enumerate(recent):
            logger.info(f"{i+1}. {exec_info['timestamp']} - {exec_info['node_name']} (retry: {exec_info['retry_count']})")
    
    def log_execution_pattern(self):
        """Log the full execution pattern to identify loops"""
        logger.info("=== FULL WORKFLOW EXECUTION PATTERN ===")
        for i, exec_info in enumerate(self.node_executions):
            logger.info(f"{i+1:2d}. {exec_info['node_name']} @ {exec_info['timestamp']} (retry: {exec_info['retry_count']})")
        
        logger.info("=== NODE EXECUTION COUNTS ===")
        for node_name, count in sorted(self.node_counts.items()):
            logger.info(f"{node_name}: {count} executions")

# Global tracker instance
execution_tracker = WorkflowExecutionTracker()

def patch_workflow_nodes(workflow_instance):
    """Monkey patch workflow nodes to add execution tracking"""
    
    # Get all node methods that might exist
    potential_node_methods = [
        'store_source_file',
        'validate_document', 
        'validate_document_quality',
        'parse_document',
        'check_for_duplicates',
        'extract_data',
        'transform_data',
        'validate_extracted_data',
        'process_prorating',
        'load_to_neo4j',
        'index_document',
        'complete_processing',
        'handle_error',
        'handle_rejection',
        'initialize_risk_assessment',
        'perform_risk_assessment',
        'store_risk_results',
        'finalize_risk_assessment',
        'handle_risk_error'
    ]
    
    patched_count = 0
    
    for method_name in potential_node_methods:
        if hasattr(workflow_instance, method_name):
            original_method = getattr(workflow_instance, method_name)
            
            def create_tracked_method(orig_method, node_name):
                def tracked_method(state):
                    logger.info(f"\n{'='*60}")
                    logger.info(f"EXECUTING NODE: {node_name}")
                    logger.info(f"{'='*60}")
                    
                    # Track the execution
                    execution_tracker.track_node_execution(node_name, state)
                    
                    try:
                        result = orig_method(state)
                        logger.info(f"NODE COMPLETED: {node_name}")
                        logger.info(f"Result status: {result.get('status') if isinstance(result, dict) else 'N/A'}")
                        return result
                    except Exception as e:
                        logger.error(f"NODE FAILED: {node_name} - {str(e)}")
                        raise
                
                return tracked_method
            
            # Replace the method
            setattr(workflow_instance, method_name, create_tracked_method(original_method, method_name))
            patched_count += 1
            
    logger.info(f"Patched {patched_count} workflow nodes for execution tracking")

def run_debug_workflow():
    """Run the workflow with comprehensive debugging"""
    
    logger.info("="*80)
    logger.info("STARTING EHS AI INGESTION WORKFLOW DEBUG SESSION")
    logger.info("="*80)
    
    try:
        # Step by step import testing
        WorkflowClass = test_imports_step_by_step()
        logger.info(f"Successfully imported workflow class: {WorkflowClass.__name__}")
        
        # Create test document
        test_file_path = create_debug_document()
        
        # Initialize workflow with debugging enabled
        logger.info("Initializing workflow...")
        
        # Use simpler initialization for base workflow if risk assessment workflow fails
        if "Enhanced" in WorkflowClass.__name__:
            # Base enhanced workflow
            workflow = WorkflowClass(
                llama_parse_api_key=os.environ["LLAMA_PARSE_API_KEY"],
                neo4j_uri=os.environ["NEO4J_URI"],
                neo4j_username=os.environ["NEO4J_USERNAME"],
                neo4j_password=os.environ["NEO4J_PASSWORD"],
                neo4j_database=os.environ["NEO4J_DATABASE"],
                llm_model="gpt-4o-mini",  # Use cheaper model for debugging
                max_retries=1,  # Reduced retries to fail faster
                enable_phase1_features=True,
                storage_path="/tmp/debug_storage/"
            )
        else:
            # Risk assessment integrated workflow
            workflow = WorkflowClass(
                llama_parse_api_key=os.environ["LLAMA_PARSE_API_KEY"],
                neo4j_uri=os.environ["NEO4J_URI"],
                neo4j_username=os.environ["NEO4J_USERNAME"],
                neo4j_password=os.environ["NEO4J_PASSWORD"],
                neo4j_database=os.environ["NEO4J_DATABASE"],
                llm_model="gpt-4o-mini",  # Use cheaper model for debugging
                max_retries=1,  # Reduced retries to fail faster
                enable_phase1_features=True,
                enable_risk_assessment=True,
                storage_path="/tmp/debug_storage/",
                risk_assessment_methodology="comprehensive"
            )
        
        logger.info("Workflow initialized successfully")
        
        # Patch workflow nodes for tracking
        patch_workflow_nodes(workflow)
        
        # Prepare test metadata
        test_metadata = {
            "facility_id": "TEST-FACILITY-001",
            "source": "debug_test",
            "uploaded_by": "debug_script",
            "original_filename": "test_safety_report.md"
        }
        
        logger.info(f"Starting document processing: {test_file_path}")
        
        # Process the document with timeout protection
        final_state = workflow.process_document(
            file_path=test_file_path,
            document_id="debug_test_doc_001",
            metadata=test_metadata
        )
        
        logger.info("="*80)
        logger.info("WORKFLOW COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
        # Log final state
        logger.info("Final workflow state:")
        safe_state = {}
        for key, value in final_state.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                safe_state[key] = value
            elif isinstance(value, (list, dict)):
                safe_state[key] = f"<{type(value).__name__} with {len(value)} items>"
            else:
                safe_state[key] = f"<{type(value).__name__}>"
        
        logger.info(json.dumps(safe_state, indent=2))
        
        # Log execution summary
        execution_tracker.log_execution_pattern()
        
    except RecursionError as e:
        logger.error("="*80)
        logger.error("RECURSION ERROR CAUGHT!")
        logger.error("="*80)
        logger.error(f"Recursion error: {str(e)}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        
        # Log workflow execution pattern
        execution_tracker.log_execution_pattern()
        
        # Log recent node executions
        execution_tracker.log_recent_executions(15)
        
        raise
        
    except Exception as e:
        logger.error("="*80)
        logger.error("WORKFLOW EXECUTION ERROR!")
        logger.error("="*80)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        
        # Log execution pattern even on error
        execution_tracker.log_execution_pattern()
        
        raise
    
    finally:
        # Clean up test file
        try:
            if 'test_file_path' in locals():
                os.unlink(test_file_path)
                logger.info(f"Cleaned up test file: {test_file_path}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to clean up test file: {cleanup_error}")
        
        # Close workflow
        try:
            if 'workflow' in locals():
                workflow.close()
                logger.info("Workflow closed")
        except Exception as close_error:
            logger.warning(f"Error closing workflow: {close_error}")

def main():
    """Main debug execution"""
    
    logger.info("Debug script starting...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current recursion limit: {sys.getrecursionlimit()}")
    logger.info(f"Python path: {sys.path[:3]}...")  # Show first 3 entries
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Environment variables set:")
    for key in ["OPENAI_API_KEY", "LLAMA_PARSE_API_KEY", "NEO4J_URI", "LANGCHAIN_TRACING_V2"]:
        value = os.environ.get(key, "NOT SET")
        masked_value = value[:10] + "..." + value[-10:] if len(value) > 20 else value
        logger.info(f"  {key}: {masked_value}")
    
    try:
        run_debug_workflow()
        logger.info("Debug session completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Debug session interrupted by user")
        
    except Exception as e:
        logger.error(f"Debug session failed: {str(e)}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    finally:
        # Restore original recursion limit
        sys.setrecursionlimit(original_recursion_limit)
        logger.info(f"Restored recursion limit to {original_recursion_limit}")

if __name__ == "__main__":
    main()