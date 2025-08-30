#!/usr/bin/env python3
"""Test script with path fix for the module conflict."""

import sys
import os

# Remove the src directory from the Python path to avoid conflicts
# We'll add it back after imports
original_path = sys.path.copy()
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')

# Remove src path temporarily to avoid conflicts with llama_index imports
sys.path = [p for p in sys.path if p != src_path]

try:
    print("Importing with modified path to avoid conflicts...")
    
    # Import llama_index first (before adding src to path)
    from llama_index.core.async_utils import asyncio_run
    from llama_parse import LlamaParse
    print("✓ LlamaIndex imports successful")
    
    # Now add src back to the path
    sys.path.insert(0, src_path)
    
    # Import our modules
    from ehs_workflows.ingestion_workflow_with_risk_assessment import RiskAssessmentIntegratedWorkflow
    print("✓ Workflow imports successful")
    
    print("\n✅ All imports successful with path fix!")
    
    # Test instantiation
    print("\nTesting workflow instantiation...")
    workflow = RiskAssessmentIntegratedWorkflow(
        llama_parse_api_key=os.getenv("LLAMA_CLOUD_API_KEY", "test_key"),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
        enable_risk_assessment=True
    )
    print("✓ Workflow instantiated successfully")
    
except Exception as e:
    print(f"\n❌ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Restore original path
    sys.path = original_path