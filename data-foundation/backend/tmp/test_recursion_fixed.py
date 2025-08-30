#!/usr/bin/env python3
"""Test if recursion issue is fixed after renaming workflows to ehs_workflows."""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

print("Testing import of risk assessment workflow...")
print(f"Python path: {sys.path[0]}")

try:
    # This import was causing the recursion error before
    from ehs_workflows.ingestion_workflow_with_risk_assessment import RiskAssessmentIntegratedWorkflow
    print("✅ Successfully imported RiskAssessmentIntegratedWorkflow!")
    
    # Test instantiation
    print("\nTesting workflow instantiation...")
    workflow = RiskAssessmentIntegratedWorkflow(
        llama_parse_api_key="test_key",
        neo4j_uri="bolt://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password="password",
        enable_risk_assessment=True
    )
    print("✅ Workflow instantiated successfully!")
    
    # Test that we can access the workflow graph
    print("\nTesting workflow graph access...")
    graph = workflow.workflow
    print(f"✅ Workflow graph created: {type(graph)}")
    
    print("\n🎉 SUCCESS: The recursion issue has been resolved!")
    print("The module rename from 'workflows' to 'ehs_workflows' fixed the import conflict.")
    
except RecursionError as e:
    print(f"\n❌ RecursionError still occurs: {e}")
    import traceback
    traceback.print_exc()
    
except Exception as e:
    print(f"\n❌ Other error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()