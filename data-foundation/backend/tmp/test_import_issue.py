#!/usr/bin/env python3
"""Test script to isolate the import recursion issue."""

import sys
print(f"Python version: {sys.version}")
print(f"Recursion limit: {sys.getrecursionlimit()}")

try:
    print("\n1. Testing basic imports...")
    import os
    import json
    print("✓ Basic imports OK")
    
    print("\n2. Testing langchain imports...")
    from langchain_core.documents import Document
    print("✓ langchain_core.documents OK")
    
    print("\n3. Testing langgraph imports...")
    from langgraph.graph import StateGraph, END
    print("✓ langgraph.graph OK")
    
    print("\n4. Testing workflow imports...")
    # First try the base workflow
    from ehs_workflows.ingestion_workflow_enhanced import EnhancedIngestionWorkflow
    print("✓ ingestion_workflow_enhanced OK")
    
    # Then try the risk assessment workflow
    from ehs_workflows.ingestion_workflow_with_risk_assessment import RiskAssessmentIntegratedWorkflow
    print("✓ ingestion_workflow_with_risk_assessment OK")
    
    print("\n5. Testing agent imports...")
    from agents.risk_assessment.agent import RiskAssessmentAgent
    print("✓ risk_assessment.agent OK")
    
    print("\n✅ All imports successful!")
    
except RecursionError as e:
    print(f"\n❌ RecursionError during import: {e}")
    import traceback
    traceback.print_exc()
    
except Exception as e:
    print(f"\n❌ Other error during import: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()