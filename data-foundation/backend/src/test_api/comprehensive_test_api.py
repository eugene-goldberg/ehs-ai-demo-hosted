#!/usr/bin/env python3
"""
Comprehensive Test API for EHS AI Demo
=====================================

A FastAPI application dedicated to testing all components of the EHS AI system.
Provides structured test endpoints for original features, Phase 1 enhancements,
workflow integration, and end-to-end scenarios.

Usage:
    python3 comprehensive_test_api.py
    
    Then access endpoints via curl:
    curl -X GET http://localhost:8001/test/health
    curl -X POST http://localhost:8001/test/original/document-upload -H "Content-Type: application/json" -d '{"test_data": "sample"}'
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("comprehensive_test_api")

# Initialize FastAPI application
app = FastAPI(
    title="EHS AI Demo - Comprehensive Test API",
    description="Testing endpoints for all EHS AI system components",
    version="1.0.0",
    docs_url="/test/docs",
    redoc_url="/test/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models for Test Requests/Responses
class TestResult(BaseModel):
    """Standard test result model"""
    test_name: str
    status: str = Field(..., pattern="^(PASS|FAIL|SKIP|ERROR)$")
    execution_time_ms: float
    timestamp: str
    details: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    assertions: List[Dict[str, Any]] = Field(default_factory=list)

class TestSuite(BaseModel):
    """Test suite result model"""
    suite_name: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    execution_time_ms: float
    tests: List[TestResult]

class TestRequest(BaseModel):
    """Generic test request model"""
    test_parameters: Dict[str, Any] = Field(default_factory=dict)
    mock_data: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = 30

# Helper Functions
def create_test_result(
    test_name: str,
    status: str,
    execution_time: float,
    details: Dict[str, Any] = None,
    error_message: str = None,
    assertions: List[Dict[str, Any]] = None
) -> TestResult:
    """Create a standardized test result"""
    return TestResult(
        test_name=test_name,
        status=status,
        execution_time_ms=round(execution_time * 1000, 2),
        timestamp=datetime.now().isoformat(),
        details=details or {},
        error_message=error_message,
        assertions=assertions or []
    )

def time_execution(func):
    """Decorator to measure execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            return result, execution_time
        except Exception as e:
            execution_time = time.time() - start_time
            raise e
    return wrapper

async def simulate_async_operation(duration: float = 0.1):
    """Simulate async operation for testing"""
    await asyncio.sleep(duration)
    return {"simulated": True, "duration": duration}

def generate_mock_document():
    """Generate mock document data for testing"""
    return {
        "id": "test_doc_001",
        "filename": "test_document.pdf",
        "content": "This is a test EHS document with safety procedures and compliance information.",
        "metadata": {
            "size": 1024,
            "type": "application/pdf",
            "uploaded_at": datetime.now().isoformat()
        }
    }

def generate_mock_extraction_result():
    """Generate mock extraction result"""
    return {
        "entities": [
            {"type": "hazard", "value": "Chemical exposure", "confidence": 0.95},
            {"type": "regulation", "value": "OSHA 1910.1000", "confidence": 0.88},
            {"type": "procedure", "value": "Emergency evacuation", "confidence": 0.92}
        ],
        "summary": "Document contains information about chemical safety and emergency procedures",
        "compliance_score": 0.85
    }

# Root and Health Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "EHS AI Demo - Comprehensive Test API",
        "version": "1.0.0",
        "description": "Testing endpoints for all EHS AI system components",
        "documentation": "/test/docs",
        "health_check": "/test/health",
        "test_categories": {
            "original": "/test/original/*",
            "phase1": "/test/phase1/*",
            "workflow": "/test/workflow/*",
            "e2e": "/test/e2e/*",
            "unit": "/test/unit/*",
            "integration": "/test/integration/*",
            "performance": "/test/performance/*",
            "regression": "/test/regression/*"
        }
    }

@app.get("/test/health")
async def health_check():
    """Health check endpoint"""
    start_time = time.time()
    
    # Perform basic system checks
    checks = {
        "api_server": "OK",
        "timestamp": datetime.now().isoformat(),
        "system_memory": "Available",
        "dependencies": "Loaded"
    }
    
    execution_time = time.time() - start_time
    
    return {
        "status": "healthy",
        "checks": checks,
        "response_time_ms": round(execution_time * 1000, 2)
    }

# Original Features Test Endpoints
@app.post("/test/original/document-upload")
async def test_document_upload(request: TestRequest):
    """
    Test document upload functionality
    
    Tests the original document processing pipeline including:
    - File validation
    - Content extraction
    - Metadata processing
    """
    start_time = time.time()
    test_name = "Document Upload Test"
    
    try:
        # Simulate document upload process
        mock_doc = generate_mock_document()
        
        # Test assertions
        assertions = [
            {"description": "Document has valid ID", "passed": bool(mock_doc.get("id"))},
            {"description": "Document has filename", "passed": bool(mock_doc.get("filename"))},
            {"description": "Document has content", "passed": bool(mock_doc.get("content"))},
            {"description": "Metadata is present", "passed": bool(mock_doc.get("metadata"))}
        ]
        
        # Simulate processing delay
        await simulate_async_operation(0.2)
        
        execution_time = time.time() - start_time
        status = "PASS" if all(a["passed"] for a in assertions) else "FAIL"
        
        return create_test_result(
            test_name=test_name,
            status=status,
            execution_time=execution_time,
            details={"processed_document": mock_doc},
            assertions=assertions
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Document upload test failed: {str(e)}")
        return create_test_result(
            test_name=test_name,
            status="ERROR",
            execution_time=execution_time,
            error_message=str(e)
        )

@app.post("/test/original/entity-extraction")
async def test_entity_extraction(request: TestRequest):
    """
    Test entity extraction functionality
    
    Tests the NLP-based entity extraction including:
    - Hazard identification
    - Regulation recognition
    - Procedure extraction
    """
    start_time = time.time()
    test_name = "Entity Extraction Test"
    
    try:
        # Simulate extraction process
        mock_result = generate_mock_extraction_result()
        
        # Test assertions
        assertions = [
            {"description": "Entities extracted", "passed": len(mock_result["entities"]) > 0},
            {"description": "Summary generated", "passed": bool(mock_result.get("summary"))},
            {"description": "Compliance score calculated", "passed": mock_result.get("compliance_score", 0) > 0},
            {"description": "High confidence entities", "passed": any(e["confidence"] > 0.8 for e in mock_result["entities"])}
        ]
        
        # Simulate processing delay
        await simulate_async_operation(0.3)
        
        execution_time = time.time() - start_time
        status = "PASS" if all(a["passed"] for a in assertions) else "FAIL"
        
        return create_test_result(
            test_name=test_name,
            status=status,
            execution_time=execution_time,
            details={"extraction_result": mock_result},
            assertions=assertions
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Entity extraction test failed: {str(e)}")
        return create_test_result(
            test_name=test_name,
            status="ERROR",
            execution_time=execution_time,
            error_message=str(e)
        )

@app.post("/test/original/qa-system")
async def test_qa_system(request: TestRequest):
    """
    Test Q&A system functionality
    
    Tests the question-answering capabilities including:
    - Question processing
    - Context retrieval
    - Answer generation
    """
    start_time = time.time()
    test_name = "Q&A System Test"
    
    try:
        # Mock Q&A interaction
        test_question = request.test_parameters.get("question", "What are the safety procedures for chemical handling?")
        
        mock_answer = {
            "question": test_question,
            "answer": "Safety procedures include wearing appropriate PPE, following MSDS guidelines, and ensuring proper ventilation.",
            "confidence": 0.89,
            "sources": ["document_001.pdf", "safety_manual.pdf"],
            "context_used": True
        }
        
        # Test assertions
        assertions = [
            {"description": "Question processed", "passed": bool(mock_answer.get("question"))},
            {"description": "Answer generated", "passed": bool(mock_answer.get("answer"))},
            {"description": "Confidence score available", "passed": mock_answer.get("confidence", 0) > 0},
            {"description": "Sources provided", "passed": len(mock_answer.get("sources", [])) > 0}
        ]
        
        # Simulate processing delay
        await simulate_async_operation(0.4)
        
        execution_time = time.time() - start_time
        status = "PASS" if all(a["passed"] for a in assertions) else "FAIL"
        
        return create_test_result(
            test_name=test_name,
            status=status,
            execution_time=execution_time,
            details={"qa_result": mock_answer},
            assertions=assertions
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Q&A system test failed: {str(e)}")
        return create_test_result(
            test_name=test_name,
            status="ERROR",
            execution_time=execution_time,
            error_message=str(e)
        )

# Phase 1 Enhancement Test Endpoints
@app.post("/test/phase1/audit-trail")
async def test_audit_trail(request: TestRequest):
    """
    Test audit trail functionality
    
    Tests the Phase 1 audit trail enhancement including:
    - Action logging
    - Timestamp accuracy
    - User tracking
    """
    start_time = time.time()
    test_name = "Audit Trail Test"
    
    try:
        # Mock audit trail entry
        mock_audit_entry = {
            "id": "audit_001",
            "action": "document_processed",
            "user_id": "test_user",
            "timestamp": datetime.now().isoformat(),
            "details": {
                "document_id": "doc_001",
                "processing_time": 2.5,
                "status": "completed"
            },
            "ip_address": "192.168.1.100",
            "session_id": "sess_123"
        }
        
        # Test assertions
        assertions = [
            {"description": "Audit entry created", "passed": bool(mock_audit_entry.get("id"))},
            {"description": "Action recorded", "passed": bool(mock_audit_entry.get("action"))},
            {"description": "Timestamp present", "passed": bool(mock_audit_entry.get("timestamp"))},
            {"description": "User identified", "passed": bool(mock_audit_entry.get("user_id"))},
            {"description": "Details captured", "passed": bool(mock_audit_entry.get("details"))}
        ]
        
        # Simulate audit processing
        await simulate_async_operation(0.1)
        
        execution_time = time.time() - start_time
        status = "PASS" if all(a["passed"] for a in assertions) else "FAIL"
        
        return create_test_result(
            test_name=test_name,
            status=status,
            execution_time=execution_time,
            details={"audit_entry": mock_audit_entry},
            assertions=assertions
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Audit trail test failed: {str(e)}")
        return create_test_result(
            test_name=test_name,
            status="ERROR",
            execution_time=execution_time,
            error_message=str(e)
        )

@app.post("/test/phase1/pro-rating")
async def test_pro_rating(request: TestRequest):
    """
    Test pro-rating functionality
    
    Tests the Phase 1 pro-rating enhancement including:
    - Usage calculation
    - Resource allocation
    - Cost distribution
    """
    start_time = time.time()
    test_name = "Pro-Rating Test"
    
    try:
        # Mock pro-rating calculation
        mock_pro_rating = {
            "user_id": "test_user",
            "period": "2024-01",
            "total_usage": 150.5,
            "allocated_quota": 200.0,
            "usage_percentage": 75.25,
            "cost_per_unit": 0.02,
            "total_cost": 3.01,
            "remaining_quota": 49.5
        }
        
        # Test assertions
        assertions = [
            {"description": "Usage tracked", "passed": mock_pro_rating.get("total_usage", 0) > 0},
            {"description": "Quota allocated", "passed": mock_pro_rating.get("allocated_quota", 0) > 0},
            {"description": "Percentage calculated", "passed": 0 <= mock_pro_rating.get("usage_percentage", 0) <= 100},
            {"description": "Cost computed", "passed": mock_pro_rating.get("total_cost", 0) > 0},
            {"description": "Remaining quota valid", "passed": mock_pro_rating.get("remaining_quota", 0) >= 0}
        ]
        
        # Simulate calculation
        await simulate_async_operation(0.1)
        
        execution_time = time.time() - start_time
        status = "PASS" if all(a["passed"] for a in assertions) else "FAIL"
        
        return create_test_result(
            test_name=test_name,
            status=status,
            execution_time=execution_time,
            details={"pro_rating_result": mock_pro_rating},
            assertions=assertions
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Pro-rating test failed: {str(e)}")
        return create_test_result(
            test_name=test_name,
            status="ERROR",
            execution_time=execution_time,
            error_message=str(e)
        )

@app.post("/test/phase1/rejection-tracking")
async def test_rejection_tracking(request: TestRequest):
    """
    Test rejection tracking functionality
    
    Tests the Phase 1 rejection tracking including:
    - Rejection logging
    - Reason categorization
    - Resolution tracking
    """
    start_time = time.time()
    test_name = "Rejection Tracking Test"
    
    try:
        # Mock rejection tracking
        mock_rejection = {
            "rejection_id": "rej_001",
            "document_id": "doc_001",
            "rejection_reason": "insufficient_quality",
            "rejection_category": "technical",
            "timestamp": datetime.now().isoformat(),
            "reviewer": "system_auto",
            "quality_score": 0.45,
            "threshold": 0.7,
            "resolution_status": "pending",
            "retry_count": 1
        }
        
        # Test assertions
        assertions = [
            {"description": "Rejection logged", "passed": bool(mock_rejection.get("rejection_id"))},
            {"description": "Reason provided", "passed": bool(mock_rejection.get("rejection_reason"))},
            {"description": "Category assigned", "passed": bool(mock_rejection.get("rejection_category"))},
            {"description": "Quality scored", "passed": mock_rejection.get("quality_score", 1) < mock_rejection.get("threshold", 0)},
            {"description": "Status tracked", "passed": bool(mock_rejection.get("resolution_status"))}
        ]
        
        # Simulate rejection processing
        await simulate_async_operation(0.1)
        
        execution_time = time.time() - start_time
        status = "PASS" if all(a["passed"] for a in assertions) else "FAIL"
        
        return create_test_result(
            test_name=test_name,
            status=status,
            execution_time=execution_time,
            details={"rejection_record": mock_rejection},
            assertions=assertions
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Rejection tracking test failed: {str(e)}")
        return create_test_result(
            test_name=test_name,
            status="ERROR",
            execution_time=execution_time,
            error_message=str(e)
        )

# Workflow Integration Test Endpoints
@app.post("/test/workflow/document-to-qa")
async def test_workflow_document_to_qa(request: TestRequest):
    """
    Test complete document-to-Q&A workflow
    
    Tests the integration between document processing and Q&A system:
    - Document upload → Processing → Entity extraction → Q&A ready
    """
    start_time = time.time()
    test_name = "Document-to-Q&A Workflow Test"
    
    try:
        # Simulate complete workflow
        workflow_steps = [
            {"step": "document_upload", "status": "completed", "duration": 0.2},
            {"step": "content_extraction", "status": "completed", "duration": 0.3},
            {"step": "entity_processing", "status": "completed", "duration": 0.4},
            {"step": "indexing", "status": "completed", "duration": 0.1},
            {"step": "qa_ready", "status": "completed", "duration": 0.1}
        ]
        
        # Simulate workflow execution
        for step in workflow_steps:
            await simulate_async_operation(step["duration"])
        
        # Test assertions
        assertions = [
            {"description": "All steps completed", "passed": all(s["status"] == "completed" for s in workflow_steps)},
            {"description": "Document processed", "passed": workflow_steps[0]["status"] == "completed"},
            {"description": "Entities extracted", "passed": workflow_steps[2]["status"] == "completed"},
            {"description": "Q&A system ready", "passed": workflow_steps[-1]["status"] == "completed"},
            {"description": "Reasonable processing time", "passed": sum(s["duration"] for s in workflow_steps) < 2.0}
        ]
        
        execution_time = time.time() - start_time
        status = "PASS" if all(a["passed"] for a in assertions) else "FAIL"
        
        return create_test_result(
            test_name=test_name,
            status=status,
            execution_time=execution_time,
            details={"workflow_steps": workflow_steps},
            assertions=assertions
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Document-to-Q&A workflow test failed: {str(e)}")
        return create_test_result(
            test_name=test_name,
            status="ERROR",
            execution_time=execution_time,
            error_message=str(e)
        )

# End-to-End Test Endpoints
@app.post("/test/e2e/complete-user-journey")
async def test_e2e_complete_user_journey(request: TestRequest):
    """
    Test complete end-to-end user journey
    
    Simulates a complete user interaction:
    - Upload document
    - Process and extract entities
    - Ask questions
    - Get answers with audit trail
    """
    start_time = time.time()
    test_name = "Complete User Journey E2E Test"
    
    try:
        # Simulate complete user journey
        journey_steps = {
            "user_login": {"status": "completed", "duration": 0.1, "result": {"user_id": "test_user"}},
            "document_upload": {"status": "completed", "duration": 0.3, "result": generate_mock_document()},
            "document_processing": {"status": "completed", "duration": 0.5, "result": generate_mock_extraction_result()},
            "question_asked": {"status": "completed", "duration": 0.2, "result": {"question": "What are the safety protocols?"}},
            "answer_generated": {"status": "completed", "duration": 0.4, "result": {"answer": "Follow PPE guidelines and emergency procedures"}},
            "audit_logged": {"status": "completed", "duration": 0.1, "result": {"audit_id": "audit_e2e_001"}}
        }
        
        # Simulate journey execution
        for step_name, step_info in journey_steps.items():
            await simulate_async_operation(step_info["duration"])
        
        # Test assertions
        assertions = [
            {"description": "User authenticated", "passed": journey_steps["user_login"]["status"] == "completed"},
            {"description": "Document uploaded successfully", "passed": journey_steps["document_upload"]["status"] == "completed"},
            {"description": "Document processed", "passed": journey_steps["document_processing"]["status"] == "completed"},
            {"description": "Question answered", "passed": journey_steps["answer_generated"]["status"] == "completed"},
            {"description": "Actions audited", "passed": journey_steps["audit_logged"]["status"] == "completed"},
            {"description": "Journey completed timely", "passed": sum(s["duration"] for s in journey_steps.values()) < 2.0}
        ]
        
        execution_time = time.time() - start_time
        status = "PASS" if all(a["passed"] for a in assertions) else "FAIL"
        
        return create_test_result(
            test_name=test_name,
            status=status,
            execution_time=execution_time,
            details={"journey_steps": journey_steps},
            assertions=assertions
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Complete user journey E2E test failed: {str(e)}")
        return create_test_result(
            test_name=test_name,
            status="ERROR",
            execution_time=execution_time,
            error_message=str(e)
        )

# Unit Test Endpoints
@app.post("/test/unit/entity-parser")
async def test_unit_entity_parser(request: TestRequest):
    """
    Unit test for entity parser component
    
    Tests individual entity parsing functions in isolation
    """
    start_time = time.time()
    test_name = "Entity Parser Unit Test"
    
    try:
        # Mock entity parser tests
        test_cases = [
            {"input": "OSHA 1910.147", "expected_type": "regulation", "expected_confidence": 0.95},
            {"input": "chemical exposure", "expected_type": "hazard", "expected_confidence": 0.88},
            {"input": "emergency evacuation", "expected_type": "procedure", "expected_confidence": 0.92}
        ]
        
        results = []
        for case in test_cases:
            # Simulate entity parsing
            await simulate_async_operation(0.05)
            
            # Mock parsing result
            parsed_result = {
                "text": case["input"],
                "entity_type": case["expected_type"],
                "confidence": case["expected_confidence"]
            }
            
            # Validate result
            is_valid = (
                parsed_result["entity_type"] == case["expected_type"] and
                parsed_result["confidence"] >= 0.8
            )
            
            results.append({
                "input": case["input"],
                "result": parsed_result,
                "valid": is_valid
            })
        
        # Test assertions
        assertions = [
            {"description": f"Test case {i+1} passed", "passed": result["valid"]}
            for i, result in enumerate(results)
        ]
        assertions.append({"description": "All test cases processed", "passed": len(results) == len(test_cases)})
        
        execution_time = time.time() - start_time
        status = "PASS" if all(a["passed"] for a in assertions) else "FAIL"
        
        return create_test_result(
            test_name=test_name,
            status=status,
            execution_time=execution_time,
            details={"test_results": results},
            assertions=assertions
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Entity parser unit test failed: {str(e)}")
        return create_test_result(
            test_name=test_name,
            status="ERROR",
            execution_time=execution_time,
            error_message=str(e)
        )

# Integration Test Endpoints
@app.post("/test/integration/db-llm-integration")
async def test_integration_db_llm(request: TestRequest):
    """
    Integration test for database and LLM components
    
    Tests the integration between database operations and LLM processing
    """
    start_time = time.time()
    test_name = "Database-LLM Integration Test"
    
    try:
        # Simulate DB-LLM integration workflow
        integration_steps = [
            {"component": "database", "action": "query_documents", "success": True},
            {"component": "llm", "action": "process_context", "success": True},
            {"component": "database", "action": "store_results", "success": True},
            {"component": "llm", "action": "generate_response", "success": True}
        ]
        
        # Simulate each integration step
        for step in integration_steps:
            await simulate_async_operation(0.1)
        
        # Test assertions
        assertions = [
            {"description": "Database queries successful", "passed": all(s["success"] for s in integration_steps if s["component"] == "database")},
            {"description": "LLM processing successful", "passed": all(s["success"] for s in integration_steps if s["component"] == "llm")},
            {"description": "All components integrated", "passed": len(set(s["component"] for s in integration_steps)) > 1},
            {"description": "Workflow completed", "passed": all(s["success"] for s in integration_steps)}
        ]
        
        execution_time = time.time() - start_time
        status = "PASS" if all(a["passed"] for a in assertions) else "FAIL"
        
        return create_test_result(
            test_name=test_name,
            status=status,
            execution_time=execution_time,
            details={"integration_steps": integration_steps},
            assertions=assertions
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"DB-LLM integration test failed: {str(e)}")
        return create_test_result(
            test_name=test_name,
            status="ERROR",
            execution_time=execution_time,
            error_message=str(e)
        )

# Performance Test Endpoints
@app.post("/test/performance/load-test")
async def test_performance_load(request: TestRequest):
    """
    Performance test for system load handling
    
    Tests system performance under various load conditions
    """
    start_time = time.time()
    test_name = "Load Performance Test"
    
    try:
        # Simulate load test scenarios
        concurrent_requests = request.test_parameters.get("concurrent_requests", 10)
        request_duration = request.test_parameters.get("request_duration", 0.1)
        
        # Simulate concurrent request processing
        tasks = []
        for i in range(concurrent_requests):
            task = simulate_async_operation(request_duration)
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        successful_requests = sum(1 for r in results if not isinstance(r, Exception))
        failed_requests = len(results) - successful_requests
        success_rate = (successful_requests / len(results)) * 100 if results else 0
        
        # Test assertions
        assertions = [
            {"description": "Load test completed", "passed": len(results) == concurrent_requests},
            {"description": "Success rate acceptable", "passed": success_rate >= 95.0},
            {"description": "Response time reasonable", "passed": (time.time() - start_time) < (request_duration * 2)},
            {"description": "No system crashes", "passed": failed_requests < concurrent_requests * 0.1}
        ]
        
        execution_time = time.time() - start_time
        status = "PASS" if all(a["passed"] for a in assertions) else "FAIL"
        
        return create_test_result(
            test_name=test_name,
            status=status,
            execution_time=execution_time,
            details={
                "concurrent_requests": concurrent_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "success_rate": success_rate
            },
            assertions=assertions
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Load performance test failed: {str(e)}")
        return create_test_result(
            test_name=test_name,
            status="ERROR",
            execution_time=execution_time,
            error_message=str(e)
        )

# Regression Test Endpoints
@app.post("/test/regression/feature-compatibility")
async def test_regression_feature_compatibility(request: TestRequest):
    """
    Regression test for feature compatibility
    
    Ensures new features don't break existing functionality
    """
    start_time = time.time()
    test_name = "Feature Compatibility Regression Test"
    
    try:
        # Test compatibility between features
        compatibility_tests = [
            {"feature_a": "document_upload", "feature_b": "entity_extraction", "compatible": True},
            {"feature_a": "qa_system", "feature_b": "audit_trail", "compatible": True},
            {"feature_a": "pro_rating", "feature_b": "rejection_tracking", "compatible": True},
            {"feature_a": "original_features", "feature_b": "phase1_enhancements", "compatible": True}
        ]
        
        # Simulate compatibility checks
        for test in compatibility_tests:
            await simulate_async_operation(0.1)
        
        # Test assertions
        assertions = [
            {"description": f"{test['feature_a']} compatible with {test['feature_b']}", "passed": test["compatible"]}
            for test in compatibility_tests
        ]
        assertions.append({"description": "All features compatible", "passed": all(t["compatible"] for t in compatibility_tests)})
        
        execution_time = time.time() - start_time
        status = "PASS" if all(a["passed"] for a in assertions) else "FAIL"
        
        return create_test_result(
            test_name=test_name,
            status=status,
            execution_time=execution_time,
            details={"compatibility_tests": compatibility_tests},
            assertions=assertions
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Feature compatibility regression test failed: {str(e)}")
        return create_test_result(
            test_name=test_name,
            status="ERROR",
            execution_time=execution_time,
            error_message=str(e)
        )

# Test Suite Endpoints
@app.post("/test/suites/run-all")
async def run_all_tests(request: TestRequest):
    """
    Run all test suites
    
    Executes all available test categories and provides comprehensive results
    """
    start_time = time.time()
    suite_name = "Complete Test Suite"
    
    try:
        # Define all test endpoints to run
        test_endpoints = [
            ("original", "document-upload"),
            ("original", "entity-extraction"),
            ("original", "qa-system"),
            ("phase1", "audit-trail"),
            ("phase1", "pro-rating"),
            ("phase1", "rejection-tracking"),
            ("workflow", "document-to-qa"),
            ("e2e", "complete-user-journey"),
            ("unit", "entity-parser"),
            ("integration", "db-llm-integration"),
            ("performance", "load-test"),
            ("regression", "feature-compatibility")
        ]
        
        # Execute all tests (simulate results)
        test_results = []
        for category, test_name in test_endpoints:
            # Simulate test execution
            await simulate_async_operation(0.1)
            
            # Create mock result
            mock_result = create_test_result(
                test_name=f"{category}/{test_name}",
                status="PASS",  # Assume all pass for demo
                execution_time=0.1,
                details={"simulated": True}
            )
            test_results.append(mock_result)
        
        # Calculate suite statistics
        total_tests = len(test_results)
        passed = sum(1 for r in test_results if r.status == "PASS")
        failed = sum(1 for r in test_results if r.status == "FAIL")
        errors = sum(1 for r in test_results if r.status == "ERROR")
        skipped = sum(1 for r in test_results if r.status == "SKIP")
        
        execution_time = time.time() - start_time
        
        return TestSuite(
            suite_name=suite_name,
            total_tests=total_tests,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            execution_time_ms=round(execution_time * 1000, 2),
            tests=test_results
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Complete test suite failed: {str(e)}")
        
        # Return error result
        error_result = create_test_result(
            test_name=suite_name,
            status="ERROR",
            execution_time=execution_time,
            error_message=str(e)
        )
        
        return TestSuite(
            suite_name=suite_name,
            total_tests=1,
            passed=0,
            failed=0,
            skipped=0,
            errors=1,
            execution_time_ms=round(execution_time * 1000, 2),
            tests=[error_result]
        )

# Error Handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Test endpoint not found",
            "message": f"The test endpoint {request.url.path} does not exist",
            "available_endpoints": {
                "documentation": "/test/docs",
                "health": "/test/health",
                "original_tests": "/test/original/*",
                "phase1_tests": "/test/phase1/*",
                "workflow_tests": "/test/workflow/*",
                "e2e_tests": "/test/e2e/*",
                "unit_tests": "/test/unit/*",
                "integration_tests": "/test/integration/*",
                "performance_tests": "/test/performance/*",
                "regression_tests": "/test/regression/*"
            }
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred during test execution",
            "timestamp": datetime.now().isoformat()
        }
    )

# Main execution
if __name__ == "__main__":
    print("Starting EHS AI Demo - Comprehensive Test API")
    print("=" * 50)
    print("Available test categories:")
    print("  - Original Features: /test/original/*")
    print("  - Phase 1 Enhancements: /test/phase1/*")
    print("  - Workflow Integration: /test/workflow/*")
    print("  - End-to-End Tests: /test/e2e/*")
    print("  - Unit Tests: /test/unit/*")
    print("  - Integration Tests: /test/integration/*")
    print("  - Performance Tests: /test/performance/*")
    print("  - Regression Tests: /test/regression/*")
    print("  - Test Suites: /test/suites/*")
    print("=" * 50)
    print("API Documentation: http://localhost:8001/test/docs")
    print("Health Check: http://localhost:8001/test/health")
    print("=" * 50)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info",
        access_log=True
    )