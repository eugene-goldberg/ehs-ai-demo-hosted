#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite for Chatbot API

This test suite provides comprehensive coverage of the chatbot API,
including all endpoints, intent analysis, session management, EHS data integration,
error handling, and response formatting.

Features Tested:
- All API endpoints (/api/chatbot/chat, /api/chatbot/health, /api/chatbot/clear-session)
- Chat message processing and intent analysis
- Session management and conversation context
- EHS data integration (electricity, water, waste)
- Error handling scenarios
- Response format validation
- Neo4j integration
- Security and validation aspects

Created: 2025-09-15
Version: 1.0.0
Author: Claude Code Agent
"""

import os
import sys
import json
import pytest
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass
from contextlib import asynccontextmanager

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from dotenv import load_dotenv

# Import the API module
from api.chatbot_api import (
    router as chatbot_router,
    ChatMessage, ChatRequest, ChatResponse, HealthResponse, SessionClearResponse,
    chat_sessions, generate_session_id, get_or_create_session,
    analyze_user_intent, generate_ehs_response,
    format_electricity_response, format_water_response, format_waste_response
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test Configuration
@dataclass
class TestConfig:
    """Test configuration constants"""
    API_BASE_URL = "/api/chatbot"
    TEST_TIMEOUT = 30
    MAX_RESPONSE_TIME = 5.0
    SAMPLE_SESSION_ID = "test-session-12345"
    TEST_SITE_ALGONQUIN = "algonquin_il"
    TEST_SITE_HOUSTON = "houston_tx"

# Test Data
class TestDataGenerator:
    """Generate test data for various scenarios"""
    
    @staticmethod
    def get_sample_chat_request(message: str = "Hello", session_id: str = None, site_filter: str = None) -> Dict:
        """Generate sample chat request"""
        request = {
            "message": message,
            "context": {"test": True}
        }
        if session_id:
            request["session_id"] = session_id
        if site_filter:
            request["site_filter"] = site_filter
        return request
    
    @staticmethod
    def get_sample_consumption_data() -> Dict:
        """Generate sample consumption data"""
        return {
            "total_sum": 12500.75,
            "average": 416.69,
            "max_value": 850.25,
            "min_value": 125.50,
            "count": 30
        }
    
    @staticmethod
    def get_sample_multi_site_data() -> Dict:
        """Generate sample multi-site data"""
        return {
            "algonquin_il": TestDataGenerator.get_sample_consumption_data(),
            "houston_tx": {
                "total_sum": 8750.25,
                "average": 291.67,
                "max_value": 625.00,
                "min_value": 95.75,
                "count": 30
            }
        }

# Fixtures
@pytest.fixture
def test_app():
    """Create test FastAPI application"""
    app = FastAPI()
    app.include_router(chatbot_router)
    return app

@pytest.fixture
def client(test_app):
    """Create test client"""
    return TestClient(test_app)

@pytest.fixture
def clean_sessions():
    """Clean chat sessions before each test"""
    original_sessions = chat_sessions.copy()
    chat_sessions.clear()
    yield
    chat_sessions.clear()
    chat_sessions.update(original_sessions)

@pytest.fixture
def mock_neo4j_client():
    """Mock Neo4j client"""
    with patch('api.chatbot_api.get_neo4j_client') as mock:
        mock_client = AsyncMock()
        mock_client.execute_query.return_value = [{"test": 1}]
        mock.return_value = mock_client
        yield mock_client

@pytest.fixture
def mock_env_service():
    """Mock Environmental Assessment Service"""
    with patch('api.chatbot_api.get_environmental_service') as mock:
        mock_service = AsyncMock()
        
        # Setup default return values
        sample_data = TestDataGenerator.get_sample_consumption_data()
        mock_service.get_electricity_consumption_facts.return_value = sample_data
        mock_service.get_water_consumption_facts.return_value = sample_data
        mock_service.get_waste_generation_facts.return_value = sample_data
        
        mock.return_value = mock_service
        yield mock_service

# Test Classes
class TestChatbotModels:
    """Test Pydantic model validation"""
    
    def test_chat_message_validation(self):
        """Test ChatMessage model validation"""
        # Valid message
        message = ChatMessage(role="user", content="Hello chatbot")
        assert message.role == "user"
        assert message.content == "Hello chatbot"
        assert isinstance(message.timestamp, datetime)
        
        # Invalid role
        with pytest.raises(ValueError):
            ChatMessage(role="invalid", content="Hello")
    
    def test_chat_request_validation(self):
        """Test ChatRequest model validation"""
        # Valid request
        request = ChatRequest(message="How much electricity did we use?")
        assert request.message == "How much electricity did we use?"
        assert request.session_id is None
        assert request.context == {}
        
        # Request with all fields
        request = ChatRequest(
            message="Test message",
            session_id="test-123",
            context={"key": "value"},
            site_filter="algonquin_il"
        )
        assert request.session_id == "test-123"
        assert request.site_filter == "algonquin_il"
        
        # Empty message should fail
        with pytest.raises(ValueError):
            ChatRequest(message="")
    
    def test_chat_response_validation(self):
        """Test ChatResponse model validation"""
        response = ChatResponse(
            response="Test response",
            session_id="test-123",
            data_sources=["Neo4j"],
            suggestions=["Follow up question?"]
        )
        assert response.response == "Test response"
        assert response.session_id == "test-123"
        assert len(response.data_sources) == 1
        assert len(response.suggestions) == 1

class TestSessionManagement:
    """Test session management functionality"""
    
    def test_generate_session_id(self):
        """Test session ID generation"""
        session_id1 = generate_session_id()
        session_id2 = generate_session_id()
        
        assert isinstance(session_id1, str)
        assert isinstance(session_id2, str)
        assert session_id1 != session_id2
        assert len(session_id1) > 10  # UUID should be longer
    
    def test_get_or_create_session_new(self, clean_sessions):
        """Test creating new session"""
        session_id = get_or_create_session(None)
        
        assert session_id in chat_sessions
        assert "messages" in chat_sessions[session_id]
        assert "created_at" in chat_sessions[session_id]
        assert "last_activity" in chat_sessions[session_id]
        assert "context" in chat_sessions[session_id]
    
    def test_get_or_create_session_existing(self, clean_sessions):
        """Test getting existing session"""
        # Create a session first
        original_session_id = get_or_create_session(None)
        
        # Try to get the same session
        retrieved_session_id = get_or_create_session(original_session_id)
        
        assert original_session_id == retrieved_session_id
        assert len(chat_sessions) == 1

class TestIntentAnalysis:
    """Test intent analysis and entity extraction"""
    
    @pytest.mark.asyncio
    async def test_electricity_intent_detection(self):
        """Test electricity intent detection"""
        messages = [
            "How much electricity did we use?",
            "Show me power consumption",
            "What's our kWh usage?",
            "Electric bill analysis"
        ]
        
        for message in messages:
            result = await analyze_user_intent(message)
            assert "electricity_query" in result["intents"]
    
    @pytest.mark.asyncio
    async def test_water_intent_detection(self):
        """Test water intent detection"""
        messages = [
            "How much water did we consume?",
            "Show me H2O usage",
            "Water consumption trends",
            "Gallons used last month"
        ]
        
        for message in messages:
            result = await analyze_user_intent(message)
            assert "water_query" in result["intents"]
    
    @pytest.mark.asyncio
    async def test_waste_intent_detection(self):
        """Test waste intent detection"""
        messages = [
            "How much waste did we generate?",
            "Garbage disposal data",
            "Recycling statistics",
            "Trash output analysis"
        ]
        
        for message in messages:
            result = await analyze_user_intent(message)
            assert "waste_query" in result["intents"]
    
    @pytest.mark.asyncio
    async def test_site_detection(self):
        """Test site detection in messages"""
        test_cases = [
            ("Algonquin electricity usage", "algonquin_il"),
            ("Houston water consumption", "houston_tx"),
            ("Illinois facility data", "algonquin_il"),
            ("Texas location metrics", "houston_tx")
        ]
        
        for message, expected_site in test_cases:
            result = await analyze_user_intent(message)
            assert expected_site in result["sites"]
    
    @pytest.mark.asyncio
    async def test_multiple_intents(self):
        """Test detection of multiple intents"""
        message = "Compare electricity and water usage between Algonquin and Houston"
        result = await analyze_user_intent(message)
        
        assert "electricity_query" in result["intents"]
        assert "water_query" in result["intents"]
        assert "comparison_query" in result["intents"]
        assert "algonquin_il" in result["sites"]
        assert "houston_tx" in result["sites"]

class TestResponseFormatting:
    """Test response formatting functions"""
    
    def test_format_electricity_response_single_site(self):
        """Test electricity response formatting for single site"""
        data = TestDataGenerator.get_sample_consumption_data()
        
        response = format_electricity_response(data, "algonquin_il", {})
        
        assert "Algonquin, IL" in response
        assert "12500.75 kWh" in response
        assert "416.69 kWh" in response
        assert "850.25 kWh" in response
    
    def test_format_electricity_response_multi_site(self):
        """Test electricity response formatting for multiple sites"""
        data = TestDataGenerator.get_sample_multi_site_data()
        
        response = format_electricity_response(data, None, {})
        
        assert "Algonquin, IL" in response
        assert "Houston, TX" in response
        assert "12500.75 kWh" in response
        assert "8750.25 kWh" in response
    
    def test_format_water_response(self):
        """Test water response formatting"""
        data = TestDataGenerator.get_sample_consumption_data()
        
        response = format_water_response(data, "houston_tx", {})
        
        assert "Houston, TX" in response
        assert "gallons" in response
        assert "12500.75" in response
    
    def test_format_waste_response(self):
        """Test waste response formatting"""
        data = TestDataGenerator.get_sample_consumption_data()
        
        response = format_waste_response(data, None, {})
        
        assert "tons" in response
        assert "12500.75" in response

class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_endpoint_success(self, client, mock_neo4j_client):
        """Test successful health check"""
        response = client.get(f"{TestConfig.API_BASE_URL}/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data
        assert "dependencies" in data
        assert "neo4j" in data["dependencies"]
    
    def test_health_endpoint_neo4j_failure(self, client):
        """Test health check with Neo4j failure"""
        with patch('api.chatbot_api.get_neo4j_client') as mock:
            mock.side_effect = Exception("Neo4j connection failed")
            
            response = client.get(f"{TestConfig.API_BASE_URL}/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"

class TestChatEndpoint:
    """Test main chat endpoint"""
    
    def test_chat_basic_conversation(self, client, clean_sessions, mock_env_service):
        """Test basic chat conversation"""
        request_data = TestDataGenerator.get_sample_chat_request("Hello chatbot")
        
        response = client.post(f"{TestConfig.API_BASE_URL}/chat", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "response" in data
        assert "session_id" in data
        assert "timestamp" in data
        assert "suggestions" in data
        assert len(data["session_id"]) > 10  # Valid UUID
    
    def test_chat_electricity_query(self, client, clean_sessions, mock_env_service):
        """Test electricity-specific query"""
        request_data = TestDataGenerator.get_sample_chat_request(
            "How much electricity did we use at Algonquin?",
            site_filter="algonquin_il"
        )
        
        response = client.post(f"{TestConfig.API_BASE_URL}/chat", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "electricity" in data["response"].lower()
        assert "algonquin" in data["response"].lower()
        assert "Neo4j Database" in data.get("data_sources", [])
        mock_env_service.get_electricity_consumption_facts.assert_called_once()
    
    def test_chat_water_query(self, client, clean_sessions, mock_env_service):
        """Test water-specific query"""
        request_data = TestDataGenerator.get_sample_chat_request(
            "Show me water consumption for Houston",
            site_filter="houston_tx"
        )
        
        response = client.post(f"{TestConfig.API_BASE_URL}/chat", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "water" in data["response"].lower()
        assert "houston" in data["response"].lower()
        mock_env_service.get_water_consumption_facts.assert_called_once()
    
    def test_chat_waste_query(self, client, clean_sessions, mock_env_service):
        """Test waste-specific query"""
        request_data = TestDataGenerator.get_sample_chat_request("How much waste did we generate?")
        
        response = client.post(f"{TestConfig.API_BASE_URL}/chat", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "waste" in data["response"].lower()
        mock_env_service.get_waste_generation_facts.assert_called_once()
    
    def test_chat_session_continuity(self, client, clean_sessions, mock_env_service):
        """Test session continuity across multiple messages"""
        # First message
        request1 = TestDataGenerator.get_sample_chat_request("Hello")
        response1 = client.post(f"{TestConfig.API_BASE_URL}/chat", json=request1)
        session_id = response1.json()["session_id"]
        
        # Second message with same session
        request2 = TestDataGenerator.get_sample_chat_request("Show electricity usage", session_id)
        response2 = client.post(f"{TestConfig.API_BASE_URL}/chat", json=request2)
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response2.json()["session_id"] == session_id
        
        # Verify session has messages
        assert session_id in chat_sessions
        assert len(chat_sessions[session_id]["messages"]) == 4  # 2 user + 2 assistant messages
    
    def test_chat_invalid_request(self, client):
        """Test chat with invalid request data"""
        # Empty message
        response = client.post(f"{TestConfig.API_BASE_URL}/chat", json={"message": ""})
        assert response.status_code == 422
        
        # Missing message
        response = client.post(f"{TestConfig.API_BASE_URL}/chat", json={})
        assert response.status_code == 422
    
    def test_chat_service_error(self, client, clean_sessions):
        """Test chat with service error"""
        with patch('api.chatbot_api.get_environmental_service') as mock:
            mock.side_effect = Exception("Service unavailable")
            
            request_data = TestDataGenerator.get_sample_chat_request("Hello")
            response = client.post(f"{TestConfig.API_BASE_URL}/chat", json=request_data)
            
            assert response.status_code == 500

class TestSessionEndpoints:
    """Test session management endpoints"""
    
    def test_clear_session_existing(self, client, clean_sessions):
        """Test clearing existing session"""
        # Create a session first
        session_id = get_or_create_session(None)
        
        response = client.post(f"{TestConfig.API_BASE_URL}/clear-session?session_id={session_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["session_id"] == session_id
        assert "successfully cleared" in data["message"]
        assert session_id not in chat_sessions
    
    def test_clear_session_nonexistent(self, client, clean_sessions):
        """Test clearing non-existent session"""
        fake_session_id = "fake-session-id"
        
        response = client.post(f"{TestConfig.API_BASE_URL}/clear-session?session_id={fake_session_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["session_id"] == fake_session_id
        assert "not found" in data["message"]
    
    def test_list_active_sessions(self, client, clean_sessions):
        """Test listing active sessions"""
        # Create some sessions
        session1 = get_or_create_session(None)
        session2 = get_or_create_session(None)
        
        response = client.get(f"{TestConfig.API_BASE_URL}/sessions")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["active_sessions"] == 2
        assert session1 in data["sessions"]
        assert session2 in data["sessions"]
    
    def test_get_session_history(self, client, clean_sessions):
        """Test getting session history"""
        # Create a session and add some messages
        session_id = get_or_create_session(None)
        
        # Add messages via chat endpoint
        request_data = TestDataGenerator.get_sample_chat_request("Hello", session_id)
        with patch('api.chatbot_api.get_environmental_service'):
            client.post(f"{TestConfig.API_BASE_URL}/chat", json=request_data)
        
        response = client.get(f"{TestConfig.API_BASE_URL}/sessions/{session_id}/history")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["session_id"] == session_id
        assert "messages" in data
        assert len(data["messages"]) >= 2  # User + assistant message
    
    def test_get_session_history_not_found(self, client, clean_sessions):
        """Test getting history for non-existent session"""
        fake_session_id = "fake-session-id"
        
        response = client.get(f"{TestConfig.API_BASE_URL}/sessions/{fake_session_id}/history")
        
        assert response.status_code == 404

# Test Runner
if __name__ == "__main__":
    """Run the test suite"""
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--log-cli-level=INFO",
        "--cov=api.chatbot_api",
        "--cov-report=html:htmlcov",
        "--cov-report=term-missing"
    ])
