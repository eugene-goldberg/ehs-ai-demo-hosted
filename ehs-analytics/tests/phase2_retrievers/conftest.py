"""
Test configuration and fixtures for Phase 2 retriever tests.
"""

import pytest
import pytest_asyncio
import asyncio
import os
from typing import AsyncGenerator

# Set environment variables for testing
os.environ.setdefault("TESTING", "true")
os.environ.setdefault("LOG_LEVEL", "INFO")

@pytest_asyncio.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_environment():
    """Setup test environment variables."""
    env_vars = {
        "NEO4J_URI": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "NEO4J_USERNAME": os.getenv("NEO4J_USERNAME", "neo4j"),
        "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD", "EhsAI2024!"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "TESTING": "true"
    }
    
    # Validate required environment variables
    missing_vars = [key for key, value in env_vars.items() if not value and key != "TESTING"]
    if missing_vars:
        pytest.skip(f"Missing required environment variables: {missing_vars}")
    
    return env_vars
