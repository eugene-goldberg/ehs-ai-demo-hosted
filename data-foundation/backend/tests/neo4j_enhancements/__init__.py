"""
Neo4j Enhancements Test Package

This package contains comprehensive integration tests for the enhanced Neo4j schema
implementation, including:

- Schema creation and constraint validation
- Node lifecycle management (Goals, Targets, Recommendations, etc.)
- Trend analysis system testing
- Forecast generation and validation
- LLM query template functionality
- Analytics aggregation operations
- End-to-end workflow validation

Test Configuration:
- Uses separate test database to avoid data conflicts
- Includes fixtures for data setup and teardown
- Supports both individual test categories and full test suite execution

Usage:
    # Run all tests
    pytest tests/neo4j_enhancements/

    # Run specific test category
    python test_enhanced_schema_integration.py schema
    python test_enhanced_schema_integration.py goals
    python test_enhanced_schema_integration.py trends
    etc.

Author: Claude AI Assistant
Created: 2025-08-28
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Claude AI Assistant"

# Test configuration constants
TEST_DATABASE_NAME = "test_enhanced_schema"
TEST_FACILITY_NAME = "Test Manufacturing Facility"
TEST_DEPARTMENT_NAME = "Safety Department"

# Export main test classes for easier imports
from .test_enhanced_schema_integration import (
    TestSchemaCreationAndIndexes,
    TestGoalAndTargetLifecycle,
    TestTrendAnalysisWorkflow,
    TestRecommendationSystem,
    TestForecastGeneration,
    TestLLMQueryTemplates,
    TestAnalyticsAggregation,
    TestEndToEndWorkflows,
    TestConfiguration
)

__all__ = [
    "TestSchemaCreationAndIndexes",
    "TestGoalAndTargetLifecycle", 
    "TestTrendAnalysisWorkflow",
    "TestRecommendationSystem",
    "TestForecastGeneration",
    "TestLLMQueryTemplates",
    "TestAnalyticsAggregation",
    "TestEndToEndWorkflows",
    "TestConfiguration",
    "TEST_DATABASE_NAME",
    "TEST_FACILITY_NAME",
    "TEST_DEPARTMENT_NAME"
]