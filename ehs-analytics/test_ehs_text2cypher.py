#!/usr/bin/env python3
"""
Test script for EHS Text2Cypher implementation.

This script demonstrates the EHS-specific Text2Cypher retriever capabilities
including intent detection, query optimization, and example usage.
"""

import asyncio
import json
from typing import Dict, Any

from ehs_analytics.retrieval.strategies.ehs_text2cypher import (
    EHSText2CypherRetriever, 
    EHSQueryIntent
)
from ehs_analytics.retrieval.ehs_examples import (
    get_example_summary,
    get_consumption_analysis_examples,
    get_compliance_check_examples,
    get_equipment_efficiency_examples
)
from ehs_analytics.retrieval.base import QueryType


def test_intent_detection():
    """Test EHS intent detection functionality."""
    print("=" * 60)
    print("Testing EHS Intent Detection")
    print("=" * 60)
    
    # Create a mock retriever for testing intent detection
    config = {
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "password",
        "openai_api_key": "test-key",
        "use_graphrag": False  # Disable GraphRAG for basic testing
    }
    
    retriever = EHSText2CypherRetriever(config)
    
    test_queries = [
        ("What is the water consumption for Plant A last month?", "consumption_analysis"),
        ("Which permits are expiring in the next 30 days?", "compliance_check"),
        ("Show me incidents with high severity levels", "risk_assessment"),
        ("What are the CO2 emissions for all facilities?", "emission_tracking"),
        ("Which equipment has the lowest efficiency rating?", "equipment_efficiency"),
        ("What is the status of environmental permits?", "permit_status"),
        ("Show me all facilities and their types", "general_inquiry")
    ]
    
    for query, expected_intent in test_queries:
        detected_intent = retriever._detect_ehs_intent(query, QueryType.GENERAL)
        print(f"Query: {query}")
        print(f"Expected: {expected_intent}")
        print(f"Detected: {detected_intent.value}")
        print(f"Match: {'✓' if detected_intent.value == expected_intent else '✗'}")
        print("-" * 40)


def test_query_validation():
    """Test EHS query validation functionality."""
    print("\n" + "=" * 60)
    print("Testing EHS Query Validation")
    print("=" * 60)
    
    config = {
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_user": "neo4j", 
        "neo4j_password": "password",
        "openai_api_key": "test-key",
        "use_graphrag": False
    }
    
    retriever = EHSText2CypherRetriever(config)
    
    test_cases = [
        # Valid consumption queries
        ("Show water usage for all facilities", EHSQueryIntent.CONSUMPTION_ANALYSIS, True),
        ("What is the electricity consumption trend?", EHSQueryIntent.CONSUMPTION_ANALYSIS, True),
        
        # Valid compliance queries
        ("Which permits are expiring soon?", EHSQueryIntent.COMPLIANCE_CHECK, True),
        ("Show compliance status for facility X", EHSQueryIntent.COMPLIANCE_CHECK, True),
        
        # Invalid queries
        ("", EHSQueryIntent.GENERAL_INQUIRY, False),
        ("Hi", EHSQueryIntent.GENERAL_INQUIRY, False),
        ("Random text without EHS context", EHSQueryIntent.CONSUMPTION_ANALYSIS, False),
    ]
    
    async def run_validation_tests():
        for query, intent, expected_valid in test_cases:
            is_valid = await retriever._validate_ehs_query(query, intent)
            print(f"Query: '{query}'")
            print(f"Intent: {intent.value}")
            print(f"Expected Valid: {expected_valid}")
            print(f"Actual Valid: {is_valid}")
            print(f"Result: {'✓' if is_valid == expected_valid else '✗'}")
            print("-" * 40)
    
    asyncio.run(run_validation_tests())


def test_query_optimization():
    """Test EHS query optimization functionality."""
    print("\n" + "=" * 60)
    print("Testing EHS Query Optimization")
    print("=" * 60)
    
    config = {
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "password", 
        "openai_api_key": "test-key",
        "query_optimization": True,
        "use_graphrag": False
    }
    
    retriever = EHSText2CypherRetriever(config)
    
    test_queries = [
        ("Show water consumption", EHSQueryIntent.CONSUMPTION_ANALYSIS),
        ("Check permit status", EHSQueryIntent.COMPLIANCE_CHECK),
        ("Analyze equipment efficiency", EHSQueryIntent.EQUIPMENT_EFFICIENCY),
    ]
    
    async def run_optimization_tests():
        for query, intent in test_queries:
            optimized = await retriever._optimize_query(query, intent, QueryType.GENERAL)
            print(f"Original: {query}")
            print(f"Intent: {intent.value}")
            print(f"Optimized: {optimized}")
            print(f"Improvement: {len(optimized) - len(query)} characters added")
            print("-" * 40)
    
    asyncio.run(run_optimization_tests())


def test_examples_module():
    """Test the EHS examples module functionality."""
    print("\n" + "=" * 60)
    print("Testing EHS Examples Module")
    print("=" * 60)
    
    # Get summary
    summary = get_example_summary()
    print(f"Total Examples: {summary['total_examples']}")
    print(f"Intent Types: {len(summary['intent_types'])}")
    print("\nExamples by Intent:")
    for intent, count in summary['examples_by_intent'].items():
        print(f"  {intent}: {count} examples")
    
    print("\n" + "-" * 40)
    print("Sample Examples:")
    
    # Show sample from each major category
    categories = [
        ("Consumption Analysis", get_consumption_analysis_examples),
        ("Compliance Check", get_compliance_check_examples),
        ("Equipment Efficiency", get_equipment_efficiency_examples)
    ]
    
    for category_name, get_examples_func in categories:
        examples = get_examples_func()
        if examples:
            example = examples[0]
            print(f"\n{category_name} Example:")
            print(f"Question: {example['question']}")
            print(f"Description: {example['description']}")
            print(f"Cypher (first 100 chars): {example['cypher'][:100]}...")


def test_complexity_calculation():
    """Test query complexity calculation."""
    print("\n" + "=" * 60)
    print("Testing Query Complexity Calculation")
    print("=" * 60)
    
    config = {
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "password",
        "openai_api_key": "test-key",
        "use_graphrag": False
    }
    
    retriever = EHSText2CypherRetriever(config)
    
    test_queries = [
        "Show facilities",  # Simple
        "Count total water consumption for all facilities",  # Medium
        "Calculate average efficiency for equipment related to water systems during last month",  # Complex
        "Show sum of emissions connected to facilities between January and March with equipment breakdown"  # Very complex
    ]
    
    for query in test_queries:
        complexity = retriever._calculate_query_complexity(query)
        print(f"Query: {query}")
        print(f"Complexity Score: {complexity}")
        print(f"Category: {'Simple' if complexity <= 2 else 'Medium' if complexity <= 5 else 'Complex'}")
        print("-" * 40)


def main():
    """Run all tests."""
    print("EHS Text2Cypher Retriever Test Suite")
    print("=" * 60)
    
    try:
        test_intent_detection()
        test_query_validation()
        test_query_optimization()
        test_examples_module()
        test_complexity_calculation()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()