"""
Example usage of the Text2Cypher retriever for EHS Analytics.

This script demonstrates how to initialize and use the Text2CypherRetriever
to convert natural language queries into Cypher queries for Neo4j.
"""

import asyncio
import os
from typing import Dict, Any

from ehs_analytics.retrieval import Text2CypherRetriever, QueryType


async def main():
    """Demonstrate Text2Cypher retriever usage."""
    
    # Configuration for the retriever
    config = {
        "neo4j_uri": os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
        "neo4j_user": os.getenv("NEO4J_USER", "neo4j"),
        "neo4j_password": os.getenv("NEO4J_PASSWORD", "password"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "model_name": "gpt-3.5-turbo",
        "temperature": 0.0,
        "max_tokens": 2000,
        "cypher_validation": True
    }
    
    # Initialize the retriever
    retriever = Text2CypherRetriever(config)
    
    try:
        # Initialize connections
        print("Initializing Text2Cypher retriever...")
        await retriever.initialize()
        print("✅ Retriever initialized successfully")
        
        # Health check
        health = await retriever.health_check()
        print(f"Health status: {health}")
        
        # Example queries for different EHS scenarios
        test_queries = [
            {
                "query": "Show me the total water consumption for all facilities last month",
                "type": QueryType.CONSUMPTION,
                "description": "Water consumption analysis"
            },
            {
                "query": "Which permits are expiring in the next 30 days?",
                "type": QueryType.COMPLIANCE,
                "description": "Permit compliance check"
            },
            {
                "query": "Find equipment with efficiency rating below 80%",
                "type": QueryType.EFFICIENCY,
                "description": "Equipment efficiency analysis"
            },
            {
                "query": "List all CO2 emissions recorded this year by facility",
                "type": QueryType.EMISSIONS,
                "description": "Emission tracking query"
            },
            {
                "query": "What facilities have had safety incidents in the past 6 months?",
                "type": QueryType.RISK,
                "description": "Risk assessment query"
            }
        ]
        
        # Execute test queries
        for i, test_case in enumerate(test_queries, 1):
            print(f"\n{'='*60}")
            print(f"Test Query {i}: {test_case['description']}")
            print(f"Query: {test_case['query']}")
            print(f"Type: {test_case['type'].value}")
            print("="*60)
            
            # Validate query first
            is_valid = await retriever.validate_query(test_case["query"])
            print(f"Query validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
            
            if is_valid:
                try:
                    # Execute retrieval
                    result = await retriever.retrieve(
                        query=test_case["query"],
                        query_type=test_case["type"],
                        limit=10
                    )
                    
                    print(f"Success: {result.success}")
                    print(f"Message: {result.message}")
                    print(f"Results count: {len(result.data)}")
                    print(f"Execution time: {result.metadata.execution_time_ms:.2f}ms")
                    print(f"Confidence score: {result.metadata.confidence_score:.2f}")
                    
                    if result.metadata.cypher_query:
                        print(f"Generated Cypher: {result.metadata.cypher_query}")
                    
                    if result.data:
                        print("Sample results:")
                        for j, item in enumerate(result.data[:3], 1):
                            print(f"  {j}. {item}")
                    
                    if not result.success and result.metadata.error_message:
                        print(f"Error: {result.metadata.error_message}")
                        
                except Exception as e:
                    print(f"❌ Query execution failed: {e}")
            else:
                print("Skipping invalid query")
        
        print(f"\n{'='*60}")
        print("Testing completed successfully!")
        
    except Exception as e:
        print(f"❌ Failed to initialize or run retriever: {e}")
        
    finally:
        # Cleanup
        print("\nCleaning up resources...")
        await retriever.cleanup()
        print("✅ Cleanup completed")


def print_schema_info():
    """Print EHS schema information."""
    from ehs_analytics.retrieval.base import EHSSchemaAware
    
    schema = EHSSchemaAware()
    
    print("EHS Database Schema Information:")
    print("=" * 50)
    
    print("\nNode Types:")
    for node_type, info in schema.NODE_TYPES.items():
        print(f"  {node_type}: {info['description']}")
        print(f"    Properties: {', '.join(info['properties'])}")
    
    print("\nRelationships:")
    for rel, desc in schema.RELATIONSHIPS.items():
        print(f"  {rel}: {desc}")
    
    print("\nQuery Type Mappings:")
    for query_type in QueryType:
        relevant_nodes = schema.get_relevant_nodes(query_type)
        print(f"  {query_type.value}: {', '.join(relevant_nodes)}")


if __name__ == "__main__":
    print("EHS Analytics - Text2Cypher Retriever Example")
    print("=" * 50)
    
    # Print schema information
    print_schema_info()
    
    print("\nRunning retriever tests...")
    
    # Run the async main function
    asyncio.run(main())