"""
Example script showing how to use the Neo4j index definitions and management utilities.

This script demonstrates:
1. Creating indexes with priority levels
2. Checking index status
3. Generating creation scripts
4. Performance analysis

Usage:
    python index_setup_example.py
"""

import logging
import sys
import os
from datetime import datetime

# Add the src directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from langchain_neo4j import Neo4jGraph
from src.neo4j_enhancements.schema.indexes import (
    EHSIndexDefinitions,
    IndexManager,
    get_high_priority_indexes,
    get_indexes_by_node_type,
    generate_index_creation_script
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_neo4j_connection():
    """Setup Neo4j connection using environment variables."""
    try:
        # These would typically come from environment variables
        graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
            database=os.getenv("NEO4J_DATABASE", "neo4j")
        )
        return graph
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        return None


def demonstrate_index_creation():
    """Demonstrate creating indexes with different priorities."""
    logger.info("=== Index Creation Demonstration ===")
    
    # Get graph connection
    graph = setup_neo4j_connection()
    if not graph:
        logger.error("Cannot proceed without database connection")
        return
    
    # Create index manager
    index_manager = IndexManager(graph)
    
    # Create high priority indexes first (priority = 1)
    logger.info("Creating high priority indexes...")
    high_priority_results = index_manager.create_all_indexes(priority_filter=1)
    logger.info(f"High priority results: {high_priority_results}")
    
    # Create medium priority indexes (priority = 2)
    logger.info("Creating medium priority indexes...")
    medium_priority_results = index_manager.create_all_indexes(priority_filter=2)
    logger.info(f"Medium priority results: {medium_priority_results}")
    
    # Create all constraints
    logger.info("Creating constraints...")
    constraint_results = index_manager.create_all_constraints()
    logger.info(f"Constraint results: {constraint_results}")


def demonstrate_index_analysis():
    """Demonstrate index status analysis."""
    logger.info("=== Index Analysis Demonstration ===")
    
    graph = setup_neo4j_connection()
    if not graph:
        return
    
    index_manager = IndexManager(graph)
    
    # Get current index status
    logger.info("Current index status:")
    index_status = index_manager.get_index_status()
    for idx in index_status[:5]:  # Show first 5
        logger.info(f"  {idx.get('name', 'N/A')}: {idx.get('state', 'N/A')}")
    
    # Get constraint status
    logger.info("Current constraint status:")
    constraint_status = index_manager.get_constraint_status()
    for const in constraint_status[:5]:  # Show first 5
        logger.info(f"  {const.get('name', 'N/A')}: {const.get('type', 'N/A')}")


def demonstrate_query_analysis():
    """Demonstrate query performance analysis."""
    logger.info("=== Query Performance Analysis ===")
    
    graph = setup_neo4j_connection()
    if not graph:
        return
    
    index_manager = IndexManager(graph)
    
    # Example queries to analyze
    sample_queries = [
        """
        MATCH (m:EnhancedHistoricalMetric)
        WHERE m.facility_name = 'Plant A' 
        AND m.metric_type = 'incident_rate'
        AND m.reporting_period >= '2024-01-01'
        RETURN m.value, m.reporting_period
        ORDER BY m.reporting_period DESC
        LIMIT 10
        """,
        
        """
        MATCH (g:Goal)-[:HAS_TARGET]->(t:Target)
        WHERE g.status = 'active' 
        AND t.on_track_status = 'at_risk'
        RETURN g.goal_name, t.target_name, t.achievement_percentage
        """,
        
        """
        MATCH (r:Recommendation)
        WHERE r.priority = 'high' 
        AND r.status = 'pending'
        AND r.facility_name = 'Plant B'
        RETURN r.recommendation_title, r.description
        ORDER BY r.created_at DESC
        """
    ]
    
    for i, query in enumerate(sample_queries, 1):
        logger.info(f"Analyzing query {i}...")
        analysis = index_manager.analyze_query_performance(query.strip())
        if 'error' not in analysis:
            logger.info(f"  Query {i} suggestions: {analysis.get('suggestions', [])}")
        else:
            logger.warning(f"  Query {i} analysis failed: {analysis['error']}")


def demonstrate_index_utilities():
    """Demonstrate utility functions."""
    logger.info("=== Index Utilities Demonstration ===")
    
    # Get high priority indexes
    high_priority = get_high_priority_indexes()
    logger.info(f"High priority indexes count: {len(high_priority)}")
    for idx in high_priority[:3]:  # Show first 3
        logger.info(f"  - {idx.name}: {idx.description}")
    
    # Get indexes for specific node types
    metric_indexes = get_indexes_by_node_type("EnhancedHistoricalMetric")
    logger.info(f"EnhancedHistoricalMetric indexes: {len(metric_indexes)}")
    
    goal_indexes = get_indexes_by_node_type("Goal")
    logger.info(f"Goal indexes: {len(goal_indexes)}")
    
    # Get all index types
    all_indexes = EHSIndexDefinitions.get_all_indexes()
    index_types = {}
    for idx in all_indexes:
        idx_type = idx.index_type.value
        index_types[idx_type] = index_types.get(idx_type, 0) + 1
    
    logger.info("Index types distribution:")
    for idx_type, count in index_types.items():
        logger.info(f"  {idx_type}: {count}")


def generate_and_save_script():
    """Generate and save index creation script."""
    logger.info("=== Script Generation ===")
    
    script = generate_index_creation_script()
    
    # Save to file
    output_file = f"neo4j_indexes_setup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.cypher"
    output_path = os.path.join(os.path.dirname(__file__), output_file)
    
    try:
        with open(output_path, 'w') as f:
            f.write(script)
        logger.info(f"Index creation script saved to: {output_path}")
        logger.info(f"Script length: {len(script)} characters")
        
        # Show first few lines
        lines = script.split('\n')
        logger.info("Script preview (first 10 lines):")
        for line in lines[:10]:
            logger.info(f"  {line}")
        
    except Exception as e:
        logger.error(f"Failed to save script: {e}")


def main():
    """Main demonstration function."""
    logger.info("Starting Neo4j Index Definitions Demonstration")
    logger.info("=" * 60)
    
    try:
        # Run demonstrations
        demonstrate_index_utilities()
        print()
        
        generate_and_save_script()
        print()
        
        # Only run database operations if connection is available
        graph = setup_neo4j_connection()
        if graph:
            demonstrate_index_creation()
            print()
            
            demonstrate_index_analysis()
            print()
            
            demonstrate_query_analysis()
        else:
            logger.warning("Skipping database operations - no connection available")
        
        logger.info("Demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()