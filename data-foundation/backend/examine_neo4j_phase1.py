#!/usr/bin/env python3
"""
Neo4j Database Phase 1 Examination Script

This script connects to the existing Neo4j database and examines what Phase 1 objects
are already present, providing a comprehensive report on the database structure.

Author: AI Assistant
Date: 2025-08-23
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from collections import defaultdict

# Add the virtual environment path to sys.path if needed
venv_path = "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/web-app/backend/venv/lib/python3.11/site-packages"
if venv_path not in sys.path:
    sys.path.insert(0, venv_path)

try:
    from neo4j import GraphDatabase
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure you're running this script in the virtual environment with neo4j and python-dotenv installed")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('neo4j_examination.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Neo4jExaminer:
    """Neo4j Database Examiner for Phase 1 objects"""
    
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """Initialize the Neo4j connection"""
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver = None
        self.connection_verified = False
        
    def connect(self) -> bool:
        """Establish connection to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            # Verify connection
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    self.connection_verified = True
                    logger.info(f"Successfully connected to Neo4j at {self.uri}")
                    return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False
        
        return False
    
    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def get_all_labels(self) -> List[str]:
        """Get all node labels in the database"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("CALL db.labels()")
                labels = [record["label"] for record in result]
                logger.info(f"Found {len(labels)} node labels")
                return sorted(labels)
        except Exception as e:
            logger.error(f"Error getting labels: {e}")
            return []
    
    def get_label_counts(self, labels: List[str]) -> Dict[str, int]:
        """Get count of nodes for each label"""
        counts = {}
        for label in labels:
            try:
                with self.driver.session(database=self.database) as session:
                    query = f"MATCH (n:`{label}`) RETURN COUNT(n) as count"
                    result = session.run(query)
                    count = result.single()["count"]
                    counts[label] = count
            except Exception as e:
                logger.error(f"Error counting nodes for label {label}: {e}")
                counts[label] = 0
        
        return counts
    
    def get_all_relationship_types(self) -> List[str]:
        """Get all relationship types in the database"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("CALL db.relationshipTypes()")
                rel_types = [record["relationshipType"] for record in result]
                logger.info(f"Found {len(rel_types)} relationship types")
                return sorted(rel_types)
        except Exception as e:
            logger.error(f"Error getting relationship types: {e}")
            return []
    
    def get_relationship_counts(self, rel_types: List[str]) -> Dict[str, int]:
        """Get count of relationships for each type"""
        counts = {}
        for rel_type in rel_types:
            try:
                with self.driver.session(database=self.database) as session:
                    query = f"MATCH ()-[r:`{rel_type}`]-() RETURN COUNT(r) as count"
                    result = session.run(query)
                    count = result.single()["count"]
                    counts[rel_type] = count
            except Exception as e:
                logger.error(f"Error counting relationships for type {rel_type}: {e}")
                counts[rel_type] = 0
        
        return counts
    
    def get_sample_nodes(self, label: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get sample nodes for a given label"""
        try:
            with self.driver.session(database=self.database) as session:
                query = f"MATCH (n:`{label}`) RETURN n LIMIT {limit}"
                result = session.run(query)
                nodes = []
                for record in result:
                    node = record["n"]
                    node_dict = {
                        "id": node.id,
                        "labels": list(node.labels),
                        "properties": dict(node.items())
                    }
                    nodes.append(node_dict)
                return nodes
        except Exception as e:
            logger.error(f"Error getting sample nodes for label {label}: {e}")
            return []
    
    def get_node_properties(self, label: str) -> Dict[str, List[str]]:
        """Get all properties for nodes with a given label"""
        try:
            with self.driver.session(database=self.database) as session:
                query = f"MATCH (n:`{label}`) UNWIND keys(n) AS key RETURN DISTINCT key ORDER BY key"
                result = session.run(query)
                properties = [record["key"] for record in result]
                
                # Get property types by sampling
                property_types = {}
                for prop in properties:
                    sample_query = f"MATCH (n:`{label}`) WHERE n.`{prop}` IS NOT NULL RETURN DISTINCT type(n.`{prop}`) AS propType LIMIT 10"
                    type_result = session.run(sample_query)
                    types = [record["propType"] for record in type_result]
                    property_types[prop] = types
                
                return property_types
        except Exception as e:
            logger.error(f"Error getting properties for label {label}: {e}")
            return {}
    
    def get_constraints(self) -> List[Dict[str, Any]]:
        """Get all constraints in the database"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("SHOW CONSTRAINTS")
                constraints = []
                for record in result:
                    constraint = dict(record)
                    constraints.append(constraint)
                return constraints
        except Exception as e:
            logger.error(f"Error getting constraints: {e}")
            return []
    
    def get_indexes(self) -> List[Dict[str, Any]]:
        """Get all indexes in the database"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("SHOW INDEXES")
                indexes = []
                for record in result:
                    index = dict(record)
                    indexes.append(index)
                return indexes
        except Exception as e:
            logger.error(f"Error getting indexes: {e}")
            return []
    
    def get_relationship_patterns(self) -> List[Dict[str, Any]]:
        """Get common relationship patterns in the database"""
        try:
            with self.driver.session(database=self.database) as session:
                query = """
                MATCH (a)-[r]->(b)
                RETURN 
                    labels(a)[0] as source_label,
                    type(r) as relationship_type,
                    labels(b)[0] as target_label,
                    COUNT(*) as frequency
                ORDER BY frequency DESC
                LIMIT 20
                """
                result = session.run(query)
                patterns = []
                for record in result:
                    pattern = {
                        "source_label": record["source_label"],
                        "relationship_type": record["relationship_type"],
                        "target_label": record["target_label"],
                        "frequency": record["frequency"]
                    }
                    patterns.append(pattern)
                return patterns
        except Exception as e:
            logger.error(f"Error getting relationship patterns: {e}")
            return []
    
    def search_phase1_objects(self) -> Dict[str, Any]:
        """Search for Phase 1 specific objects"""
        phase1_labels = ["AuditTrail", "ProRating", "RejectionTracking", "Document", "Entity", "Chunk"]
        phase1_data = {}
        
        for label in phase1_labels:
            try:
                with self.driver.session(database=self.database) as session:
                    # Check if label exists
                    count_query = f"MATCH (n:`{label}`) RETURN COUNT(n) as count"
                    result = session.run(count_query)
                    count = result.single()["count"]
                    
                    if count > 0:
                        phase1_data[label] = {
                            "count": count,
                            "sample_nodes": self.get_sample_nodes(label, 3),
                            "properties": self.get_node_properties(label)
                        }
                    else:
                        phase1_data[label] = {"count": 0, "sample_nodes": [], "properties": {}}
            except Exception as e:
                logger.error(f"Error searching for {label}: {e}")
                phase1_data[label] = {"count": 0, "sample_nodes": [], "properties": {}}
        
        return phase1_data
    
    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive examination report"""
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("NEO4J DATABASE PHASE 1 EXAMINATION REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Database: {self.uri}/{self.database}")
        report_lines.append("")
        
        # Database Overview
        report_lines.append("1. DATABASE OVERVIEW")
        report_lines.append("-" * 40)
        
        labels = self.get_all_labels()
        label_counts = self.get_label_counts(labels)
        rel_types = self.get_all_relationship_types()
        rel_counts = self.get_relationship_counts(rel_types)
        
        total_nodes = sum(label_counts.values())
        total_relationships = sum(rel_counts.values())
        
        report_lines.append(f"Total Node Labels: {len(labels)}")
        report_lines.append(f"Total Nodes: {total_nodes}")
        report_lines.append(f"Total Relationship Types: {len(rel_types)}")
        report_lines.append(f"Total Relationships: {total_relationships}")
        report_lines.append("")
        
        # Node Labels and Counts
        report_lines.append("2. NODE LABELS AND COUNTS")
        report_lines.append("-" * 40)
        for label in sorted(label_counts.keys(), key=lambda x: label_counts[x], reverse=True):
            report_lines.append(f"  {label:25} : {label_counts[label]:>10,} nodes")
        report_lines.append("")
        
        # Relationship Types and Counts
        report_lines.append("3. RELATIONSHIP TYPES AND COUNTS")
        report_lines.append("-" * 40)
        for rel_type in sorted(rel_counts.keys(), key=lambda x: rel_counts[x], reverse=True):
            report_lines.append(f"  {rel_type:25} : {rel_counts[rel_type]:>10,} relationships")
        report_lines.append("")
        
        # Phase 1 Specific Analysis
        report_lines.append("4. PHASE 1 OBJECTS ANALYSIS")
        report_lines.append("-" * 40)
        phase1_data = self.search_phase1_objects()
        
        for label, data in phase1_data.items():
            report_lines.append(f"\n{label}:")
            report_lines.append(f"  Count: {data['count']}")
            
            if data['count'] > 0:
                report_lines.append(f"  Properties: {', '.join(data['properties'].keys())}")
                
                if data['sample_nodes']:
                    report_lines.append("  Sample Node:")
                    sample = data['sample_nodes'][0]
                    for key, value in sample['properties'].items():
                        # Truncate long values
                        str_value = str(value)
                        if len(str_value) > 100:
                            str_value = str_value[:97] + "..."
                        report_lines.append(f"    {key}: {str_value}")
        
        # Node Properties Schema
        report_lines.append("\n\n5. NODE PROPERTIES SCHEMA")
        report_lines.append("-" * 40)
        for label in labels:
            if label_counts[label] > 0:
                properties = self.get_node_properties(label)
                if properties:
                    report_lines.append(f"\n{label}:")
                    for prop, types in properties.items():
                        report_lines.append(f"  {prop}: {', '.join(types)}")
        
        # Relationship Patterns
        report_lines.append("\n\n6. RELATIONSHIP PATTERNS")
        report_lines.append("-" * 40)
        patterns = self.get_relationship_patterns()
        for pattern in patterns[:10]:  # Top 10 patterns
            report_lines.append(
                f"  ({pattern['source_label']})-[{pattern['relationship_type']}]->({pattern['target_label']}) "
                f": {pattern['frequency']} times"
            )
        
        # Constraints
        report_lines.append("\n\n7. DATABASE CONSTRAINTS")
        report_lines.append("-" * 40)
        constraints = self.get_constraints()
        if constraints:
            for constraint in constraints:
                report_lines.append(f"  {constraint}")
        else:
            report_lines.append("  No constraints found")
        
        # Indexes
        report_lines.append("\n\n8. DATABASE INDEXES")
        report_lines.append("-" * 40)
        indexes = self.get_indexes()
        if indexes:
            for index in indexes:
                report_lines.append(f"  {index}")
        else:
            report_lines.append("  No indexes found")
        
        report_lines.append("\n" + "="*80)
        report_lines.append("END OF REPORT")
        report_lines.append("="*80)
        
        return "\n".join(report_lines)


def main():
    """Main execution function"""
    # Load environment variables
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(env_path)
    
    # Get Neo4j credentials from environment
    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    username = os.getenv('NEO4J_USERNAME', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD', 'EhsAI2024!')
    database = os.getenv('NEO4J_DATABASE', 'neo4j')
    
    logger.info("Starting Neo4j Phase 1 Database Examination")
    logger.info(f"Connecting to: {uri}")
    
    # Initialize examiner
    examiner = Neo4jExaminer(uri, username, password, database)
    
    try:
        # Connect to database
        if not examiner.connect():
            logger.error("Failed to establish connection to Neo4j database")
            sys.exit(1)
        
        # Generate comprehensive report
        logger.info("Generating comprehensive examination report...")
        report = examiner.generate_comprehensive_report()
        
        # Write report to file
        report_filename = f"neo4j_phase1_examination_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_filename, 'w') as f:
            f.write(report)
        
        # Display report
        print(report)
        
        logger.info(f"Report saved to: {report_filename}")
        logger.info("Neo4j Phase 1 Database Examination completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred during examination: {e}")
        sys.exit(1)
    
    finally:
        # Clean up
        examiner.close()


if __name__ == "__main__":
    main()