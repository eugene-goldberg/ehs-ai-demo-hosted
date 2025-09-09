#!/usr/bin/env python3
"""
Neo4j Database Catalog Script

This script connects to a Neo4j database and catalogs ALL objects and relationships,
generating a comprehensive markdown report without modifying any data.

Requirements:
- neo4j Python driver
- python-dotenv for environment variables

Author: Claude Code
Created: 2025-09-09
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
from pathlib import Path

try:
    from neo4j import GraphDatabase
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error: Missing required packages. Please install with:")
    print("pip install neo4j python-dotenv")
    sys.exit(1)


class Neo4jCatalogger:
    """Class to catalog all Neo4j database objects and relationships."""
    
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """Initialize the Neo4j catalogger with connection parameters."""
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver = None
        self.catalog_data = {}
        
    def connect(self) -> bool:
        """Connect to the Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            # Test the connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            print(f"✓ Successfully connected to Neo4j at {self.uri}")
            return True
        except Exception as e:
            print(f"✗ Failed to connect to Neo4j: {str(e)}")
            return False
    
    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            print("✓ Connection closed")
    
    def get_node_labels_and_counts(self) -> Dict[str, int]:
        """Get all node labels and their counts."""
        print("📊 Cataloging node labels and counts...")
        
        query = """
        CALL db.labels() YIELD label
        CALL {
            WITH label
            CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) as count', {})
            YIELD value
            RETURN value.count as count
        }
        RETURN label, count
        ORDER BY label
        """
        
        # Fallback query if APOC is not available
        fallback_query = """
        CALL db.labels() YIELD label
        RETURN label, 0 as count
        ORDER BY label
        """
        
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(query)
                labels_counts = {record["label"]: record["count"] for record in result}
                
                # If APOC didn't work or returned 0 counts, get individual counts
                if not labels_counts or all(count == 0 for count in labels_counts.values()):
                    labels_counts = {}
                    result = session.run("CALL db.labels() YIELD label RETURN label")
                    for record in result:
                        label = record["label"]
                        count_query = f"MATCH (n:`{label}`) RETURN count(n) as count"
                        count_result = session.run(count_query)
                        count = count_result.single()["count"]
                        labels_counts[label] = count
                        
            except Exception as e:
                print(f"⚠️ Warning: Could not get exact counts, using fallback: {str(e)}")
                result = session.run(fallback_query)
                labels_counts = {record["label"]: 0 for record in result}
                
                # Try to get individual counts
                for label in labels_counts.keys():
                    try:
                        count_query = f"MATCH (n:`{label}`) RETURN count(n) as count"
                        count_result = session.run(count_query)
                        labels_counts[label] = count_result.single()["count"]
                    except:
                        labels_counts[label] = "Unknown"
        
        return labels_counts
    
    def get_relationship_types_and_counts(self) -> Dict[str, int]:
        """Get all relationship types and their counts."""
        print("🔗 Cataloging relationship types and counts...")
        
        query = """
        CALL db.relationshipTypes() YIELD relationshipType
        CALL {
            WITH relationshipType
            CALL apoc.cypher.run('MATCH ()-[r:' + relationshipType + ']-() RETURN count(r) as count', {})
            YIELD value
            RETURN value.count as count
        }
        RETURN relationshipType, count
        ORDER BY relationshipType
        """
        
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(query)
                rel_counts = {record["relationshipType"]: record["count"] for record in result}
                
                # If APOC didn't work or returned 0 counts, get individual counts
                if not rel_counts or all(count == 0 for count in rel_counts.values()):
                    rel_counts = {}
                    result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
                    for record in result:
                        rel_type = record["relationshipType"]
                        count_query = f"MATCH ()-[r:`{rel_type}`]-() RETURN count(r) as count"
                        count_result = session.run(count_query)
                        count = count_result.single()["count"]
                        rel_counts[rel_type] = count
                        
            except Exception as e:
                print(f"⚠️ Warning: Could not get relationship counts: {str(e)}")
                result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
                rel_counts = {record["relationshipType"]: "Unknown" for record in result}
        
        return rel_counts
    
    def get_property_keys_for_labels(self, labels: List[str]) -> Dict[str, List[str]]:
        """Get all property keys for each node label."""
        print("🔑 Cataloging property keys for node labels...")
        
        label_properties = {}
        
        with self.driver.session(database=self.database) as session:
            for label in labels:
                try:
                    query = f"""
                    MATCH (n:`{label}`)
                    WITH keys(n) as props
                    UNWIND props as prop
                    RETURN DISTINCT prop
                    ORDER BY prop
                    """
                    result = session.run(query)
                    properties = [record["prop"] for record in result]
                    label_properties[label] = properties
                except Exception as e:
                    print(f"⚠️ Warning: Could not get properties for label {label}: {str(e)}")
                    label_properties[label] = []
        
        return label_properties
    
    def get_property_keys_for_relationships(self, rel_types: List[str]) -> Dict[str, List[str]]:
        """Get all property keys for each relationship type."""
        print("🔗🔑 Cataloging property keys for relationship types...")
        
        rel_properties = {}
        
        with self.driver.session(database=self.database) as session:
            for rel_type in rel_types:
                try:
                    query = f"""
                    MATCH ()-[r:`{rel_type}`]-()
                    WITH keys(r) as props
                    UNWIND props as prop
                    RETURN DISTINCT prop
                    ORDER BY prop
                    """
                    result = session.run(query)
                    properties = [record["prop"] for record in result]
                    rel_properties[rel_type] = properties
                except Exception as e:
                    print(f"⚠️ Warning: Could not get properties for relationship {rel_type}: {str(e)}")
                    rel_properties[rel_type] = []
        
        return rel_properties
    
    def get_sample_nodes(self, labels: List[str], limit: int = 3) -> Dict[str, List[Dict]]:
        """Get sample data for each node label."""
        print(f"📝 Getting sample nodes (limit: {limit})...")
        
        sample_nodes = {}
        
        with self.driver.session(database=self.database) as session:
            for label in labels:
                try:
                    query = f"MATCH (n:`{label}`) RETURN n LIMIT {limit}"
                    result = session.run(query)
                    nodes = []
                    for record in result:
                        node = record["n"]
                        node_data = dict(node)
                        # Convert any non-serializable values to strings
                        for key, value in node_data.items():
                            if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                                node_data[key] = str(value)
                        nodes.append(node_data)
                    sample_nodes[label] = nodes
                except Exception as e:
                    print(f"⚠️ Warning: Could not get sample nodes for label {label}: {str(e)}")
                    sample_nodes[label] = []
        
        return sample_nodes
    
    def get_sample_relationships(self, rel_types: List[str], limit: int = 3) -> Dict[str, List[Dict]]:
        """Get sample data for each relationship type."""
        print(f"🔗📝 Getting sample relationships (limit: {limit})...")
        
        sample_rels = {}
        
        with self.driver.session(database=self.database) as session:
            for rel_type in rel_types:
                try:
                    query = f"""
                    MATCH (start)-[r:`{rel_type}`]->(end)
                    RETURN 
                        labels(start) as start_labels,
                        labels(end) as end_labels,
                        r,
                        id(start) as start_id,
                        id(end) as end_id
                    LIMIT {limit}
                    """
                    result = session.run(query)
                    relationships = []
                    for record in result:
                        rel_data = dict(record["r"])
                        # Convert any non-serializable values to strings
                        for key, value in rel_data.items():
                            if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                                rel_data[key] = str(value)
                        
                        relationships.append({
                            "properties": rel_data,
                            "start_labels": record["start_labels"],
                            "end_labels": record["end_labels"],
                            "start_id": record["start_id"],
                            "end_id": record["end_id"]
                        })
                    sample_rels[rel_type] = relationships
                except Exception as e:
                    print(f"⚠️ Warning: Could not get sample relationships for type {rel_type}: {str(e)}")
                    sample_rels[rel_type] = []
        
        return sample_rels
    
    def get_constraints_and_indexes(self) -> Dict[str, List[Dict]]:
        """Get all constraints and indexes in the database."""
        print("⚙️ Cataloging constraints and indexes...")
        
        constraints_indexes = {"constraints": [], "indexes": []}
        
        with self.driver.session(database=self.database) as session:
            # Get constraints
            try:
                result = session.run("SHOW CONSTRAINTS")
                for record in result:
                    constraint_data = dict(record)
                    # Convert any non-serializable values to strings
                    for key, value in constraint_data.items():
                        if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                            constraint_data[key] = str(value)
                    constraints_indexes["constraints"].append(constraint_data)
            except Exception as e:
                print(f"⚠️ Warning: Could not get constraints: {str(e)}")
            
            # Get indexes
            try:
                result = session.run("SHOW INDEXES")
                for record in result:
                    index_data = dict(record)
                    # Convert any non-serializable values to strings
                    for key, value in index_data.items():
                        if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                            index_data[key] = str(value)
                    constraints_indexes["indexes"].append(index_data)
            except Exception as e:
                print(f"⚠️ Warning: Could not get indexes: {str(e)}")
        
        return constraints_indexes
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get general database information."""
        print("ℹ️ Getting database information...")
        
        db_info = {}
        
        with self.driver.session(database=self.database) as session:
            try:
                # Get Neo4j version
                result = session.run("CALL dbms.components() YIELD name, versions, edition")
                components = [dict(record) for record in result]
                db_info["components"] = components
            except Exception as e:
                print(f"⚠️ Warning: Could not get database components: {str(e)}")
                db_info["components"] = []
            
            try:
                # Get database size information
                result = session.run("CALL apoc.monitor.store()")
                store_info = result.single()
                if store_info:
                    db_info["store_info"] = dict(store_info)
            except Exception as e:
                print(f"⚠️ Warning: Could not get store information (APOC may not be available): {str(e)}")
                db_info["store_info"] = {}
        
        return db_info
    
    def catalog_all(self) -> Dict[str, Any]:
        """Run complete catalog of the database."""
        print("🚀 Starting comprehensive Neo4j database catalog...")
        
        # Get node labels and counts
        node_labels_counts = self.get_node_labels_and_counts()
        labels = list(node_labels_counts.keys())
        
        # Get relationship types and counts
        rel_types_counts = self.get_relationship_types_and_counts()
        rel_types = list(rel_types_counts.keys())
        
        # Get property keys
        label_properties = self.get_property_keys_for_labels(labels)
        rel_properties = self.get_property_keys_for_relationships(rel_types)
        
        # Get sample data
        sample_nodes = self.get_sample_nodes(labels)
        sample_relationships = self.get_sample_relationships(rel_types)
        
        # Get constraints and indexes
        constraints_indexes = self.get_constraints_and_indexes()
        
        # Get database info
        db_info = self.get_database_info()
        
        self.catalog_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "neo4j_uri": self.uri,
                "database": self.database,
                "total_node_labels": len(labels),
                "total_relationship_types": len(rel_types)
            },
            "database_info": db_info,
            "node_labels": {
                "counts": node_labels_counts,
                "properties": label_properties,
                "samples": sample_nodes
            },
            "relationship_types": {
                "counts": rel_types_counts,
                "properties": rel_properties,
                "samples": sample_relationships
            },
            "constraints_and_indexes": constraints_indexes
        }
        
        print("✅ Catalog complete!")
        return self.catalog_data
    
    def generate_markdown_report(self, output_path: str) -> bool:
        """Generate a comprehensive markdown report."""
        if not self.catalog_data:
            print("❌ No catalog data available. Run catalog_all() first.")
            return False
        
        print(f"📄 Generating markdown report: {output_path}")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Header
                f.write("# Neo4j Database Catalog\n\n")
                f.write(f"**Generated:** {self.catalog_data['metadata']['generated_at']}\n")
                f.write(f"**Neo4j URI:** {self.catalog_data['metadata']['neo4j_uri']}\n")
                f.write(f"**Database:** {self.catalog_data['metadata']['database']}\n\n")
                
                # Table of Contents
                f.write("## Table of Contents\n\n")
                f.write("- [Database Information](#database-information)\n")
                f.write("- [Summary Statistics](#summary-statistics)\n")
                f.write("- [Node Labels](#node-labels)\n")
                f.write("- [Relationship Types](#relationship-types)\n")
                f.write("- [Constraints and Indexes](#constraints-and-indexes)\n")
                f.write("- [Sample Data](#sample-data)\n\n")
                
                # Database Information
                f.write("## Database Information\n\n")
                db_info = self.catalog_data['database_info']
                if db_info.get('components'):
                    f.write("### Neo4j Components\n\n")
                    for component in db_info['components']:
                        f.write(f"- **{component.get('name', 'Unknown')}:** {component.get('versions', 'Unknown')} ({component.get('edition', 'Unknown')})\n")
                    f.write("\n")
                
                if db_info.get('store_info'):
                    f.write("### Store Information\n\n")
                    store_info = db_info['store_info']
                    for key, value in store_info.items():
                        f.write(f"- **{key}:** {value}\n")
                    f.write("\n")
                
                # Summary Statistics
                f.write("## Summary Statistics\n\n")
                f.write(f"- **Total Node Labels:** {self.catalog_data['metadata']['total_node_labels']}\n")
                f.write(f"- **Total Relationship Types:** {self.catalog_data['metadata']['total_relationship_types']}\n")
                
                total_nodes = sum(count for count in self.catalog_data['node_labels']['counts'].values() 
                                if isinstance(count, int))
                total_relationships = sum(count for count in self.catalog_data['relationship_types']['counts'].values() 
                                        if isinstance(count, int))
                
                f.write(f"- **Total Nodes:** {total_nodes}\n")
                f.write(f"- **Total Relationships:** {total_relationships}\n\n")
                
                # Node Labels
                f.write("## Node Labels\n\n")
                f.write("### Node Label Counts\n\n")
                f.write("| Label | Count |\n")
                f.write("|-------|-------|\n")
                for label, count in self.catalog_data['node_labels']['counts'].items():
                    f.write(f"| {label} | {count} |\n")
                f.write("\n")
                
                f.write("### Node Properties\n\n")
                for label, properties in self.catalog_data['node_labels']['properties'].items():
                    f.write(f"#### {label}\n")
                    if properties:
                        f.write("Properties:\n")
                        for prop in properties:
                            f.write(f"- {prop}\n")
                    else:
                        f.write("No properties found.\n")
                    f.write("\n")
                
                # Relationship Types
                f.write("## Relationship Types\n\n")
                f.write("### Relationship Type Counts\n\n")
                f.write("| Type | Count |\n")
                f.write("|------|-------|\n")
                for rel_type, count in self.catalog_data['relationship_types']['counts'].items():
                    f.write(f"| {rel_type} | {count} |\n")
                f.write("\n")
                
                f.write("### Relationship Properties\n\n")
                for rel_type, properties in self.catalog_data['relationship_types']['properties'].items():
                    f.write(f"#### {rel_type}\n")
                    if properties:
                        f.write("Properties:\n")
                        for prop in properties:
                            f.write(f"- {prop}\n")
                    else:
                        f.write("No properties found.\n")
                    f.write("\n")
                
                # Constraints and Indexes
                f.write("## Constraints and Indexes\n\n")
                
                constraints = self.catalog_data['constraints_and_indexes']['constraints']
                if constraints:
                    f.write("### Constraints\n\n")
                    for constraint in constraints:
                        f.write(f"- **{constraint.get('name', 'Unnamed')}:** {constraint.get('description', 'No description')}\n")
                    f.write("\n")
                else:
                    f.write("### Constraints\n\nNo constraints found.\n\n")
                
                indexes = self.catalog_data['constraints_and_indexes']['indexes']
                if indexes:
                    f.write("### Indexes\n\n")
                    for index in indexes:
                        f.write(f"- **{index.get('name', 'Unnamed')}:** {index.get('description', 'No description')}\n")
                    f.write("\n")
                else:
                    f.write("### Indexes\n\nNo indexes found.\n\n")
                
                # Sample Data
                f.write("## Sample Data\n\n")
                
                f.write("### Sample Nodes\n\n")
                for label, samples in self.catalog_data['node_labels']['samples'].items():
                    f.write(f"#### {label} (Sample)\n\n")
                    if samples:
                        for i, sample in enumerate(samples, 1):
                            f.write(f"**Sample {i}:**\n")
                            f.write("```json\n")
                            f.write(json.dumps(sample, indent=2))
                            f.write("\n```\n\n")
                    else:
                        f.write("No sample data available.\n\n")
                
                f.write("### Sample Relationships\n\n")
                for rel_type, samples in self.catalog_data['relationship_types']['samples'].items():
                    f.write(f"#### {rel_type} (Sample)\n\n")
                    if samples:
                        for i, sample in enumerate(samples, 1):
                            f.write(f"**Sample {i}:**\n")
                            f.write(f"- **Start Node:** {sample['start_labels']} (ID: {sample['start_id']})\n")
                            f.write(f"- **End Node:** {sample['end_labels']} (ID: {sample['end_id']})\n")
                            f.write(f"- **Properties:**\n")
                            f.write("```json\n")
                            f.write(json.dumps(sample['properties'], indent=2))
                            f.write("\n```\n\n")
                    else:
                        f.write("No sample data available.\n\n")
            
            print(f"✅ Markdown report generated successfully: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error generating markdown report: {str(e)}")
            return False
    
    def save_json_catalog(self, output_path: str) -> bool:
        """Save the catalog data as JSON."""
        if not self.catalog_data:
            print("❌ No catalog data available. Run catalog_all() first.")
            return False
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.catalog_data, f, indent=2, ensure_ascii=False)
            print(f"✅ JSON catalog saved: {output_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving JSON catalog: {str(e)}")
            return False


def main():
    """Main function to run the Neo4j cataloger."""
    # Load environment variables
    env_path = "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/.env"
    load_dotenv(env_path)
    
    # Get Neo4j connection parameters
    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    username = os.getenv('NEO4J_USERNAME', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD')
    database = os.getenv('NEO4J_DATABASE', 'neo4j')
    
    if not password:
        print("❌ Error: NEO4J_PASSWORD not found in environment variables")
        sys.exit(1)
    
    print(f"🔌 Connecting to Neo4j at {uri}")
    print(f"📊 Database: {database}")
    
    # Create output directory if it doesn't exist
    output_dir = Path("/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/neo4j-data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize catalogger
    catalogger = Neo4jCatalogger(uri, username, password, database)
    
    try:
        # Connect to Neo4j
        if not catalogger.connect():
            sys.exit(1)
        
        # Run complete catalog
        catalog_data = catalogger.catalog_all()
        
        # Generate reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        markdown_path = output_dir / f"neo4j_catalog_{timestamp}.md"
        json_path = output_dir / f"neo4j_catalog_{timestamp}.json"
        
        # Generate markdown report
        if catalogger.generate_markdown_report(str(markdown_path)):
            print(f"📄 Markdown report: {markdown_path}")
        
        # Save JSON catalog
        if catalogger.save_json_catalog(str(json_path)):
            print(f"📊 JSON catalog: {json_path}")
        
        print("\n🎉 Neo4j database catalog completed successfully!")
        print(f"📁 Output files saved in: {output_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Cataloging interrupted by user")
    except Exception as e:
        print(f"❌ Error during cataloging: {str(e)}")
        sys.exit(1)
    finally:
        catalogger.close()


if __name__ == "__main__":
    main()