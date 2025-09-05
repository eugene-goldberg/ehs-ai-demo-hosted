#!/usr/bin/env python3
"""
Neo4j Full Database Backup Script

This script creates a comprehensive backup of a Neo4j database by:
1. Connecting to the Neo4j database using environment variables
2. Exporting ALL nodes with all their properties
3. Exporting ALL relationships with all their properties
4. Saving everything to a timestamped JSON file
5. Including backup metadata (date, node count, relationship count, execution time)

Usage:
    python3 backup_neo4j_full.py

Output:
    Creates neo4j_backup_YYYYMMDD_HHMMSS.json in the backend directory
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import time
from pathlib import Path

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, AuthError, ConfigurationError
except ImportError:
    print("Error: neo4j package not found. Install it with: pip install neo4j")
    exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("Error: python-dotenv package not found. Install it with: pip install python-dotenv")
    exit(1)


class Neo4jBackupManager:
    """Handles Neo4j database backup operations."""
    
    def __init__(self):
        """Initialize the backup manager with environment variables."""
        # Load environment variables
        load_dotenv()
        
        # Neo4j connection parameters
        self.uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.username = os.getenv('NEO4J_USERNAME', 'neo4j')
        self.password = os.getenv('NEO4J_PASSWORD', '')
        self.database = os.getenv('NEO4J_DATABASE', 'neo4j')
        
        # Validate required parameters
        if not self.password:
            raise ValueError("NEO4J_PASSWORD environment variable is required")
        
        # Initialize driver
        self.driver = None
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('backup_neo4j_full.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def connect(self) -> bool:
        """
        Establish connection to Neo4j database.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.logger.info(f"Connecting to Neo4j at {self.uri}")
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            
            # Test connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            
            self.logger.info("Successfully connected to Neo4j")
            return True
            
        except AuthError as e:
            self.logger.error(f"Authentication failed: {e}")
            return False
        except ServiceUnavailable as e:
            self.logger.error(f"Neo4j service unavailable: {e}")
            return False
        except ConfigurationError as e:
            self.logger.error(f"Configuration error: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error connecting to Neo4j: {e}")
            return False
    
    def disconnect(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            self.logger.info("Neo4j connection closed")
    
    def get_all_node_labels(self) -> List[str]:
        """
        Get all node labels in the database.
        
        Returns:
            List[str]: List of all node labels
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("CALL db.labels()")
                labels = [record["label"] for record in result]
                self.logger.info(f"Found {len(labels)} node labels: {labels}")
                return labels
        except Exception as e:
            self.logger.error(f"Error getting node labels: {e}")
            return []
    
    def get_all_relationship_types(self) -> List[str]:
        """
        Get all relationship types in the database.
        
        Returns:
            List[str]: List of all relationship types
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("CALL db.relationshipTypes()")
                types = [record["relationshipType"] for record in result]
                self.logger.info(f"Found {len(types)} relationship types: {types}")
                return types
        except Exception as e:
            self.logger.error(f"Error getting relationship types: {e}")
            return []
    
    def export_all_nodes(self) -> List[Dict[str, Any]]:
        """
        Export all nodes with their properties.
        
        Returns:
            List[Dict[str, Any]]: List of all nodes with properties
        """
        nodes = []
        try:
            with self.driver.session(database=self.database) as session:
                # Get all nodes with their properties and labels
                query = """
                MATCH (n)
                RETURN 
                    id(n) as node_id,
                    labels(n) as labels,
                    properties(n) as properties
                ORDER BY id(n)
                """
                
                result = session.run(query)
                
                for record in result:
                    node_data = {
                        'node_id': record['node_id'],
                        'labels': record['labels'],
                        'properties': dict(record['properties'])
                    }
                    nodes.append(node_data)
                
                self.logger.info(f"Exported {len(nodes)} nodes")
                return nodes
                
        except Exception as e:
            self.logger.error(f"Error exporting nodes: {e}")
            return []
    
    def export_all_relationships(self) -> List[Dict[str, Any]]:
        """
        Export all relationships with their properties.
        
        Returns:
            List[Dict[str, Any]]: List of all relationships with properties
        """
        relationships = []
        try:
            with self.driver.session(database=self.database) as session:
                # Get all relationships with their properties
                query = """
                MATCH (start_node)-[r]->(end_node)
                RETURN 
                    id(r) as rel_id,
                    id(start_node) as start_node_id,
                    id(end_node) as end_node_id,
                    type(r) as relationship_type,
                    properties(r) as properties
                ORDER BY id(r)
                """
                
                result = session.run(query)
                
                for record in result:
                    rel_data = {
                        'rel_id': record['rel_id'],
                        'start_node_id': record['start_node_id'],
                        'end_node_id': record['end_node_id'],
                        'relationship_type': record['relationship_type'],
                        'properties': dict(record['properties'])
                    }
                    relationships.append(rel_data)
                
                self.logger.info(f"Exported {len(relationships)} relationships")
                return relationships
                
        except Exception as e:
            self.logger.error(f"Error exporting relationships: {e}")
            return []
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics for backup metadata.
        
        Returns:
            Dict[str, Any]: Database statistics
        """
        stats = {}
        try:
            with self.driver.session(database=self.database) as session:
                # Get node count
                result = session.run("MATCH (n) RETURN count(n) as node_count")
                stats['total_nodes'] = result.single()['node_count']
                
                # Get relationship count
                result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                stats['total_relationships'] = result.single()['rel_count']
                
                # Get labels and their counts
                result = session.run("""
                    CALL db.labels() YIELD label
                    CALL apoc.cypher.run('MATCH (n:`' + label + '`) RETURN count(n) as count', {}) YIELD value
                    RETURN label, value.count as count
                    ORDER BY count DESC
                """)
                
                # If apoc is not available, get label counts differently
                try:
                    label_counts = {}
                    for record in result:
                        label_counts[record['label']] = record['count']
                    stats['label_counts'] = label_counts
                except:
                    # Fallback method without APOC
                    labels = self.get_all_node_labels()
                    label_counts = {}
                    for label in labels:
                        result = session.run(f"MATCH (n:`{label}`) RETURN count(n) as count")
                        label_counts[label] = result.single()['count']
                    stats['label_counts'] = label_counts
                
                # Get relationship type counts
                rel_types = self.get_all_relationship_types()
                rel_type_counts = {}
                for rel_type in rel_types:
                    result = session.run(f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) as count")
                    rel_type_counts[rel_type] = result.single()['count']
                stats['relationship_type_counts'] = rel_type_counts
                
                self.logger.info(f"Database statistics: {stats}")
                return stats
                
        except Exception as e:
            self.logger.error(f"Error getting database statistics: {e}")
            return {}
    
    def create_backup(self, output_dir: Optional[str] = None) -> Optional[str]:
        """
        Create a full backup of the Neo4j database.
        
        Args:
            output_dir (Optional[str]): Directory to save backup file. 
                                      Defaults to backend directory.
        
        Returns:
            Optional[str]: Path to backup file if successful, None otherwise
        """
        start_time = time.time()
        
        try:
            # Determine output directory
            if output_dir is None:
                # Default to backend directory (parent of scripts directory)
                current_dir = Path(__file__).parent
                output_dir = current_dir.parent
            
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"neo4j_backup_{timestamp}.json"
            backup_filepath = output_dir / backup_filename
            
            self.logger.info(f"Starting backup to {backup_filepath}")
            
            # Get database statistics
            stats = self.get_database_statistics()
            
            # Export nodes and relationships
            self.logger.info("Exporting nodes...")
            nodes = self.export_all_nodes()
            
            self.logger.info("Exporting relationships...")
            relationships = self.export_all_relationships()
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Create backup data structure
            backup_data = {
                'metadata': {
                    'backup_timestamp': datetime.now().isoformat(),
                    'neo4j_uri': self.uri,
                    'neo4j_database': self.database,
                    'backup_version': '1.0',
                    'execution_time_seconds': round(execution_time, 2),
                    'statistics': stats
                },
                'nodes': nodes,
                'relationships': relationships
            }
            
            # Write backup to JSON file
            self.logger.info(f"Writing backup to {backup_filepath}")
            with open(backup_filepath, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False, default=str)
            
            # Log success
            total_time = time.time() - start_time
            self.logger.info(f"Backup completed successfully in {total_time:.2f} seconds")
            self.logger.info(f"Backup file: {backup_filepath}")
            self.logger.info(f"Total nodes: {len(nodes)}")
            self.logger.info(f"Total relationships: {len(relationships)}")
            self.logger.info(f"File size: {backup_filepath.stat().st_size / 1024 / 1024:.2f} MB")
            
            return str(backup_filepath)
            
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return None


def main():
    """Main function to run the backup."""
    print("Neo4j Full Database Backup Tool")
    print("=" * 40)
    
    backup_manager = Neo4jBackupManager()
    
    try:
        # Connect to Neo4j
        if not backup_manager.connect():
            print("Failed to connect to Neo4j. Check your connection settings.")
            return 1
        
        # Create backup
        backup_file = backup_manager.create_backup()
        
        if backup_file:
            print(f"\n✓ Backup completed successfully!")
            print(f"✓ Backup file: {backup_file}")
            return 0
        else:
            print("\n✗ Backup failed!")
            return 1
    
    except KeyboardInterrupt:
        print("\n\nBackup interrupted by user")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return 1
    finally:
        backup_manager.disconnect()


if __name__ == "__main__":
    exit(main())