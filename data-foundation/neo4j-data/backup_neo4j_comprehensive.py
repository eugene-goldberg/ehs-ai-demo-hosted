#!/usr/bin/env python3
"""
Neo4j Comprehensive Database Backup Script

This script creates a comprehensive backup of a Neo4j database by:
1. Connecting to the Neo4j database using environment variables
2. Exporting ALL nodes with all their properties and labels
3. Exporting ALL relationships with all their properties
4. Providing detailed progress tracking for large datasets
5. Validating backup counts against database catalog (only for non-empty types)
6. Saving everything to a timestamped JSON file
7. Including backup metadata (date, node count, relationship count, execution time)

Usage:
    python3 backup_neo4j_comprehensive.py [batch_size] [--force]
    
Options:
    batch_size: Number of items to process in each batch (default: 1000)
    --force: Skip interactive validation prompt and continue anyway

Output:
    Creates neo4j_backup_YYYYMMDD_HHMMSS.json in the current directory
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import time
from pathlib import Path
import sys
import argparse

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


class ProgressTracker:
    """Handles progress tracking for large dataset operations."""
    
    def __init__(self, total: int, operation: str):
        self.total = total
        self.operation = operation
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0
        
    def update(self, increment: int = 1):
        """Update progress and log if significant progress made."""
        self.current += increment
        current_time = time.time()
        
        # Log every 10% progress or every 5 seconds
        progress_pct = (self.current / self.total) * 100
        if (current_time - self.last_update > 5) or (progress_pct >= self.last_update + 10):
            elapsed = current_time - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            remaining = (self.total - self.current) / rate if rate > 0 else 0
            
            print(f"{self.operation}: {self.current:,}/{self.total:,} ({progress_pct:.1f}%) "
                  f"- Rate: {rate:.1f}/sec - ETA: {remaining:.0f}s")
            
            self.last_update = progress_pct
    
    def complete(self):
        """Mark operation as complete and log final stats."""
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        print(f"{self.operation} completed: {self.current:,} items in {elapsed:.1f}s "
              f"(avg rate: {rate:.1f}/sec)")


class Neo4jBackupManager:
    """Handles comprehensive Neo4j database backup operations."""
    
    def __init__(self, force_mode: bool = False):
        """Initialize the backup manager with environment variables."""
        # Load environment variables
        load_dotenv()
        
        # Neo4j connection parameters
        self.uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.username = os.getenv('NEO4J_USERNAME', 'neo4j')
        self.password = os.getenv('NEO4J_PASSWORD', '')
        self.database = os.getenv('NEO4J_DATABASE', 'neo4j')
        self.force_mode = force_mode
        
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
                logging.FileHandler('backup_neo4j_comprehensive.log'),
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
    
    def get_database_catalog(self) -> Dict[str, Any]:
        """
        Get comprehensive database catalog information for validation.
        
        Returns:
            Dict[str, Any]: Complete catalog with counts and types
        """
        catalog = {}
        try:
            with self.driver.session(database=self.database) as session:
                # Get total counts first
                result = session.run("MATCH (n) RETURN count(n) as node_count")
                catalog['total_nodes'] = result.single()['node_count']
                
                result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                catalog['total_relationships'] = result.single()['rel_count']
                
                # Get all node labels
                result = session.run("CALL db.labels()")
                labels = [record["label"] for record in result]
                catalog['node_labels'] = labels
                catalog['node_label_count'] = len(labels)
                
                # Get node label counts
                label_counts = {}
                non_empty_labels = []
                empty_labels = []
                for label in labels:
                    # Use parameterized query to avoid injection issues
                    result = session.run(
                        "MATCH (n) WHERE $label IN labels(n) RETURN count(n) as count",
                        label=label
                    )
                    count = result.single()['count']
                    label_counts[label] = count
                    if count > 0:
                        non_empty_labels.append(label)
                    else:
                        empty_labels.append(label)
                
                catalog['node_label_counts'] = label_counts
                catalog['non_empty_labels'] = non_empty_labels
                catalog['empty_labels'] = empty_labels
                
                # Get all relationship types
                result = session.run("CALL db.relationshipTypes()")
                rel_types = [record["relationshipType"] for record in result]
                catalog['relationship_types'] = rel_types
                catalog['relationship_type_count'] = len(rel_types)
                
                # Get relationship type counts
                rel_type_counts = {}
                non_empty_rel_types = []
                empty_rel_types = []
                for rel_type in rel_types:
                    result = session.run(
                        "MATCH ()-[r]->() WHERE type(r) = $rel_type RETURN count(r) as count",
                        rel_type=rel_type
                    )
                    count = result.single()['count']
                    rel_type_counts[rel_type] = count
                    if count > 0:
                        non_empty_rel_types.append(rel_type)
                    else:
                        empty_rel_types.append(rel_type)
                
                catalog['relationship_type_counts'] = rel_type_counts
                catalog['non_empty_relationship_types'] = non_empty_rel_types
                catalog['empty_relationship_types'] = empty_rel_types
                
                self.logger.info(f"Database catalog: {catalog['total_nodes']:,} nodes, "
                               f"{catalog['total_relationships']:,} relationships, "
                               f"{catalog['node_label_count']} node labels, "
                               f"{catalog['relationship_type_count']} relationship types")
                
                # Log empty types for visibility
                if empty_labels:
                    self.logger.info(f"Empty node labels (0 count): {sorted(empty_labels)}")
                if empty_rel_types:
                    self.logger.info(f"Empty relationship types (0 count): {sorted(empty_rel_types)}")
                
                return catalog
                
        except Exception as e:
            self.logger.error(f"Error getting database catalog: {e}")
            return {}
    
    def export_all_nodes_batched(self, batch_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Export all nodes with their properties in batches with progress tracking.
        
        Args:
            batch_size (int): Number of nodes to process in each batch
        
        Returns:
            List[Dict[str, Any]]: List of all nodes with properties
        """
        nodes = []
        try:
            with self.driver.session(database=self.database) as session:
                # Get total count first
                result = session.run("MATCH (n) RETURN count(n) as total")
                total_nodes = result.single()['total']
                
                if total_nodes == 0:
                    self.logger.warning("No nodes found in database")
                    return []
                
                self.logger.info(f"Starting export of {total_nodes:,} nodes in batches of {batch_size:,}")
                progress = ProgressTracker(total_nodes, "Exporting nodes")
                
                # Export nodes in batches
                skip = 0
                while skip < total_nodes:
                    query = """
                    MATCH (n)
                    RETURN 
                        id(n) as node_id,
                        labels(n) as labels,
                        properties(n) as properties
                    ORDER BY id(n)
                    SKIP $skip LIMIT $limit
                    """
                    
                    result = session.run(query, skip=skip, limit=batch_size)
                    batch_count = 0
                    
                    for record in result:
                        node_data = {
                            'node_id': record['node_id'],
                            'labels': record['labels'],
                            'properties': dict(record['properties'])
                        }
                        nodes.append(node_data)
                        batch_count += 1
                    
                    if batch_count == 0:
                        break  # No more results
                    
                    progress.update(batch_count)
                    skip += batch_size
                
                progress.complete()
                self.logger.info(f"Successfully exported {len(nodes):,} nodes")
                return nodes
                
        except Exception as e:
            self.logger.error(f"Error exporting nodes: {e}")
            return []
    
    def export_all_relationships_batched(self, batch_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Export all relationships with their properties in batches with progress tracking.
        
        Args:
            batch_size (int): Number of relationships to process in each batch
        
        Returns:
            List[Dict[str, Any]]: List of all relationships with properties
        """
        relationships = []
        try:
            with self.driver.session(database=self.database) as session:
                # Get total count first
                result = session.run("MATCH ()-[r]->() RETURN count(r) as total")
                total_rels = result.single()['total']
                
                if total_rels == 0:
                    self.logger.warning("No relationships found in database")
                    return []
                
                self.logger.info(f"Starting export of {total_rels:,} relationships in batches of {batch_size:,}")
                progress = ProgressTracker(total_rels, "Exporting relationships")
                
                # Export relationships in batches
                skip = 0
                while skip < total_rels:
                    query = """
                    MATCH (start_node)-[r]->(end_node)
                    RETURN 
                        id(r) as rel_id,
                        id(start_node) as start_node_id,
                        id(end_node) as end_node_id,
                        type(r) as relationship_type,
                        properties(r) as properties
                    ORDER BY id(r)
                    SKIP $skip LIMIT $limit
                    """
                    
                    result = session.run(query, skip=skip, limit=batch_size)
                    batch_count = 0
                    
                    for record in result:
                        rel_data = {
                            'rel_id': record['rel_id'],
                            'start_node_id': record['start_node_id'],
                            'end_node_id': record['end_node_id'],
                            'type': record['relationship_type'],
                            'properties': dict(record['properties'])
                        }
                        relationships.append(rel_data)
                        batch_count += 1
                    
                    if batch_count == 0:
                        break  # No more results
                    
                    progress.update(batch_count)
                    skip += batch_size
                
                progress.complete()
                self.logger.info(f"Successfully exported {len(relationships):,} relationships")
                return relationships
                
        except Exception as e:
            self.logger.error(f"Error exporting relationships: {e}")
            return []
    
    def validate_backup_completeness(self, nodes: List[Dict], relationships: List[Dict], 
                                   catalog: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate that backup is complete by comparing counts with catalog.
        Only validates non-empty node labels and relationship types.
        
        Args:
            nodes: Exported nodes list
            relationships: Exported relationships list
            catalog: Database catalog information
        
        Returns:
            Tuple[bool, List[str]]: (is_complete, list_of_issues)
        """
        issues = []
        
        # Validate total counts
        if len(nodes) != catalog.get('total_nodes', 0):
            issues.append(f"Node count mismatch: exported {len(nodes):,}, "
                         f"catalog shows {catalog.get('total_nodes', 0):,}")
        
        if len(relationships) != catalog.get('total_relationships', 0):
            issues.append(f"Relationship count mismatch: exported {len(relationships):,}, "
                         f"catalog shows {catalog.get('total_relationships', 0):,}")
        
        # Only validate non-empty node labels
        exported_labels = set()
        for node in nodes:
            exported_labels.update(node.get('labels', []))
        
        non_empty_labels = set(catalog.get('non_empty_labels', []))
        missing_labels = non_empty_labels - exported_labels
        if missing_labels:
            issues.append(f"Missing non-empty node labels in backup: {sorted(missing_labels)}")
        
        # Only validate non-empty relationship types (using 'type' key now)
        exported_rel_types = set(rel.get('type') for rel in relationships if rel.get('type'))
        non_empty_rel_types = set(catalog.get('non_empty_relationship_types', []))
        missing_rel_types = non_empty_rel_types - exported_rel_types
        if missing_rel_types:
            issues.append(f"Missing non-empty relationship types in backup: {sorted(missing_rel_types)}")
        
        # Count nodes by label in backup
        backup_label_counts = {}
        for node in nodes:
            for label in node.get('labels', []):
                backup_label_counts[label] = backup_label_counts.get(label, 0) + 1
        
        # Validate label counts (only for non-empty labels)
        catalog_label_counts = catalog.get('node_label_counts', {})
        for label in catalog.get('non_empty_labels', []):
            catalog_count = catalog_label_counts.get(label, 0)
            backup_count = backup_label_counts.get(label, 0)
            if backup_count != catalog_count:
                issues.append(f"Label count mismatch for '{label}': "
                             f"exported {backup_count:,}, catalog shows {catalog_count:,}")
        
        # Count relationships by type in backup (using 'type' key now)
        backup_rel_type_counts = {}
        for rel in relationships:
            rel_type = rel.get('type')
            if rel_type:
                backup_rel_type_counts[rel_type] = backup_rel_type_counts.get(rel_type, 0) + 1
        
        # Validate relationship type counts (only for non-empty types)
        catalog_rel_type_counts = catalog.get('relationship_type_counts', {})
        for rel_type in catalog.get('non_empty_relationship_types', []):
            catalog_count = catalog_rel_type_counts.get(rel_type, 0)
            backup_count = backup_rel_type_counts.get(rel_type, 0)
            if backup_count != catalog_count:
                issues.append(f"Relationship type count mismatch for '{rel_type}': "
                             f"exported {backup_count:,}, catalog shows {catalog_count:,}")
        
        is_complete = len(issues) == 0
        
        if is_complete:
            self.logger.info("✓ Backup validation successful - all counts match catalog")
        else:
            self.logger.warning(f"✗ Backup validation found {len(issues)} issues:")
            for issue in issues:
                self.logger.warning(f"  - {issue}")
        
        return is_complete, issues
    
    def create_comprehensive_backup(self, output_dir: Optional[str] = None, 
                                  batch_size: int = 1000) -> Optional[str]:
        """
        Create a comprehensive backup of the Neo4j database with validation.
        
        Args:
            output_dir (Optional[str]): Directory to save backup file. 
                                      Defaults to current directory.
            batch_size (int): Batch size for processing large datasets
        
        Returns:
            Optional[str]: Path to backup file if successful, None otherwise
        """
        start_time = time.time()
        
        try:
            # Determine output directory
            if output_dir is None:
                output_dir = Path.cwd()
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"neo4j_backup_{timestamp}.json"
            backup_filepath = output_dir / backup_filename
            
            self.logger.info(f"Starting comprehensive backup to {backup_filepath}")
            print(f"Starting comprehensive backup to {backup_filepath}")
            
            # Step 1: Get database catalog for validation
            print("\n1. Analyzing database catalog...")
            catalog = self.get_database_catalog()
            if not catalog:
                self.logger.error("Failed to get database catalog")
                return None
            
            print(f"   Database contains:")
            print(f"   - {catalog['total_nodes']:,} total nodes")
            print(f"   - {catalog['total_relationships']:,} total relationships")
            print(f"   - {catalog['node_label_count']} different node labels")
            print(f"   - {catalog['relationship_type_count']} different relationship types")
            
            # Show empty labels/types if any
            empty_labels = catalog.get('empty_labels', [])
            empty_rel_types = catalog.get('empty_relationship_types', [])
            
            if empty_labels:
                print(f"   - {len(empty_labels)} node labels with 0 count: {sorted(empty_labels)}")
            
            if empty_rel_types:
                print(f"   - {len(empty_rel_types)} relationship types with 0 count: {sorted(empty_rel_types)}")
            
            non_empty_labels = len(catalog.get('non_empty_labels', []))
            non_empty_rel_types = len(catalog.get('non_empty_relationship_types', []))
            print(f"   - {non_empty_labels} non-empty node labels will be validated")
            print(f"   - {non_empty_rel_types} non-empty relationship types will be validated")
            
            # Step 2: Export all nodes with progress tracking
            print("\n2. Exporting all nodes...")
            nodes = self.export_all_nodes_batched(batch_size)
            if not nodes and catalog['total_nodes'] > 0:
                self.logger.error("Failed to export nodes")
                return None
            
            # Step 3: Export all relationships with progress tracking
            print("\n3. Exporting all relationships...")
            relationships = self.export_all_relationships_batched(batch_size)
            if not relationships and catalog['total_relationships'] > 0:
                self.logger.error("Failed to export relationships")
                return None
            
            # Step 4: Validate backup completeness
            print("\n4. Validating backup completeness...")
            is_complete, issues = self.validate_backup_completeness(nodes, relationships, catalog)
            
            if not is_complete:
                print("⚠️  WARNING: Backup validation found issues:")
                for issue in issues:
                    print(f"   - {issue}")
                
                # Ask user if they want to continue despite issues (unless force mode)
                if not self.force_mode:
                    response = input("\nContinue with backup despite validation issues? (y/N): ")
                    if response.lower() != 'y':
                        self.logger.info("Backup cancelled by user due to validation issues")
                        return None
                else:
                    print("   (Continuing anyway due to --force flag)")
                    self.logger.info("Continuing backup despite validation issues due to --force flag")
            
            # Calculate execution time for data export
            export_time = time.time() - start_time
            
            # Step 5: Create backup data structure
            print("\n5. Creating backup file...")
            backup_data = {
                'metadata': {
                    'backup_timestamp': datetime.now().isoformat(),
                    'neo4j_uri': self.uri,
                    'neo4j_database': self.database,
                    'backup_version': '2.1',
                    'export_time_seconds': round(export_time, 2),
                    'batch_size': batch_size,
                    'validation_passed': is_complete,
                    'validation_issues': issues,
                    'force_mode': self.force_mode,
                    'catalog': catalog
                },
                'nodes': nodes,
                'relationships': relationships
            }
            
            # Step 6: Write backup to JSON file
            write_start = time.time()
            with open(backup_filepath, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False, default=str)
            write_time = time.time() - write_start
            
            # Calculate final statistics
            total_time = time.time() - start_time
            file_size_mb = backup_filepath.stat().st_size / 1024 / 1024
            
            # Log and display success
            print(f"\n✓ Backup completed successfully!")
            print(f"✓ Backup file: {backup_filepath}")
            print(f"✓ Total nodes exported: {len(nodes):,}")
            print(f"✓ Total relationships exported: {len(relationships):,}")
            print(f"✓ File size: {file_size_mb:.2f} MB")
            print(f"✓ Export time: {export_time:.1f}s")
            print(f"✓ Write time: {write_time:.1f}s")
            print(f"✓ Total time: {total_time:.1f}s")
            print(f"✓ Validation: {'PASSED' if is_complete else 'ISSUES FOUND'}")
            if self.force_mode:
                print(f"✓ Force mode: {'ENABLED' if self.force_mode else 'DISABLED'}")
            
            self.logger.info(f"Comprehensive backup completed successfully")
            self.logger.info(f"File: {backup_filepath}")
            self.logger.info(f"Nodes: {len(nodes):,}, Relationships: {len(relationships):,}")
            self.logger.info(f"Size: {file_size_mb:.2f} MB, Time: {total_time:.1f}s")
            self.logger.info(f"Validation: {'PASSED' if is_complete else 'ISSUES'}")
            self.logger.info(f"Force mode: {self.force_mode}")
            
            return str(backup_filepath)
            
        except KeyboardInterrupt:
            self.logger.warning("Backup interrupted by user")
            print("\n⚠️  Backup interrupted by user")
            return None
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            print(f"\n✗ Backup failed: {e}")
            return None


def main():
    """Main function to run the comprehensive backup."""
    print("Neo4j Comprehensive Database Backup Tool")
    print("=" * 45)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Neo4j Comprehensive Database Backup Tool")
    parser.add_argument('batch_size', type=int, nargs='?', default=1000,
                       help='Number of items to process in each batch (default: 1000)')
    parser.add_argument('--force', action='store_true',
                       help='Skip interactive validation prompt and continue anyway')
    
    args = parser.parse_args()
    
    print(f"Using batch size: {args.batch_size:,}")
    if args.force:
        print("Force mode: ENABLED (will skip validation prompts)")
    
    backup_manager = Neo4jBackupManager(force_mode=args.force)
    
    try:
        # Connect to Neo4j
        if not backup_manager.connect():
            print("✗ Failed to connect to Neo4j. Check your connection settings.")
            return 1
        
        # Create comprehensive backup
        backup_file = backup_manager.create_comprehensive_backup(batch_size=args.batch_size)
        
        if backup_file:
            return 0
        else:
            print("\n✗ Backup failed!")
            return 1
    
    except KeyboardInterrupt:
        print("\n\nBackup interrupted by user")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        backup_manager.logger.error(f"Unexpected error in main: {e}")
        return 1
    finally:
        backup_manager.disconnect()


if __name__ == "__main__":
    exit(main())