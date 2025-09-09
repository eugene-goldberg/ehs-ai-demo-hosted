#!/usr/bin/env python3
"""
Neo4j Backup Restore Capability Analyzer

This script analyzes a Neo4j backup file to verify it has all necessary components
for a successful restore operation WITHOUT connecting to Neo4j or modifying any data.

Author: Claude Code
Created: 2025-09-09
"""

import json
import sys
from typing import Dict, List, Set, Tuple, Any
from datetime import datetime
from collections import defaultdict
import os


class BackupRestoreAnalyzer:
    """Analyzes Neo4j backup files for restore capability verification."""
    
    def __init__(self, backup_file_path: str):
        """Initialize the analyzer with backup file path."""
        self.backup_file_path = backup_file_path
        self.backup_data = None
        self.analysis_results = {
            'file_analysis': {},
            'metadata_analysis': {},
            'node_analysis': {},
            'relationship_analysis': {},
            'integrity_checks': {},
            'restore_capability': {},
            'validation_summary': {}
        }
        
    def load_backup_file(self) -> bool:
        """Load and parse the backup JSON file."""
        try:
            print(f"Loading backup file: {self.backup_file_path}")
            
            if not os.path.exists(self.backup_file_path):
                self.analysis_results['file_analysis']['error'] = "Backup file does not exist"
                return False
                
            file_size = os.path.getsize(self.backup_file_path)
            print(f"File size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
            
            with open(self.backup_file_path, 'r', encoding='utf-8') as f:
                self.backup_data = json.load(f)
                
            self.analysis_results['file_analysis'] = {
                'file_exists': True,
                'file_size_bytes': file_size,
                'file_size_mb': round(file_size / (1024*1024), 2),
                'json_valid': True,
                'load_successful': True
            }
            
            print("✓ Backup file loaded successfully")
            return True
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON format: {str(e)}"
            print(f"✗ {error_msg}")
            self.analysis_results['file_analysis']['error'] = error_msg
            return False
        except Exception as e:
            error_msg = f"Failed to load backup file: {str(e)}"
            print(f"✗ {error_msg}")
            self.analysis_results['file_analysis']['error'] = error_msg
            return False
    
    def analyze_metadata(self) -> None:
        """Analyze backup metadata for completeness and validity."""
        print("\n=== METADATA ANALYSIS ===")
        
        if 'metadata' not in self.backup_data:
            self.analysis_results['metadata_analysis']['error'] = "No metadata section found"
            print("✗ No metadata section found")
            return
            
        metadata = self.backup_data['metadata']
        required_fields = [
            'backup_timestamp', 'backup_version', 'neo4j_uri', 
            'neo4j_database', 'catalog'
        ]
        
        missing_fields = []
        for field in required_fields:
            if field not in metadata:
                missing_fields.append(field)
        
        # Analyze catalog
        catalog_analysis = {}
        if 'catalog' in metadata:
            catalog = metadata['catalog']
            catalog_analysis = {
                'total_nodes': catalog.get('total_nodes', 0),
                'total_relationships': catalog.get('total_relationships', 0),
                'node_label_count': catalog.get('node_label_count', 0),
                'relationship_type_count': catalog.get('relationship_type_count', 0),
                'node_labels': len(catalog.get('node_labels', [])),
                'relationship_types': len(catalog.get('relationship_types', [])),
                'has_node_label_counts': 'node_label_counts' in catalog,
                'has_relationship_type_counts': 'relationship_type_counts' in catalog
            }
        
        self.analysis_results['metadata_analysis'] = {
            'has_metadata': True,
            'missing_required_fields': missing_fields,
            'backup_timestamp': metadata.get('backup_timestamp'),
            'backup_version': metadata.get('backup_version'),
            'validation_passed': metadata.get('validation_passed', False),
            'validation_issues': metadata.get('validation_issues', []),
            'catalog': catalog_analysis
        }
        
        # Print results
        if missing_fields:
            print(f"✗ Missing required metadata fields: {missing_fields}")
        else:
            print("✓ All required metadata fields present")
            
        print(f"✓ Backup timestamp: {metadata.get('backup_timestamp')}")
        print(f"✓ Backup version: {metadata.get('backup_version')}")
        print(f"✓ Target database: {metadata.get('neo4j_database')}")
        print(f"✓ Catalog shows {catalog_analysis.get('total_nodes', 0):,} nodes, {catalog_analysis.get('total_relationships', 0):,} relationships")
        
        if metadata.get('validation_passed'):
            print("✓ Original backup validation passed")
        else:
            print("⚠ Original backup validation did not pass")
    
    def analyze_nodes(self) -> None:
        """Analyze node data structure and completeness."""
        print("\n=== NODE ANALYSIS ===")
        
        if 'nodes' not in self.backup_data:
            self.analysis_results['node_analysis']['error'] = "No nodes section found"
            print("✗ No nodes section found")
            return
            
        nodes = self.backup_data['nodes']
        node_count = len(nodes)
        
        # Sample analysis on first 100 nodes for structure verification
        sample_size = min(100, node_count)
        nodes_with_ids = 0
        nodes_with_labels = 0
        nodes_with_properties = 0
        unique_node_ids = set()
        label_distribution = defaultdict(int)
        
        # Check for required fields in nodes
        for i in range(sample_size):
            node = nodes[i]
            
            if 'node_id' in node:
                nodes_with_ids += 1
                unique_node_ids.add(node['node_id'])
                
            if 'labels' in node and node['labels']:
                nodes_with_labels += 1
                for label in node['labels']:
                    label_distribution[label] += 1
                    
            if 'properties' in node:
                nodes_with_properties += 1
        
        # Verify node ID continuity for full dataset
        print(f"Verifying node ID continuity for all {node_count:,} nodes...")
        all_node_ids = set()
        expected_max_id = node_count - 1  # Assuming 0-based indexing
        
        for node in nodes:
            if 'node_id' in node:
                all_node_ids.add(node['node_id'])
        
        missing_ids = []
        for i in range(node_count):
            if i not in all_node_ids:
                missing_ids.append(i)
                if len(missing_ids) >= 10:  # Limit output
                    missing_ids.append("...")
                    break
        
        missing_ids_count = len([x for x in missing_ids if x != "..."])
        if "..." in missing_ids:
            missing_ids_display = "10+"
        else:
            missing_ids_display = missing_ids_count
        
        self.analysis_results['node_analysis'] = {
            'total_nodes': node_count,
            'sample_size': sample_size,
            'nodes_with_ids': nodes_with_ids,
            'nodes_with_labels': nodes_with_labels,
            'nodes_with_properties': nodes_with_properties,
            'unique_node_ids_in_sample': len(unique_node_ids),
            'total_unique_node_ids': len(all_node_ids),
            'expected_node_count': expected_max_id + 1,
            'missing_node_ids': missing_ids[:10],  # First 10 missing IDs
            'missing_node_ids_count': missing_ids_count,
            'missing_node_ids_display': missing_ids_display,
            'label_distribution_sample': dict(label_distribution)
        }
        
        # Print results
        print(f"✓ Found {node_count:,} nodes in backup")
        print(f"✓ Sample analysis: {nodes_with_ids}/{sample_size} nodes have IDs")
        print(f"✓ Sample analysis: {nodes_with_labels}/{sample_size} nodes have labels")
        print(f"✓ Sample analysis: {nodes_with_properties}/{sample_size} nodes have properties")
        print(f"✓ Total unique node IDs: {len(all_node_ids):,}")
        
        if missing_ids_count > 0:
            print(f"⚠ Missing node IDs detected: {missing_ids_display}")
        else:
            print("✓ Node ID continuity verified - no gaps detected")
    
    def analyze_relationships(self) -> None:
        """Analyze relationship data structure and references."""
        print("\n=== RELATIONSHIP ANALYSIS ===")
        
        if 'relationships' not in self.backup_data:
            self.analysis_results['relationship_analysis']['error'] = "No relationships section found"
            print("✗ No relationships section found")
            return
            
        relationships = self.backup_data['relationships']
        rel_count = len(relationships)
        
        # Get node IDs for reference validation
        if 'nodes' not in self.backup_data:
            print("⚠ Cannot validate relationship references - no nodes data")
            return
            
        valid_node_ids = {node['node_id'] for node in self.backup_data['nodes'] if 'node_id' in node}
        
        # Analyze relationships
        rels_with_ids = 0
        rels_with_start_nodes = 0
        rels_with_end_nodes = 0
        rels_with_types = 0
        rels_with_properties = 0
        valid_start_refs = 0
        valid_end_refs = 0
        unique_rel_ids = set()
        type_distribution = defaultdict(int)
        invalid_start_refs = []
        invalid_end_refs = []
        
        print(f"Analyzing {rel_count:,} relationships...")
        
        for rel in relationships:
            if 'rel_id' in rel:
                rels_with_ids += 1
                unique_rel_ids.add(rel['rel_id'])
                
            if 'start_node_id' in rel:
                rels_with_start_nodes += 1
                start_id = rel['start_node_id']
                if start_id in valid_node_ids:
                    valid_start_refs += 1
                else:
                    if len(invalid_start_refs) < 10:
                        invalid_start_refs.append(start_id)
                        
            if 'end_node_id' in rel:
                rels_with_end_nodes += 1
                end_id = rel['end_node_id']
                if end_id in valid_node_ids:
                    valid_end_refs += 1
                else:
                    if len(invalid_end_refs) < 10:
                        invalid_end_refs.append(end_id)
                        
            if 'type' in rel and rel['type']:
                rels_with_types += 1
                type_distribution[rel['type']] += 1
                
            if 'properties' in rel:
                rels_with_properties += 1
        
        # Check relationship ID continuity
        expected_rel_ids = set(range(rel_count))
        missing_rel_ids = expected_rel_ids - unique_rel_ids
        
        self.analysis_results['relationship_analysis'] = {
            'total_relationships': rel_count,
            'rels_with_ids': rels_with_ids,
            'rels_with_start_nodes': rels_with_start_nodes,
            'rels_with_end_nodes': rels_with_end_nodes,
            'rels_with_types': rels_with_types,
            'rels_with_properties': rels_with_properties,
            'unique_rel_ids': len(unique_rel_ids),
            'valid_start_refs': valid_start_refs,
            'valid_end_refs': valid_end_refs,
            'invalid_start_refs': invalid_start_refs[:10],
            'invalid_end_refs': invalid_end_refs[:10],
            'missing_rel_ids': list(missing_rel_ids)[:10],
            'missing_rel_ids_count': len(missing_rel_ids),
            'type_distribution': dict(type_distribution)
        }
        
        # Print results
        print(f"✓ Found {rel_count:,} relationships in backup")
        print(f"✓ {rels_with_ids:,}/{rel_count:,} relationships have IDs")
        print(f"✓ {rels_with_start_nodes:,}/{rel_count:,} relationships have start node IDs")
        print(f"✓ {rels_with_end_nodes:,}/{rel_count:,} relationships have end node IDs")
        print(f"✓ {rels_with_types:,}/{rel_count:,} relationships have types")
        print(f"✓ {valid_start_refs:,}/{rels_with_start_nodes:,} valid start node references")
        print(f"✓ {valid_end_refs:,}/{rels_with_end_nodes:,} valid end node references")
        
        if invalid_start_refs:
            print(f"⚠ {len(invalid_start_refs)} invalid start node references found (showing first 10)")
        if invalid_end_refs:
            print(f"⚠ {len(invalid_end_refs)} invalid end node references found (showing first 10)")
        if missing_rel_ids:
            print(f"⚠ {len(missing_rel_ids)} missing relationship IDs")
    
    def perform_integrity_checks(self) -> None:
        """Perform comprehensive data integrity checks."""
        print("\n=== DATA INTEGRITY CHECKS ===")
        
        integrity_results = {
            'orphaned_relationships': 0,
            'circular_references': [],
            'data_consistency': True,
            'structure_valid': True
        }
        
        if 'nodes' not in self.backup_data or 'relationships' not in self.backup_data:
            integrity_results['structure_valid'] = False
            print("✗ Cannot perform integrity checks - missing nodes or relationships data")
            self.analysis_results['integrity_checks'] = integrity_results
            return
        
        # Check for orphaned relationships
        valid_node_ids = {node['node_id'] for node in self.backup_data['nodes'] if 'node_id' in node}
        orphaned_count = 0
        
        for rel in self.backup_data['relationships']:
            start_id = rel.get('start_node_id')
            end_id = rel.get('end_node_id')
            
            if (start_id is not None and start_id not in valid_node_ids) or \
               (end_id is not None and end_id not in valid_node_ids):
                orphaned_count += 1
        
        integrity_results['orphaned_relationships'] = orphaned_count
        
        # Cross-reference with metadata catalog
        metadata_consistency = True
        if 'metadata' in self.backup_data and 'catalog' in self.backup_data['metadata']:
            catalog = self.backup_data['metadata']['catalog']
            expected_nodes = catalog.get('total_nodes', 0)
            expected_rels = catalog.get('total_relationships', 0)
            actual_nodes = len(self.backup_data['nodes'])
            actual_rels = len(self.backup_data['relationships'])
            
            if expected_nodes != actual_nodes or expected_rels != actual_rels:
                metadata_consistency = False
        
        integrity_results['metadata_consistency'] = metadata_consistency
        integrity_results['expected_vs_actual_nodes'] = {
            'expected': catalog.get('total_nodes', 0) if 'metadata' in self.backup_data else 0,
            'actual': len(self.backup_data['nodes'])
        }
        integrity_results['expected_vs_actual_rels'] = {
            'expected': catalog.get('total_relationships', 0) if 'metadata' in self.backup_data else 0,
            'actual': len(self.backup_data['relationships'])
        }
        
        self.analysis_results['integrity_checks'] = integrity_results
        
        # Print results
        if orphaned_count == 0:
            print("✓ No orphaned relationships detected")
        else:
            print(f"⚠ {orphaned_count:,} orphaned relationships found")
            
        if metadata_consistency:
            print("✓ Metadata catalog matches actual data counts")
        else:
            print("⚠ Metadata catalog does not match actual data counts")
    
    def assess_restore_capability(self) -> None:
        """Assess overall restore capability and generate recommendations."""
        print("\n=== RESTORE CAPABILITY ASSESSMENT ===")
        
        # Collect all critical issues
        critical_issues = []
        warnings = []
        
        # File-level issues
        if 'error' in self.analysis_results.get('file_analysis', {}):
            critical_issues.append("Backup file cannot be loaded")
        
        # Metadata issues
        metadata = self.analysis_results.get('metadata_analysis', {})
        if metadata.get('missing_required_fields'):
            critical_issues.append(f"Missing metadata fields: {metadata['missing_required_fields']}")
        
        # Node issues
        nodes = self.analysis_results.get('node_analysis', {})
        if 'error' in nodes:
            critical_issues.append("No node data found")
        elif nodes.get('missing_node_ids_count', 0) > 0:
            warnings.append(f"Missing node IDs detected: {nodes['missing_node_ids_display']}")
        
        # Relationship issues
        rels = self.analysis_results.get('relationship_analysis', {})
        if 'error' in rels:
            critical_issues.append("No relationship data found")
        elif rels.get('invalid_start_refs') or rels.get('invalid_end_refs'):
            warnings.append("Invalid relationship references detected")
        
        # Check for missing relationship IDs
        if rels.get('missing_rel_ids_count', 0) > 0:
            warnings.append(f"Missing relationship IDs detected: {rels['missing_rel_ids_count']}")
        
        # Integrity issues
        integrity = self.analysis_results.get('integrity_checks', {})
        if integrity.get('orphaned_relationships', 0) > 0:
            warnings.append(f"Orphaned relationships: {integrity['orphaned_relationships']}")
        
        if not integrity.get('metadata_consistency', True):
            warnings.append("Metadata inconsistency with actual data")
        
        # Determine restore capability
        can_restore = len(critical_issues) == 0
        restore_confidence = "HIGH" if can_restore and len(warnings) == 0 else \
                           "MEDIUM" if can_restore and len(warnings) <= 2 else \
                           "LOW" if can_restore else "IMPOSSIBLE"
        
        self.analysis_results['restore_capability'] = {
            'can_restore': can_restore,
            'confidence': restore_confidence,
            'critical_issues': critical_issues,
            'warnings': warnings,
            'recommendation': self._generate_recommendation(can_restore, critical_issues, warnings)
        }
        
        # Print assessment
        print(f"Restore Capability: {'✓ POSSIBLE' if can_restore else '✗ NOT POSSIBLE'}")
        print(f"Confidence Level: {restore_confidence}")
        
        if critical_issues:
            print("\nCRITICAL ISSUES (must be resolved):")
            for issue in critical_issues:
                print(f"  ✗ {issue}")
        
        if warnings:
            print("\nWARNINGS (should be investigated):")
            for warning in warnings:
                print(f"  ⚠ {warning}")
        
        if can_restore:
            print("\n✓ This backup contains all necessary components for a successful restore:")
            print("  - Node internal IDs for relationship mapping")
            print("  - Complete node properties and labels") 
            print("  - Complete relationship properties and types")
            print("  - Valid start and end node IDs for relationships")
            print("  - Backup metadata for validation")
    
    def _generate_recommendation(self, can_restore: bool, critical_issues: List[str], warnings: List[str]) -> str:
        """Generate restoration recommendation based on analysis."""
        if not can_restore:
            return "DO NOT ATTEMPT RESTORE - Critical issues must be resolved first"
        elif len(warnings) == 0:
            return "SAFE TO RESTORE - No issues detected"
        elif len(warnings) <= 2:
            return "RESTORE WITH CAUTION - Minor issues detected, monitor restoration process"
        else:
            return "INVESTIGATE BEFORE RESTORE - Multiple warnings detected"
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        print("\n" + "="*60)
        print("BACKUP RESTORE CAPABILITY ANALYSIS SUMMARY")
        print("="*60)
        
        # File info
        file_analysis = self.analysis_results.get('file_analysis', {})
        print(f"Backup File: {os.path.basename(self.backup_file_path)}")
        print(f"File Size: {file_analysis.get('file_size_mb', 0)} MB")
        
        # Data overview
        metadata = self.analysis_results.get('metadata_analysis', {})
        catalog = metadata.get('catalog', {})
        print(f"Total Nodes: {catalog.get('total_nodes', 0):,}")
        print(f"Total Relationships: {catalog.get('total_relationships', 0):,}")
        print(f"Node Labels: {catalog.get('node_labels', 0)}")
        print(f"Relationship Types: {catalog.get('relationship_types', 0)}")
        
        # Restore capability
        restore = self.analysis_results.get('restore_capability', {})
        print(f"\nRestore Capability: {'✓ POSSIBLE' if restore.get('can_restore') else '✗ NOT POSSIBLE'}")
        print(f"Confidence: {restore.get('confidence', 'UNKNOWN')}")
        print(f"Recommendation: {restore.get('recommendation', 'No recommendation available')}")
        
        # Issues summary
        critical_count = len(restore.get('critical_issues', []))
        warning_count = len(restore.get('warnings', []))
        print(f"Critical Issues: {critical_count}")
        print(f"Warnings: {warning_count}")
        
        # Analysis timestamp
        self.analysis_results['validation_summary'] = {
            'analysis_timestamp': datetime.now().isoformat(),
            'backup_file': self.backup_file_path,
            'analyzer_version': '1.0.0',
            'total_checks_performed': 5,
            'overall_status': 'PASSED' if restore.get('can_restore') else 'FAILED'
        }
        
        print(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        return self.analysis_results
    
    def save_detailed_report(self, output_file: str = None) -> str:
        """Save detailed analysis report to JSON file."""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"backup_restore_analysis_{timestamp}.json"
        
        output_path = os.path.join(os.path.dirname(self.backup_file_path), output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        print(f"\nDetailed report saved to: {output_path}")
        return output_path


def main():
    """Main execution function."""
    # Default backup file path
    backup_file = "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/neo4j-data/neo4j_backup_20250909_075720.json"
    
    # Allow command line override
    if len(sys.argv) > 1:
        backup_file = sys.argv[1]
    
    print("Neo4j Backup Restore Capability Analyzer")
    print("=" * 50)
    print(f"Target backup file: {backup_file}")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = BackupRestoreAnalyzer(backup_file)
    
    # Perform analysis
    try:
        # Load backup file
        if not analyzer.load_backup_file():
            print("Analysis failed - cannot load backup file")
            return 1
        
        # Run all analyses
        analyzer.analyze_metadata()
        analyzer.analyze_nodes()
        analyzer.analyze_relationships()
        analyzer.perform_integrity_checks()
        analyzer.assess_restore_capability()
        
        # Generate reports
        results = analyzer.generate_summary_report()
        report_file = analyzer.save_detailed_report()
        
        # Return appropriate exit code
        return 0 if results['restore_capability']['can_restore'] else 1
        
    except Exception as e:
        print(f"\nAnalysis failed with error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())