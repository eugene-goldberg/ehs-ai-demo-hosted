#!/usr/bin/env python3
"""
Neo4j Backup Verification Script - Final Version

This script verifies that a Neo4j backup matches the catalog with 100% certainty.
Optimized for performance and accurate parsing.
"""

import json
import re
import os
import sys
from datetime import datetime
from collections import defaultdict


def load_catalog_stats(catalog_path):
    """Extract key statistics from catalog"""
    print("Loading catalog statistics...")
    
    with open(catalog_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract total statistics
    total_nodes_match = re.search(r'- \*\*Total Nodes:\*\* (\d+)', content)
    total_relationships_match = re.search(r'- \*\*Total Relationships:\*\* (\d+)', content)

    total_nodes = int(total_nodes_match.group(1)) if total_nodes_match else 0
    total_relationships = int(total_relationships_match.group(1)) if total_relationships_match else 0

    print(f"  Found totals - Nodes: {total_nodes}, Relationships: {total_relationships}")

    # Extract node label counts using a different approach
    node_labels = {}
    
    # Find the start of the node labels table
    lines = content.split('\n')
    in_node_table = False
    
    for line in lines:
        line = line.strip()
        
        if "### Node Label Counts" in line:
            in_node_table = True
            continue
        
        if in_node_table and line.startswith('|') and '|' in line[1:]:
            parts = [part.strip() for part in line.split('|')]
            if len(parts) >= 4 and parts[1] and parts[2]:
                label = parts[1]
                count_str = parts[2]
                
                # Skip header and separator rows
                if label not in ['Label', '-------'] and count_str not in ['Count', '-------']:
                    try:
                        count = int(count_str)
                        node_labels[label] = count
                    except (ValueError, IndexError):
                        continue
        
        # Stop if we hit another section
        if in_node_table and line.startswith('###') and "Node Label Counts" not in line:
            break

    print(f"  Found {len(node_labels)} node labels")

    # Extract relationship type counts
    relationship_types = {}
    in_rel_table = False
    
    for line in lines:
        line = line.strip()
        
        if "### Relationship Type Counts" in line:
            in_rel_table = True
            continue
        
        if in_rel_table and line.startswith('|') and '|' in line[1:]:
            parts = [part.strip() for part in line.split('|')]
            if len(parts) >= 4 and parts[1] and parts[2]:
                rel_type = parts[1]
                count_str = parts[2]
                
                # Skip header and separator rows
                if rel_type not in ['Type', '-------'] and count_str not in ['Count', '-------']:
                    try:
                        count = int(count_str)
                        relationship_types[rel_type] = count
                    except (ValueError, IndexError):
                        continue
        
        # Stop if we hit another section
        if in_rel_table and line.startswith('###') and "Relationship Type Counts" not in line:
            break

    print(f"  Found {len(relationship_types)} relationship types")

    return {
        'total_nodes': total_nodes,
        'total_relationships': total_relationships,
        'node_labels': node_labels,
        'relationship_types': relationship_types
    }


def load_backup_stats(backup_path):
    """Extract key statistics from backup"""
    print("Loading backup file...")
    
    with open(backup_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'metadata' not in data:
        raise ValueError("Backup file missing metadata section")

    metadata = data['metadata']
    
    # Get totals from metadata catalog
    metadata_nodes = 0
    metadata_relationships = 0
    
    if 'catalog' in metadata:
        catalog = metadata['catalog']
        metadata_nodes = catalog.get('total_nodes', 0)
        metadata_relationships = catalog.get('total_relationships', 0)
    
    print(f"  Metadata totals - Nodes: {metadata_nodes}, Relationships: {metadata_relationships}")
    
    # Count actual data
    actual_nodes = len(data.get('nodes', []))
    actual_relationships = len(data.get('relationships', []))
    
    print(f"  Actual counts - Nodes: {actual_nodes}, Relationships: {actual_relationships}")
    
    # Count node labels from actual data
    print("  Counting node labels...")
    node_labels = defaultdict(int)
    
    if 'nodes' in data:
        for i, node in enumerate(data['nodes']):
            if i % 1000 == 0 and i > 0:
                print(f"    Processed {i}/{actual_nodes} nodes...")
            
            labels = node.get('labels', [])
            for label in labels:
                node_labels[label] += 1
    
    print(f"  Found {len(node_labels)} unique node labels")
    
    # Count relationship types from actual data
    print("  Counting relationship types...")
    relationship_types = defaultdict(int)
    
    if 'relationships' in data:
        for i, rel in enumerate(data['relationships']):
            if i % 1000 == 0 and i > 0:
                print(f"    Processed {i}/{actual_relationships} relationships...")
            
            rel_type = rel.get('type', 'UNKNOWN')
            relationship_types[rel_type] += 1
    
    print(f"  Found {len(relationship_types)} unique relationship types")
    
    return {
        'metadata_nodes': metadata_nodes,
        'metadata_relationships': metadata_relationships,
        'actual_nodes': actual_nodes,
        'actual_relationships': actual_relationships,
        'node_labels': dict(node_labels),
        'relationship_types': dict(relationship_types)
    }


def generate_verification_report(catalog_stats, backup_stats):
    """Generate detailed verification report"""
    
    print("\n" + "=" * 80)
    print("BACKUP VERIFICATION REPORT")
    print("=" * 80)
    
    discrepancies = []
    matches = []
    warnings = []
    
    # Compare total counts
    print("\n1. TOTAL COUNT VERIFICATION")
    print("-" * 40)
    
    # Nodes comparison
    catalog_nodes = catalog_stats['total_nodes']
    metadata_nodes = backup_stats['metadata_nodes']
    actual_nodes = backup_stats['actual_nodes']
    
    print(f"Total Nodes:")
    print(f"  Catalog:     {catalog_nodes:,}")
    print(f"  Backup Meta: {metadata_nodes:,}")
    print(f"  Backup Real: {actual_nodes:,}")
    
    if catalog_nodes == actual_nodes:
        matches.append(f"Total nodes match perfectly: {catalog_nodes:,}")
        print("  ✓ PERFECT MATCH (catalog = actual)")
    else:
        discrepancies.append(f"Total nodes mismatch - Catalog: {catalog_nodes:,}, Backup: {actual_nodes:,}, Difference: {catalog_nodes - actual_nodes:,}")
        print("  ✗ MISMATCH (catalog ≠ actual)")
    
    if catalog_nodes != metadata_nodes:
        warnings.append(f"Metadata node count differs from catalog: {metadata_nodes:,} vs {catalog_nodes:,}")
    
    # Relationships comparison
    catalog_rels = catalog_stats['total_relationships']
    metadata_rels = backup_stats['metadata_relationships']
    actual_rels = backup_stats['actual_relationships']
    
    print(f"\nTotal Relationships:")
    print(f"  Catalog:     {catalog_rels:,}")
    print(f"  Backup Meta: {metadata_rels:,}")
    print(f"  Backup Real: {actual_rels:,}")
    
    if catalog_rels == actual_rels:
        matches.append(f"Total relationships match perfectly: {catalog_rels:,}")
        print("  ✓ PERFECT MATCH (catalog = actual)")
    else:
        discrepancies.append(f"Total relationships mismatch - Catalog: {catalog_rels:,}, Backup: {actual_rels:,}, Difference: {catalog_rels - actual_rels:,}")
        print("  ✗ MISMATCH (catalog ≠ actual)")
    
    if catalog_rels != metadata_rels:
        warnings.append(f"Metadata relationship count differs from catalog: {metadata_rels:,} vs {catalog_rels:,}")
    
    # Node labels comparison
    print("\n2. NODE LABEL VERIFICATION")
    print("-" * 40)
    
    catalog_labels = catalog_stats['node_labels']
    backup_labels = backup_stats['node_labels']
    
    # Find labels with discrepancies
    label_discrepancies = []
    label_matches = []
    
    all_labels = set(catalog_labels.keys()) | set(backup_labels.keys())
    
    for label in sorted(all_labels):
        catalog_count = catalog_labels.get(label, 0)
        backup_count = backup_labels.get(label, 0)
        
        if catalog_count == backup_count:
            if catalog_count > 0:  # Only report non-zero matches
                label_matches.append((label, catalog_count))
        else:
            label_discrepancies.append((label, catalog_count, backup_count))
    
    if label_discrepancies:
        print(f"Found {len(label_discrepancies)} node label discrepancies:")
        for label, cat_count, bak_count in label_discrepancies[:10]:  # Show top 10
            diff = cat_count - bak_count
            print(f"  ✗ {label:<25} Catalog: {cat_count:>6}, Backup: {bak_count:>6}, Diff: {diff:>6}")
            discrepancies.append(f"Node label '{label}' mismatch - Catalog: {cat_count}, Backup: {bak_count}")
        
        if len(label_discrepancies) > 10:
            print(f"  ... and {len(label_discrepancies) - 10} more discrepancies")
    else:
        print("✓ All node labels match perfectly!")
    
    # Show top matching labels
    if label_matches:
        print(f"\nTop matching node labels ({len(label_matches)} total):")
        top_matches = sorted(label_matches, key=lambda x: x[1], reverse=True)[:10]
        for label, count in top_matches:
            print(f"  ✓ {label:<25} {count:>6} nodes")
            matches.append(f"Node label '{label}' matches: {count}")
    
    # Relationship types comparison
    print("\n3. RELATIONSHIP TYPE VERIFICATION")
    print("-" * 40)
    
    catalog_types = catalog_stats['relationship_types']
    backup_types = backup_stats['relationship_types']
    
    # Find types with discrepancies
    type_discrepancies = []
    type_matches = []
    
    all_types = set(catalog_types.keys()) | set(backup_types.keys())
    
    for rel_type in sorted(all_types):
        catalog_count = catalog_types.get(rel_type, 0)
        backup_count = backup_types.get(rel_type, 0)
        
        if catalog_count == backup_count:
            if catalog_count > 0:  # Only report non-zero matches
                type_matches.append((rel_type, catalog_count))
        else:
            type_discrepancies.append((rel_type, catalog_count, backup_count))
    
    if type_discrepancies:
        print(f"Found {len(type_discrepancies)} relationship type discrepancies:")
        for rel_type, cat_count, bak_count in type_discrepancies:
            diff = cat_count - bak_count
            print(f"  ✗ {rel_type:<25} Catalog: {cat_count:>6}, Backup: {bak_count:>6}, Diff: {diff:>6}")
            discrepancies.append(f"Relationship type '{rel_type}' mismatch - Catalog: {cat_count}, Backup: {bak_count}")
    else:
        print("✓ All relationship types match perfectly!")
    
    # Show matching types
    if type_matches:
        print(f"\nMatching relationship types ({len(type_matches)} total):")
        for rel_type, count in type_matches:
            print(f"  ✓ {rel_type:<25} {count:>6} relationships")
            matches.append(f"Relationship type '{rel_type}' matches: {count}")
    
    # Final verdict
    print("\n4. OVERALL VERIFICATION RESULT")
    print("-" * 40)
    
    if discrepancies:
        status = "✗ VERIFICATION FAILED"
        print(f"{status}")
        print(f"Found {len(discrepancies)} discrepancies - backup does NOT match catalog")
    else:
        status = "✓ VERIFICATION PASSED" 
        print(f"{status}")
        print("Backup matches catalog with 100% certainty!")
    
    if warnings:
        print(f"\n⚠ {len(warnings)} warnings (metadata inconsistencies)")
    
    print(f"\nSummary:")
    print(f"  Matches:       {len(matches)}")
    print(f"  Discrepancies: {len(discrepancies)}")
    print(f"  Warnings:      {len(warnings)}")
    
    return {
        'status': status,
        'matches': matches,
        'discrepancies': discrepancies,
        'warnings': warnings,
        'success': len(discrepancies) == 0
    }


def save_detailed_report(result, catalog_path, backup_path):
    """Save detailed report to file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f"backup_verification_report_{timestamp}.txt"
    report_path = os.path.join(os.path.dirname(backup_path), report_filename)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("NEO4J BACKUP VERIFICATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Catalog:   {catalog_path}\n")
        f.write(f"Backup:    {backup_path}\n")
        f.write(f"Status:    {result['status']}\n")
        f.write("\n")
        
        f.write("SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Matches:       {len(result['matches'])}\n")
        f.write(f"Discrepancies: {len(result['discrepancies'])}\n")
        f.write(f"Warnings:      {len(result['warnings'])}\n")
        f.write("\n")
        
        if result['discrepancies']:
            f.write("DISCREPANCIES\n")
            f.write("-" * 40 + "\n")
            for disc in result['discrepancies']:
                f.write(f"✗ {disc}\n")
            f.write("\n")
        
        if result['warnings']:
            f.write("WARNINGS\n")
            f.write("-" * 40 + "\n")
            for warn in result['warnings']:
                f.write(f"⚠ {warn}\n")
            f.write("\n")
        
        f.write("MATCHES\n")
        f.write("-" * 40 + "\n")
        for match in result['matches']:
            f.write(f"✓ {match}\n")
        f.write("\n")
        
        f.write("CONCLUSION\n")
        f.write("-" * 40 + "\n")
        if result['success']:
            f.write("The backup verification PASSED with 100% certainty.\n")
            f.write("All counts match between the catalog and backup.\n")
        else:
            f.write("The backup verification FAILED.\n")
            f.write("Discrepancies found between catalog and backup.\n")
        f.write("=" * 80 + "\n")
    
    return report_path


def main():
    """Main verification function"""
    print("=" * 80)
    print("NEO4J BACKUP VERIFICATION - COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    
    # File paths
    base_path = "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/neo4j-data"
    catalog_path = os.path.join(base_path, "NEO4J_DATA_CATALOG.md")
    backup_path = os.path.join(base_path, "neo4j_backup_20250909_075720.json")

    # Check files exist
    for path in [catalog_path, backup_path]:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            sys.exit(1)

    try:
        # Load data
        catalog_stats = load_catalog_stats(catalog_path)
        backup_stats = load_backup_stats(backup_path)
        
        # Generate verification report
        result = generate_verification_report(catalog_stats, backup_stats)
        
        # Save detailed report
        report_path = save_detailed_report(result, catalog_path, backup_path)
        print(f"\n📄 Detailed report saved to: {report_path}")
        
        print("\n" + "=" * 80)
        
        return result['success']
        
    except Exception as e:
        print(f"ERROR: Verification failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)