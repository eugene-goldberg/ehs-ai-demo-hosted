"""
Neo4j Schema Indexes Package

This package provides comprehensive index definitions and management utilities
for the Neo4j EHS AI platform, ensuring optimal query performance across all
enhanced node types and query patterns.

Classes:
    IndexDefinition: Definition of a Neo4j index with metadata
    ConstraintDefinition: Definition of a Neo4j constraint
    EHSIndexDefinitions: Central registry of all EHS platform indexes
    IndexManager: Utility for creating and managing indexes

Functions:
    get_high_priority_indexes: Get only high priority indexes
    get_indexes_by_node_type: Get indexes for specific node types
    generate_index_creation_script: Generate complete Cypher creation script
"""

from .index_definitions import (
    IndexType,
    IndexStatus,
    IndexDefinition,
    ConstraintDefinition,
    EHSIndexDefinitions,
    IndexManager,
    get_high_priority_indexes,
    get_indexes_by_node_type,
    generate_index_creation_script
)

__all__ = [
    "IndexType",
    "IndexStatus",
    "IndexDefinition", 
    "ConstraintDefinition",
    "EHSIndexDefinitions",
    "IndexManager",
    "get_high_priority_indexes",
    "get_indexes_by_node_type",
    "generate_index_creation_script"
]