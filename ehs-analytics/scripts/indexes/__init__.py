"""
EHS Analytics Index Creation Scripts

This package provides scripts for creating and managing Neo4j indexes
for the EHS Analytics system, including vector and fulltext indexes
for optimal search performance.

Scripts:
- create_vector_indexes.py: Creates vector indexes for semantic search
- create_fulltext_indexes.py: Creates fulltext indexes for text search  
- create_all_indexes.py: Orchestrates creation of all indexes
- populate_embeddings.py: Generates embeddings for existing documents
"""

__version__ = "1.0.0"
__author__ = "EHS Analytics Team"

from .create_vector_indexes import VectorIndexCreator
from .create_fulltext_indexes import FulltextIndexCreator

__all__ = [
    "VectorIndexCreator",
    "FulltextIndexCreator"
]