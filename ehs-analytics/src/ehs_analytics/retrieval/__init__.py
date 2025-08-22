"""
EHS Analytics Retrieval System.

This package provides retrieval strategies for the EHS Analytics platform,
including Text2Cypher, Vector, and Hybrid approaches for querying EHS data
from Neo4j graph database.
"""

from .base import (
    BaseRetriever,
    RetrievalStrategy,
    QueryType,
    RetrievalResult,
    RetrievalMetadata,
    EHSSchemaAware
)

from .config import (
    RetrieverConfig,
    Neo4jConfig,
    LLMConfig,
    VectorStoreConfig,
    DatabaseType,
    LLMProvider,
    VectorStoreType
)

from .strategies.text2cypher import Text2CypherRetriever

__all__ = [
    # Base classes and interfaces
    "BaseRetriever",
    "RetrievalStrategy", 
    "QueryType",
    "RetrievalResult",
    "RetrievalMetadata",
    "EHSSchemaAware",
    
    # Configuration classes
    "RetrieverConfig",
    "Neo4jConfig", 
    "LLMConfig",
    "VectorStoreConfig",
    "DatabaseType",
    "LLMProvider",
    "VectorStoreType",
    
    # Retriever implementations
    "Text2CypherRetriever"
]