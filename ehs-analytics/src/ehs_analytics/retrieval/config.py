"""
Configuration utilities for EHS Analytics retrieval system.

This module provides configuration helpers and validation for different
retrieval strategies and their dependencies.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, validator


class DatabaseType(str, Enum):
    """Supported database types."""
    NEO4J = "neo4j"
    AURA_DB = "aura_db"


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    ANTHROPIC = "anthropic"


class VectorStoreType(str, Enum):
    """Supported vector store types."""
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    NEO4J_VECTOR = "neo4j_vector"


@dataclass
class Neo4jConfig:
    """Configuration for Neo4j database connection."""
    
    uri: str
    username: str
    password: str
    database: str = "neo4j"
    max_connection_lifetime: int = 30 * 60  # 30 minutes
    max_connection_pool_size: int = 50
    connection_timeout: int = 30
    
    @classmethod
    def from_env(cls) -> 'Neo4jConfig':
        """Create Neo4jConfig from environment variables."""
        return cls(
            uri=os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
            username=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", ""),
            database=os.getenv("NEO4J_DATABASE", "neo4j")
        )


@dataclass
class LLMConfig:
    """Configuration for Language Model."""
    
    provider: LLMProvider
    model_name: str
    api_key: str
    temperature: float = 0.0
    max_tokens: int = 2000
    timeout: int = 60
    base_url: Optional[str] = None  # For Azure OpenAI or custom endpoints
    api_version: Optional[str] = None  # For Azure OpenAI
    
    @classmethod
    def openai_from_env(cls, model_name: str = "gpt-3.5-turbo") -> 'LLMConfig':
        """Create OpenAI LLMConfig from environment variables."""
        return cls(
            provider=LLMProvider.OPENAI,
            model_name=model_name,
            api_key=os.getenv("OPENAI_API_KEY", ""),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
        )


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    
    provider: VectorStoreType
    api_key: Optional[str] = None
    environment: Optional[str] = None
    index_name: Optional[str] = None
    dimension: int = 1536  # Default for OpenAI embeddings
    metric: str = "cosine"
    
    # Provider-specific settings
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def pinecone_from_env(cls) -> 'VectorStoreConfig':
        """Create Pinecone VectorStoreConfig from environment variables."""
        return cls(
            provider=VectorStoreType.PINECONE,
            api_key=os.getenv("PINECONE_API_KEY", ""),
            environment=os.getenv("PINECONE_ENVIRONMENT", ""),
            index_name=os.getenv("PINECONE_INDEX_NAME", "ehs-analytics")
        )


class RetrieverConfig(BaseModel):
    """
    Comprehensive configuration for retrieval strategies.
    
    This class provides validation and standardized configuration
    for different retrieval approaches.
    """
    
    # Database configuration
    neo4j: Dict[str, Any]
    
    # LLM configuration
    llm: Dict[str, Any]
    
    # Vector store configuration (optional)
    vector_store: Optional[Dict[str, Any]] = None
    
    # Strategy-specific settings
    text2cypher_settings: Dict[str, Any] = {}
    vector_settings: Dict[str, Any] = {}
    hybrid_settings: Dict[str, Any] = {}
    
    # General settings
    default_limit: int = 10
    query_timeout: int = 30
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutes
    
    # Logging and monitoring
    log_queries: bool = True
    log_level: str = "INFO"
    enable_metrics: bool = True
    
    @validator('neo4j')
    def validate_neo4j_config(cls, v):
        """Validate Neo4j configuration."""
        required_fields = ['uri', 'username', 'password']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required Neo4j field: {field}")
        return v
    
    @validator('llm')
    def validate_llm_config(cls, v):
        """Validate LLM configuration."""
        required_fields = ['provider', 'model_name', 'api_key']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Missing required LLM field: {field}")
        return v
    
    @classmethod
    def from_env(cls) -> 'RetrieverConfig':
        """Create RetrieverConfig from environment variables."""
        
        neo4j_config = Neo4jConfig.from_env()
        llm_config = LLMConfig.openai_from_env()
        
        config_dict = {
            "neo4j": {
                "uri": neo4j_config.uri,
                "username": neo4j_config.username,
                "password": neo4j_config.password,
                "database": neo4j_config.database,
                "max_connection_lifetime": neo4j_config.max_connection_lifetime,
                "max_connection_pool_size": neo4j_config.max_connection_pool_size,
                "connection_timeout": neo4j_config.connection_timeout
            },
            "llm": {
                "provider": llm_config.provider.value,
                "model_name": llm_config.model_name,
                "api_key": llm_config.api_key,
                "temperature": llm_config.temperature,
                "max_tokens": llm_config.max_tokens,
                "timeout": llm_config.timeout
            },
            "text2cypher_settings": {
                "cypher_validation": os.getenv("TEXT2CYPHER_VALIDATION", "true").lower() == "true",
                "include_schema_info": os.getenv("TEXT2CYPHER_INCLUDE_SCHEMA", "true").lower() == "true",
                "max_retries": int(os.getenv("TEXT2CYPHER_MAX_RETRIES", "3"))
            },
            "default_limit": int(os.getenv("DEFAULT_QUERY_LIMIT", "10")),
            "query_timeout": int(os.getenv("QUERY_TIMEOUT", "30")),
            "enable_caching": os.getenv("ENABLE_CACHING", "true").lower() == "true",
            "cache_ttl": int(os.getenv("CACHE_TTL", "300")),
            "log_queries": os.getenv("LOG_QUERIES", "true").lower() == "true",
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "enable_metrics": os.getenv("ENABLE_METRICS", "true").lower() == "true"
        }
        
        # Add vector store config if available
        if os.getenv("PINECONE_API_KEY"):
            vector_config = VectorStoreConfig.pinecone_from_env()
            config_dict["vector_store"] = {
                "provider": vector_config.provider.value,
                "api_key": vector_config.api_key,
                "environment": vector_config.environment,
                "index_name": vector_config.index_name,
                "dimension": vector_config.dimension,
                "metric": vector_config.metric
            }
        
        return cls(**config_dict)
    
    def get_text2cypher_config(self) -> Dict[str, Any]:
        """Get configuration specifically for Text2Cypher retriever."""
        base_config = {
            "neo4j_uri": self.neo4j["uri"],
            "neo4j_user": self.neo4j["username"],
            "neo4j_password": self.neo4j["password"],
            "neo4j_database": self.neo4j.get("database", "neo4j"),
            "openai_api_key": self.llm["api_key"],
            "model_name": self.llm["model_name"],
            "temperature": self.llm["temperature"],
            "max_tokens": self.llm["max_tokens"],
            "timeout": self.query_timeout
        }
        
        # Add Text2Cypher specific settings
        base_config.update(self.text2cypher_settings)
        
        return base_config
    
    def get_vector_config(self) -> Optional[Dict[str, Any]]:
        """Get configuration for vector retriever."""
        if not self.vector_store:
            return None
        
        base_config = {
            "llm_config": self.llm,
            "vector_store": self.vector_store,
            "timeout": self.query_timeout
        }
        
        base_config.update(self.vector_settings)
        return base_config