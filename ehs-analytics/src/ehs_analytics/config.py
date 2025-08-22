"""
Configuration settings for EHS Analytics

This module provides configuration management using Pydantic Settings
for environment variables and application configuration.
"""

import os
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    
    # Application settings
    app_name: str = Field(default="EHS Analytics", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    environment: str = Field(default="development", description="Environment (development, staging, production)")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # API settings
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_timeout: int = Field(default=300, description="API request timeout in seconds")
    max_request_size: int = Field(default=10 * 1024 * 1024, description="Maximum request size in bytes")
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:3001", 
            "http://localhost:8080"
        ],
        description="Allowed CORS origins"
    )
    cors_allow_credentials: bool = Field(default=True, description="Allow CORS credentials")
    
    # Security settings
    enable_auth: bool = Field(default=False, description="Enable authentication")
    jwt_secret_key: str = Field(default="dev-secret-key-change-in-production", description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration_hours: int = Field(default=24, description="JWT token expiration in hours")
    
    # Database settings - Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j database URI")
    neo4j_username: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="password", description="Neo4j password")
    neo4j_database: str = Field(default="neo4j", description="Neo4j database name")
    neo4j_max_pool_size: int = Field(default=50, description="Neo4j connection pool size")
    neo4j_connection_timeout: int = Field(default=30, description="Neo4j connection timeout in seconds")
    
    # OpenAI settings
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-3.5-turbo", description="OpenAI model to use")
    openai_temperature: float = Field(default=0.1, description="OpenAI temperature setting")
    openai_max_tokens: int = Field(default=1000, description="OpenAI max tokens")
    
    # LLM settings (for workflow components)
    llm_model_name: str = Field(default="gpt-3.5-turbo", description="LLM model name for workflow")
    llm_temperature: float = Field(default=0.0, description="LLM temperature for workflow")
    llm_max_tokens: int = Field(default=2000, description="LLM max tokens for workflow")
    
    # Cypher generation settings
    cypher_validation: bool = Field(default=True, description="Enable Cypher query validation")
    cypher_timeout: int = Field(default=30, description="Cypher query timeout in seconds")
    
    # Additional API keys that might be in .env but not used yet
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    llama_parse_api_key: str = Field(default="", description="LlamaParse API key")
    
    # Vector store settings (optional)
    vector_store_type: str = Field(default="none", description="Vector store type (pinecone, weaviate, qdrant, none)")
    pinecone_api_key: Optional[str] = Field(default=None, description="Pinecone API key")
    pinecone_environment: Optional[str] = Field(default=None, description="Pinecone environment")
    pinecone_index_name: Optional[str] = Field(default="ehs-analytics", description="Pinecone index name")
    
    # Weaviate settings
    weaviate_url: Optional[str] = Field(default=None, description="Weaviate URL")
    weaviate_api_key: Optional[str] = Field(default=None, description="Weaviate API key")
    
    # Qdrant settings
    qdrant_url: Optional[str] = Field(default=None, description="Qdrant URL")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant API key")
    qdrant_collection_name: str = Field(default="ehs_documents", description="Qdrant collection name")
    
    # Cache settings
    redis_url: Optional[str] = Field(default=None, description="Redis URL for caching")
    
    # Workflow settings
    workflow_timeout: int = Field(default=300, description="Workflow processing timeout in seconds")
    max_concurrent_queries: int = Field(default=100, description="Maximum concurrent queries")
    enable_caching: bool = Field(default=True, description="Enable result caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    
    # Monitoring settings
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(default=9090, description="Metrics server port")
    enable_tracing: bool = Field(default=False, description="Enable distributed tracing")
    
    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests_per_minute: int = Field(default=60, description="Requests per minute per IP")
    rate_limit_burst: int = Field(default=20, description="Rate limit burst size")
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v):
        """Validate environment value."""
        allowed_environments = ['development', 'staging', 'production']
        if v not in allowed_environments:
            raise ValueError(f'Environment must be one of: {allowed_environments}')
        return v
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        allowed_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in allowed_levels:
            raise ValueError(f'Log level must be one of: {allowed_levels}')
        return v.upper()
    
    @field_validator('vector_store_type')
    @classmethod
    def validate_vector_store_type(cls, v):
        """Validate vector store type."""
        allowed_types = ['pinecone', 'weaviate', 'qdrant', 'none']
        if v not in allowed_types:
            raise ValueError(f'Vector store type must be one of: {allowed_types}')
        return v
    
    @field_validator('openai_temperature', 'llm_temperature')
    @classmethod
    def validate_temperature(cls, v):
        """Validate temperature values."""
        if not 0.0 <= v <= 2.0:
            raise ValueError('Temperature must be between 0.0 and 2.0')
        return v
    
    @field_validator('cors_origins', mode='before')
    @classmethod
    def validate_cors_origins(cls, v):
        """Parse CORS origins from environment variable."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @property
    def database_url(self) -> str:
        """Get complete database URL."""
        return f"{self.neo4j_uri}"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"
    
    def get_openai_config(self) -> dict:
        """Get OpenAI configuration."""
        return {
            "api_key": self.openai_api_key,
            "model": self.openai_model,
            "temperature": self.openai_temperature,
            "max_tokens": self.openai_max_tokens
        }
    
    def get_llm_config(self) -> dict:
        """Get LLM configuration for workflow components."""
        return {
            "api_key": self.openai_api_key,  # Use same API key
            "model_name": self.llm_model_name,
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_tokens
        }
    
    def get_neo4j_config(self) -> dict:
        """Get Neo4j configuration."""
        return {
            "uri": self.neo4j_uri,
            "username": self.neo4j_username,
            "password": self.neo4j_password,
            "database": self.neo4j_database,
            "max_pool_size": self.neo4j_max_pool_size,
            "connection_timeout": self.neo4j_connection_timeout
        }
    
    def get_text2cypher_config(self) -> dict:
        """Get Text2Cypher retriever configuration."""
        return {
            "neo4j_uri": self.neo4j_uri,
            "neo4j_user": self.neo4j_username,
            "neo4j_password": self.neo4j_password,
            "openai_api_key": self.openai_api_key,
            "model_name": self.llm_model_name,
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_tokens,
            "cypher_validation": self.cypher_validation,
            "cypher_timeout": self.cypher_timeout
        }
    
    def get_vector_store_config(self) -> dict:
        """Get vector store configuration."""
        config = {"type": self.vector_store_type}
        
        if self.vector_store_type == "pinecone":
            config.update({
                "api_key": self.pinecone_api_key,
                "environment": self.pinecone_environment,
                "index_name": self.pinecone_index_name
            })
        elif self.vector_store_type == "weaviate":
            config.update({
                "url": self.weaviate_url,
                "api_key": self.weaviate_api_key
            })
        elif self.vector_store_type == "qdrant":
            config.update({
                "url": self.qdrant_url,
                "api_key": self.qdrant_api_key,
                "collection_name": self.qdrant_collection_name
            })
        
        return config
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        # Environment variable prefixes
        "env_prefix": "",
        # Allow extra fields to be ignored instead of raising errors
        "extra": "ignore"
    }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings


# Export commonly used configurations
def get_database_config() -> dict:
    """Get database configuration."""
    return settings.get_neo4j_config()


def get_api_config() -> dict:
    """Get API configuration."""
    return {
        "host": settings.api_host,
        "port": settings.api_port,
        "timeout": settings.api_timeout,
        "cors_origins": settings.cors_origins,
        "cors_allow_credentials": settings.cors_allow_credentials,
        "debug": settings.debug
    }


def get_security_config() -> dict:
    """Get security configuration."""
    return {
        "enable_auth": settings.enable_auth,
        "jwt_secret_key": settings.jwt_secret_key,
        "jwt_algorithm": settings.jwt_algorithm,
        "jwt_expiration_hours": settings.jwt_expiration_hours
    }


def get_workflow_config() -> dict:
    """Get workflow configuration."""
    return {
        "timeout": settings.workflow_timeout,
        "max_concurrent_queries": settings.max_concurrent_queries,
        "enable_caching": settings.enable_caching,
        "cache_ttl": settings.cache_ttl,
        "llm_config": settings.get_llm_config(),
        "text2cypher_config": settings.get_text2cypher_config()
    }