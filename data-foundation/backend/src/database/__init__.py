"""
Database module for EHS AI Demo Data Foundation

This module provides database clients and utilities for the EHS AI system.
"""

from .neo4j_client import (
    Neo4jClient,
    ConnectionConfig,
    QueryStats,
    HealthCheckResult,
    create_neo4j_client,
    neo4j_session
)

__all__ = [
    'Neo4jClient',
    'ConnectionConfig', 
    'QueryStats',
    'HealthCheckResult',
    'create_neo4j_client',
    'neo4j_session'
]