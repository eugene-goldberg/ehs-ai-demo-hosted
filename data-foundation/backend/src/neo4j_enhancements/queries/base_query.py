"""
Base Query Class for Neo4j Operations

This module provides a base class for all Neo4j query operations,
handling connection management and common query patterns.

Created: 2025-08-28
Version: 1.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Union
from neo4j import GraphDatabase, Transaction, Session
import os

logger = logging.getLogger(__name__)


class BaseQuery:
    """
    Base class for all Neo4j query operations
    """
    
    def __init__(self, driver: GraphDatabase.driver):
        """
        Initialize the base query class
        
        Args:
            driver: Neo4j database driver
        """
        self.driver = driver
        self.database = os.getenv('NEO4J_DATABASE', 'neo4j')
    
    def execute_read(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a read query
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of query results as dictionaries
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
                
        except Exception as e:
            logger.error(f"Read query failed: {e}")
            raise
    
    def execute_write(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a write query
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of query results as dictionaries
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
                
        except Exception as e:
            logger.error(f"Write query failed: {e}")
            raise
    
    def execute_transaction(self, transaction_func, **kwargs) -> Any:
        """
        Execute a transaction
        
        Args:
            transaction_func: Function to execute in transaction
            **kwargs: Additional arguments
            
        Returns:
            Transaction result
        """
        try:
            with self.driver.session(database=self.database) as session:
                return session.execute_write(transaction_func, **kwargs)
                
        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            raise
    
    def health_check(self) -> bool:
        """
        Check database connection health
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                result.single()
                return True
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False