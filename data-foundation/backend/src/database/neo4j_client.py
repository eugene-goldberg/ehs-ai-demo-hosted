#!/usr/bin/env python3
"""
Neo4j Database Client

This module provides a comprehensive Neo4j client for managing database connections,
executing queries, and handling database operations with proper error handling,
retry logic, and performance tracking.

Features:
- Connection management with retry logic and exponential backoff
- Session context managers for proper resource cleanup
- Query execution methods with performance tracking
- Error handling for Neo4j-specific errors
- Statistics and monitoring capabilities
- Batch operations support
- Health check functionality

Author: AI Assistant
Date: 2025-08-30
"""

import os
import sys
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Generator
from datetime import datetime
import json
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import traceback
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from neo4j import GraphDatabase, Driver, Session, Result
    from neo4j.exceptions import TransientError, ServiceUnavailable, DatabaseError, ClientError, AuthError
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure you're running this script in the virtual environment with required packages installed")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ConnectionConfig:
    """Configuration for Neo4j connection"""
    uri: str
    username: str
    password: str
    database: str = "neo4j"
    max_connection_lifetime: int = 30 * 60  # 30 minutes
    max_connection_pool_size: int = 50
    connection_acquisition_timeout: int = 60  # 60 seconds

    resolver: Optional[Any] = None
    user_agent: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'ConnectionConfig':
        """Create configuration from environment variables"""
        return cls(
            uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            username=os.getenv('NEO4J_USERNAME', 'neo4j'),
            password=os.getenv('NEO4J_PASSWORD', 'EhsAI2024!'),
            database=os.getenv('NEO4J_DATABASE', 'neo4j'),
            user_agent=os.getenv('NEO4J_USER_AGENT')
        )


@dataclass
class QueryStats:
    """Statistics for query execution"""
    query_count: int = 0
    total_execution_time: float = 0.0
    total_result_count: int = 0
    error_count: int = 0
    retry_count: int = 0
    
    @property
    def average_execution_time(self) -> float:
        """Calculate average query execution time"""
        return self.total_execution_time / max(1, self.query_count)
    
    @property
    def average_result_count(self) -> float:
        """Calculate average number of results per query"""
        return self.total_result_count / max(1, self.query_count)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting"""
        return {
            'query_count': self.query_count,
            'total_execution_time': self.total_execution_time,
            'average_execution_time': self.average_execution_time,
            'total_result_count': self.total_result_count,
            'average_result_count': self.average_result_count,
            'error_count': self.error_count,
            'retry_count': self.retry_count
        }


@dataclass
class HealthCheckResult:
    """Result of Neo4j health check"""
    is_healthy: bool
    connection_time: float
    server_info: Optional[Dict[str, Any]] = None
    database_info: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting"""
        return {
            'is_healthy': self.is_healthy,
            'connection_time': self.connection_time,
            'server_info': self.server_info,
            'database_info': self.database_info,
            'error_message': self.error_message,
            'timestamp': self.timestamp.isoformat(),
            'checked_at': self.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }


class Neo4jClient:
    """
    Comprehensive Neo4j database client with advanced features.
    
    This client provides:
    - Robust connection management with automatic retry
    - Session context managers for resource cleanup
    - Query execution with performance tracking
    - Error handling for all Neo4j exception types
    - Health monitoring and statistics
    - Batch operation support
    """
    
    def __init__(self, config: Optional[ConnectionConfig] = None, 
                 max_retries: int = 3, retry_delay: float = 1.0,
                 enable_logging: bool = True):
        """
        Initialize the Neo4j client.
        
        Args:
            config: Connection configuration (defaults to environment variables)
            max_retries: Maximum number of retry attempts for failed operations
            retry_delay: Base delay between retries (exponential backoff applied)
            enable_logging: Whether to enable detailed logging
        """
        self.config = config or ConnectionConfig.from_env()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_logging = enable_logging
        
        self.driver: Optional[Driver] = None
        self.connection_verified = False
        
        # Performance and statistics tracking
        self.stats = {
            'connections_created': 0,
            'connections_failed': 0,
            'sessions_created': 0,
            'sessions_failed': 0,
            'last_connection_time': None,
            'total_connection_time': 0.0
        }
        
        self.query_stats = QueryStats()
        
        # Setup logging if enabled
        if self.enable_logging and not logger.handlers:
            self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    def connect(self) -> bool:
        """
        Establish connection to Neo4j database with retry logic.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                # Create driver with configuration
                driver_args = {
                    'auth': (self.config.username, self.config.password),
                    'max_connection_lifetime': self.config.max_connection_lifetime,
                    'max_connection_pool_size': self.config.max_connection_pool_size,
                    'connection_acquisition_timeout': self.config.connection_acquisition_timeout,

                }
                
                # Add optional parameters if present
                if self.config.resolver:
                    driver_args['resolver'] = self.config.resolver
                if self.config.user_agent:
                    driver_args['user_agent'] = self.config.user_agent
                
                self.driver = GraphDatabase.driver(self.config.uri, **driver_args)
                
                # Verify connection with test query
                with self.driver.session(database=self.config.database) as session:
                    result = session.run("RETURN 1 as test")
                    test_value = result.single()["test"]
                    
                    if test_value == 1:
                        connection_time = time.time() - start_time
                        self.connection_verified = True
                        self.stats['connections_created'] += 1
                        self.stats['last_connection_time'] = connection_time
                        self.stats['total_connection_time'] += connection_time
                        
                        if self.enable_logging:
                            logger.info(
                                f"Successfully connected to Neo4j at {self.config.uri} "
                                f"(attempt {attempt + 1}, time: {connection_time:.3f}s)"
                            )
                        return True
                        
            except (AuthError, ClientError) as e:
                # Don't retry for authentication or client errors
                error_msg = f"Connection failed due to auth/client error: {e}"
                if self.enable_logging:
                    logger.error(error_msg)
                self.stats['connections_failed'] += 1
                return False
                
            except (TransientError, ServiceUnavailable, DatabaseError) as e:
                self.stats['connections_failed'] += 1
                if self.enable_logging:
                    logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    sleep_time = self.retry_delay * (2 ** attempt)
                    time.sleep(sleep_time)
                else:
                    if self.enable_logging:
                        logger.error(f"Failed to connect to Neo4j after {self.max_retries} attempts: {e}")
                    return False
                    
            except Exception as e:
                self.stats['connections_failed'] += 1
                error_msg = f"Unexpected connection error: {e}"
                if self.enable_logging:
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())
                return False
        
        return False
    
    def close(self):
        """Close the Neo4j connection and cleanup resources"""
        if self.driver:
            try:
                self.driver.close()
                if self.enable_logging:
                    logger.info("Neo4j connection closed")
            except Exception as e:
                if self.enable_logging:
                    logger.warning(f"Error closing driver: {e}")
            finally:
                self.driver = None
                self.connection_verified = False
    
    def __enter__(self):
        """Context manager entry"""
        if not self.connection_verified:
            if not self.connect():
                raise ConnectionError("Failed to establish Neo4j connection")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    @contextmanager
    def session_scope(self, access_mode: str = "WRITE", bookmarks: Optional[List] = None,
                     fetch_size: int = 1000) -> Generator[Session, None, None]:
        """
        Context manager for Neo4j sessions with proper resource cleanup.
        
        Args:
            access_mode: Session access mode ("READ" or "WRITE")
            bookmarks: Session bookmarks for consistency
            fetch_size: Result fetch size for memory efficiency
            
        Yields:
            Session: Neo4j session object
        """
        if not self.driver:
            raise ConnectionError("Neo4j driver not initialized. Call connect() first.")
        
        session = None
        try:
            session_args = {
                'database': self.config.database,
                'default_access_mode': access_mode,
                'fetch_size': fetch_size
            }
            
            if bookmarks:
                session_args['bookmarks'] = bookmarks
            
            session = self.driver.session(**session_args)
            self.stats['sessions_created'] += 1
            
            yield session
            
        except Exception as e:
            self.stats['sessions_failed'] += 1
            if self.enable_logging:
                logger.error(f"Session error: {e}")
            raise
        finally:
            if session:
                try:
                    session.close()
                except Exception as e:
                    if self.enable_logging:
                        logger.warning(f"Error closing session: {e}")
    
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None,
                     access_mode: str = "WRITE", fetch_all: bool = True,
                     timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Execute a Neo4j query with retry logic and performance tracking.
        
        Args:
            query: Cypher query string
            parameters: Query parameters dictionary
            access_mode: Session access mode ("READ" or "WRITE")
            fetch_all: Whether to fetch all results immediately
            timeout: Query timeout in seconds
            
        Returns:
            List of result records as dictionaries
        """
        if parameters is None:
            parameters = {}
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                with self.session_scope(access_mode=access_mode) as session:
                    # Execute query with optional timeout
                    if timeout:
                        result = session.run(query, parameters, timeout=timeout)
                    else:
                        result = session.run(query, parameters)
                    
                    # Fetch results
                    if fetch_all:
                        result_data = [record.data() for record in result]
                    else:
                        result_data = result
                    
                    # Track performance
                    execution_time = time.time() - start_time
                    self.query_stats.query_count += 1
                    self.query_stats.total_execution_time += execution_time
                    
                    if fetch_all:
                        self.query_stats.total_result_count += len(result_data)
                    
                    if self.enable_logging and execution_time > 1.0:  # Log slow queries
                        logger.warning(
                            f"Slow query executed in {execution_time:.3f}s: "
                            f"{query[:100]}{'...' if len(query) > 100 else ''}"
                        )
                    
                    return result_data
                    
            except (TransientError, ServiceUnavailable) as e:
                self.query_stats.retry_count += 1
                self.query_stats.error_count += 1
                
                if self.enable_logging:
                    logger.warning(f"Transient error on attempt {attempt + 1}: {e}")
                
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    time.sleep(sleep_time)
                else:
                    error_msg = f"Query failed after {self.max_retries} attempts: {e}"
                    if self.enable_logging:
                        logger.error(error_msg)
                    raise ConnectionError(error_msg)
                    
            except (ClientError, DatabaseError, AuthError) as e:
                # Don't retry for these types of errors
                self.query_stats.error_count += 1
                error_msg = f"Query execution error: {e}"
                if self.enable_logging:
                    logger.error(error_msg)
                    logger.error(f"Query: {query[:200]}{'...' if len(query) > 200 else ''}")
                raise
                
            except Exception as e:
                self.query_stats.error_count += 1
                error_msg = f"Unexpected query error: {e}"
                if self.enable_logging:
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())
                raise
        
        # This should never be reached
        raise RuntimeError("Query execution failed unexpectedly")
    
    def execute_read_query(self, query: str, parameters: Optional[Dict[str, Any]] = None,
                          timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Execute a read-only query.
        
        Args:
            query: Cypher query string
            parameters: Query parameters dictionary
            timeout: Query timeout in seconds
            
        Returns:
            List of result records as dictionaries
        """
        return self.execute_query(query, parameters, access_mode="READ", timeout=timeout)
    
    def execute_write_query(self, query: str, parameters: Optional[Dict[str, Any]] = None,
                           timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Execute a write query.
        
        Args:
            query: Cypher query string
            parameters: Query parameters dictionary
            timeout: Query timeout in seconds
            
        Returns:
            List of result records as dictionaries
        """
        return self.execute_query(query, parameters, access_mode="WRITE", timeout=timeout)
    
    def execute_batch_queries(self, queries: List[Tuple[str, Dict[str, Any]]], 
                             access_mode: str = "WRITE",
                             transaction: bool = True) -> List[List[Dict[str, Any]]]:
        """
        Execute multiple queries in batch, optionally within a transaction.
        
        Args:
            queries: List of (query, parameters) tuples
            access_mode: Session access mode ("READ" or "WRITE")
            transaction: Whether to execute in a single transaction
            
        Returns:
            List of result lists for each query
        """
        results = []
        
        if transaction:
            # Execute all queries in a single transaction
            with self.session_scope(access_mode=access_mode) as session:
                with session.begin_transaction() as tx:
                    for query, parameters in queries:
                        start_time = time.time()
                        
                        try:
                            result = tx.run(query, parameters)
                            result_data = [record.data() for record in result]
                            results.append(result_data)
                            
                            # Track performance
                            execution_time = time.time() - start_time
                            self.query_stats.query_count += 1
                            self.query_stats.total_execution_time += execution_time
                            self.query_stats.total_result_count += len(result_data)
                            
                        except Exception as e:
                            self.query_stats.error_count += 1
                            if self.enable_logging:
                                logger.error(f"Batch query failed: {e}")
                            raise
        else:
            # Execute queries individually
            for query, parameters in queries:
                result = self.execute_query(query, parameters, access_mode)
                results.append(result)
        
        return results
    
    def execute_procedure(self, procedure_name: str, parameters: Optional[Dict[str, Any]] = None,
                         yield_items: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Execute a Neo4j stored procedure.
        
        Args:
            procedure_name: Name of the procedure to call
            parameters: Procedure parameters
            yield_items: Specific items to yield from procedure results
            
        Returns:
            List of result records
        """
        if parameters is None:
            parameters = {}
        
        # Build CALL query
        param_str = ", ".join(f"${key}" for key in parameters.keys()) if parameters else ""
        
        if yield_items:
            yield_str = f" YIELD {', '.join(yield_items)}"
        else:
            yield_str = ""
        
        query = f"CALL {procedure_name}({param_str}){yield_str}"
        
        return self.execute_query(query, parameters, access_mode="READ")
    
    def run_cypher_script(self, script_path: Union[str, Path], 
                         parameters: Optional[Dict[str, Any]] = None) -> List[List[Dict[str, Any]]]:
        """
        Execute a Cypher script file containing multiple statements.
        
        Args:
            script_path: Path to the Cypher script file
            parameters: Parameters to use for all queries in the script
            
        Returns:
            List of results for each statement in the script
        """
        script_path = Path(script_path)
        
        if not script_path.exists():
            raise FileNotFoundError(f"Script file not found: {script_path}")
        
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        # Split script into individual statements
        statements = [stmt.strip() for stmt in script_content.split(';') if stmt.strip()]
        
        # Execute each statement
        results = []
        for i, statement in enumerate(statements):
            try:
                if self.enable_logging:
                    logger.info(f"Executing script statement {i + 1}/{len(statements)}")
                
                result = self.execute_query(statement, parameters)
                results.append(result)
                
            except Exception as e:
                if self.enable_logging:
                    logger.error(f"Error in script statement {i + 1}: {e}")
                raise RuntimeError(f"Script execution failed at statement {i + 1}: {e}")
        
        return results
    
    def health_check(self, include_server_info: bool = True, 
                    include_database_info: bool = True) -> HealthCheckResult:
        """
        Perform comprehensive health check of the Neo4j connection.
        
        Args:
            include_server_info: Whether to include server information
            include_database_info: Whether to include database information
            
        Returns:
            HealthCheckResult with detailed health information
        """
        start_time = time.time()
        
        try:
            # Basic connectivity test
            result = self.execute_read_query("RETURN 1 as test")
            
            if not result or result[0].get('test') != 1:
                return HealthCheckResult(
                    is_healthy=False,
                    connection_time=time.time() - start_time,
                    error_message="Basic connectivity test failed"
                )
            
            health_result = HealthCheckResult(
                is_healthy=True,
                connection_time=time.time() - start_time
            )
            
            # Collect server information
            if include_server_info:
                try:
                    server_info_result = self.execute_procedure("dbms.components")
                    health_result.server_info = {
                        'components': server_info_result,
                        'neo4j_version': None,
                        'edition': None
                    }
                    
                    # Extract version info
                    for component in server_info_result:
                        if component.get('name') == 'Neo4j Kernel':
                            health_result.server_info['neo4j_version'] = component.get('versions', [])
                            health_result.server_info['edition'] = component.get('edition')
                            break
                            
                except Exception as e:
                    if self.enable_logging:
                        logger.warning(f"Could not retrieve server info: {e}")
            
            # Collect database information
            if include_database_info:
                try:
                    db_info_query = """
                    CALL db.info() YIELD name, value
                    RETURN collect({name: name, value: value}) as info
                    """
                    db_info_result = self.execute_read_query(db_info_query)
                    
                    if db_info_result:
                        health_result.database_info = {
                            'info': db_info_result[0].get('info', []),
                            'database_name': self.config.database
                        }
                        
                except Exception as e:
                    if self.enable_logging:
                        logger.warning(f"Could not retrieve database info: {e}")
            
            return health_result
            
        except Exception as e:
            return HealthCheckResult(
                is_healthy=False,
                connection_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive client statistics.
        
        Returns:
            Dictionary containing connection and query statistics
        """
        return {
            'connection_stats': self.stats.copy(),
            'query_stats': self.query_stats.to_dict(),
            'configuration': {
                'uri': self.config.uri,
                'database': self.config.database,
                'max_retries': self.max_retries,
                'retry_delay': self.retry_delay,
                'connection_pool_size': self.config.max_connection_pool_size,
                'connection_timeout': self.config.connection_acquisition_timeout
            },
            'generated_at': datetime.now().isoformat()
        }
    
    def reset_statistics(self):
        """Reset all performance statistics"""
        self.stats = {
            'connections_created': 0,
            'connections_failed': 0,
            'sessions_created': 0,
            'sessions_failed': 0,
            'last_connection_time': None,
            'total_connection_time': 0.0
        }
        self.query_stats = QueryStats()
        
        if self.enable_logging:
            logger.info("Statistics reset")
    
    def test_connection(self) -> bool:
        """
        Simple connection test.
        
        Returns:
            bool: True if connection is working, False otherwise
        """
        try:
            result = self.execute_read_query("RETURN 1 as test", timeout=5.0)
            return len(result) == 1 and result[0].get('test') == 1
        except Exception:
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get basic database information and statistics.
        
        Returns:
            Dictionary with database information
        """
        try:
            # Get node and relationship counts
            count_query = """
            CALL apoc.meta.stats() YIELD nodeCount, relCount, labelCount, relTypeCount
            RETURN nodeCount, relCount, labelCount, relTypeCount
            """
            
            try:
                count_result = self.execute_read_query(count_query)
                if count_result:
                    counts = count_result[0]
                else:
                    # Fallback if APOC is not available
                    counts = self._get_basic_counts()
            except Exception:
                # Fallback if APOC is not available
                counts = self._get_basic_counts()
            
            # Get labels and relationship types
            labels_query = "CALL db.labels() YIELD label RETURN collect(label) as labels"
            rel_types_query = "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types"
            
            labels_result = self.execute_read_query(labels_query)
            rel_types_result = self.execute_read_query(rel_types_query)
            
            return {
                'database_name': self.config.database,
                'node_count': counts.get('nodeCount', 0),
                'relationship_count': counts.get('relCount', 0),
                'label_count': counts.get('labelCount', 0),
                'relationship_type_count': counts.get('relTypeCount', 0),
                'labels': labels_result[0].get('labels', []) if labels_result else [],
                'relationship_types': rel_types_result[0].get('types', []) if rel_types_result else [],
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Error getting database info: {e}")
            return {
                'error': str(e),
                'generated_at': datetime.now().isoformat()
            }
    
    def _get_basic_counts(self) -> Dict[str, int]:
        """Get basic node and relationship counts without APOC"""
        try:
            # Count nodes
            node_result = self.execute_read_query("MATCH (n) RETURN count(n) as nodeCount")
            node_count = node_result[0].get('nodeCount', 0) if node_result else 0
            
            # Count relationships
            rel_result = self.execute_read_query("MATCH ()-[r]->() RETURN count(r) as relCount")
            rel_count = rel_result[0].get('relCount', 0) if rel_result else 0
            
            # Count labels
            labels_result = self.execute_read_query("CALL db.labels() YIELD label RETURN count(label) as labelCount")
            label_count = labels_result[0].get('labelCount', 0) if labels_result else 0
            
            # Count relationship types
            rel_types_result = self.execute_read_query("CALL db.relationshipTypes() YIELD relationshipType RETURN count(relationshipType) as relTypeCount")
            rel_type_count = rel_types_result[0].get('relTypeCount', 0) if rel_types_result else 0
            
            return {
                'nodeCount': node_count,
                'relCount': rel_count,
                'labelCount': label_count,
                'relTypeCount': rel_type_count
            }
            
        except Exception as e:
            if self.enable_logging:
                logger.warning(f"Error getting basic counts: {e}")
            return {
                'nodeCount': 0,
                'relCount': 0,
                'labelCount': 0,
                'relTypeCount': 0
            }
    
    def create_constraint(self, constraint_query: str) -> bool:
        """
        Create a database constraint.
        
        Args:
            constraint_query: Cypher constraint creation query
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.execute_write_query(constraint_query)
            if self.enable_logging:
                logger.info(f"Constraint created: {constraint_query[:100]}...")
            return True
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Error creating constraint: {e}")
            return False
    
    def create_index(self, index_query: str) -> bool:
        """
        Create a database index.
        
        Args:
            index_query: Cypher index creation query
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.execute_write_query(index_query)
            if self.enable_logging:
                logger.info(f"Index created: {index_query[:100]}...")
            return True
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Error creating index: {e}")
            return False
    
    def drop_constraint(self, constraint_name: str) -> bool:
        """
        Drop a database constraint by name.
        
        Args:
            constraint_name: Name of the constraint to drop
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            query = f"DROP CONSTRAINT {constraint_name}"
            self.execute_write_query(query)
            if self.enable_logging:
                logger.info(f"Constraint dropped: {constraint_name}")
            return True
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Error dropping constraint: {e}")
            return False
    
    def drop_index(self, index_name: str) -> bool:
        """
        Drop a database index by name.
        
        Args:
            index_name: Name of the index to drop
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            query = f"DROP INDEX {index_name}"
            self.execute_write_query(query)
            if self.enable_logging:
                logger.info(f"Index dropped: {index_name}")
            return True
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Error dropping index: {e}")
            return False
    
    def list_constraints(self) -> List[Dict[str, Any]]:
        """
        List all database constraints.
        
        Returns:
            List of constraint information dictionaries
        """
        try:
            query = "SHOW CONSTRAINTS"
            return self.execute_read_query(query)
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Error listing constraints: {e}")
            return []
    
    def list_indexes(self) -> List[Dict[str, Any]]:
        """
        List all database indexes.
        
        Returns:
            List of index information dictionaries
        """
        try:
            query = "SHOW INDEXES"
            return self.execute_read_query(query)
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Error listing indexes: {e}")
            return []
    
    def clear_database(self, confirm: bool = False) -> bool:
        """
        Clear all data from the database (DANGEROUS OPERATION).
        
        Args:
            confirm: Must be True to actually perform the operation
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not confirm:
            if self.enable_logging:
                logger.warning("Database clear operation requires explicit confirmation")
            return False
        
        try:
            # Delete all relationships first
            self.execute_write_query("MATCH ()-[r]->() DELETE r")
            
            # Then delete all nodes
            self.execute_write_query("MATCH (n) DELETE n")
            
            if self.enable_logging:
                logger.warning("Database cleared - all nodes and relationships deleted")
            return True
            
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Error clearing database: {e}")
            return False
    
    def export_data(self, query: str, output_file: Union[str, Path], 
                   format: str = 'json', parameters: Optional[Dict[str, Any]] = None) -> bool:
        """
        Export query results to file.
        
        Args:
            query: Cypher query to execute
            output_file: Output file path
            format: Export format ('json', 'csv')
            parameters: Query parameters
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            results = self.execute_read_query(query, parameters)
            output_file = Path(output_file)
            
            if format.lower() == 'json':
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            elif format.lower() == 'csv':
                import pandas as pd
                df = pd.DataFrame(results)
                df.to_csv(output_file, index=False)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            if self.enable_logging:
                logger.info(f"Data exported to {output_file} ({len(results)} records)")
            return True
            
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Error exporting data: {e}")
            return False


# Factory function for easy client creation
def create_neo4j_client(uri: Optional[str] = None, username: Optional[str] = None,
                       password: Optional[str] = None, database: Optional[str] = None,
                       **kwargs) -> Neo4jClient:
    """
    Factory function to create a Neo4j client with optional parameter overrides.
    
    Args:
        uri: Neo4j URI (defaults to environment variable)
        username: Username (defaults to environment variable)
        password: Password (defaults to environment variable)
        database: Database name (defaults to environment variable)
        **kwargs: Additional arguments passed to Neo4jClient constructor
        
    Returns:
        Neo4jClient: Configured Neo4j client instance
    """
    config = ConnectionConfig.from_env()
    
    # Override with provided parameters
    if uri:
        config.uri = uri
    if username:
        config.username = username
    if password:
        config.password = password
    if database:
        config.database = database
    
    return Neo4jClient(config=config, **kwargs)


# Context manager for quick operations
@contextmanager
def neo4j_session(uri: Optional[str] = None, username: Optional[str] = None,
                 password: Optional[str] = None, database: Optional[str] = None):
    """
    Context manager for quick Neo4j operations.
    
    Example:
        with neo4j_session() as client:
            results = client.execute_read_query("MATCH (n) RETURN count(n)")
    """
    client = create_neo4j_client(uri, username, password, database)
    try:
        if not client.connect():
            raise ConnectionError("Failed to connect to Neo4j")
        yield client
    finally:
        client.close()


def main():
    """Main function for testing and demonstration"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Neo4j Client Testing and Operations')
    parser.add_argument('--health-check', action='store_true', help='Perform health check')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    parser.add_argument('--test-query', type=str, help='Execute a test query')
    parser.add_argument('--list-indexes', action='store_true', help='List all indexes')
    parser.add_argument('--list-constraints', action='store_true', help='List all constraints')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        with neo4j_session() as client:
            print(f"Connected to Neo4j at {client.config.uri}")
            
            if args.health_check:
                print("\nPerforming health check...")
                health = client.health_check()
                print(json.dumps(health.to_dict(), indent=2, default=str))
            
            if args.stats:
                print("\nDatabase information:")
                db_info = client.get_database_info()
                print(json.dumps(db_info, indent=2, default=str))
                
                print("\nClient statistics:")
                stats = client.get_statistics()
                print(json.dumps(stats, indent=2, default=str))
            
            if args.test_query:
                print(f"\nExecuting test query: {args.test_query}")
                results = client.execute_read_query(args.test_query)
                print(f"Results ({len(results)} records):")
                for i, result in enumerate(results[:5]):  # Show first 5 results
                    print(f"  {i + 1}: {result}")
                if len(results) > 5:
                    print(f"  ... and {len(results) - 5} more records")
            
            if args.list_indexes:
                print("\nDatabase indexes:")
                indexes = client.list_indexes()
                for idx in indexes:
                    print(f"  - {idx}")
            
            if args.list_constraints:
                print("\nDatabase constraints:")
                constraints = client.list_constraints()
                for constraint in constraints:
                    print(f"  - {constraint}")
                    
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()