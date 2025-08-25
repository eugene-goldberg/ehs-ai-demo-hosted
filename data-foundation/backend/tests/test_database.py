"""
Test database configuration for EHS AI Platform tests.

Provides real Neo4j connections and fixtures for testing without mocks.
Uses credentials from .env file and supports data isolation and cleanup.
"""

import os
import sys
import json
import pytest
import logging
from typing import Dict, List, Any, Optional, Generator, Tuple
from datetime import datetime, date, timedelta
from pathlib import Path
import uuid

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from neo4j import GraphDatabase, Driver, Session, Transaction
from neo4j.graph import Node, Relationship
from neo4j.exceptions import ServiceUnavailable, AuthError

# Import project modules
from workflows.extraction_workflow import DataExtractionWorkflow, QueryType, ExtractionState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDatabaseError(Exception):
    """Custom exception for test database operations."""
    pass


class Neo4jTestClient:
    """
    Neo4j test client that provides real database connections with data isolation.
    """
    
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """
        Initialize Neo4j test client.
        
        Args:
            uri: Neo4j connection URI
            username: Neo4j username  
            password: Neo4j password
            database: Neo4j database name
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver: Optional[Driver] = None
        self.test_session_id = str(uuid.uuid4())
        self.test_labels = [f"TEST_{self.test_session_id}"]
        
    def connect(self) -> None:
        """Establish connection to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            # Test connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Neo4j at {self.uri}")
        except (ServiceUnavailable, AuthError) as e:
            raise TestDatabaseError(f"Failed to connect to Neo4j: {str(e)}")
    
    def disconnect(self) -> None:
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Disconnected from Neo4j")
    
    def get_session(self) -> Session:
        """Get a Neo4j session."""
        if not self.driver:
            raise TestDatabaseError("Not connected to Neo4j. Call connect() first.")
        return self.driver.session(database=self.database)
    
    def execute_query(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """
        Execute a Cypher query and return results.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result records as dictionaries
        """
        with self.get_session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    def create_test_node(self, labels: List[str], properties: Dict) -> str:
        """
        Create a test node with test session labels.
        
        Args:
            labels: Node labels
            properties: Node properties
            
        Returns:
            Created node ID
        """
        # Add test labels for isolation
        all_labels = labels + self.test_labels
        label_str = ":".join(all_labels)
        
        # Add test session ID to properties
        test_properties = {**properties, "test_session_id": self.test_session_id}
        
        query = f"""
        CREATE (n:{label_str} $properties)
        RETURN elementId(n) as node_id
        """
        
        result = self.execute_query(query, {"properties": test_properties})
        return result[0]["node_id"] if result else None
    
    def create_test_relationship(self, from_node_id: str, to_node_id: str, 
                               rel_type: str, properties: Optional[Dict] = None) -> str:
        """
        Create a test relationship between nodes.
        
        Args:
            from_node_id: Source node ID
            to_node_id: Target node ID
            rel_type: Relationship type
            properties: Relationship properties
            
        Returns:
            Created relationship ID
        """
        rel_properties = {**(properties or {}), "test_session_id": self.test_session_id}
        
        query = """
        MATCH (a), (b)
        WHERE elementId(a) = $from_id AND elementId(b) = $to_id
        CREATE (a)-[r:`{rel_type}` $properties]->(b)
        RETURN elementId(r) as rel_id
        """.format(rel_type=rel_type)
        
        result = self.execute_query(query, {
            "from_id": from_node_id,
            "to_id": to_node_id,
            "properties": rel_properties
        })
        return result[0]["rel_id"] if result else None
    
    def cleanup_test_data(self) -> int:
        """
        Clean up all test data for this session.
        
        Returns:
            Number of nodes/relationships deleted
        """
        # Delete relationships first
        rel_query = f"""
        MATCH ()-[r]-()
        WHERE r.test_session_id = $session_id
        DELETE r
        RETURN count(r) as deleted_relationships
        """
        
        rel_result = self.execute_query(rel_query, {"session_id": self.test_session_id})
        deleted_rels = rel_result[0]["deleted_relationships"] if rel_result else 0
        
        # Delete nodes
        node_query = f"""
        MATCH (n)
        WHERE n.test_session_id = $session_id
        DELETE n
        RETURN count(n) as deleted_nodes
        """
        
        node_result = self.execute_query(node_query, {"session_id": self.test_session_id})
        deleted_nodes = node_result[0]["deleted_nodes"] if node_result else 0
        
        total_deleted = deleted_nodes + deleted_rels
        if total_deleted > 0:
            logger.info(f"Cleaned up {deleted_nodes} test nodes and {deleted_rels} test relationships")
        
        return total_deleted


class TestDataFactory:
    """Factory for creating test data objects."""
    
    def __init__(self, neo4j_client: Neo4jTestClient):
        self.neo4j_client = neo4j_client
    
    def create_test_document(self, 
                           doc_type: str = "environmental_report",
                           content: Optional[str] = None,
                           metadata: Optional[Dict] = None) -> str:
        """
        Create a test Document node.
        
        Args:
            doc_type: Type of document
            content: Document content
            metadata: Additional metadata
            
        Returns:
            Document node ID
        """
        properties = {
            "type": doc_type,
            "content": content or f"Test {doc_type} content",
            "created_at": datetime.utcnow().isoformat(),
            "status": "processed",
            **(metadata or {})
        }
        
        return self.neo4j_client.create_test_node(["Document"], properties)
    
    def create_monthly_usage_allocation(self,
                                      facility_id: str,
                                      month: int,
                                      year: int,
                                      allocation_data: Dict) -> str:
        """
        Create a MonthlyUsageAllocation node for pro-rating tests.
        
        Args:
            facility_id: Facility identifier
            month: Month (1-12)
            year: Year
            allocation_data: Allocation percentages and data
            
        Returns:
            MonthlyUsageAllocation node ID
        """
        properties = {
            "facility_id": facility_id,
            "month": month,
            "year": year,
            "allocation_percentage": allocation_data.get("allocation_percentage", 100.0),
            "utility_type": allocation_data.get("utility_type", "electricity"),
            "consumption_kwh": allocation_data.get("consumption_kwh", 1000.0),
            "cost_usd": allocation_data.get("cost_usd", 120.0),
            "created_at": datetime.utcnow().isoformat()
        }
        
        return self.neo4j_client.create_test_node(["MonthlyUsageAllocation"], properties)
    
    def create_user_node(self,
                        user_id: str,
                        user_type: str = "analyst",
                        metadata: Optional[Dict] = None) -> str:
        """
        Create a User node for rejection tracking tests.
        
        Args:
            user_id: User identifier
            user_type: Type of user (analyst, admin, etc.)
            metadata: Additional user metadata
            
        Returns:
            User node ID
        """
        properties = {
            "user_id": user_id,
            "user_type": user_type,
            "created_at": datetime.utcnow().isoformat(),
            "status": "active",
            **(metadata or {})
        }
        
        return self.neo4j_client.create_test_node(["User"], properties)
    
    def create_facility_node(self,
                           facility_id: str,
                           facility_name: Optional[str] = None,
                           location: Optional[str] = None) -> str:
        """
        Create a Facility node for testing.
        
        Args:
            facility_id: Facility identifier
            facility_name: Human-readable facility name
            location: Facility location
            
        Returns:
            Facility node ID
        """
        properties = {
            "facility_id": facility_id,
            "facility_name": facility_name or f"Test Facility {facility_id}",
            "location": location or "Test Location",
            "created_at": datetime.utcnow().isoformat(),
            "status": "active"
        }
        
        return self.neo4j_client.create_test_node(["Facility"], properties)
    
    def create_emissions_record(self,
                              facility_node_id: str,
                              emission_type: str,
                              amount: float,
                              unit: str = "tons_co2e",
                              date: Optional[date] = None) -> str:
        """
        Create an emissions record linked to a facility.
        
        Args:
            facility_node_id: Facility node ID to link to
            emission_type: Type of emission (scope1, scope2, etc.)
            amount: Emission amount
            unit: Unit of measurement
            date: Date of emission record
            
        Returns:
            Emissions record node ID
        """
        record_date = date or datetime.utcnow().date()
        
        properties = {
            "emission_type": emission_type,
            "amount": amount,
            "unit": unit,
            "date": record_date.isoformat(),
            "created_at": datetime.utcnow().isoformat()
        }
        
        emissions_node_id = self.neo4j_client.create_test_node(["EmissionsRecord"], properties)
        
        # Create relationship to facility
        self.neo4j_client.create_test_relationship(
            facility_node_id, 
            emissions_node_id, 
            "HAS_EMISSION",
            {"created_at": datetime.utcnow().isoformat()}
        )
        
        return emissions_node_id


# Test configuration and fixtures
@pytest.fixture(scope="session")
def test_db_config():
    """Load test database configuration from environment."""
    config = {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "username": os.getenv("NEO4J_USERNAME", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", "EhsAI2024!"),
        "database": os.getenv("NEO4J_DATABASE", "neo4j")
    }
    
    # Verify all required config is present
    missing_config = [k for k, v in config.items() if not v]
    if missing_config:
        pytest.skip(f"Missing Neo4j configuration: {missing_config}")
    
    return config


@pytest.fixture(scope="session")
def neo4j_driver(test_db_config):
    """Create a Neo4j driver for the test session."""
    driver = GraphDatabase.driver(
        test_db_config["uri"],
        auth=(test_db_config["username"], test_db_config["password"])
    )
    
    # Test connection
    try:
        with driver.session(database=test_db_config["database"]) as session:
            session.run("RETURN 1")
    except Exception as e:
        pytest.skip(f"Cannot connect to Neo4j: {str(e)}")
    
    yield driver
    driver.close()


@pytest.fixture
def neo4j_test_client(test_db_config):
    """Create a Neo4j test client with data isolation."""
    client = Neo4jTestClient(
        uri=test_db_config["uri"],
        username=test_db_config["username"],
        password=test_db_config["password"],
        database=test_db_config["database"]
    )
    
    client.connect()
    yield client
    
    # Cleanup test data
    client.cleanup_test_data()
    client.disconnect()


@pytest.fixture
def test_data_factory(neo4j_test_client):
    """Create a test data factory."""
    return TestDataFactory(neo4j_test_client)


@pytest.fixture
def real_extraction_workflow(test_db_config):
    """Create a real DataExtractionWorkflow instance for testing."""
    workflow = DataExtractionWorkflow(
        neo4j_uri=test_db_config["uri"],
        neo4j_username=test_db_config["username"],
        neo4j_password=test_db_config["password"],
        llm_model="gpt-4",
        output_dir="./test_reports"
    )
    
    yield workflow
    
    # Cleanup
    workflow.driver.close()


@pytest.fixture
def sample_facility_with_data(test_data_factory):
    """Create a sample facility with associated data for testing."""
    # Create facility
    facility_id = test_data_factory.create_facility_node(
        facility_id="TEST_FAC_001",
        facility_name="Test Manufacturing Plant",
        location="Test City, USA"
    )
    
    # Create monthly usage allocations
    allocations = []
    for month in range(1, 13):  # Full year
        allocation_id = test_data_factory.create_monthly_usage_allocation(
            facility_id="TEST_FAC_001",
            month=month,
            year=2024,
            allocation_data={
                "allocation_percentage": 85.0,
                "utility_type": "electricity",
                "consumption_kwh": 1000 + (month * 50),
                "cost_usd": 120 + (month * 6)
            }
        )
        allocations.append(allocation_id)
    
    # Create emissions records
    emissions = []
    emission_types = ["scope1", "scope2", "scope3"]
    for i, emission_type in enumerate(emission_types):
        emission_id = test_data_factory.create_emissions_record(
            facility_node_id=facility_id,
            emission_type=emission_type,
            amount=100.0 + (i * 50),
            unit="tons_co2e",
            date=date(2024, 6, 15)
        )
        emissions.append(emission_id)
    
    # Create associated documents
    documents = []
    doc_types = ["environmental_report", "utility_bill", "emissions_inventory"]
    for doc_type in doc_types:
        doc_id = test_data_factory.create_test_document(
            doc_type=doc_type,
            content=f"Test {doc_type} for facility TEST_FAC_001",
            metadata={"facility_id": "TEST_FAC_001"}
        )
        documents.append(doc_id)
    
    return {
        "facility_id": facility_id,
        "allocations": allocations,
        "emissions": emissions,
        "documents": documents
    }


@pytest.fixture
def sample_users(test_data_factory):
    """Create sample users for rejection tracking tests."""
    users = []
    user_types = ["analyst", "admin", "reviewer"]
    
    for i, user_type in enumerate(user_types):
        user_id = test_data_factory.create_user_node(
            user_id=f"test_user_{i+1}",
            user_type=user_type,
            metadata={
                "name": f"Test {user_type.title()} {i+1}",
                "email": f"test_{user_type}_{i+1}@example.com",
                "permissions": ["read", "write"] if user_type != "reviewer" else ["read"]
            }
        )
        users.append(user_id)
    
    return users


# Helper functions for test assertions
def assert_node_exists(neo4j_client: Neo4jTestClient, node_id: str) -> bool:
    """Assert that a node exists in the database."""
    query = "MATCH (n) WHERE elementId(n) = $node_id RETURN count(n) as count"
    result = neo4j_client.execute_query(query, {"node_id": node_id})
    return result[0]["count"] > 0 if result else False


def assert_relationship_exists(neo4j_client: Neo4jTestClient, 
                             from_id: str, to_id: str, rel_type: str) -> bool:
    """Assert that a relationship exists between two nodes."""
    query = f"""
    MATCH (a)-[r:`{rel_type}`]->(b)
    WHERE elementId(a) = $from_id AND elementId(b) = $to_id
    RETURN count(r) as count
    """
    result = neo4j_client.execute_query(query, {"from_id": from_id, "to_id": to_id})
    return result[0]["count"] > 0 if result else False


def get_node_properties(neo4j_client: Neo4jTestClient, node_id: str) -> Optional[Dict]:
    """Get properties of a node by ID."""
    query = "MATCH (n) WHERE elementId(n) = $node_id RETURN properties(n) as props"
    result = neo4j_client.execute_query(query, {"node_id": node_id})
    return result[0]["props"] if result else None


# Test database connection and basic operations
def test_neo4j_connection(neo4j_test_client):
    """Test that Neo4j connection works."""
    result = neo4j_test_client.execute_query("RETURN 'connection_test' as test")
    assert result[0]["test"] == "connection_test"


def test_data_isolation(neo4j_test_client):
    """Test that test data is properly isolated."""
    # Create a test node
    node_id = neo4j_test_client.create_test_node(
        labels=["TestNode"], 
        properties={"name": "isolation_test"}
    )
    
    # Verify it has test session ID
    props = get_node_properties(neo4j_test_client, node_id)
    assert props["test_session_id"] == neo4j_test_client.test_session_id
    assert props["name"] == "isolation_test"


def test_data_cleanup(neo4j_test_client):
    """Test that test data cleanup works."""
    # Create test nodes and relationships
    node1_id = neo4j_test_client.create_test_node(["TestNode"], {"name": "node1"})
    node2_id = neo4j_test_client.create_test_node(["TestNode"], {"name": "node2"})
    rel_id = neo4j_test_client.create_test_relationship(node1_id, node2_id, "CONNECTS_TO")
    
    # Verify they exist
    assert assert_node_exists(neo4j_test_client, node1_id)
    assert assert_node_exists(neo4j_test_client, node2_id)
    assert assert_relationship_exists(neo4j_test_client, node1_id, node2_id, "CONNECTS_TO")
    
    # Cleanup and verify deletion
    deleted_count = neo4j_test_client.cleanup_test_data()
    assert deleted_count >= 3  # At least 2 nodes + 1 relationship


if __name__ == "__main__":
    # Run basic connection test
    from dotenv import load_dotenv
    load_dotenv()
    
    config = {
        "uri": os.getenv("NEO4J_URI"),
        "username": os.getenv("NEO4J_USERNAME"), 
        "password": os.getenv("NEO4J_PASSWORD"),
        "database": os.getenv("NEO4J_DATABASE", "neo4j")
    }
    
    client = Neo4jTestClient(**config)
    try:
        client.connect()
        print("✓ Neo4j connection successful")
        
        # Test basic operations
        result = client.execute_query("RETURN 'test' as value")
        print(f"✓ Query execution successful: {result}")
        
        # Test data factory
        factory = TestDataFactory(client)
        doc_id = factory.create_test_document()
        print(f"✓ Test document created: {doc_id}")
        
        # Cleanup
        deleted = client.cleanup_test_data()
        print(f"✓ Cleanup completed: {deleted} items deleted")
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
    finally:
        client.disconnect()