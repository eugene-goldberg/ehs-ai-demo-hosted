"""
Unit tests for Text2CypherRetriever.

These tests cover Cypher query generation, validation, error handling,
and Neo4j interaction mocking for the Text2Cypher retrieval strategy.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ehs_analytics.retrieval.strategies.text2cypher import Text2CypherRetriever
from ehs_analytics.retrieval.base import (
    RetrievalStrategy, QueryType, RetrievalResult, RetrievalMetadata
)
from neo4j.exceptions import Neo4jError, ServiceUnavailable, AuthError


class TestText2CypherRetriever:
    """Test suite for Text2CypherRetriever."""

    @pytest.fixture
    def basic_config(self):
        """Basic configuration for Text2CypherRetriever."""
        return {
            "neo4j_uri": "bolt://localhost:7687",
            "neo4j_user": "neo4j", 
            "neo4j_password": "test_password",
            "openai_api_key": "test_api_key",
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.0,
            "max_tokens": 2000,
            "cypher_validation": True
        }

    @pytest.fixture
    def mock_neo4j_driver(self):
        """Mock Neo4j driver."""
        driver = Mock()
        session = Mock()
        result = Mock()
        record = Mock()
        
        record.get.return_value = 1
        record.__getitem__ = Mock(return_value=1)
        result.single.return_value = record
        
        session.run.return_value = result
        session.__enter__.return_value = session
        session.__exit__.return_value = None
        
        driver.session.return_value = session
        driver.close = Mock()
        
        return driver

    @pytest.fixture
    def mock_neo4j_graph(self):
        """Mock LangChain Neo4j graph."""
        graph = Mock()
        graph.schema = {
            "node_props": {
                "Facility": {"name": "STRING", "location": "STRING"},
                "Equipment": {"name": "STRING", "type": "STRING"},
                "UtilityBill": {"amount": "FLOAT", "billing_period": "DATE"}
            },
            "rel_props": {},
            "relationships": [
                {"start": "Facility", "type": "CONTAINS", "end": "Equipment"},
                {"start": "UtilityBill", "type": "RECORDED_AT", "end": "Facility"}
            ]
        }
        graph.refresh_schema = Mock()
        return graph

    @pytest.fixture
    def mock_llm(self):
        """Mock OpenAI LLM."""
        llm = Mock()
        llm.model_name = "gpt-3.5-turbo"
        return llm

    @pytest.fixture
    def mock_cypher_chain(self):
        """Mock GraphCypherQAChain."""
        chain = Mock()
        chain.invoke = AsyncMock(return_value={
            "result": [
                {"facility_name": "Plant A", "total_consumption": 1500.0},
                {"facility_name": "Plant B", "total_consumption": 1200.0}
            ],
            "intermediate_steps": [{
                "query": "MATCH (f:Facility)-[:RECORDED_AT]-(u:UtilityBill) RETURN f.name as facility_name, SUM(u.amount) as total_consumption"
            }]
        })
        return chain

    @pytest.fixture
    async def text2cypher_retriever(self, basic_config, mock_neo4j_driver, mock_neo4j_graph, mock_llm, mock_cypher_chain):
        """Text2CypherRetriever instance with mocked dependencies."""
        with patch('ehs_analytics.retrieval.strategies.text2cypher.GraphDatabase.driver', return_value=mock_neo4j_driver), \
             patch('ehs_analytics.retrieval.strategies.text2cypher.Neo4jGraph', return_value=mock_neo4j_graph), \
             patch('ehs_analytics.retrieval.strategies.text2cypher.ChatOpenAI', return_value=mock_llm), \
             patch('ehs_analytics.retrieval.strategies.text2cypher.GraphCypherQAChain.from_llm', return_value=mock_cypher_chain):
            
            retriever = Text2CypherRetriever(basic_config)
            await retriever.initialize()
            return retriever

    def test_init_with_config(self, basic_config):
        """Test Text2CypherRetriever initialization with configuration."""
        retriever = Text2CypherRetriever(basic_config)
        
        assert retriever.neo4j_uri == basic_config["neo4j_uri"]
        assert retriever.neo4j_user == basic_config["neo4j_user"]
        assert retriever.neo4j_password == basic_config["neo4j_password"]
        assert retriever.openai_api_key == basic_config["openai_api_key"]
        assert retriever.model_name == basic_config["model_name"]
        assert retriever.temperature == basic_config["temperature"]
        assert retriever.max_tokens == basic_config["max_tokens"]
        assert retriever.cypher_validation == basic_config["cypher_validation"]

    def test_init_with_defaults(self):
        """Test Text2CypherRetriever initialization with default values."""
        config = {
            "neo4j_uri": "bolt://localhost:7687",
            "neo4j_user": "neo4j",
            "neo4j_password": "password",
            "openai_api_key": "api_key"
        }
        
        retriever = Text2CypherRetriever(config)
        
        assert retriever.model_name == "gpt-3.5-turbo"  # default
        assert retriever.temperature == 0.0  # default
        assert retriever.max_tokens == 2000  # default
        assert retriever.cypher_validation is True  # default

    @pytest.mark.asyncio
    async def test_initialize_success(self, basic_config, mock_neo4j_driver, mock_neo4j_graph, mock_llm, mock_cypher_chain):
        """Test successful initialization of Text2CypherRetriever."""
        with patch('ehs_analytics.retrieval.strategies.text2cypher.GraphDatabase.driver', return_value=mock_neo4j_driver), \
             patch('ehs_analytics.retrieval.strategies.text2cypher.Neo4jGraph', return_value=mock_neo4j_graph), \
             patch('ehs_analytics.retrieval.strategies.text2cypher.ChatOpenAI', return_value=mock_llm), \
             patch('ehs_analytics.retrieval.strategies.text2cypher.GraphCypherQAChain.from_llm', return_value=mock_cypher_chain):
            
            retriever = Text2CypherRetriever(basic_config)
            await retriever.initialize()
            
            assert retriever._initialized is True
            assert retriever.driver == mock_neo4j_driver
            assert retriever.graph == mock_neo4j_graph
            assert retriever.llm == mock_llm
            assert retriever.cypher_chain == mock_cypher_chain

    @pytest.mark.asyncio
    async def test_initialize_neo4j_connection_failure(self, basic_config):
        """Test initialization failure due to Neo4j connection issues."""
        with patch('ehs_analytics.retrieval.strategies.text2cypher.GraphDatabase.driver') as mock_driver:
            mock_driver.side_effect = ServiceUnavailable("Connection failed")
            
            retriever = Text2CypherRetriever(basic_config)
            
            with pytest.raises(ServiceUnavailable):
                await retriever.initialize()
            
            assert retriever._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_auth_failure(self, basic_config):
        """Test initialization failure due to Neo4j authentication issues."""
        with patch('ehs_analytics.retrieval.strategies.text2cypher.GraphDatabase.driver') as mock_driver:
            mock_driver.side_effect = AuthError("Authentication failed")
            
            retriever = Text2CypherRetriever(basic_config)
            
            with pytest.raises(AuthError):
                await retriever.initialize()

    @pytest.mark.asyncio
    async def test_retrieve_success(self, text2cypher_retriever):
        """Test successful query retrieval."""
        query = "Show electricity consumption for all facilities"
        
        result = await text2cypher_retriever.retrieve(
            query=query,
            query_type=QueryType.CONSUMPTION,
            limit=10
        )
        
        assert isinstance(result, RetrievalResult)
        assert result.success is True
        assert result.data is not None
        assert len(result.data) > 0
        assert result.metadata.strategy == RetrievalStrategy.TEXT2CYPHER
        assert result.metadata.query_type == QueryType.CONSUMPTION
        assert result.metadata.confidence_score > 0.0
        assert result.metadata.execution_time_ms > 0
        assert result.metadata.cypher_query is not None

    @pytest.mark.asyncio
    async def test_retrieve_different_query_types(self, text2cypher_retriever):
        """Test retrieval with different query types."""
        query_types = [
            QueryType.CONSUMPTION,
            QueryType.EFFICIENCY,
            QueryType.COMPLIANCE,
            QueryType.EMISSIONS,
            QueryType.RISK
        ]
        
        for query_type in query_types:
            result = await text2cypher_retriever.retrieve(
                query="Test query",
                query_type=query_type
            )
            
            assert isinstance(result, RetrievalResult)
            assert result.metadata.query_type == query_type

    @pytest.mark.asyncio
    async def test_retrieve_with_limit(self, text2cypher_retriever):
        """Test retrieval with result limit."""
        query = "Show all facilities"
        limit = 5
        
        result = await text2cypher_retriever.retrieve(
            query=query,
            query_type=QueryType.GENERAL,
            limit=limit
        )
        
        assert isinstance(result, RetrievalResult)
        assert result.success is True
        # The limit should be incorporated into the enhanced query

    @pytest.mark.asyncio
    async def test_retrieve_not_initialized(self, basic_config):
        """Test retrieval fails when retriever not initialized."""
        retriever = Text2CypherRetriever(basic_config)
        
        with pytest.raises(RuntimeError, match="not initialized"):
            await retriever.retrieve("test query")

    @pytest.mark.asyncio
    async def test_retrieve_empty_query(self, text2cypher_retriever):
        """Test retrieval with empty query."""
        result = await text2cypher_retriever.retrieve("")
        
        assert isinstance(result, RetrievalResult)
        assert result.success is False
        assert "Invalid" in result.message

    @pytest.mark.asyncio
    async def test_retrieve_neo4j_error(self, text2cypher_retriever):
        """Test retrieval handling Neo4j errors."""
        text2cypher_retriever.cypher_chain.invoke.side_effect = Neo4jError("Database error", "Neo.DatabaseError.General.UnknownError")
        
        result = await text2cypher_retriever.retrieve("test query")
        
        assert isinstance(result, RetrievalResult)
        assert result.success is False
        assert result.metadata.error_message is not None

    @pytest.mark.asyncio
    async def test_validate_query_success(self, text2cypher_retriever):
        """Test successful query validation."""
        ehs_queries = [
            "Show electricity consumption for all facilities",
            "Find equipment maintenance schedules",
            "List environmental permits expiring soon",
            "What is the water usage trend?",
            "How many safety incidents occurred?"
        ]
        
        for query in ehs_queries:
            is_valid = await text2cypher_retriever.validate_query(query)
            assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_query_failure(self, text2cypher_retriever):
        """Test query validation failure for non-EHS queries."""
        non_ehs_queries = [
            "What's the weather like?",
            "Tell me a joke",
            "How to cook pasta?",
            "",
            "   ",
            "a"  # Too short
        ]
        
        for query in non_ehs_queries:
            is_valid = await text2cypher_retriever.validate_query(query)
            assert is_valid is False

    def test_validate_query_input_valid(self, text2cypher_retriever):
        """Test basic query input validation."""
        valid_queries = [
            "test query",
            "A longer test query with more words",
            "Query with numbers 123 and symbols!"
        ]
        
        for query in valid_queries:
            assert text2cypher_retriever._validate_query_input(query) is True

    def test_validate_query_input_invalid(self, text2cypher_retriever):
        """Test basic query input validation with invalid inputs."""
        invalid_queries = [
            "",
            "   ",
            None,
            "ab"  # Too short (less than 3 chars)
        ]
        
        for query in invalid_queries:
            assert text2cypher_retriever._validate_query_input(query) is False

    def test_get_strategy(self, text2cypher_retriever):
        """Test retrieval strategy identifier."""
        assert text2cypher_retriever.get_strategy() == RetrievalStrategy.TEXT2CYPHER

    def test_build_ehs_cypher_prompt(self, text2cypher_retriever):
        """Test EHS-specific Cypher prompt building."""
        prompt = text2cypher_retriever._build_ehs_cypher_prompt()
        
        assert isinstance(prompt, str)
        assert len(prompt) > 100  # Should be substantial
        assert "EHS" in prompt
        assert "Cypher" in prompt
        assert "Neo4j" in prompt
        assert "Equipment" in prompt
        assert "Facility" in prompt
        assert "UtilityBill" in prompt

    def test_enhance_query_with_context_consumption(self, text2cypher_retriever):
        """Test query enhancement for consumption queries."""
        query = "Show electricity usage"
        query_type = QueryType.CONSUMPTION
        
        enhanced = text2cypher_retriever._enhance_query_with_context(query, query_type)
        
        assert len(enhanced) > len(query)
        assert "consumption" in enhanced.lower() or "usage" in enhanced.lower()

    def test_enhance_query_with_context_compliance(self, text2cypher_retriever):
        """Test query enhancement for compliance queries."""
        query = "Check regulations"
        query_type = QueryType.COMPLIANCE
        
        enhanced = text2cypher_retriever._enhance_query_with_context(query, query_type)
        
        assert len(enhanced) > len(query)
        assert "compliance" in enhanced.lower() or "permit" in enhanced.lower()

    def test_enhance_query_with_context_unknown_type(self, text2cypher_retriever):
        """Test query enhancement for unknown query type."""
        query = "Test query"
        query_type = QueryType.GENERAL
        
        enhanced = text2cypher_retriever._enhance_query_with_context(query, query_type)
        
        # Should return original query when no specific enhancement is available
        assert enhanced == query

    @pytest.mark.asyncio
    async def test_execute_cypher_chain_success(self, text2cypher_retriever):
        """Test successful Cypher chain execution."""
        query = "Enhanced test query"
        
        result = await text2cypher_retriever._execute_cypher_chain(query)
        
        assert isinstance(result, dict)
        assert "result" in result
        assert "intermediate_steps" in result

    @pytest.mark.asyncio
    async def test_execute_cypher_chain_neo4j_error(self, text2cypher_retriever):
        """Test Cypher chain execution with Neo4j error."""
        text2cypher_retriever.cypher_chain.invoke.side_effect = Neo4jError("Query failed", "Neo.ClientError.Statement.SyntaxError")
        
        with pytest.raises(Neo4jError):
            await text2cypher_retriever._execute_cypher_chain("test query")

    @pytest.mark.asyncio
    async def test_execute_cypher_chain_general_error(self, text2cypher_retriever):
        """Test Cypher chain execution with general error."""
        text2cypher_retriever.cypher_chain.invoke.side_effect = Exception("Unexpected error")
        
        with pytest.raises(Exception):
            await text2cypher_retriever._execute_cypher_chain("test query")

    def test_structure_results_list_input(self, text2cypher_retriever):
        """Test result structuring with list input."""
        raw_results = [
            {"facility_name": "Plant A", "consumption": 1500.0},
            {"facility_name": "Plant B", "consumption": 1200.0}
        ]
        
        structured = text2cypher_retriever._structure_results(raw_results, QueryType.CONSUMPTION)
        
        assert isinstance(structured, list)
        assert len(structured) == 2
        
        for result in structured:
            assert isinstance(result, dict)
            assert result["query_type"] == QueryType.CONSUMPTION.value

    def test_structure_results_dict_input(self, text2cypher_retriever):
        """Test result structuring with dictionary input."""
        raw_results = {"facility_name": "Plant A", "consumption": 1500.0}
        
        structured = text2cypher_retriever._structure_results(raw_results, QueryType.CONSUMPTION)
        
        assert isinstance(structured, list)
        assert len(structured) == 1
        assert structured[0]["query_type"] == QueryType.CONSUMPTION.value

    def test_structure_results_scalar_input(self, text2cypher_retriever):
        """Test result structuring with scalar input."""
        raw_results = 42
        
        structured = text2cypher_retriever._structure_results(raw_results, QueryType.GENERAL)
        
        assert isinstance(structured, list)
        assert len(structured) == 1
        assert structured[0]["result"] == 42
        assert structured[0]["type"] == "scalar"

    def test_structure_results_empty_input(self, text2cypher_retriever):
        """Test result structuring with empty input."""
        raw_results = None
        
        structured = text2cypher_retriever._structure_results(raw_results, QueryType.GENERAL)
        
        assert isinstance(structured, list)
        assert len(structured) == 0

    def test_structure_single_result_node(self, text2cypher_retriever):
        """Test structuring single result as Neo4j node."""
        # Mock Neo4j Node
        mock_node = Mock()
        mock_node._properties = {"name": "Plant A", "location": "City A"}
        mock_node.labels = ["Facility"]
        
        result = text2cypher_retriever._structure_single_result(mock_node, QueryType.GENERAL)
        
        assert isinstance(result, dict)
        assert result["name"] == "Plant A"
        assert result["location"] == "City A"
        assert result["_node_type"] == "Facility"
        assert result["query_type"] == QueryType.GENERAL.value

    def test_structure_single_result_relationship(self, text2cypher_retriever):
        """Test structuring single result as Neo4j relationship."""
        # Mock Neo4j Relationship
        mock_rel = Mock()
        mock_rel.type = "CONTAINS"
        mock_rel._properties = {"since": "2020-01-01"}
        mock_rel._start_node = Mock(_properties={"name": "Plant A"})
        mock_rel._end_node = Mock(_properties={"name": "Equipment 1"})
        
        result = text2cypher_retriever._structure_single_result(mock_rel, QueryType.GENERAL)
        
        assert isinstance(result, dict)
        assert result["relationship_type"] == "CONTAINS"
        assert result["properties"] == {"since": "2020-01-01"}
        assert result["start_node"] == {"name": "Plant A"}
        assert result["end_node"] == {"name": "Equipment 1"}

    def test_structure_single_result_dict(self, text2cypher_retriever):
        """Test structuring single result as dictionary."""
        item = {"facility": "Plant A", "value": 100}
        
        result = text2cypher_retriever._structure_single_result(item, QueryType.CONSUMPTION)
        
        assert result["facility"] == "Plant A"
        assert result["value"] == 100
        assert result["query_type"] == QueryType.CONSUMPTION.value

    def test_structure_single_result_other(self, text2cypher_retriever):
        """Test structuring single result as other type."""
        item = "simple string"
        
        result = text2cypher_retriever._structure_single_result(item, QueryType.GENERAL)
        
        assert result["value"] == "simple string"
        assert result["query_type"] == QueryType.GENERAL.value

    def test_calculate_confidence_score_high_quality(self, text2cypher_retriever):
        """Test confidence calculation for high-quality query and results."""
        cypher_query = "MATCH (f:Facility)-[:RECORDED_AT]-(u:UtilityBill) WHERE u.utility_type = 'electricity' RETURN f.name, SUM(u.amount) LIMIT 10"
        results = [{"facility": "Plant A", "consumption": 1500}] * 5  # 5 results
        
        confidence = text2cypher_retriever._calculate_confidence_score(cypher_query, results)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be high due to WHERE, SUM, LIMIT and results

    def test_calculate_confidence_score_low_quality(self, text2cypher_retriever):
        """Test confidence calculation for low-quality query and results."""
        cypher_query = "MATCH (f:Facility) RETURN f"  # Simple query
        results = []  # No results
        
        confidence = text2cypher_retriever._calculate_confidence_score(cypher_query, results)
        
        assert confidence == 0.0  # No results

    def test_calculate_confidence_score_empty_inputs(self, text2cypher_retriever):
        """Test confidence calculation with empty inputs."""
        confidence1 = text2cypher_retriever._calculate_confidence_score("", [])
        confidence2 = text2cypher_retriever._calculate_confidence_score(None, None)
        
        assert confidence1 == 0.0
        assert confidence2 == 0.0

    def test_count_relationships_with_relationships(self, text2cypher_retriever):
        """Test relationship counting with relationship results."""
        results = [
            {"relationship_type": "CONTAINS", "other": "data"},
            {"facility": "Plant A"},  # Not a relationship
            {"relationship_type": "RECORDED_AT", "other": "data"}
        ]
        
        count = text2cypher_retriever._count_relationships(results)
        
        assert count == 2

    def test_count_relationships_no_relationships(self, text2cypher_retriever):
        """Test relationship counting with no relationship results."""
        results = [
            {"facility": "Plant A"},
            {"equipment": "Boiler 1"},
            {"consumption": 1500}
        ]
        
        count = text2cypher_retriever._count_relationships(results)
        
        assert count == 0

    def test_count_relationships_empty_results(self, text2cypher_retriever):
        """Test relationship counting with empty results."""
        count = text2cypher_retriever._count_relationships([])
        assert count == 0

    @pytest.mark.asyncio
    async def test_cleanup(self, text2cypher_retriever):
        """Test cleanup of retriever resources."""
        # Verify driver exists
        assert text2cypher_retriever.driver is not None
        
        await text2cypher_retriever.cleanup()
        
        # Verify driver close was called
        text2cypher_retriever.driver.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_performance_within_bounds(self, text2cypher_retriever):
        """Test that retrieval completes within reasonable time bounds."""
        query = "Show facility consumption data"
        start_time = time.time()
        
        result = await text2cypher_retriever.retrieve(query)
        
        end_time = time.time()
        duration_seconds = end_time - start_time
        
        # Should complete within 5 seconds for unit test
        assert duration_seconds < 5.0
        assert result.metadata.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_multiple_concurrent_retrievals(self, text2cypher_retriever):
        """Test concurrent retrievals don't interfere with each other."""
        queries = [
            "Show electricity consumption",
            "List equipment maintenance",
            "Check permit status"
        ]
        
        # Run concurrent retrievals
        tasks = [text2cypher_retriever.retrieve(query) for query in queries]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, RetrievalResult)

    @pytest.mark.asyncio
    async def test_retrieve_with_retry_logic(self, text2cypher_retriever):
        """Test retrieval handles transient errors with retry."""
        # Make first call fail, second succeed
        text2cypher_retriever.cypher_chain.invoke.side_effect = [
            Neo4jError("Temporary failure", "Neo.TransientError.Network.UnknownHostException"),
            {
                "result": [{"test": "data"}],
                "intermediate_steps": [{"query": "MATCH (n) RETURN n"}]
            }
        ]
        
        # Note: Current implementation doesn't have retry logic, 
        # so this test documents expected behavior for future implementation
        result = await text2cypher_retriever.retrieve("test query")
        
        # Currently this will fail, but documents expected retry behavior
        assert isinstance(result, RetrievalResult)
        assert result.success is False  # Will be False until retry logic is added

    def test_schema_awareness_integration(self, text2cypher_retriever):
        """Test that retriever integrates with EHS schema awareness."""
        # Test that the prompt includes schema context
        prompt = text2cypher_retriever._build_ehs_cypher_prompt()
        
        # Should include schema information from mock graph
        assert "Facility" in prompt
        assert "Equipment" in prompt 
        assert "UtilityBill" in prompt