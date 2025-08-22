"""
Unit tests for QueryRouterAgent.

These tests cover intent classification, entity extraction, confidence scoring,
and error handling for the EHS query router.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ehs_analytics.agents.query_router import (
    QueryRouterAgent, QueryClassification, EntityExtraction,
    IntentType, RetrieverType
)
from langchain.schema import HumanMessage, SystemMessage, AIMessage


class TestQueryRouterAgent:
    """Test suite for QueryRouterAgent."""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing."""
        llm = Mock()
        # Mock response that simulates ChatOpenAI response
        mock_response = Mock()
        mock_response.content = '{"intent": "consumption_analysis", "confidence": 0.85, "reasoning": "Query focuses on energy usage patterns"}'
        llm.invoke.return_value = mock_response
        llm.model_name = "gpt-3.5-turbo"
        return llm

    @pytest.fixture
    def router_agent(self, mock_llm):
        """QueryRouterAgent instance with mocked LLM."""
        return QueryRouterAgent(llm=mock_llm, temperature=0.1, max_tokens=1000)

    @pytest.fixture
    def sample_queries(self):
        """Sample queries for testing different intent types."""
        return {
            "consumption_analysis": "What is the electricity consumption trend for Plant A over the last quarter?",
            "compliance_check": "Are we compliant with EPA air quality standards?",
            "risk_assessment": "What are the environmental risks at our chemical facility?",
            "emission_tracking": "Track our carbon footprint for Scope 1 emissions",
            "equipment_efficiency": "How efficient is our HVAC system performing?",
            "permit_status": "When do our environmental permits expire?",
            "general_inquiry": "What EHS data do we have available?"
        }

    def test_init_with_default_llm(self):
        """Test QueryRouterAgent initialization with default LLM."""
        with patch('ehs_analytics.agents.query_router.ChatOpenAI') as mock_openai:
            agent = QueryRouterAgent()
            mock_openai.assert_called_once_with(
                model="gpt-3.5-turbo",
                temperature=0.1,
                max_tokens=1000
            )
            assert agent.llm is not None

    def test_init_with_custom_llm(self, mock_llm):
        """Test QueryRouterAgent initialization with custom LLM."""
        agent = QueryRouterAgent(llm=mock_llm, temperature=0.2, max_tokens=500)
        assert agent.llm == mock_llm
        assert hasattr(agent, 'intent_patterns')
        assert hasattr(agent, 'entity_patterns')
        assert hasattr(agent, 'intent_retriever_map')

    def test_intent_patterns_coverage(self, router_agent):
        """Test that all intent types have patterns defined."""
        intent_types = {intent for intent in IntentType}
        pattern_intent_types = set(router_agent.intent_patterns.keys())
        
        # General inquiry might not have patterns (it's the fallback)
        expected_patterns = intent_types - {IntentType.GENERAL_INQUIRY}
        assert pattern_intent_types >= expected_patterns

    def test_entity_patterns_coverage(self, router_agent):
        """Test that all expected entity types have patterns defined."""
        expected_entities = {
            'facilities', 'date_ranges', 'equipment', 'pollutants',
            'regulations', 'departments', 'metrics'
        }
        actual_entities = set(router_agent.entity_patterns.keys())
        assert actual_entities == expected_entities

    def test_intent_retriever_mapping(self, router_agent):
        """Test that all intent types have retriever mappings."""
        for intent in IntentType:
            assert intent in router_agent.intent_retriever_map
            assert isinstance(router_agent.intent_retriever_map[intent], RetrieverType)

    @pytest.mark.asyncio
    async def test_classify_query_consumption_analysis(self, router_agent, sample_queries):
        """Test classification of consumption analysis queries."""
        query = sample_queries["consumption_analysis"]
        
        result = router_agent.classify_query(query)
        
        assert isinstance(result, QueryClassification)
        assert result.intent_type == IntentType.CONSUMPTION_ANALYSIS
        assert 0.0 <= result.confidence_score <= 1.0
        assert result.suggested_retriever == RetrieverType.CONSUMPTION_RETRIEVER
        assert result.reasoning is not None
        assert isinstance(result.entities_identified, EntityExtraction)

    @pytest.mark.asyncio
    async def test_classify_query_compliance_check(self, router_agent, sample_queries):
        """Test classification of compliance check queries."""
        query = sample_queries["compliance_check"]
        
        result = router_agent.classify_query(query)
        
        assert isinstance(result, QueryClassification)
        assert result.intent_type == IntentType.COMPLIANCE_CHECK
        assert result.suggested_retriever == RetrieverType.COMPLIANCE_RETRIEVER
        assert result.confidence_score > 0.0

    @pytest.mark.asyncio
    async def test_classify_query_risk_assessment(self, router_agent, sample_queries):
        """Test classification of risk assessment queries."""
        query = sample_queries["risk_assessment"]
        
        result = router_agent.classify_query(query)
        
        assert isinstance(result, QueryClassification)
        assert result.intent_type == IntentType.RISK_ASSESSMENT
        assert result.suggested_retriever == RetrieverType.RISK_RETRIEVER

    @pytest.mark.asyncio
    async def test_classify_query_emission_tracking(self, router_agent, sample_queries):
        """Test classification of emission tracking queries."""
        query = sample_queries["emission_tracking"]
        
        result = router_agent.classify_query(query)
        
        assert isinstance(result, QueryClassification)
        assert result.intent_type == IntentType.EMISSION_TRACKING
        assert result.suggested_retriever == RetrieverType.EMISSION_RETRIEVER

    @pytest.mark.asyncio
    async def test_classify_query_equipment_efficiency(self, router_agent, sample_queries):
        """Test classification of equipment efficiency queries."""
        query = sample_queries["equipment_efficiency"]
        
        result = router_agent.classify_query(query)
        
        assert isinstance(result, QueryClassification)
        assert result.intent_type == IntentType.EQUIPMENT_EFFICIENCY
        assert result.suggested_retriever == RetrieverType.EQUIPMENT_RETRIEVER

    @pytest.mark.asyncio
    async def test_classify_query_permit_status(self, router_agent, sample_queries):
        """Test classification of permit status queries."""
        query = sample_queries["permit_status"]
        
        result = router_agent.classify_query(query)
        
        assert isinstance(result, QueryClassification)
        assert result.intent_type == IntentType.PERMIT_STATUS
        assert result.suggested_retriever == RetrieverType.PERMIT_RETRIEVER

    @pytest.mark.asyncio
    async def test_classify_query_general_inquiry(self, router_agent, sample_queries):
        """Test classification of general inquiry queries."""
        query = sample_queries["general_inquiry"]
        
        result = router_agent.classify_query(query)
        
        assert isinstance(result, QueryClassification)
        assert result.intent_type == IntentType.GENERAL_INQUIRY
        assert result.suggested_retriever == RetrieverType.GENERAL_RETRIEVER

    def test_calculate_pattern_scores(self, router_agent):
        """Test pattern-based scoring calculation."""
        query = "What is the electricity consumption trend for Plant A?"
        
        scores = router_agent._calculate_pattern_scores(query)
        
        assert isinstance(scores, dict)
        assert len(scores) == len(IntentType) - 1  # Excluding GENERAL_INQUIRY
        
        # Consumption query should score high for consumption intent
        assert scores[IntentType.CONSUMPTION_ANALYSIS] > 0.0
        
        # All scores should be between 0 and 1
        for score in scores.values():
            assert 0.0 <= score <= 1.0

    def test_extract_entities_facilities(self, router_agent):
        """Test entity extraction for facilities."""
        query = "Show consumption for Plant A and Manufacturing Site B"
        
        entities = router_agent._extract_entities(query)
        
        assert isinstance(entities, EntityExtraction)
        assert len(entities.facilities) >= 1
        # Should find "Plant A" or similar facility names

    def test_extract_entities_date_ranges(self, router_agent):
        """Test entity extraction for date ranges."""
        query = "Show data for January 2024 and last month"
        
        entities = router_agent._extract_entities(query)
        
        assert isinstance(entities, EntityExtraction)
        # Should find "January 2024" and/or "last month"

    def test_extract_entities_equipment(self, router_agent):
        """Test entity extraction for equipment."""
        query = "Check the efficiency of boiler B-1 and HVAC system"
        
        entities = router_agent._extract_entities(query)
        
        assert isinstance(entities, EntityExtraction)
        # Should find "boiler" and/or "HVAC"

    def test_extract_entities_pollutants(self, router_agent):
        """Test entity extraction for pollutants."""
        query = "Track CO2 and NOx emissions from the facility"
        
        entities = router_agent._extract_entities(query)
        
        assert isinstance(entities, EntityExtraction)
        assert "CO2" in entities.pollutants or "NOx" in entities.pollutants

    def test_extract_entities_regulations(self, router_agent):
        """Test entity extraction for regulations."""
        query = "Check compliance with EPA and OSHA requirements"
        
        entities = router_agent._extract_entities(query)
        
        assert isinstance(entities, EntityExtraction)
        assert "EPA" in entities.regulations or "OSHA" in entities.regulations

    def test_extract_entities_metrics(self, router_agent):
        """Test entity extraction for metrics."""
        query = "What's the efficiency percentage and kWh consumption?"
        
        entities = router_agent._extract_entities(query)
        
        assert isinstance(entities, EntityExtraction)
        # Should find "efficiency" and/or "kWh"

    def test_llm_classify_success(self, router_agent):
        """Test successful LLM classification."""
        query = "What is electricity consumption?"
        pattern_scores = {IntentType.CONSUMPTION_ANALYSIS: 0.8}
        
        result = router_agent._llm_classify(query, pattern_scores)
        
        assert isinstance(result, dict)
        assert "intent" in result
        assert "confidence" in result
        assert "reasoning" in result
        assert isinstance(result["intent"], IntentType)

    def test_llm_classify_fallback_on_error(self, router_agent):
        """Test LLM classification fallback when LLM fails."""
        # Mock LLM to raise an exception
        router_agent.llm.invoke.side_effect = Exception("LLM API error")
        
        query = "What is electricity consumption?"
        pattern_scores = {
            IntentType.CONSUMPTION_ANALYSIS: 0.8,
            IntentType.COMPLIANCE_CHECK: 0.2
        }
        
        result = router_agent._llm_classify(query, pattern_scores)
        
        assert isinstance(result, dict)
        assert result["intent"] == IntentType.CONSUMPTION_ANALYSIS  # Highest pattern score
        assert result["confidence"] == 0.8
        assert "failed" in result["reasoning"].lower()

    def test_parse_llm_response_valid_json(self, router_agent):
        """Test parsing valid JSON LLM response."""
        response = '{"intent": "consumption_analysis", "confidence": 0.85, "reasoning": "Test"}'
        
        result = router_agent._parse_llm_response(response)
        
        assert result["intent"] == "consumption_analysis"
        assert result["confidence"] == 0.85
        assert result["reasoning"] == "Test"

    def test_parse_llm_response_invalid_json(self, router_agent):
        """Test parsing invalid JSON LLM response."""
        response = "This is not valid JSON"
        
        result = router_agent._parse_llm_response(response)
        
        assert result["intent"] == "general_inquiry"
        assert result["confidence"] == 0.5
        assert "fallback" in result["reasoning"].lower()

    def test_parse_llm_response_missing_fields(self, router_agent):
        """Test parsing LLM response with missing required fields."""
        response = '{"intent": "consumption_analysis"}'  # Missing confidence and reasoning
        
        result = router_agent._parse_llm_response(response)
        
        assert result["intent"] == "general_inquiry"
        assert result["confidence"] == 0.5

    def test_calculate_final_confidence(self, router_agent):
        """Test final confidence calculation combining pattern and LLM scores."""
        pattern_scores = {IntentType.CONSUMPTION_ANALYSIS: 0.6}
        llm_confidence = 0.8
        
        final_confidence = router_agent._calculate_final_confidence(pattern_scores, llm_confidence)
        
        # Should be weighted average: 0.3 * 0.6 + 0.7 * 0.8 = 0.74
        expected = 0.3 * 0.6 + 0.7 * 0.8
        assert abs(final_confidence - expected) < 0.01
        assert 0.0 <= final_confidence <= 1.0

    def test_rewrite_query_consumption_analysis(self, router_agent):
        """Test query rewriting for consumption analysis."""
        original_query = "Show me electricity usage"
        intent = IntentType.CONSUMPTION_ANALYSIS
        entities = EntityExtraction(
            facilities=["Plant A"],
            date_ranges=["last month"],
            equipment=[],
            pollutants=[],
            regulations=[],
            departments=[],
            metrics=["electricity"]
        )
        
        rewritten = router_agent._rewrite_query(original_query, intent, entities)
        
        assert rewritten is not None
        assert "consumption" in rewritten.lower()
        assert "plant a" in rewritten.lower()
        assert "last month" in rewritten.lower()

    def test_rewrite_query_no_entities(self, router_agent):
        """Test query rewriting with no entities."""
        original_query = "Show me data"
        intent = IntentType.GENERAL_INQUIRY
        entities = EntityExtraction([], [], [], [], [], [], [])
        
        rewritten = router_agent._rewrite_query(original_query, intent, entities)
        
        # Should return None when no meaningful rewrite can be done
        assert rewritten is None

    def test_classify_query_empty_query(self, router_agent):
        """Test classification of empty query raises ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            router_agent.classify_query("")

    def test_classify_query_whitespace_only(self, router_agent):
        """Test classification of whitespace-only query raises ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            router_agent.classify_query("   \n\t   ")

    def test_classify_query_with_user_id(self, router_agent, sample_queries):
        """Test query classification with user ID for logging context."""
        query = sample_queries["consumption_analysis"]
        user_id = "test_user_123"
        
        result = router_agent.classify_query(query, user_id=user_id)
        
        assert isinstance(result, QueryClassification)
        assert result.intent_type is not None

    @patch('ehs_analytics.agents.query_router.get_ehs_monitor')
    @patch('ehs_analytics.agents.query_router.logger')
    def test_classify_query_logging_and_monitoring(self, mock_logger, mock_monitor, router_agent, sample_queries):
        """Test that classification includes proper logging and monitoring."""
        mock_monitor_instance = Mock()
        mock_monitor.return_value = mock_monitor_instance
        
        query = sample_queries["consumption_analysis"]
        
        result = router_agent.classify_query(query)
        
        # Verify monitoring was called
        mock_monitor_instance.record_query.assert_called_once()
        
        # Verify logging occurred (check if logger methods were called)
        assert mock_logger.info.call_count > 0

    def test_get_intent_examples_coverage(self, router_agent):
        """Test that intent examples cover all intent types."""
        examples = router_agent.get_intent_examples()
        
        # Should have examples for all intent types
        assert len(examples) == len(IntentType)
        
        for intent_type in IntentType:
            assert intent_type in examples
            assert isinstance(examples[intent_type], list)
            assert len(examples[intent_type]) > 0
            
            # Each example should be a non-empty string
            for example in examples[intent_type]:
                assert isinstance(example, str)
                assert len(example.strip()) > 0

    def test_get_classification_stats(self, router_agent):
        """Test getting classification statistics."""
        stats = router_agent.get_classification_stats()
        
        assert isinstance(stats, dict)
        assert "total_classifications" in stats
        assert "intent_distribution" in stats
        assert "average_confidence" in stats
        assert "average_processing_time_ms" in stats
        
        # Intent distribution should include all intent types
        intent_dist = stats["intent_distribution"]
        for intent in IntentType:
            assert intent.value in intent_dist

    def test_confidence_score_bounds(self, router_agent, sample_queries):
        """Test that confidence scores are always within valid bounds."""
        for intent_name, query in sample_queries.items():
            result = router_agent.classify_query(query)
            assert 0.0 <= result.confidence_score <= 1.0

    def test_entity_extraction_all_empty_initially(self, router_agent):
        """Test that EntityExtraction initializes with empty lists."""
        entities = EntityExtraction([], [], [], [], [], [], [])
        
        assert entities.facilities == []
        assert entities.date_ranges == []
        assert entities.equipment == []
        assert entities.pollutants == []
        assert entities.regulations == []
        assert entities.departments == []
        assert entities.metrics == []

    @pytest.mark.parametrize("intent_type,expected_retriever", [
        (IntentType.CONSUMPTION_ANALYSIS, RetrieverType.CONSUMPTION_RETRIEVER),
        (IntentType.COMPLIANCE_CHECK, RetrieverType.COMPLIANCE_RETRIEVER),
        (IntentType.RISK_ASSESSMENT, RetrieverType.RISK_RETRIEVER),
        (IntentType.EMISSION_TRACKING, RetrieverType.EMISSION_RETRIEVER),
        (IntentType.EQUIPMENT_EFFICIENCY, RetrieverType.EQUIPMENT_RETRIEVER),
        (IntentType.PERMIT_STATUS, RetrieverType.PERMIT_RETRIEVER),
        (IntentType.GENERAL_INQUIRY, RetrieverType.GENERAL_RETRIEVER),
    ])
    def test_intent_to_retriever_mapping(self, router_agent, intent_type, expected_retriever):
        """Test that each intent type maps to the correct retriever."""
        assert router_agent.intent_retriever_map[intent_type] == expected_retriever

    def test_classification_with_special_characters(self, router_agent):
        """Test classification works with special characters and unicode."""
        query = "What's the CO₂ emission data for Plant-A (Building #1)?"
        
        result = router_agent.classify_query(query)
        
        assert isinstance(result, QueryClassification)
        assert result.intent_type is not None
        assert result.confidence_score >= 0.0

    def test_classification_performance(self, router_agent, sample_queries):
        """Test that classification completes within reasonable time."""
        import time
        
        query = sample_queries["consumption_analysis"]
        start_time = time.time()
        
        result = router_agent.classify_query(query)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within 10 seconds (generous for unit test)
        assert duration < 10.0
        assert isinstance(result, QueryClassification)

    def test_thread_safety_concurrent_classifications(self, router_agent, sample_queries):
        """Test that concurrent classifications don't interfere with each other."""
        import threading
        import concurrent.futures
        
        def classify_query_wrapper(query):
            return router_agent.classify_query(query)
        
        queries = list(sample_queries.values())
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(classify_query_wrapper, query) for query in queries[:3]]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All results should be valid QueryClassification objects
        assert len(results) == 3
        for result in results:
            assert isinstance(result, QueryClassification)
            assert result.intent_type is not None