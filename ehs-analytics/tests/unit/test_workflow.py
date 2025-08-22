"""
Unit tests for EHSWorkflow.

These tests cover workflow state management, node execution, routing,
error handling, and external dependency mocking.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ehs_analytics.workflows.ehs_workflow import (
    EHSWorkflow, EHSWorkflowState, create_ehs_workflow
)
from ehs_analytics.agents.query_router import (
    QueryRouterAgent, QueryClassification, EntityExtraction, IntentType, RetrieverType
)
from ehs_analytics.api.dependencies import DatabaseManager


class TestEHSWorkflowState:
    """Test suite for EHSWorkflowState."""

    @pytest.fixture
    def sample_state(self):
        """Sample workflow state for testing."""
        return EHSWorkflowState(
            query_id="test-query-123",
            original_query="What is the electricity consumption trend?",
            user_id="user-456"
        )

    def test_init_state(self, sample_state):
        """Test workflow state initialization."""
        assert sample_state.query_id == "test-query-123"
        assert sample_state.original_query == "What is the electricity consumption trend?"
        assert sample_state.user_id == "user-456"
        assert sample_state.classification is None
        assert sample_state.retrieval_results is None
        assert sample_state.analysis_results is None
        assert sample_state.recommendations is None
        assert sample_state.error is None
        assert sample_state.metadata == {}
        assert sample_state.workflow_trace == []
        assert isinstance(sample_state.created_at, datetime)
        assert isinstance(sample_state.updated_at, datetime)
        assert sample_state.step_durations == {}
        assert sample_state.total_duration_ms is None

    def test_update_state(self, sample_state):
        """Test state update functionality."""
        original_updated_at = sample_state.updated_at
        
        # Wait a bit to ensure timestamp changes
        import time
        time.sleep(0.001)
        
        sample_state.update_state(
            error="Test error",
            metadata={"test": "value"}
        )
        
        assert sample_state.error == "Test error"
        assert sample_state.metadata == {"test": "value"}
        assert sample_state.updated_at > original_updated_at

    def test_add_trace(self, sample_state):
        """Test adding trace messages to workflow."""
        sample_state.add_trace("Starting classification")
        sample_state.add_trace("Classification completed", "CLASSIFY")
        
        assert len(sample_state.workflow_trace) == 2
        assert "Starting classification" in sample_state.workflow_trace[0]
        assert "[CLASSIFY]" in sample_state.workflow_trace[1]
        assert "Classification completed" in sample_state.workflow_trace[1]

    def test_record_step_duration(self, sample_state):
        """Test recording step durations."""
        sample_state.record_step_duration("classification", 150.5)
        sample_state.record_step_duration("retrieval", 300.2)
        
        assert sample_state.step_durations["classification"] == 150.5
        assert sample_state.step_durations["retrieval"] == 300.2

    def test_to_dict(self, sample_state):
        """Test converting state to dictionary."""
        # Add some data to the state
        mock_classification = Mock()
        mock_classification.__dict__ = {
            "intent_type": IntentType.CONSUMPTION_ANALYSIS,
            "confidence_score": 0.8
        }
        sample_state.classification = mock_classification
        sample_state.add_trace("Test trace")
        sample_state.record_step_duration("test_step", 100.0)
        
        state_dict = sample_state.to_dict()
        
        assert isinstance(state_dict, dict)
        assert state_dict["query_id"] == "test-query-123"
        assert state_dict["original_query"] == "What is the electricity consumption trend?"
        assert state_dict["user_id"] == "user-456"
        assert state_dict["classification"] is not None
        assert state_dict["workflow_trace"] == sample_state.workflow_trace
        assert state_dict["step_durations"] == {"test_step": 100.0}
        assert "created_at" in state_dict
        assert "updated_at" in state_dict


class TestEHSWorkflow:
    """Test suite for EHSWorkflow."""

    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager."""
        db_manager = Mock(spec=DatabaseManager)
        db_manager.is_connected = True
        db_manager.health_check = AsyncMock(return_value=True)
        return db_manager

    @pytest.fixture
    def mock_query_router(self):
        """Mock query router agent."""
        router = Mock(spec=QueryRouterAgent)
        
        # Mock classification result
        mock_classification = QueryClassification(
            intent_type=IntentType.CONSUMPTION_ANALYSIS,
            confidence_score=0.85,
            entities_identified=EntityExtraction(
                facilities=["Plant A"],
                date_ranges=["last quarter"],
                equipment=[],
                pollutants=[],
                regulations=[],
                departments=[],
                metrics=["electricity"]
            ),
            suggested_retriever=RetrieverType.CONSUMPTION_RETRIEVER,
            reasoning="Query focuses on electricity consumption trends"
        )
        
        router.classify_query.return_value = mock_classification
        return router

    @pytest.fixture
    def mock_profiler(self):
        """Mock profiler."""
        profiler = Mock()
        profiler.profile_operation = Mock()
        profiler.profile_operation.return_value.__enter__ = Mock()
        profiler.profile_operation.return_value.__exit__ = Mock()
        return profiler

    @pytest.fixture
    def mock_monitor(self):
        """Mock monitor."""
        monitor = Mock()
        monitor.record_query = Mock()
        monitor.record_retrieval = Mock()
        monitor.record_analysis = Mock()
        return monitor

    @pytest.fixture
    def workflow(self, mock_db_manager, mock_query_router, mock_profiler, mock_monitor):
        """EHSWorkflow instance with mocked dependencies."""
        with patch('ehs_analytics.workflows.ehs_workflow.get_ehs_profiler', return_value=mock_profiler), \
             patch('ehs_analytics.workflows.ehs_workflow.get_ehs_monitor', return_value=mock_monitor):
            
            workflow = EHSWorkflow(mock_db_manager, mock_query_router)
            return workflow

    @pytest.fixture
    async def initialized_workflow(self, workflow):
        """Initialized EHSWorkflow instance."""
        await workflow.initialize()
        return workflow

    def test_init_workflow(self, mock_db_manager, mock_query_router):
        """Test EHSWorkflow initialization."""
        workflow = EHSWorkflow(mock_db_manager, mock_query_router)
        
        assert workflow.db_manager == mock_db_manager
        assert workflow.query_router == mock_query_router
        assert workflow.is_initialized is False

    @pytest.mark.asyncio
    async def test_initialize_success(self, workflow):
        """Test successful workflow initialization."""
        await workflow.initialize()
        
        assert workflow.is_initialized is True

    @pytest.mark.asyncio 
    async def test_initialize_failure(self, workflow):
        """Test workflow initialization failure."""
        # Mock an exception during initialization
        with patch.object(workflow, 'is_initialized', side_effect=Exception("Init failed")):
            with pytest.raises(Exception, match="Init failed"):
                await workflow.initialize()

    @pytest.mark.asyncio
    async def test_process_query_success(self, initialized_workflow):
        """Test successful query processing through complete workflow."""
        query_id = "test-query-123"
        query = "What is the electricity consumption trend for Plant A?"
        user_id = "user-456"
        
        result = await initialized_workflow.process_query(
            query_id=query_id,
            query=query,
            user_id=user_id
        )
        
        assert isinstance(result, EHSWorkflowState)
        assert result.query_id == query_id
        assert result.original_query == query
        assert result.user_id == user_id
        assert result.classification is not None
        assert result.retrieval_results is not None
        assert result.analysis_results is not None
        assert result.total_duration_ms is not None
        assert result.total_duration_ms > 0
        assert len(result.step_durations) > 0
        assert len(result.workflow_trace) > 0

    @pytest.mark.asyncio
    async def test_process_query_with_options(self, initialized_workflow):
        """Test query processing with custom options."""
        query_id = "test-query-123"
        query = "Test query"
        options = {"include_recommendations": True, "timeout": 300}
        
        result = await initialized_workflow.process_query(
            query_id=query_id,
            query=query,
            options=options
        )
        
        assert result.recommendations is not None
        assert result.recommendations["recommendations_count"] >= 0

    @pytest.mark.asyncio
    async def test_process_query_no_recommendations(self, initialized_workflow):
        """Test query processing without recommendations."""
        query_id = "test-query-123"
        query = "Test query"
        options = {"include_recommendations": False}
        
        result = await initialized_workflow.process_query(
            query_id=query_id,
            query=query,
            options=options
        )
        
        assert result.recommendations is None

    @pytest.mark.asyncio
    async def test_process_query_failure(self, initialized_workflow):
        """Test query processing failure handling."""
        # Mock router to raise exception
        initialized_workflow.query_router.classify_query.side_effect = Exception("Classification failed")
        
        query_id = "test-query-123"
        query = "Test query"
        
        with pytest.raises(Exception, match="Classification failed"):
            await initialized_workflow.process_query(query_id, query)

    @pytest.mark.asyncio
    async def test_step_classify_query_success(self, initialized_workflow):
        """Test successful query classification step."""
        state = EHSWorkflowState("test-123", "Test query", "user-456")
        
        await initialized_workflow._step_classify_query(state)
        
        assert state.classification is not None
        assert state.classification.intent_type is not None
        assert state.classification.confidence_score > 0
        assert "classification" in state.step_durations
        assert any("classified" in trace for trace in state.workflow_trace)

    @pytest.mark.asyncio
    async def test_step_classify_query_failure(self, initialized_workflow):
        """Test query classification step failure."""
        initialized_workflow.query_router.classify_query.side_effect = Exception("Router error")
        
        state = EHSWorkflowState("test-123", "Test query", "user-456")
        
        with pytest.raises(Exception, match="Router error"):
            await initialized_workflow._step_classify_query(state)
        
        # Should still record step duration even on failure
        assert "classification" in state.step_durations

    @pytest.mark.asyncio
    async def test_step_retrieve_data_success(self, initialized_workflow):
        """Test successful data retrieval step."""
        state = EHSWorkflowState("test-123", "Test query", "user-456")
        
        # Set up classification first
        mock_classification = Mock()
        mock_classification.suggested_retriever = RetrieverType.CONSUMPTION_RETRIEVER
        state.classification = mock_classification
        
        await initialized_workflow._step_retrieve_data(state)
        
        assert state.retrieval_results is not None
        assert "retrieval" in state.step_durations
        assert any("retrieval" in trace for trace in state.workflow_trace)

    @pytest.mark.asyncio
    async def test_step_retrieve_data_failure(self, initialized_workflow):
        """Test data retrieval step failure."""
        state = EHSWorkflowState("test-123", "Test query", "user-456")
        state.classification = Mock(suggested_retriever=RetrieverType.GENERAL_RETRIEVER)
        
        # Mock asyncio.sleep to raise exception
        with patch('asyncio.sleep', side_effect=Exception("Retrieval error")):
            with pytest.raises(Exception, match="Retrieval error"):
                await initialized_workflow._step_retrieve_data(state)

    @pytest.mark.asyncio
    async def test_step_analyze_data_success(self, initialized_workflow):
        """Test successful data analysis step."""
        state = EHSWorkflowState("test-123", "Test query", "user-456")
        
        # Set up classification for analysis
        mock_classification = Mock()
        mock_classification.intent_type = IntentType.RISK_ASSESSMENT
        state.classification = mock_classification
        
        await initialized_workflow._step_analyze_data(state)
        
        assert state.analysis_results is not None
        assert len(state.analysis_results) > 0
        assert "analysis" in state.step_durations
        assert any("analysis" in trace.lower() for trace in state.workflow_trace)

    @pytest.mark.asyncio
    async def test_step_analyze_data_failure(self, initialized_workflow):
        """Test data analysis step failure."""
        state = EHSWorkflowState("test-123", "Test query", "user-456")
        
        # Mock _perform_analysis to raise exception
        with patch.object(initialized_workflow, '_perform_analysis', side_effect=Exception("Analysis error")):
            with pytest.raises(Exception, match="Analysis error"):
                await initialized_workflow._step_analyze_data(state)

    @pytest.mark.asyncio
    async def test_step_generate_recommendations_success(self, initialized_workflow):
        """Test successful recommendation generation step."""
        state = EHSWorkflowState("test-123", "Test query", "user-456")
        
        await initialized_workflow._step_generate_recommendations(state)
        
        assert state.recommendations is not None
        assert "recommendations_count" in state.recommendations
        assert "recommendations" in state.step_durations
        assert any("recommendation" in trace.lower() for trace in state.workflow_trace)

    @pytest.mark.asyncio
    async def test_step_generate_recommendations_failure(self, initialized_workflow):
        """Test recommendation generation step failure."""
        state = EHSWorkflowState("test-123", "Test query", "user-456")
        
        # Mock _generate_recommendations to raise exception
        with patch.object(initialized_workflow, '_generate_recommendations', side_effect=Exception("Recommendation error")):
            with pytest.raises(Exception, match="Recommendation error"):
                await initialized_workflow._step_generate_recommendations(state)

    @pytest.mark.asyncio
    async def test_perform_analysis_risk_assessment(self, initialized_workflow):
        """Test analysis for risk assessment intent."""
        state = EHSWorkflowState("test-123", "Test query", "user-456")
        mock_classification = Mock()
        mock_classification.intent_type = IntentType.RISK_ASSESSMENT
        state.classification = mock_classification
        
        results = await initialized_workflow._perform_analysis(state)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert results[0]["analysis_type"] == "risk_assessment"
        assert "risk_level" in results[0]
        assert "risk_score" in results[0]

    @pytest.mark.asyncio
    async def test_perform_analysis_compliance_check(self, initialized_workflow):
        """Test analysis for compliance check intent."""
        state = EHSWorkflowState("test-123", "Test query", "user-456")
        mock_classification = Mock()
        mock_classification.intent_type = IntentType.COMPLIANCE_CHECK
        state.classification = mock_classification
        
        results = await initialized_workflow._perform_analysis(state)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert results[0]["analysis_type"] == "compliance_status"
        assert "compliant" in results[0]
        assert "compliance_score" in results[0]

    @pytest.mark.asyncio
    async def test_perform_analysis_general(self, initialized_workflow):
        """Test analysis for general intent."""
        state = EHSWorkflowState("test-123", "Test query", "user-456")
        mock_classification = Mock()
        mock_classification.intent_type = IntentType.CONSUMPTION_ANALYSIS
        state.classification = mock_classification
        
        results = await initialized_workflow._perform_analysis(state)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert results[0]["analysis_type"] == "general_analysis"

    @pytest.mark.asyncio
    async def test_perform_analysis_no_classification(self, initialized_workflow):
        """Test analysis with no classification."""
        state = EHSWorkflowState("test-123", "Test query", "user-456")
        state.classification = None
        
        results = await initialized_workflow._perform_analysis(state)
        
        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_generate_recommendations(self, initialized_workflow):
        """Test recommendation generation."""
        state = EHSWorkflowState("test-123", "Test query", "user-456")
        
        recommendations = await initialized_workflow._generate_recommendations(state)
        
        assert isinstance(recommendations, dict)
        assert "recommendations" in recommendations
        assert "total_estimated_cost" in recommendations
        assert "total_estimated_savings" in recommendations
        assert "recommendations_count" in recommendations
        assert len(recommendations["recommendations"]) > 0
        
        # Check recommendation structure
        rec = recommendations["recommendations"][0]
        assert "id" in rec
        assert "title" in rec
        assert "description" in rec
        assert "priority" in rec
        assert "estimated_cost" in rec
        assert "estimated_savings" in rec

    @pytest.mark.asyncio
    async def test_health_check_success(self, initialized_workflow):
        """Test successful workflow health check."""
        is_healthy = await initialized_workflow.health_check()
        
        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self, workflow):
        """Test health check when workflow not initialized."""
        is_healthy = await workflow.health_check()
        
        assert is_healthy is False

    @pytest.mark.asyncio
    async def test_health_check_router_failure(self, initialized_workflow):
        """Test health check when router fails."""
        initialized_workflow.query_router.classify_query.side_effect = Exception("Router failed")
        
        is_healthy = await initialized_workflow.health_check()
        
        assert is_healthy is False

    @pytest.mark.asyncio
    async def test_health_check_router_low_confidence(self, initialized_workflow):
        """Test health check when router returns low confidence."""
        mock_result = Mock()
        mock_result.confidence_score = -0.1  # Invalid confidence
        initialized_workflow.query_router.classify_query.return_value = mock_result
        
        is_healthy = await initialized_workflow.health_check()
        
        assert is_healthy is False

    def test_get_workflow_stats(self, initialized_workflow):
        """Test getting workflow statistics."""
        stats = initialized_workflow.get_workflow_stats()
        
        assert isinstance(stats, dict)
        assert "total_queries_processed" in stats
        assert "average_processing_time_ms" in stats
        assert "success_rate" in stats
        assert "step_performance" in stats
        assert "intent_distribution" in stats
        
        # Check step performance structure
        step_perf = stats["step_performance"]
        expected_steps = ["classification", "retrieval", "analysis", "recommendations"]
        for step in expected_steps:
            assert step in step_perf
            assert "avg_ms" in step_perf[step]
            assert "success_rate" in step_perf[step]

    @pytest.mark.asyncio
    async def test_concurrent_query_processing(self, initialized_workflow):
        """Test concurrent query processing."""
        queries = [
            ("query-1", "What is electricity consumption?"),
            ("query-2", "Show equipment efficiency"),
            ("query-3", "Check permit status")
        ]
        
        # Process queries concurrently
        tasks = [
            initialized_workflow.process_query(query_id, query)
            for query_id, query in queries
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, EHSWorkflowState)
            assert result.total_duration_ms is not None

    @pytest.mark.asyncio
    async def test_workflow_step_ordering(self, initialized_workflow):
        """Test that workflow steps execute in correct order."""
        state = EHSWorkflowState("test-123", "Test query", "user-456")
        
        # Execute individual steps
        await initialized_workflow._step_classify_query(state)
        await initialized_workflow._step_retrieve_data(state)
        await initialized_workflow._step_analyze_data(state)
        await initialized_workflow._step_generate_recommendations(state)
        
        # Check that all steps completed successfully
        assert state.classification is not None
        assert state.retrieval_results is not None
        assert state.analysis_results is not None
        assert state.recommendations is not None
        
        # Check step durations are recorded
        expected_steps = ["classification", "retrieval", "analysis", "recommendations"]
        for step in expected_steps:
            assert step in state.step_durations
            assert state.step_durations[step] > 0

    @pytest.mark.asyncio 
    async def test_workflow_state_persistence(self, initialized_workflow):
        """Test that workflow state persists correctly through steps."""
        query_id = "persistent-test"
        query = "Test persistence"
        user_id = "test-user"
        
        result = await initialized_workflow.process_query(query_id, query, user_id)
        
        # Verify state consistency
        assert result.query_id == query_id
        assert result.original_query == query
        assert result.user_id == user_id
        assert result.created_at <= result.updated_at
        
        # Check that metadata accumulated through workflow
        assert len(result.workflow_trace) > 0
        assert len(result.step_durations) > 0

    @pytest.mark.asyncio
    async def test_workflow_timeout_handling(self, initialized_workflow):
        """Test workflow handles timeout scenarios."""
        # Mock a slow step
        original_sleep = asyncio.sleep
        
        async def slow_sleep(*args):
            if args[0] == 0.1:  # Our test sleep duration
                await original_sleep(2.0)  # Make it slower
            else:
                await original_sleep(*args)
        
        with patch('asyncio.sleep', side_effect=slow_sleep):
            start_time = datetime.utcnow()
            
            result = await initialized_workflow.process_query("timeout-test", "Test query")
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            # Should still complete but take longer
            assert isinstance(result, EHSWorkflowState)
            assert duration > 1.0  # Due to our slow mock

    @pytest.mark.asyncio
    async def test_monitoring_integration(self, initialized_workflow, mock_monitor):
        """Test workflow integrates properly with monitoring."""
        query_id = "monitor-test"
        query = "Test monitoring"
        
        result = await initialized_workflow.process_query(query_id, query)
        
        # Verify monitoring calls were made
        assert mock_monitor.record_query.call_count >= 1
        assert mock_monitor.record_retrieval.call_count >= 1
        
        # Check that monitor was called with correct parameters
        monitor_calls = mock_monitor.record_query.call_args_list
        assert len(monitor_calls) > 0


class TestCreateEHSWorkflow:
    """Test suite for create_ehs_workflow factory function."""

    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager."""
        return Mock(spec=DatabaseManager)

    @pytest.fixture
    def mock_query_router(self):
        """Mock query router."""
        return Mock(spec=QueryRouterAgent)

    @pytest.mark.asyncio
    async def test_create_workflow_success(self, mock_db_manager, mock_query_router):
        """Test successful workflow creation."""
        with patch('ehs_analytics.workflows.ehs_workflow.get_ehs_profiler'), \
             patch('ehs_analytics.workflows.ehs_workflow.get_ehs_monitor'):
            
            workflow = await create_ehs_workflow(mock_db_manager, mock_query_router)
            
            assert isinstance(workflow, EHSWorkflow)
            assert workflow.is_initialized is True
            assert workflow.db_manager == mock_db_manager
            assert workflow.query_router == mock_query_router

    @pytest.mark.asyncio
    async def test_create_workflow_initialization_failure(self, mock_db_manager, mock_query_router):
        """Test workflow creation with initialization failure."""
        with patch('ehs_analytics.workflows.ehs_workflow.get_ehs_profiler'), \
             patch('ehs_analytics.workflows.ehs_workflow.get_ehs_monitor'), \
             patch.object(EHSWorkflow, 'initialize', side_effect=Exception("Init failed")):
            
            with pytest.raises(Exception, match="Init failed"):
                await create_ehs_workflow(mock_db_manager, mock_query_router)

    @pytest.mark.asyncio
    async def test_create_workflow_with_logging_context(self, mock_db_manager, mock_query_router):
        """Test workflow creation includes proper logging context."""
        with patch('ehs_analytics.workflows.ehs_workflow.get_ehs_profiler'), \
             patch('ehs_analytics.workflows.ehs_workflow.get_ehs_monitor'), \
             patch('ehs_analytics.workflows.ehs_workflow.log_context') as mock_log_context, \
             patch('ehs_analytics.workflows.ehs_workflow.logger') as mock_logger:
            
            workflow = await create_ehs_workflow(mock_db_manager, mock_query_router)
            
            # Verify logging context was used
            mock_log_context.assert_called_with(
                component="workflow_factory",
                operation="create_workflow"
            )
            
            # Verify logging calls were made
            assert mock_logger.info.call_count >= 2