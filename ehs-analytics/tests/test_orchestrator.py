"""
Tests for EHS Analytics Retrieval Orchestrator.

This module tests the orchestration components including strategy selection,
result merging, and the main orchestrator functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Dict, Any, List

from src.ehs_analytics.retrieval.orchestrator import (
    RetrievalOrchestrator,
    OrchestrationConfig,
    OrchestrationMode,
    OrchestrationMetrics
)
from src.ehs_analytics.retrieval.strategy_selector import (
    StrategySelector,
    SelectionMethod,
    QueryCharacteristics,
    SelectionResult
)
from src.ehs_analytics.retrieval.result_merger import (
    ResultMerger,
    MergerConfig,
    DeduplicationMethod,
    RankingMethod,
    MergedResult,
    DeduplicationInfo
)
from src.ehs_analytics.retrieval.base import (
    RetrievalStrategy,
    QueryType,
    RetrievalResult,
    RetrievalMetadata,
    BaseRetriever
)


class MockRetriever(BaseRetriever):
    """Mock retriever for testing."""
    
    def __init__(self, strategy: RetrievalStrategy, config: Dict[str, Any]):
        super().__init__(config)
        self.strategy = strategy
        self._initialized = True
    
    async def initialize(self) -> None:
        self._initialized = True
    
    async def retrieve(
        self,
        query: str,
        query_type: QueryType = QueryType.GENERAL,
        limit: int = 10,
        **kwargs
    ) -> RetrievalResult:
        # Return mock data based on strategy
        mock_data = []
        if self.strategy == RetrievalStrategy.TEXT2CYPHER:
            mock_data = [
                {"id": "F-001", "name": "Test Facility", "score": 0.9},
                {"id": "E-001", "name": "Test Equipment", "score": 0.8}
            ]
        elif self.strategy == RetrievalStrategy.VECTOR:
            mock_data = [
                {"id": "D-001", "name": "Test Document", "score": 0.85},
                {"id": "F-001", "name": "Test Facility", "score": 0.7}  # Duplicate
            ]
        
        return RetrievalResult(
            data=mock_data,
            metadata=RetrievalMetadata(
                strategy=self.strategy,
                query_type=query_type,
                confidence_score=0.8,
                execution_time_ms=100.0
            ),
            success=True
        )
    
    async def validate_query(self, query: str) -> bool:
        return True
    
    def get_strategy(self) -> RetrievalStrategy:
        return self.strategy


@pytest.fixture
def mock_retrievers():
    """Create mock retrievers for testing."""
    return {
        RetrievalStrategy.TEXT2CYPHER: MockRetriever(RetrievalStrategy.TEXT2CYPHER, {}),
        RetrievalStrategy.VECTOR: MockRetriever(RetrievalStrategy.VECTOR, {}),
        RetrievalStrategy.HYBRID: MockRetriever(RetrievalStrategy.HYBRID, {})
    }


@pytest.fixture
def orchestration_config():
    """Create test orchestration configuration."""
    return OrchestrationConfig(
        max_strategies=2,
        min_confidence_threshold=0.6,
        enable_parallel_execution=True,
        enable_caching=False  # Disable for testing
    )


class TestStrategySelector:
    """Test strategy selector functionality."""
    
    @pytest.fixture
    def selector(self):
        return StrategySelector()
    
    @pytest.mark.asyncio
    async def test_initialization(self, selector):
        """Test selector initialization."""
        await selector.initialize()
        assert selector.strategy_performance is not None
        assert selector.query_patterns is not None
    
    @pytest.mark.asyncio
    async def test_query_analysis(self, selector):
        """Test query characteristic analysis."""
        await selector.initialize()
        
        # Test simple lookup query
        characteristics = selector._analyze_query("What is facility F-001?", QueryType.GENERAL)
        assert characteristics.is_lookup_query
        assert not characteristics.is_temporal_query
        assert characteristics.word_count == 4
        
        # Test temporal query
        characteristics = selector._analyze_query("Show water consumption over time", QueryType.CONSUMPTION)
        assert characteristics.contains_time_references
        assert characteristics.is_temporal_query
        assert "water" in characteristics.ehs_domain_terms
    
    @pytest.mark.asyncio
    async def test_rule_based_selection(self, selector):
        """Test rule-based strategy selection."""
        await selector.initialize()
        
        available_strategies = [RetrievalStrategy.TEXT2CYPHER, RetrievalStrategy.VECTOR]
        
        # Test simple lookup query - should prefer Text2Cypher
        result = await selector.select_strategies(
            query="What is facility F-001?",
            query_type=QueryType.GENERAL,
            available_strategies=available_strategies,
            selection_method=SelectionMethod.RULE_BASED
        )
        
        assert RetrievalStrategy.TEXT2CYPHER in result.selected_strategies
        assert result.selection_method == SelectionMethod.RULE_BASED
        assert result.strategy_confidences[RetrievalStrategy.TEXT2CYPHER] > 0.5
    
    @pytest.mark.asyncio
    async def test_hybrid_selection(self, selector):
        """Test hybrid strategy selection."""
        await selector.initialize()
        
        available_strategies = list(RetrievalStrategy)
        
        result = await selector.select_strategies(
            query="Analyze emission trends and patterns",
            query_type=QueryType.EMISSIONS,
            available_strategies=available_strategies,
            selection_method=SelectionMethod.HYBRID
        )
        
        assert len(result.selected_strategies) > 0
        assert result.selection_method == SelectionMethod.HYBRID
        assert result.execution_time_ms > 0
    
    def test_complexity_calculation(self, selector):
        """Test query complexity calculation."""
        # Simple query
        simple_score = selector._calculate_complexity_score("What is facility?")
        assert 0 <= simple_score <= 1
        
        # Complex query
        complex_score = selector._calculate_complexity_score(
            "Analyze the correlation between energy consumption and emission levels "
            "across multiple facilities over the past year, considering seasonal "
            "variations and equipment efficiency ratings."
        )
        assert complex_score > simple_score


class TestResultMerger:
    """Test result merger functionality."""
    
    @pytest.fixture
    def merger(self):
        config = MergerConfig(
            deduplication_method=DeduplicationMethod.CONTENT_HASH,
            ranking_method=RankingMethod.HYBRID
        )
        return ResultMerger(config)
    
    @pytest.fixture
    def mock_results(self):
        """Create mock retrieval results for testing."""
        return [
            RetrievalResult(
                data=[
                    {"id": "F-001", "name": "Main Facility", "score": 0.9},
                    {"id": "F-002", "name": "Secondary Plant", "score": 0.7}
                ],
                metadata=RetrievalMetadata(
                    strategy=RetrievalStrategy.TEXT2CYPHER,
                    query_type=QueryType.GENERAL,
                    confidence_score=0.85,
                    execution_time_ms=150.0
                ),
                success=True
            ),
            RetrievalResult(
                data=[
                    {"id": "F-001", "name": "Main Facility", "score": 0.8},  # Duplicate
                    {"id": "F-003", "name": "Research Center", "score": 0.6}
                ],
                metadata=RetrievalMetadata(
                    strategy=RetrievalStrategy.VECTOR,
                    query_type=QueryType.GENERAL,
                    confidence_score=0.75,
                    execution_time_ms=220.0
                ),
                success=True
            )
        ]
    
    @pytest.mark.asyncio
    async def test_initialization(self, merger):
        """Test merger initialization."""
        await merger.initialize()
        assert merger.config is not None
    
    @pytest.mark.asyncio
    async def test_result_merging(self, merger, mock_results):
        """Test basic result merging."""
        await merger.initialize()
        
        merged = await merger.merge_results(
            results=mock_results,
            query="test query",
            query_type=QueryType.GENERAL,
            max_results=10
        )
        
        assert isinstance(merged, MergedResult)
        assert len(merged.data) <= sum(len(r.data) for r in mock_results)
        assert merged.confidence_score > 0
        assert len(merged.source_strategies) == 2
    
    @pytest.mark.asyncio
    async def test_deduplication(self, merger, mock_results):
        """Test result deduplication."""
        await merger.initialize()
        
        merged = await merger.merge_results(
            results=mock_results,
            query="test query",
            query_type=QueryType.GENERAL,
            max_results=10
        )
        
        # Should remove duplicates
        assert merged.deduplication_info.duplicates_removed > 0
        assert merged.deduplication_info.deduplicated_count < merged.deduplication_info.original_count
    
    def test_content_hash_extraction(self, merger):
        """Test content hash extraction."""
        item = {"id": "F-001", "name": "Test Facility", "description": "Test description"}
        content = merger._extract_content_for_hash(item)
        assert "Test Facility" in content
        assert "Test description" in content
    
    def test_similarity_calculation(self, merger):
        """Test content similarity calculation."""
        content1 = "Main water treatment facility"
        content2 = "Main water treatment plant"
        content3 = "Secondary power generation unit"
        
        similarity1 = merger._calculate_content_similarity(content1, content2)
        similarity2 = merger._calculate_content_similarity(content1, content3)
        
        assert similarity1 > similarity2  # More similar content should have higher score
        assert 0 <= similarity1 <= 1
        assert 0 <= similarity2 <= 1


class TestRetrievalOrchestrator:
    """Test retrieval orchestrator functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_retrievers, orchestration_config):
        """Test orchestrator initialization."""
        orchestrator = RetrievalOrchestrator(mock_retrievers, orchestration_config)
        await orchestrator.initialize()
        
        assert orchestrator.retrievers == mock_retrievers
        assert orchestrator.config == orchestration_config
        assert orchestrator.strategy_selector is not None
        assert orchestrator.result_merger is not None
    
    @pytest.mark.asyncio
    async def test_single_mode_retrieval(self, mock_retrievers, orchestration_config):
        """Test single mode retrieval."""
        orchestrator = RetrievalOrchestrator(mock_retrievers, orchestration_config)
        await orchestrator.initialize()
        
        result = await orchestrator.retrieve(
            query="What is facility F-001?",
            query_type=QueryType.GENERAL,
            mode=OrchestrationMode.SINGLE,
            limit=5
        )
        
        assert isinstance(result, MergedResult)
        assert len(result.data) > 0
        assert result.confidence_score > 0
        assert len(result.source_strategies) >= 1
    
    @pytest.mark.asyncio
    async def test_parallel_mode_retrieval(self, mock_retrievers, orchestration_config):
        """Test parallel mode retrieval."""
        orchestrator = RetrievalOrchestrator(mock_retrievers, orchestration_config)
        await orchestrator.initialize()
        
        result = await orchestrator.retrieve(
            query="Find energy efficiency equipment",
            query_type=QueryType.EFFICIENCY,
            mode=OrchestrationMode.PARALLEL,
            limit=10
        )
        
        assert isinstance(result, MergedResult)
        assert result.metrics.parallel_execution
        assert len(result.source_strategies) > 1  # Should use multiple strategies
    
    @pytest.mark.asyncio
    async def test_adaptive_mode_selection(self, mock_retrievers, orchestration_config):
        """Test adaptive mode selection."""
        orchestrator = RetrievalOrchestrator(mock_retrievers, orchestration_config)
        await orchestrator.initialize()
        
        # Simple query should use single mode
        result1 = await orchestrator.retrieve(
            query="What is permit P-001?",
            query_type=QueryType.COMPLIANCE,
            mode=OrchestrationMode.ADAPTIVE,
            limit=5
        )
        
        # Complex query should use parallel mode
        result2 = await orchestrator.retrieve(
            query="Analyze emission patterns and recommend improvements",
            query_type=QueryType.RECOMMENDATION,
            mode=OrchestrationMode.ADAPTIVE,
            limit=10
        )
        
        assert isinstance(result1, MergedResult)
        assert isinstance(result2, MergedResult)
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, mock_retrievers, orchestration_config):
        """Test performance metrics tracking."""
        orchestrator = RetrievalOrchestrator(mock_retrievers, orchestration_config)
        await orchestrator.initialize()
        
        # Execute a few queries
        for i in range(3):
            await orchestrator.retrieve(
                query=f"Test query {i}",
                query_type=QueryType.GENERAL,
                mode=OrchestrationMode.SINGLE,
                limit=5
            )
        
        metrics = await orchestrator.get_performance_metrics()
        assert metrics['total_operations'] == 3
        assert 'recent_performance' in metrics
        assert 'strategy_performance' in metrics
    
    @pytest.mark.asyncio
    async def test_health_check(self, mock_retrievers, orchestration_config):
        """Test health check functionality."""
        orchestrator = RetrievalOrchestrator(mock_retrievers, orchestration_config)
        await orchestrator.initialize()
        
        health = await orchestrator.health_check()
        
        assert health['orchestrator']['status'] == 'healthy'
        assert health['orchestrator']['active_retrievers'] == len(mock_retrievers)
        assert 'retrievers' in health
        
        for strategy in mock_retrievers.keys():
            assert strategy.value in health['retrievers']
    
    @pytest.mark.asyncio
    async def test_error_handling(self, orchestration_config):
        """Test error handling with failing retrievers."""
        # Create a failing retriever
        failing_retriever = Mock(spec=BaseRetriever)
        failing_retriever.retrieve = AsyncMock(side_effect=Exception("Retriever failed"))
        failing_retriever._initialized = True
        failing_retriever.get_strategy.return_value = RetrievalStrategy.TEXT2CYPHER
        
        retrievers = {RetrievalStrategy.TEXT2CYPHER: failing_retriever}
        orchestrator = RetrievalOrchestrator(retrievers, orchestration_config)
        await orchestrator.initialize()
        
        result = await orchestrator.retrieve(
            query="Test query",
            query_type=QueryType.GENERAL,
            mode=OrchestrationMode.SINGLE,
            limit=5
        )
        
        # Should handle errors gracefully
        assert isinstance(result, MergedResult)
        assert result.confidence_score == 0.0  # No successful results
    
    @pytest.mark.asyncio
    async def test_cleanup(self, mock_retrievers, orchestration_config):
        """Test orchestrator cleanup."""
        orchestrator = RetrievalOrchestrator(mock_retrievers, orchestration_config)
        await orchestrator.initialize()
        
        # Add cleanup mock to retrievers
        for retriever in mock_retrievers.values():
            retriever.cleanup = AsyncMock()
        
        await orchestrator.cleanup()
        
        # Verify all retrievers were cleaned up
        for retriever in mock_retrievers.values():
            retriever.cleanup.assert_called_once()


@pytest.mark.asyncio
async def test_integration_workflow():
    """Test the complete orchestration workflow integration."""
    
    # Create mock retrievers
    mock_retrievers = {
        RetrievalStrategy.TEXT2CYPHER: MockRetriever(RetrievalStrategy.TEXT2CYPHER, {}),
        RetrievalStrategy.VECTOR: MockRetriever(RetrievalStrategy.VECTOR, {})
    }
    
    config = OrchestrationConfig(max_strategies=2, enable_caching=False)
    orchestrator = RetrievalOrchestrator(mock_retrievers, config)
    await orchestrator.initialize()
    
    try:
        # Execute a complex query that should use multiple strategies
        result = await orchestrator.retrieve(
            query="Find facilities with high energy consumption and emission levels",
            query_type=QueryType.EMISSIONS,
            mode=OrchestrationMode.ADAPTIVE,
            limit=15
        )
        
        # Verify the complete workflow
        assert isinstance(result, MergedResult)
        assert len(result.data) > 0
        assert result.confidence_score > 0
        assert len(result.source_strategies) > 0
        assert result.metrics is not None
        assert result.metrics.total_execution_time_ms > 0
        assert result.deduplication_info is not None
        
        # Test performance metrics
        perf_metrics = await orchestrator.get_performance_metrics()
        assert perf_metrics['total_operations'] == 1
        
        # Test health check
        health = await orchestrator.health_check()
        assert health['orchestrator']['status'] == 'healthy'
        
    finally:
        await orchestrator.cleanup()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])