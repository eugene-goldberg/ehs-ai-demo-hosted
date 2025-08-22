"""
Result Merger for EHS Analytics Retrieval Orchestrator.

This module provides sophisticated result merging capabilities including
deduplication, score normalization, metadata aggregation, and ranking optimization
for results from multiple retrieval strategies.
"""

import hashlib
import time
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, Counter
import math

from .base import RetrievalResult, RetrievalStrategy, QueryType

logger = logging.getLogger(__name__)


class RankingMethod(str, Enum):
    """Methods for ranking merged results."""
    SCORE_BASED = "score_based"
    STRATEGY_WEIGHTED = "strategy_weighted"
    CONSENSUS_BASED = "consensus_based"
    RELEVANCE_OPTIMIZED = "relevance_optimized"
    HYBRID = "hybrid"


class DeduplicationMethod(str, Enum):
    """Methods for result deduplication."""
    CONTENT_HASH = "content_hash"
    SIMILARITY_THRESHOLD = "similarity_threshold"
    ENTITY_ID_BASED = "entity_id_based"
    SEMANTIC_SIMILARITY = "semantic_similarity"


@dataclass
class DeduplicationInfo:
    """Information about deduplication process."""
    
    original_count: int
    deduplicated_count: int
    duplicates_removed: int
    duplicate_sources: Dict[str, List[RetrievalStrategy]]
    deduplication_method: DeduplicationMethod
    similarity_threshold: Optional[float] = None


@dataclass
class MergedResult:
    """Result container for merged retrieval results."""
    
    data: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    confidence_score: float
    source_strategies: List[RetrievalStrategy]
    deduplication_info: DeduplicationInfo
    ranking_explanation: str
    metrics: Optional[Any] = None  # OrchestrationMetrics will be set by orchestrator


@dataclass
class MergerConfig:
    """Configuration for result merging."""
    
    # Deduplication settings
    deduplication_method: DeduplicationMethod = DeduplicationMethod.CONTENT_HASH
    similarity_threshold: float = 0.85
    enable_semantic_deduplication: bool = False
    
    # Score normalization
    enable_score_normalization: bool = True
    normalization_method: str = "min_max"  # "min_max", "z_score", "softmax"
    
    # Ranking settings
    ranking_method: RankingMethod = RankingMethod.HYBRID
    strategy_weights: Dict[RetrievalStrategy, float] = field(default_factory=lambda: {
        RetrievalStrategy.TEXT2CYPHER: 0.9,
        RetrievalStrategy.VECTOR: 0.8,
        RetrievalStrategy.HYBRID: 1.2,
        RetrievalStrategy.VECTOR_CYPHER: 1.1,
        RetrievalStrategy.HYBRID_CYPHER: 1.3
    })
    
    # Quality thresholds
    min_confidence_threshold: float = 0.1
    max_results_per_strategy: int = 20
    consensus_threshold: int = 2  # Minimum strategies that must agree
    
    # Metadata aggregation
    aggregate_metadata: bool = True
    preserve_source_metadata: bool = True


class ResultMerger:
    """
    Sophisticated result merger for retrieval orchestration.
    
    Merges results from multiple retrieval strategies with deduplication,
    score normalization, ranking optimization, and metadata aggregation.
    """
    
    def __init__(self, config: MergerConfig = None):
        """
        Initialize the result merger.
        
        Args:
            config: Merger configuration settings
        """
        self.config = config or MergerConfig()
        
        # Content similarity cache for deduplication
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        
        logger.info("ResultMerger initialized")
    
    async def initialize(self) -> None:
        """Initialize the result merger."""
        logger.info("Result merger initialized")
    
    async def merge_results(
        self,
        results: List[RetrievalResult],
        query: str,
        query_type: QueryType,
        max_results: int = 50
    ) -> MergedResult:
        """
        Merge results from multiple retrieval strategies.
        
        Args:
            results: List of retrieval results to merge
            query: Original query for context
            query_type: Type of query for strategy-specific handling
            max_results: Maximum number of results in merged output
            
        Returns:
            MergedResult with consolidated and ranked results
        """
        start_time = time.time()
        
        # Filter successful results
        successful_results = [r for r in results if r.success and r.data]
        
        if not successful_results:
            return MergedResult(
                data=[],
                metadata={},
                confidence_score=0.0,
                source_strategies=[],
                deduplication_info=DeduplicationInfo(
                    original_count=0,
                    deduplicated_count=0,
                    duplicates_removed=0,
                    duplicate_sources={},
                    deduplication_method=self.config.deduplication_method
                ),
                ranking_explanation="No successful results to merge"
            )
        
        # Step 1: Normalize scores
        normalized_results = self._normalize_scores(successful_results)
        
        # Step 2: Flatten and prepare results for merging
        flattened_results = self._flatten_results(normalized_results)
        
        # Step 3: Deduplicate results
        deduplicated_results, dedup_info = self._deduplicate_results(
            flattened_results, query_type
        )
        
        # Step 4: Rank and select top results
        ranked_results, ranking_explanation = self._rank_results(
            deduplicated_results, query, query_type, max_results
        )
        
        # Step 5: Aggregate metadata
        aggregated_metadata = self._aggregate_metadata(successful_results, ranked_results)
        
        # Step 6: Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            ranked_results, successful_results, dedup_info
        )
        
        # Get source strategies
        source_strategies = list(set(r.metadata.strategy for r in successful_results))
        
        execution_time = (time.time() - start_time) * 1000
        
        merged_result = MergedResult(
            data=ranked_results,
            metadata=aggregated_metadata,
            confidence_score=confidence_score,
            source_strategies=source_strategies,
            deduplication_info=dedup_info,
            ranking_explanation=ranking_explanation
        )
        
        logger.info(
            f"Merged {len(successful_results)} result sets into {len(ranked_results)} results "
            f"in {execution_time:.2f}ms"
        )
        
        return merged_result
    
    def _normalize_scores(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Normalize scores across different retrieval strategies."""
        
        if not self.config.enable_score_normalization:
            return results
        
        # Collect all scores for normalization
        all_scores = []
        strategy_scores = defaultdict(list)
        
        for result in results:
            for item in result.data:
                score = item.get('score', 0.0)
                all_scores.append(score)
                strategy_scores[result.metadata.strategy].append(score)
        
        if not all_scores:
            return results
        
        # Calculate normalization parameters
        if self.config.normalization_method == "min_max":
            min_score = min(all_scores)
            max_score = max(all_scores)
            score_range = max_score - min_score if max_score > min_score else 1.0
        elif self.config.normalization_method == "z_score":
            mean_score = sum(all_scores) / len(all_scores)
            variance = sum((s - mean_score) ** 2 for s in all_scores) / len(all_scores)
            std_dev = math.sqrt(variance) if variance > 0 else 1.0
        
        # Normalize scores
        normalized_results = []
        for result in results:
            normalized_data = []
            for item in result.data.copy():
                original_score = item.get('score', 0.0)
                
                if self.config.normalization_method == "min_max":
                    normalized_score = (original_score - min_score) / score_range
                elif self.config.normalization_method == "z_score":
                    normalized_score = (original_score - mean_score) / std_dev
                    # Convert to 0-1 range using sigmoid
                    normalized_score = 1 / (1 + math.exp(-normalized_score))
                else:  # softmax per strategy
                    strategy_scores_list = strategy_scores[result.metadata.strategy]
                    if len(strategy_scores_list) > 1:
                        exp_scores = [math.exp(s) for s in strategy_scores_list]
                        sum_exp = sum(exp_scores)
                        normalized_score = math.exp(original_score) / sum_exp
                    else:
                        normalized_score = 1.0
                
                item['score'] = max(0.0, min(1.0, normalized_score))
                item['original_score'] = original_score
                normalized_data.append(item)
            
            # Create new result with normalized data
            normalized_result = RetrievalResult(
                data=normalized_data,
                metadata=result.metadata,
                success=result.success,
                message=result.message
            )
            normalized_results.append(normalized_result)
        
        return normalized_results
    
    def _flatten_results(self, results: List[RetrievalResult]) -> List[Dict[str, Any]]:
        """Flatten results from multiple strategies into single list with source tracking."""
        
        flattened = []
        
        for result in results:
            strategy = result.metadata.strategy
            strategy_weight = self.config.strategy_weights.get(strategy, 1.0)
            
            for item in result.data:
                # Apply quality threshold
                score = item.get('score', 0.0)
                if score < self.config.min_confidence_threshold:
                    continue
                
                # Add source information
                enhanced_item = item.copy()
                enhanced_item['_source_strategy'] = strategy
                enhanced_item['_strategy_weight'] = strategy_weight
                enhanced_item['_original_rank'] = len(flattened)
                
                # Adjust score with strategy weight
                enhanced_item['score'] = score * strategy_weight
                
                flattened.append(enhanced_item)
        
        return flattened
    
    def _deduplicate_results(
        self,
        results: List[Dict[str, Any]],
        query_type: QueryType
    ) -> Tuple[List[Dict[str, Any]], DeduplicationInfo]:
        """Deduplicate results using configured method."""
        
        original_count = len(results)
        
        if self.config.deduplication_method == DeduplicationMethod.CONTENT_HASH:
            deduplicated, duplicate_sources = self._deduplicate_by_content_hash(results)
        elif self.config.deduplication_method == DeduplicationMethod.ENTITY_ID_BASED:
            deduplicated, duplicate_sources = self._deduplicate_by_entity_id(results)
        elif self.config.deduplication_method == DeduplicationMethod.SIMILARITY_THRESHOLD:
            deduplicated, duplicate_sources = self._deduplicate_by_similarity(results)
        else:  # SEMANTIC_SIMILARITY
            deduplicated, duplicate_sources = self._deduplicate_by_semantic_similarity(results)
        
        dedup_info = DeduplicationInfo(
            original_count=original_count,
            deduplicated_count=len(deduplicated),
            duplicates_removed=original_count - len(deduplicated),
            duplicate_sources=duplicate_sources,
            deduplication_method=self.config.deduplication_method,
            similarity_threshold=self.config.similarity_threshold
        )
        
        return deduplicated, dedup_info
    
    def _deduplicate_by_content_hash(
        self,
        results: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[RetrievalStrategy]]]:
        """Deduplicate based on content hash."""
        
        seen_hashes = {}
        deduplicated = []
        duplicate_sources = defaultdict(list)
        
        for item in results:
            # Create content hash
            content_str = self._extract_content_for_hash(item)
            content_hash = hashlib.md5(content_str.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes[content_hash] = len(deduplicated)
                item['_content_hash'] = content_hash
                item['_duplicate_sources'] = [item['_source_strategy']]
                deduplicated.append(item)
            else:
                # Merge with existing item
                existing_idx = seen_hashes[content_hash]
                existing_item = deduplicated[existing_idx]
                
                # Combine scores (take maximum)
                if item['score'] > existing_item['score']:
                    existing_item['score'] = item['score']
                
                # Track duplicate sources
                existing_item['_duplicate_sources'].append(item['_source_strategy'])
                duplicate_sources[content_hash].append(item['_source_strategy'])
        
        return deduplicated, dict(duplicate_sources)
    
    def _deduplicate_by_entity_id(
        self,
        results: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[RetrievalStrategy]]]:
        """Deduplicate based on entity IDs."""
        
        seen_ids = {}
        deduplicated = []
        duplicate_sources = defaultdict(list)
        
        for item in results:
            # Extract entity ID
            entity_id = self._extract_entity_id(item)
            
            if entity_id and entity_id not in seen_ids:
                seen_ids[entity_id] = len(deduplicated)
                item['_entity_id'] = entity_id
                item['_duplicate_sources'] = [item['_source_strategy']]
                deduplicated.append(item)
            elif entity_id:
                # Merge with existing item
                existing_idx = seen_ids[entity_id]
                existing_item = deduplicated[existing_idx]
                
                # Combine scores and metadata
                existing_item['score'] = max(existing_item['score'], item['score'])
                existing_item['_duplicate_sources'].append(item['_source_strategy'])
                duplicate_sources[entity_id].append(item['_source_strategy'])
            else:
                # No entity ID found, keep as unique
                deduplicated.append(item)
        
        return deduplicated, dict(duplicate_sources)
    
    def _deduplicate_by_similarity(
        self,
        results: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[RetrievalStrategy]]]:
        """Deduplicate based on content similarity threshold."""
        
        deduplicated = []
        duplicate_sources = defaultdict(list)
        
        for item in results:
            content = self._extract_content_for_similarity(item)
            is_duplicate = False
            
            # Check against existing items
            for idx, existing_item in enumerate(deduplicated):
                existing_content = self._extract_content_for_similarity(existing_item)
                similarity = self._calculate_content_similarity(content, existing_content)
                
                if similarity >= self.config.similarity_threshold:
                    # Merge with existing item
                    existing_item['score'] = max(existing_item['score'], item['score'])
                    existing_item['_duplicate_sources'].append(item['_source_strategy'])
                    duplicate_key = f"similarity_{idx}"
                    duplicate_sources[duplicate_key].append(item['_source_strategy'])
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                item['_duplicate_sources'] = [item['_source_strategy']]
                deduplicated.append(item)
        
        return deduplicated, dict(duplicate_sources)
    
    def _deduplicate_by_semantic_similarity(
        self,
        results: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[RetrievalStrategy]]]:
        """Deduplicate based on semantic similarity (placeholder for future implementation)."""
        
        # For now, fall back to similarity threshold method
        logger.info("Semantic similarity deduplication not yet implemented, using similarity threshold")
        return self._deduplicate_by_similarity(results)
    
    def _rank_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
        query_type: QueryType,
        max_results: int
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Rank results using configured ranking method."""
        
        if not results:
            return [], "No results to rank"
        
        if self.config.ranking_method == RankingMethod.SCORE_BASED:
            ranked, explanation = self._rank_by_score(results)
        elif self.config.ranking_method == RankingMethod.STRATEGY_WEIGHTED:
            ranked, explanation = self._rank_by_strategy_weight(results)
        elif self.config.ranking_method == RankingMethod.CONSENSUS_BASED:
            ranked, explanation = self._rank_by_consensus(results)
        elif self.config.ranking_method == RankingMethod.RELEVANCE_OPTIMIZED:
            ranked, explanation = self._rank_by_relevance(results, query, query_type)
        else:  # HYBRID
            ranked, explanation = self._rank_hybrid(results, query, query_type)
        
        # Limit results
        final_results = ranked[:max_results]
        
        # Clean up internal fields
        for item in final_results:
            item.pop('_source_strategy', None)
            item.pop('_strategy_weight', None)
            item.pop('_original_rank', None)
            item.pop('_content_hash', None)
            item.pop('_entity_id', None)
        
        return final_results, explanation
    
    def _rank_by_score(self, results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
        """Rank results by score."""
        ranked = sorted(results, key=lambda x: x.get('score', 0.0), reverse=True)
        return ranked, "Ranked by normalized score"
    
    def _rank_by_strategy_weight(self, results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
        """Rank results by strategy weight."""
        ranked = sorted(
            results,
            key=lambda x: (x.get('_strategy_weight', 1.0), x.get('score', 0.0)),
            reverse=True
        )
        return ranked, "Ranked by strategy weight and score"
    
    def _rank_by_consensus(self, results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
        """Rank results by consensus (number of strategies that found the result)."""
        
        def consensus_score(item):
            duplicate_sources = item.get('_duplicate_sources', [])
            consensus_bonus = len(duplicate_sources) / len(self.config.strategy_weights)
            return item.get('score', 0.0) + consensus_bonus
        
        ranked = sorted(results, key=consensus_score, reverse=True)
        return ranked, f"Ranked by consensus (min {self.config.consensus_threshold} strategies)"
    
    def _rank_by_relevance(
        self,
        results: List[Dict[str, Any]],
        query: str,
        query_type: QueryType
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Rank results by relevance to query and query type."""
        
        query_words = set(query.lower().split())
        
        def relevance_score(item):
            base_score = item.get('score', 0.0)
            
            # Calculate content relevance
            content = self._extract_content_for_similarity(item)
            content_words = set(content.lower().split())
            word_overlap = len(query_words.intersection(content_words))
            relevance_bonus = word_overlap / max(len(query_words), 1)
            
            # Query type specific boosting
            type_bonus = 0.0
            if query_type == QueryType.CONSUMPTION and 'consumption' in content.lower():
                type_bonus = 0.1
            elif query_type == QueryType.EFFICIENCY and 'efficiency' in content.lower():
                type_bonus = 0.1
            elif query_type == QueryType.COMPLIANCE and 'permit' in content.lower():
                type_bonus = 0.1
            
            return base_score + relevance_bonus * 0.3 + type_bonus
        
        ranked = sorted(results, key=relevance_score, reverse=True)
        return ranked, "Ranked by relevance to query and type"
    
    def _rank_hybrid(
        self,
        results: List[Dict[str, Any]],
        query: str,
        query_type: QueryType
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Hybrid ranking combining multiple methods."""
        
        def hybrid_score(item):
            base_score = item.get('score', 0.0)
            strategy_weight = item.get('_strategy_weight', 1.0)
            duplicate_sources = item.get('_duplicate_sources', [])
            
            # Score component (40%)
            score_component = base_score * 0.4
            
            # Strategy weight component (25%)
            strategy_component = (strategy_weight / max(self.config.strategy_weights.values())) * 0.25
            
            # Consensus component (20%)
            consensus_component = len(duplicate_sources) / len(self.config.strategy_weights) * 0.2
            
            # Relevance component (15%)
            query_words = set(query.lower().split())
            content = self._extract_content_for_similarity(item)
            content_words = set(content.lower().split())
            word_overlap = len(query_words.intersection(content_words))
            relevance_component = (word_overlap / max(len(query_words), 1)) * 0.15
            
            return score_component + strategy_component + consensus_component + relevance_component
        
        ranked = sorted(results, key=hybrid_score, reverse=True)
        return ranked, "Hybrid ranking (score 40%, strategy 25%, consensus 20%, relevance 15%)"
    
    def _aggregate_metadata(
        self,
        original_results: List[RetrievalResult],
        final_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate metadata from original results."""
        
        if not self.config.aggregate_metadata:
            return {}
        
        metadata = {
            'total_original_results': sum(len(r.data) for r in original_results),
            'total_final_results': len(final_results),
            'strategies_used': list(set(r.metadata.strategy.value for r in original_results)),
            'avg_confidence_scores': {},
            'execution_times_ms': {},
            'cypher_queries': [],
            'vector_similarities': []
        }
        
        # Aggregate per-strategy metadata
        for result in original_results:
            strategy = result.metadata.strategy.value
            
            metadata['avg_confidence_scores'][strategy] = result.metadata.confidence_score
            metadata['execution_times_ms'][strategy] = result.metadata.execution_time_ms
            
            if result.metadata.cypher_query:
                metadata['cypher_queries'].append({
                    'strategy': strategy,
                    'query': result.metadata.cypher_query
                })
            
            if result.metadata.vector_similarity_scores:
                metadata['vector_similarities'].extend(result.metadata.vector_similarity_scores)
        
        # Calculate overall statistics
        if metadata['avg_confidence_scores']:
            metadata['overall_avg_confidence'] = sum(metadata['avg_confidence_scores'].values()) / len(metadata['avg_confidence_scores'])
        
        if metadata['execution_times_ms']:
            metadata['total_execution_time_ms'] = sum(metadata['execution_times_ms'].values())
        
        return metadata
    
    def _calculate_confidence_score(
        self,
        final_results: List[Dict[str, Any]],
        original_results: List[RetrievalResult],
        dedup_info: DeduplicationInfo
    ) -> float:
        """Calculate overall confidence score for merged results."""
        
        if not final_results:
            return 0.0
        
        factors = []
        
        # Result count factor
        result_count_factor = min(len(final_results) / 10.0, 1.0)  # Normalize to 10 results
        factors.append(result_count_factor)
        
        # Average score factor
        avg_score = sum(item.get('score', 0.0) for item in final_results) / len(final_results)
        factors.append(avg_score)
        
        # Strategy diversity factor
        unique_strategies = len(set(r.metadata.strategy for r in original_results))
        strategy_factor = min(unique_strategies / len(RetrievalStrategy), 1.0)
        factors.append(strategy_factor)
        
        # Deduplication factor (lower duplicates = higher confidence)
        if dedup_info.original_count > 0:
            dedup_factor = dedup_info.deduplicated_count / dedup_info.original_count
            factors.append(dedup_factor)
        
        # Consensus factor
        consensus_items = sum(
            1 for item in final_results 
            if len(item.get('_duplicate_sources', [])) >= self.config.consensus_threshold
        )
        if final_results:
            consensus_factor = consensus_items / len(final_results)
            factors.append(consensus_factor)
        
        # Calculate weighted average
        weights = [0.2, 0.3, 0.2, 0.15, 0.15]  # Weights for each factor
        confidence = sum(f * w for f, w in zip(factors, weights[:len(factors)]))
        
        return max(0.0, min(1.0, confidence))
    
    def _extract_content_for_hash(self, item: Dict[str, Any]) -> str:
        """Extract content for hash-based deduplication."""
        
        # Common fields that represent content
        content_fields = ['name', 'title', 'description', 'content', 'text', 'id']
        
        content_parts = []
        for field in content_fields:
            if field in item and item[field]:
                content_parts.append(str(item[field]))
        
        return " ".join(content_parts) if content_parts else str(item)
    
    def _extract_entity_id(self, item: Dict[str, Any]) -> Optional[str]:
        """Extract entity ID for ID-based deduplication."""
        
        # Common ID fields
        id_fields = ['id', 'entity_id', 'node_id', 'equipment_id', 'facility_id', 'permit_id']
        
        for field in id_fields:
            if field in item and item[field]:
                return str(item[field])
        
        return None
    
    def _extract_content_for_similarity(self, item: Dict[str, Any]) -> str:
        """Extract content for similarity comparison."""
        
        # Extract meaningful text content
        text_fields = ['name', 'title', 'description', 'content', 'text', 'type']
        
        content_parts = []
        for field in text_fields:
            if field in item and item[field]:
                content_parts.append(str(item[field]))
        
        return " ".join(content_parts) if content_parts else ""
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings."""
        
        # Simple word-based similarity (Jaccard similarity)
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0