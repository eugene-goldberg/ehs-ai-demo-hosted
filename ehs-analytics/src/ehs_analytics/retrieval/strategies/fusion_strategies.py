"""
Result fusion strategies for hybrid search in EHS Analytics.

This module implements various strategies for combining and ranking results
from vector similarity search and fulltext search to optimize EHS document retrieval.
"""

import logging
import math
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, date
from enum import Enum

import numpy as np

from .hybrid_config import FusionMethod, WeightConfiguration

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Unified search result from vector or fulltext search."""
    
    content: str
    score: float
    metadata: Dict[str, Any]
    document_id: str
    chunk_id: Optional[str] = None
    document_type: str = "unknown"
    source: str = "unknown"  # "vector" or "fulltext"
    rank: Optional[int] = None


@dataclass
class FusionResult:
    """Result after fusion processing."""
    
    content: str
    final_score: float
    metadata: Dict[str, Any]
    document_id: str
    chunk_id: Optional[str] = None
    document_type: str = "unknown"
    fusion_details: Dict[str, Any] = None  # Details about fusion process
    
    
class ScoreNormalizer:
    """Handles score normalization for combining different search methods."""
    
    @staticmethod
    def min_max_normalize(scores: List[float]) -> List[float]:
        """
        Min-max normalization to [0, 1] range.
        
        Args:
            scores: List of scores to normalize
            
        Returns:
            Normalized scores
        """
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    @staticmethod
    def z_score_normalize(scores: List[float]) -> List[float]:
        """
        Z-score normalization (standard score).
        
        Args:
            scores: List of scores to normalize
            
        Returns:
            Normalized scores
        """
        if not scores:
            return []
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        if std_score == 0:
            return [0.0] * len(scores)
        
        return [(score - mean_score) / std_score for score in scores]
    
    @staticmethod
    def sigmoid_normalize(scores: List[float], steepness: float = 1.0) -> List[float]:
        """
        Sigmoid normalization for smooth score distribution.
        
        Args:
            scores: List of scores to normalize
            steepness: Controls sigmoid steepness
            
        Returns:
            Normalized scores
        """
        if not scores:
            return []
        
        # Apply sigmoid function
        normalized = []
        for score in scores:
            sigmoid_score = 1 / (1 + math.exp(-steepness * score))
            normalized.append(sigmoid_score)
        
        return normalized
    
    @staticmethod
    def percentile_normalize(scores: List[float]) -> List[float]:
        """
        Percentile-based normalization.
        
        Args:
            scores: List of scores to normalize
            
        Returns:
            Normalized scores as percentiles
        """
        if not scores:
            return []
        
        sorted_scores = sorted(scores)
        score_to_percentile = {}
        
        for i, score in enumerate(sorted_scores):
            percentile = i / (len(sorted_scores) - 1) if len(sorted_scores) > 1 else 1.0
            score_to_percentile[score] = percentile
        
        return [score_to_percentile[score] for score in scores]


class ResultDeduplicator:
    """Handles deduplication of search results."""
    
    def __init__(self, similarity_threshold: float = 0.9):
        """
        Initialize deduplicator.
        
        Args:
            similarity_threshold: Threshold for considering results as duplicates
        """
        self.similarity_threshold = similarity_threshold
    
    def deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Remove duplicate results based on content similarity.
        
        Args:
            results: List of search results
            
        Returns:
            Deduplicated results
        """
        if not results:
            return []
        
        deduplicated = []
        seen_documents = set()
        
        for result in results:
            # First check document ID duplicates
            if result.document_id in seen_documents:
                continue
            
            # Check content similarity with existing results
            is_duplicate = False
            for existing in deduplicated:
                if self._are_similar(result, existing):
                    # Keep the one with higher score
                    if result.score > existing.score:
                        # Replace existing with current
                        deduplicated = [r for r in deduplicated if r.document_id != existing.document_id]
                        deduplicated.append(result)
                        seen_documents.discard(existing.document_id)
                        seen_documents.add(result.document_id)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(result)
                seen_documents.add(result.document_id)
        
        return deduplicated
    
    def _are_similar(self, result1: SearchResult, result2: SearchResult) -> bool:
        """
        Check if two results are similar enough to be considered duplicates.
        
        Args:
            result1: First result
            result2: Second result
            
        Returns:
            True if results are similar
        """
        # Document ID check
        if result1.document_id == result2.document_id:
            return True
        
        # Content similarity check (simple Jaccard similarity)
        similarity = self._jaccard_similarity(result1.content, result2.content)
        return similarity >= self.similarity_threshold
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate Jaccard similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Jaccard similarity score
        """
        # Simple word-based Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0


class EHSBoostCalculator:
    """Calculates EHS-specific boosts for search results."""
    
    def __init__(self, config: WeightConfiguration):
        """
        Initialize boost calculator.
        
        Args:
            config: Weight configuration with boost factors
        """
        self.config = config
    
    def calculate_facility_boost(self, result: SearchResult, query: str) -> float:
        """
        Calculate facility relevance boost.
        
        Args:
            result: Search result
            query: Original query
            
        Returns:
            Boost factor (0.0 to config.facility_boost)
        """
        facility = result.metadata.get("facility", "").lower()
        if not facility:
            return 0.0
        
        query_lower = query.lower()
        
        # Exact facility name match in query
        if facility in query_lower:
            return self.config.facility_boost
        
        # Partial facility name match
        facility_words = facility.split()
        query_words = query_lower.split()
        
        matches = sum(1 for word in facility_words if word in query_words)
        if matches > 0:
            return self.config.facility_boost * (matches / len(facility_words))
        
        return 0.0
    
    def calculate_recency_boost(self, result: SearchResult) -> float:
        """
        Calculate recency boost for recent documents.
        
        Args:
            result: Search result
            
        Returns:
            Boost factor (0.0 to config.recency_boost)
        """
        try:
            date_created = result.metadata.get("date_created")
            if not date_created:
                return 0.0
            
            if isinstance(date_created, str):
                doc_date = datetime.fromisoformat(date_created).date()
            elif isinstance(date_created, datetime):
                doc_date = date_created.date()
            elif isinstance(date_created, date):
                doc_date = date_created
            else:
                return 0.0
            
            days_old = (datetime.now().date() - doc_date).days
            
            # More recent documents get higher boost
            if days_old <= 7:
                return self.config.recency_boost
            elif days_old <= 30:
                return self.config.recency_boost * 0.8
            elif days_old <= 90:
                return self.config.recency_boost * 0.5
            elif days_old <= 365:
                return self.config.recency_boost * 0.2
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"Error calculating recency boost: {e}")
            return 0.0
    
    def calculate_compliance_boost(self, result: SearchResult) -> float:
        """
        Calculate compliance document boost.
        
        Args:
            result: Search result
            
        Returns:
            Boost factor (0.0 to config.compliance_boost)
        """
        doc_type = result.document_type.lower()
        compliance_types = [
            "permit", "compliance", "regulatory", "license", 
            "certification", "inspection", "audit"
        ]
        
        # Check document type
        if any(comp_type in doc_type for comp_type in compliance_types):
            return self.config.compliance_boost
        
        # Check content for compliance indicators
        content_lower = result.content.lower()
        compliance_indicators = [
            "regulation", "requirement", "compliance", "permit",
            "epa", "osha", "standard", "law", "code"
        ]
        
        indicator_count = sum(1 for indicator in compliance_indicators 
                            if indicator in content_lower)
        
        if indicator_count > 0:
            # Partial boost based on indicator density
            max_indicators = len(compliance_indicators)
            boost_factor = min(1.0, indicator_count / max_indicators)
            return self.config.compliance_boost * boost_factor
        
        return 0.0
    
    def calculate_metadata_boost(self, result: SearchResult, query: str) -> float:
        """
        Calculate boost based on metadata quality and relevance.
        
        Args:
            result: Search result
            query: Original query
            
        Returns:
            Boost factor (0.0 to config.metadata_boost)
        """
        metadata = result.metadata
        boost = 0.0
        
        # Quality indicators
        quality_fields = ["title", "author", "facility", "document_type", "date_created"]
        present_fields = sum(1 for field in quality_fields if metadata.get(field))
        quality_score = present_fields / len(quality_fields)
        
        # Title relevance
        title = metadata.get("title", "").lower()
        if title and any(word in title for word in query.lower().split()):
            boost += self.config.metadata_boost * 0.5
        
        # General metadata quality
        boost += self.config.metadata_boost * quality_score * 0.5
        
        return min(self.config.metadata_boost, boost)
    
    def calculate_total_boost(self, result: SearchResult, query: str) -> float:
        """
        Calculate total boost for a search result.
        
        Args:
            result: Search result
            query: Original query
            
        Returns:
            Total boost factor
        """
        total_boost = 0.0
        
        total_boost += self.calculate_facility_boost(result, query)
        total_boost += self.calculate_recency_boost(result)
        total_boost += self.calculate_compliance_boost(result)
        total_boost += self.calculate_metadata_boost(result, query)
        
        return total_boost


class FusionStrategy:
    """Base class for result fusion strategies."""
    
    def __init__(self, config: WeightConfiguration):
        """
        Initialize fusion strategy.
        
        Args:
            config: Weight configuration
        """
        self.config = config
        self.normalizer = ScoreNormalizer()
        self.boost_calculator = EHSBoostCalculator(config)
        self.deduplicator = ResultDeduplicator()
    
    def fuse_results(
        self,
        vector_results: List[SearchResult],
        fulltext_results: List[SearchResult],
        query: str,
        **kwargs
    ) -> List[FusionResult]:
        """
        Fuse vector and fulltext search results.
        
        Args:
            vector_results: Results from vector search
            fulltext_results: Results from fulltext search
            query: Original query
            **kwargs: Additional fusion parameters
            
        Returns:
            Fused and ranked results
        """
        raise NotImplementedError("Subclasses must implement fuse_results")


class ReciprocalRankFusion(FusionStrategy):
    """Reciprocal Rank Fusion (RRF) strategy."""
    
    def __init__(self, config: WeightConfiguration, k: float = 60.0):
        """
        Initialize RRF fusion.
        
        Args:
            config: Weight configuration
            k: RRF constant (higher values reduce the impact of rank)
        """
        super().__init__(config)
        self.k = k
    
    def fuse_results(
        self,
        vector_results: List[SearchResult],
        fulltext_results: List[SearchResult],
        query: str,
        **kwargs
    ) -> List[FusionResult]:
        """
        Fuse results using Reciprocal Rank Fusion.
        
        Args:
            vector_results: Results from vector search
            fulltext_results: Results from fulltext search
            query: Original query
            **kwargs: Additional parameters
            
        Returns:
            Fused results
        """
        # Create mapping of document_id to results
        all_results = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results, 1):
            doc_id = result.document_id
            rrf_score = 1 / (self.k + rank)
            
            if doc_id not in all_results:
                all_results[doc_id] = {
                    "result": result,
                    "vector_rank": rank,
                    "vector_rrf": rrf_score,
                    "fulltext_rank": None,
                    "fulltext_rrf": 0.0
                }
            else:
                all_results[doc_id]["vector_rank"] = rank
                all_results[doc_id]["vector_rrf"] = rrf_score
        
        # Process fulltext results
        for rank, result in enumerate(fulltext_results, 1):
            doc_id = result.document_id
            rrf_score = 1 / (self.k + rank)
            
            if doc_id not in all_results:
                all_results[doc_id] = {
                    "result": result,
                    "vector_rank": None,
                    "vector_rrf": 0.0,
                    "fulltext_rank": rank,
                    "fulltext_rrf": rrf_score
                }
            else:
                all_results[doc_id]["fulltext_rank"] = rank
                all_results[doc_id]["fulltext_rrf"] = rrf_score
        
        # Calculate final RRF scores
        fusion_results = []
        for doc_id, data in all_results.items():
            result = data["result"]
            
            # Weighted RRF score
            weighted_rrf = (
                self.config.vector_weight * data["vector_rrf"] +
                self.config.fulltext_weight * data["fulltext_rrf"]
            )
            
            # Add EHS-specific boosts
            boost = self.boost_calculator.calculate_total_boost(result, query)
            final_score = weighted_rrf + boost
            
            # Create fusion result
            fusion_result = FusionResult(
                content=result.content,
                final_score=final_score,
                metadata=result.metadata,
                document_id=result.document_id,
                chunk_id=result.chunk_id,
                document_type=result.document_type,
                fusion_details={
                    "vector_rank": data["vector_rank"],
                    "fulltext_rank": data["fulltext_rank"],
                    "vector_rrf": data["vector_rrf"],
                    "fulltext_rrf": data["fulltext_rrf"],
                    "weighted_rrf": weighted_rrf,
                    "boost": boost,
                    "fusion_method": "reciprocal_rank_fusion"
                }
            )
            fusion_results.append(fusion_result)
        
        # Sort by final score
        fusion_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return fusion_results


class WeightedAverageFusion(FusionStrategy):
    """Weighted average fusion strategy."""
    
    def fuse_results(
        self,
        vector_results: List[SearchResult],
        fulltext_results: List[SearchResult],
        query: str,
        **kwargs
    ) -> List[FusionResult]:
        """
        Fuse results using weighted average of normalized scores.
        
        Args:
            vector_results: Results from vector search
            fulltext_results: Results from fulltext search
            query: Original query
            **kwargs: Additional parameters
            
        Returns:
            Fused results
        """
        # Normalize scores
        vector_scores = [r.score for r in vector_results]
        fulltext_scores = [r.score for r in fulltext_results]
        
        norm_vector_scores = self.normalizer.min_max_normalize(vector_scores)
        norm_fulltext_scores = self.normalizer.min_max_normalize(fulltext_scores)
        
        # Create mapping of document_id to normalized scores
        score_map = {}
        
        # Process vector results
        for i, result in enumerate(vector_results):
            doc_id = result.document_id
            score_map[doc_id] = {
                "result": result,
                "vector_score": norm_vector_scores[i],
                "fulltext_score": 0.0
            }
        
        # Process fulltext results
        for i, result in enumerate(fulltext_results):
            doc_id = result.document_id
            if doc_id in score_map:
                score_map[doc_id]["fulltext_score"] = norm_fulltext_scores[i]
            else:
                score_map[doc_id] = {
                    "result": result,
                    "vector_score": 0.0,
                    "fulltext_score": norm_fulltext_scores[i]
                }
        
        # Calculate weighted average scores
        fusion_results = []
        for doc_id, data in score_map.items():
            result = data["result"]
            
            # Weighted average
            weighted_score = (
                self.config.vector_weight * data["vector_score"] +
                self.config.fulltext_weight * data["fulltext_score"]
            )
            
            # Add EHS-specific boosts
            boost = self.boost_calculator.calculate_total_boost(result, query)
            final_score = weighted_score + boost
            
            # Create fusion result
            fusion_result = FusionResult(
                content=result.content,
                final_score=final_score,
                metadata=result.metadata,
                document_id=result.document_id,
                chunk_id=result.chunk_id,
                document_type=result.document_type,
                fusion_details={
                    "vector_score": data["vector_score"],
                    "fulltext_score": data["fulltext_score"],
                    "weighted_score": weighted_score,
                    "boost": boost,
                    "fusion_method": "weighted_average"
                }
            )
            fusion_results.append(fusion_result)
        
        # Sort by final score
        fusion_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return fusion_results


class MaximumScoreFusion(FusionStrategy):
    """Maximum score fusion strategy."""
    
    def fuse_results(
        self,
        vector_results: List[SearchResult],
        fulltext_results: List[SearchResult],
        query: str,
        **kwargs
    ) -> List[FusionResult]:
        """
        Fuse results by taking maximum normalized score from either search method.
        
        Args:
            vector_results: Results from vector search
            fulltext_results: Results from fulltext search
            query: Original query
            **kwargs: Additional parameters
            
        Returns:
            Fused results
        """
        # Normalize scores
        vector_scores = [r.score for r in vector_results]
        fulltext_scores = [r.score for r in fulltext_results]
        
        norm_vector_scores = self.normalizer.min_max_normalize(vector_scores)
        norm_fulltext_scores = self.normalizer.min_max_normalize(fulltext_scores)
        
        # Create mapping of document_id to scores
        score_map = {}
        
        # Process vector results
        for i, result in enumerate(vector_results):
            doc_id = result.document_id
            score_map[doc_id] = {
                "result": result,
                "vector_score": norm_vector_scores[i],
                "fulltext_score": 0.0,
                "max_source": "vector"
            }
        
        # Process fulltext results
        for i, result in enumerate(fulltext_results):
            doc_id = result.document_id
            fulltext_score = norm_fulltext_scores[i]
            
            if doc_id in score_map:
                if fulltext_score > score_map[doc_id]["vector_score"]:
                    score_map[doc_id]["result"] = result
                    score_map[doc_id]["max_source"] = "fulltext"
                score_map[doc_id]["fulltext_score"] = fulltext_score
            else:
                score_map[doc_id] = {
                    "result": result,
                    "vector_score": 0.0,
                    "fulltext_score": fulltext_score,
                    "max_source": "fulltext"
                }
        
        # Calculate maximum scores
        fusion_results = []
        for doc_id, data in score_map.items():
            result = data["result"]
            
            # Take maximum score
            max_score = max(data["vector_score"], data["fulltext_score"])
            
            # Add EHS-specific boosts
            boost = self.boost_calculator.calculate_total_boost(result, query)
            final_score = max_score + boost
            
            # Create fusion result
            fusion_result = FusionResult(
                content=result.content,
                final_score=final_score,
                metadata=result.metadata,
                document_id=result.document_id,
                chunk_id=result.chunk_id,
                document_type=result.document_type,
                fusion_details={
                    "vector_score": data["vector_score"],
                    "fulltext_score": data["fulltext_score"],
                    "max_score": max_score,
                    "max_source": data["max_source"],
                    "boost": boost,
                    "fusion_method": "maximum_score"
                }
            )
            fusion_results.append(fusion_result)
        
        # Sort by final score
        fusion_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return fusion_results


class FusionStrategyFactory:
    """Factory for creating fusion strategy instances."""
    
    @staticmethod
    def create_strategy(
        method: FusionMethod, 
        config: WeightConfiguration, 
        **kwargs
    ) -> FusionStrategy:
        """
        Create fusion strategy instance.
        
        Args:
            method: Fusion method to use
            config: Weight configuration
            **kwargs: Method-specific parameters
            
        Returns:
            Fusion strategy instance
        """
        if method == FusionMethod.RECIPROCAL_RANK_FUSION:
            rrf_k = kwargs.get("rrf_k", 60.0)
            return ReciprocalRankFusion(config, k=rrf_k)
        elif method == FusionMethod.WEIGHTED_AVERAGE:
            return WeightedAverageFusion(config)
        elif method == FusionMethod.MAXIMUM_SCORE:
            return MaximumScoreFusion(config)
        else:
            raise ValueError(f"Unsupported fusion method: {method}")