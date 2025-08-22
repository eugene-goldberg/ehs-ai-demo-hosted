"""
Configuration for hybrid retrieval strategies in EHS Analytics.

This module provides configuration classes for hybrid search that combines
vector similarity and fulltext search for optimal EHS document retrieval.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

from ..base import QueryType


class SearchStrategy(str, Enum):
    """Search strategy preferences for different query types."""
    
    VECTOR_HEAVY = "vector_heavy"      # 80% vector, 20% fulltext
    BALANCED = "balanced"              # 50% vector, 50% fulltext
    FULLTEXT_HEAVY = "fulltext_heavy"  # 20% vector, 80% fulltext
    ADAPTIVE = "adaptive"              # Dynamic based on query characteristics


class FusionMethod(str, Enum):
    """Methods for combining vector and fulltext search results."""
    
    RECIPROCAL_RANK_FUSION = "rrf"     # Reciprocal Rank Fusion
    WEIGHTED_AVERAGE = "weighted_avg"   # Weighted average of scores
    MAXIMUM_SCORE = "max_score"        # Take maximum score
    LEARNED_FUSION = "learned"         # ML-based fusion (future)


@dataclass
class QueryCharacteristics:
    """Characteristics extracted from a query to determine optimal strategy."""
    
    contains_keywords: bool = False           # Has specific technical terms
    contains_facilities: bool = False         # Mentions facility names
    contains_dates: bool = False             # Has temporal elements
    contains_regulations: bool = False       # References regulations/permits
    contains_measurements: bool = False      # Has numeric values/measurements
    semantic_complexity: float = 0.5        # 0.0 = simple keywords, 1.0 = complex concepts
    query_length: int = 0                   # Number of words
    named_entities_count: int = 0           # Count of named entities


@dataclass
class WeightConfiguration:
    """Weight configuration for hybrid search components."""
    
    vector_weight: float = 0.5              # Weight for vector similarity
    fulltext_weight: float = 0.5            # Weight for fulltext search
    metadata_boost: float = 0.1             # Boost for metadata matches
    recency_boost: float = 0.05             # Boost for recent documents
    facility_boost: float = 0.1             # Boost for facility matches
    compliance_boost: float = 0.15          # Boost for compliance documents
    
    def normalize(self) -> 'WeightConfiguration':
        """Normalize weights to sum to 1.0."""
        total = self.vector_weight + self.fulltext_weight
        if total > 0:
            factor = 1.0 / total
            return WeightConfiguration(
                vector_weight=self.vector_weight * factor,
                fulltext_weight=self.fulltext_weight * factor,
                metadata_boost=self.metadata_boost,
                recency_boost=self.recency_boost,
                facility_boost=self.facility_boost,
                compliance_boost=self.compliance_boost
            )
        return self


@dataclass
class HybridRetrieverConfig:
    """Configuration for hybrid retrieval system."""
    
    # Core search parameters
    max_results: int = 10
    vector_top_k: int = 20                  # Retrieve more for fusion
    fulltext_top_k: int = 20               # Retrieve more for fusion
    
    # Fusion configuration
    fusion_method: FusionMethod = FusionMethod.RECIPROCAL_RANK_FUSION
    rrf_constant: float = 60.0             # RRF k parameter
    
    # Default strategy
    default_strategy: SearchStrategy = SearchStrategy.BALANCED
    
    # Performance tuning
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    parallel_search: bool = True
    timeout_seconds: float = 30.0
    
    # Quality filters
    min_vector_score: float = 0.3
    min_fulltext_score: float = 0.1
    enable_deduplication: bool = True
    dedup_similarity_threshold: float = 0.9
    
    # EHS-specific configuration
    facility_boost_factor: float = 1.2
    compliance_doc_boost: float = 1.3
    recent_doc_boost_days: int = 90
    
    # Query analysis
    enable_query_analysis: bool = True
    keyword_extraction_enabled: bool = True
    named_entity_recognition: bool = True


class HybridConfigurationManager:
    """Manages hybrid search configurations and query-specific optimizations."""
    
    def __init__(self, config: Optional[HybridRetrieverConfig] = None):
        """
        Initialize configuration manager.
        
        Args:
            config: Base configuration for hybrid retrieval
        """
        self.config = config or HybridRetrieverConfig()
        
        # Query type to strategy mapping
        self.query_strategy_map = {
            QueryType.COMPLIANCE: SearchStrategy.FULLTEXT_HEAVY,    # Regulatory terms need exact match
            QueryType.CONSUMPTION: SearchStrategy.BALANCED,         # Mix of keywords and concepts
            QueryType.EFFICIENCY: SearchStrategy.VECTOR_HEAVY,      # Conceptual understanding important
            QueryType.EMISSIONS: SearchStrategy.BALANCED,           # Technical terms + concepts
            QueryType.RISK: SearchStrategy.VECTOR_HEAVY,           # Risk concepts are semantic
            QueryType.RECOMMENDATION: SearchStrategy.VECTOR_HEAVY,  # Conceptual recommendations
            QueryType.GENERAL: SearchStrategy.ADAPTIVE             # Adapt to query characteristics
        }
        
        # Document type relevance weights by query type
        self.doc_type_relevance = {
            QueryType.COMPLIANCE: {
                "permit": 1.0,
                "compliance_report": 0.9,
                "regulatory_document": 0.8,
                "incident_report": 0.6,
                "utility_bill": 0.3
            },
            QueryType.CONSUMPTION: {
                "utility_bill": 1.0,
                "consumption_report": 0.9,
                "equipment_manual": 0.7,
                "facility_report": 0.6,
                "permit": 0.4
            },
            QueryType.EFFICIENCY: {
                "equipment_manual": 1.0,
                "performance_report": 0.9,
                "utility_bill": 0.8,
                "facility_report": 0.7,
                "compliance_report": 0.5
            },
            QueryType.EMISSIONS: {
                "emission_report": 1.0,
                "permit": 0.9,
                "compliance_report": 0.8,
                "facility_report": 0.7,
                "incident_report": 0.6
            },
            QueryType.RISK: {
                "incident_report": 1.0,
                "risk_assessment": 0.9,
                "safety_report": 0.8,
                "compliance_report": 0.7,
                "facility_report": 0.6
            },
            QueryType.RECOMMENDATION: {
                "best_practices": 1.0,
                "performance_report": 0.9,
                "compliance_report": 0.8,
                "facility_report": 0.7,
                "equipment_manual": 0.6
            }
        }
        
        # EHS keyword patterns for query analysis
        self.ehs_keyword_patterns = {
            "facilities": [
                "plant", "facility", "site", "location", "building",
                "factory", "warehouse", "office", "campus"
            ],
            "regulations": [
                "permit", "license", "regulation", "compliance", "EPA",
                "OSHA", "standard", "requirement", "code", "law"
            ],
            "utilities": [
                "water", "electricity", "gas", "energy", "power",
                "consumption", "usage", "bill", "meter", "kwh"
            ],
            "emissions": [
                "emission", "carbon", "co2", "pollution", "discharge",
                "waste", "effluent", "air quality", "greenhouse gas"
            ],
            "safety": [
                "incident", "accident", "injury", "safety", "hazard",
                "risk", "emergency", "spill", "exposure"
            ],
            "equipment": [
                "equipment", "machinery", "system", "device", "instrument",
                "monitor", "sensor", "pump", "tank", "boiler"
            ]
        }
    
    def analyze_query(self, query: str) -> QueryCharacteristics:
        """
        Analyze query characteristics to determine optimal search strategy.
        
        Args:
            query: Natural language query
            
        Returns:
            QueryCharacteristics with extracted features
        """
        query_lower = query.lower()
        words = query_lower.split()
        
        characteristics = QueryCharacteristics()
        characteristics.query_length = len(words)
        
        # Check for keyword patterns
        keyword_matches = 0
        for category, keywords in self.ehs_keyword_patterns.items():
            matches = sum(1 for keyword in keywords if keyword in query_lower)
            keyword_matches += matches
            
            if category == "facilities" and matches > 0:
                characteristics.contains_facilities = True
            elif category == "regulations" and matches > 0:
                characteristics.contains_regulations = True
        
        characteristics.contains_keywords = keyword_matches > 0
        
        # Check for dates (simple pattern matching)
        date_patterns = ["2023", "2024", "2025", "january", "february", "march", 
                        "april", "may", "june", "july", "august", "september",
                        "october", "november", "december", "last month", "this year"]
        characteristics.contains_dates = any(pattern in query_lower for pattern in date_patterns)
        
        # Check for measurements (simple numeric pattern)
        import re
        numeric_pattern = r'\d+\.?\d*\s*(kwh|gallon|cubic|ton|ppm|percent|%|degree)'
        characteristics.contains_measurements = bool(re.search(numeric_pattern, query_lower))
        
        # Simple named entity count (proper nouns)
        characteristics.named_entities_count = sum(1 for word in query.split() if word[0].isupper())
        
        # Semantic complexity heuristic
        if len(words) > 10:
            characteristics.semantic_complexity += 0.3
        if any(word in query_lower for word in ["how", "why", "what", "when", "where"]):
            characteristics.semantic_complexity += 0.2
        if characteristics.contains_keywords:
            characteristics.semantic_complexity -= 0.1
        
        characteristics.semantic_complexity = max(0.0, min(1.0, characteristics.semantic_complexity))
        
        return characteristics
    
    def get_search_strategy(self, query_type: QueryType, query: str) -> SearchStrategy:
        """
        Determine optimal search strategy for given query type and content.
        
        Args:
            query_type: Type of EHS query
            query: Query content for analysis
            
        Returns:
            Recommended search strategy
        """
        base_strategy = self.query_strategy_map.get(query_type, SearchStrategy.BALANCED)
        
        if base_strategy != SearchStrategy.ADAPTIVE:
            return base_strategy
        
        # Adaptive strategy based on query characteristics
        characteristics = self.analyze_query(query)
        
        if characteristics.contains_regulations or characteristics.contains_facilities:
            return SearchStrategy.FULLTEXT_HEAVY
        elif characteristics.semantic_complexity > 0.7:
            return SearchStrategy.VECTOR_HEAVY
        elif characteristics.contains_keywords and characteristics.semantic_complexity < 0.3:
            return SearchStrategy.FULLTEXT_HEAVY
        else:
            return SearchStrategy.BALANCED
    
    def get_weight_configuration(
        self, 
        strategy: SearchStrategy, 
        query_type: QueryType,
        characteristics: Optional[QueryCharacteristics] = None
    ) -> WeightConfiguration:
        """
        Get weight configuration for given strategy and query type.
        
        Args:
            strategy: Search strategy to use
            query_type: Type of EHS query
            characteristics: Query characteristics for fine-tuning
            
        Returns:
            Configured weights for hybrid search
        """
        # Base weights by strategy
        base_weights = {
            SearchStrategy.VECTOR_HEAVY: WeightConfiguration(0.8, 0.2),
            SearchStrategy.BALANCED: WeightConfiguration(0.5, 0.5),
            SearchStrategy.FULLTEXT_HEAVY: WeightConfiguration(0.2, 0.8),
            SearchStrategy.ADAPTIVE: WeightConfiguration(0.5, 0.5)
        }
        
        weights = base_weights[strategy]
        
        # Apply query type specific adjustments
        if query_type == QueryType.COMPLIANCE:
            weights.compliance_boost = 0.2
            weights.facility_boost = 0.15
        elif query_type == QueryType.CONSUMPTION:
            weights.recency_boost = 0.1
            weights.facility_boost = 0.15
        elif query_type == QueryType.RISK:
            weights.metadata_boost = 0.15
        
        # Apply characteristics-based fine-tuning
        if characteristics:
            if characteristics.contains_facilities:
                weights.facility_boost = min(0.2, weights.facility_boost + 0.05)
            if characteristics.contains_regulations:
                weights.compliance_boost = min(0.25, weights.compliance_boost + 0.05)
        
        return weights.normalize()
    
    def get_document_type_boosts(self, query_type: QueryType) -> Dict[str, float]:
        """
        Get document type relevance boosts for query type.
        
        Args:
            query_type: Type of EHS query
            
        Returns:
            Dictionary mapping document types to boost factors
        """
        return self.doc_type_relevance.get(query_type, {})
    
    def should_use_parallel_search(self, query_complexity: float = 0.5) -> bool:
        """
        Determine if parallel search should be used based on query complexity.
        
        Args:
            query_complexity: Complexity score (0.0 to 1.0)
            
        Returns:
            True if parallel search is recommended
        """
        return self.config.parallel_search and query_complexity > 0.3
    
    def get_cache_key(
        self, 
        query: str, 
        query_type: QueryType, 
        strategy: SearchStrategy,
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate cache key for hybrid search results.
        
        Args:
            query: Search query
            query_type: Type of query
            strategy: Search strategy used
            filters: Additional filters applied
            
        Returns:
            Cache key string
        """
        import hashlib
        
        # Create deterministic hash from parameters
        cache_data = f"{query}|{query_type.value}|{strategy.value}"
        if filters:
            # Sort filters for consistent hashing
            sorted_filters = sorted(filters.items()) if isinstance(filters, dict) else []
            cache_data += f"|{sorted_filters}"
        
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    def update_strategy_performance(
        self, 
        query_type: QueryType, 
        strategy: SearchStrategy,
        performance_score: float
    ):
        """
        Update strategy performance metrics for adaptive learning.
        
        Args:
            query_type: Type of query
            strategy: Strategy that was used
            performance_score: Performance score (0.0 to 1.0)
        """
        # TODO: Implement performance tracking and adaptive strategy adjustment
        # This could use a simple moving average or more sophisticated ML
        pass