"""
Configuration for VectorCypher retrieval strategy.

This module provides configuration classes for relationship-aware vector search
that combines vector similarity with graph traversal patterns for EHS data.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from .ehs_vector_config import DocumentType, QueryType, FacilityType

logger = logging.getLogger(__name__)


class RelationshipType(Enum):
    """Types of relationships in the EHS graph."""
    # Equipment relationships
    LOCATED_AT = "LOCATED_AT"           # Equipment -> Facility
    MAINTAINS = "MAINTAINS"             # Organization -> Equipment
    INSPECTED_BY = "INSPECTED_BY"       # Equipment -> Inspector
    HAS_INCIDENT = "HAS_INCIDENT"       # Equipment -> Incident
    
    # Permit relationships
    COVERS_FACILITY = "COVERS_FACILITY" # Permit -> Facility
    COVERS_EQUIPMENT = "COVERS_EQUIPMENT" # Permit -> Equipment
    ISSUED_BY = "ISSUED_BY"             # Permit -> Authority
    REQUIRES_INSPECTION = "REQUIRES_INSPECTION" # Permit -> Inspection
    
    # Document relationships
    DOCUMENTS = "DOCUMENTS"             # Document -> Equipment/Facility
    REFERENCES = "REFERENCES"           # Document -> Other Document
    CREATED_FOR = "CREATED_FOR"         # Document -> Incident/Inspection
    
    # Temporal relationships
    PRECEDES = "PRECEDES"              # Event -> Event
    FOLLOWS = "FOLLOWS"                # Event -> Event
    CONCURRENT_WITH = "CONCURRENT_WITH" # Event -> Event
    
    # Compliance relationships
    COMPLIES_WITH = "COMPLIES_WITH"    # Entity -> Regulation
    VIOLATES = "VIOLATES"              # Entity -> Regulation
    EXEMPTED_FROM = "EXEMPTED_FROM"    # Entity -> Regulation


class TraversalDepth(Enum):
    """Traversal depth limits for different query types."""
    DIRECT = 1          # Only direct relationships
    LOCAL = 2           # Local neighborhood (2 hops)
    EXTENDED = 3        # Extended context (3 hops)
    COMPREHENSIVE = 4   # Comprehensive search (4 hops)


@dataclass
class RelationshipWeight:
    """Weight configuration for relationship types."""
    weight: float = 1.0
    decay_factor: float = 0.8  # How much weight decreases per hop
    max_traversal_depth: int = 3
    bidirectional: bool = True
    temporal_boost: float = 1.0  # Boost for recent relationships


@dataclass
class PathScoringConfig:
    """Configuration for scoring traversal paths."""
    relationship_weights: Dict[RelationshipType, RelationshipWeight]
    distance_penalty: float = 0.2  # Penalty per hop
    diversity_bonus: float = 0.1   # Bonus for diverse relationship types
    temporal_decay: float = 0.95   # Daily decay for older relationships
    min_path_score: float = 0.1    # Minimum score to include path


@dataclass
class ContextExpansionConfig:
    """Configuration for expanding context through relationships."""
    max_context_nodes: int = 50
    context_similarity_threshold: float = 0.6
    relationship_type_limits: Dict[RelationshipType, int] = field(default_factory=dict)
    prefer_recent: bool = True
    temporal_window_days: Optional[int] = 365  # Only include relationships within window


@dataclass
class VectorCypherConfig:
    """Configuration for VectorCypher retriever."""
    
    # Vector search configuration
    vector_similarity_threshold: float = 0.7
    max_vector_results: int = 20
    vector_weight: float = 0.6
    
    # Graph traversal configuration
    graph_weight: float = 0.4
    max_traversal_depth: TraversalDepth = TraversalDepth.EXTENDED
    path_scoring_config: PathScoringConfig = field(default_factory=lambda: PathScoringConfig(
        relationship_weights=get_default_relationship_weights(),
        distance_penalty=0.2,
        diversity_bonus=0.1,
        temporal_decay=0.95,
        min_path_score=0.1
    ))
    
    # Context expansion configuration
    context_expansion: ContextExpansionConfig = field(default_factory=lambda: ContextExpansionConfig(
        max_context_nodes=50,
        context_similarity_threshold=0.6,
        relationship_type_limits={
            RelationshipType.LOCATED_AT: 10,
            RelationshipType.COVERS_FACILITY: 15,
            RelationshipType.DOCUMENTS: 20,
            RelationshipType.HAS_INCIDENT: 8
        },
        prefer_recent=True,
        temporal_window_days=365
    ))
    
    # Performance optimization
    max_total_results: int = 50
    enable_result_caching: bool = True
    cache_ttl_seconds: int = 300
    parallel_search: bool = True
    batch_size: int = 10
    
    # Query-specific configurations
    query_type_configs: Dict[QueryType, 'QuerySpecificConfig'] = field(default_factory=dict)


@dataclass 
class QuerySpecificConfig:
    """Query-type specific configuration overrides."""
    relationship_priorities: List[RelationshipType]
    max_depth: TraversalDepth
    context_focus: List[str]  # Node types to focus on
    temporal_importance: float = 1.0
    facility_scope: Optional[List[FacilityType]] = None


def get_default_relationship_weights() -> Dict[RelationshipType, RelationshipWeight]:
    """Get default relationship weights for EHS domain."""
    return {
        # High priority relationships
        RelationshipType.LOCATED_AT: RelationshipWeight(
            weight=1.0, decay_factor=0.9, max_traversal_depth=2, temporal_boost=1.0
        ),
        RelationshipType.COVERS_FACILITY: RelationshipWeight(
            weight=0.95, decay_factor=0.85, max_traversal_depth=2, temporal_boost=1.2
        ),
        RelationshipType.HAS_INCIDENT: RelationshipWeight(
            weight=0.9, decay_factor=0.8, max_traversal_depth=3, temporal_boost=1.5
        ),
        RelationshipType.DOCUMENTS: RelationshipWeight(
            weight=0.85, decay_factor=0.85, max_traversal_depth=2, temporal_boost=1.1
        ),
        
        # Medium priority relationships
        RelationshipType.COVERS_EQUIPMENT: RelationshipWeight(
            weight=0.8, decay_factor=0.8, max_traversal_depth=3, temporal_boost=1.1
        ),
        RelationshipType.MAINTAINS: RelationshipWeight(
            weight=0.75, decay_factor=0.75, max_traversal_depth=2, temporal_boost=1.0
        ),
        RelationshipType.INSPECTED_BY: RelationshipWeight(
            weight=0.7, decay_factor=0.8, max_traversal_depth=2, temporal_boost=1.3
        ),
        RelationshipType.REQUIRES_INSPECTION: RelationshipWeight(
            weight=0.7, decay_factor=0.8, max_traversal_depth=2, temporal_boost=1.2
        ),
        
        # Lower priority relationships
        RelationshipType.REFERENCES: RelationshipWeight(
            weight=0.6, decay_factor=0.7, max_traversal_depth=3, temporal_boost=0.9
        ),
        RelationshipType.CREATED_FOR: RelationshipWeight(
            weight=0.65, decay_factor=0.75, max_traversal_depth=2, temporal_boost=1.1
        ),
        RelationshipType.ISSUED_BY: RelationshipWeight(
            weight=0.5, decay_factor=0.8, max_traversal_depth=2, temporal_boost=0.8
        ),
        
        # Temporal relationships
        RelationshipType.PRECEDES: RelationshipWeight(
            weight=0.4, decay_factor=0.6, max_traversal_depth=4, temporal_boost=0.7
        ),
        RelationshipType.FOLLOWS: RelationshipWeight(
            weight=0.4, decay_factor=0.6, max_traversal_depth=4, temporal_boost=0.7
        ),
        RelationshipType.CONCURRENT_WITH: RelationshipWeight(
            weight=0.3, decay_factor=0.7, max_traversal_depth=3, temporal_boost=0.8
        ),
        
        # Compliance relationships
        RelationshipType.COMPLIES_WITH: RelationshipWeight(
            weight=0.8, decay_factor=0.8, max_traversal_depth=2, temporal_boost=1.1
        ),
        RelationshipType.VIOLATES: RelationshipWeight(
            weight=0.9, decay_factor=0.8, max_traversal_depth=2, temporal_boost=1.4
        ),
        RelationshipType.EXEMPTED_FROM: RelationshipWeight(
            weight=0.6, decay_factor=0.8, max_traversal_depth=2, temporal_boost=0.9
        ),
    }


def get_query_specific_configs() -> Dict[QueryType, QuerySpecificConfig]:
    """Get query-type specific configuration overrides."""
    return {
        QueryType.EQUIPMENT_STATUS: QuerySpecificConfig(
            relationship_priorities=[
                RelationshipType.LOCATED_AT,
                RelationshipType.MAINTAINS,
                RelationshipType.INSPECTED_BY,
                RelationshipType.DOCUMENTS
            ],
            max_depth=TraversalDepth.LOCAL,
            context_focus=["Equipment", "Facility", "Inspection"],
            temporal_importance=1.2,
            facility_scope=[FacilityType.MANUFACTURING, FacilityType.WAREHOUSE]
        ),
        
        QueryType.PERMIT_COMPLIANCE: QuerySpecificConfig(
            relationship_priorities=[
                RelationshipType.COVERS_FACILITY,
                RelationshipType.COVERS_EQUIPMENT,
                RelationshipType.ISSUED_BY,
                RelationshipType.COMPLIES_WITH
            ],
            max_depth=TraversalDepth.EXTENDED,
            context_focus=["Permit", "Facility", "Equipment", "Regulation"],
            temporal_importance=1.5,
            facility_scope=None
        ),
        
        QueryType.RISK: QuerySpecificConfig(
            relationship_priorities=[
                RelationshipType.HAS_INCIDENT,
                RelationshipType.LOCATED_AT,
                RelationshipType.DOCUMENTS,
                RelationshipType.PRECEDES
            ],
            max_depth=TraversalDepth.COMPREHENSIVE,
            context_focus=["Incident", "Equipment", "Facility", "Document"],
            temporal_importance=1.8,
            facility_scope=None
        ),
        
        QueryType.SAFETY_ANALYSIS: QuerySpecificConfig(
            relationship_priorities=[
                RelationshipType.HAS_INCIDENT,
                RelationshipType.VIOLATES,
                RelationshipType.INSPECTED_BY,
                RelationshipType.COVERS_EQUIPMENT
            ],
            max_depth=TraversalDepth.EXTENDED,
            context_focus=["Incident", "Inspection", "Violation", "Equipment"],
            temporal_importance=1.6,
            facility_scope=None
        ),
        
        QueryType.ENVIRONMENTAL_IMPACT: QuerySpecificConfig(
            relationship_priorities=[
                RelationshipType.COVERS_FACILITY,
                RelationshipType.DOCUMENTS,
                RelationshipType.COMPLIES_WITH,
                RelationshipType.LOCATED_AT
            ],
            max_depth=TraversalDepth.EXTENDED,
            context_focus=["Facility", "Document", "Permit", "Regulation"],
            temporal_importance=1.3,
            facility_scope=[FacilityType.MANUFACTURING, FacilityType.CHEMICAL_PLANT]
        ),
        
        QueryType.COST_ANALYSIS: QuerySpecificConfig(
            relationship_priorities=[
                RelationshipType.DOCUMENTS,
                RelationshipType.LOCATED_AT,
                RelationshipType.MAINTAINS,
                RelationshipType.COVERS_FACILITY
            ],
            max_depth=TraversalDepth.LOCAL,
            context_focus=["Document", "Facility", "Equipment", "Cost"],
            temporal_importance=1.1,
            facility_scope=None
        ),
        
        QueryType.PREDICTIVE_ANALYSIS: QuerySpecificConfig(
            relationship_priorities=[
                RelationshipType.PRECEDES,
                RelationshipType.HAS_INCIDENT,
                RelationshipType.INSPECTED_BY,
                RelationshipType.DOCUMENTS
            ],
            max_depth=TraversalDepth.COMPREHENSIVE,
            context_focus=["Incident", "Inspection", "Equipment", "Pattern"],
            temporal_importance=2.0,
            facility_scope=None
        )
    }


class VectorCypherConfigManager:
    """Manager for VectorCypher configurations."""
    
    def __init__(self, base_config: Optional[VectorCypherConfig] = None):
        """Initialize with base configuration."""
        self.base_config = base_config or VectorCypherConfig()
        self.query_configs = get_query_specific_configs()
        
    def get_config_for_query(self, query_type: QueryType) -> VectorCypherConfig:
        """Get configuration optimized for specific query type."""
        config = VectorCypherConfig(
            # Copy base config values
            vector_similarity_threshold=self.base_config.vector_similarity_threshold,
            max_vector_results=self.base_config.max_vector_results,
            vector_weight=self.base_config.vector_weight,
            graph_weight=self.base_config.graph_weight,
            max_traversal_depth=self.base_config.max_traversal_depth,
            path_scoring_config=self.base_config.path_scoring_config,
            context_expansion=self.base_config.context_expansion,
            max_total_results=self.base_config.max_total_results,
            enable_result_caching=self.base_config.enable_result_caching,
            cache_ttl_seconds=self.base_config.cache_ttl_seconds,
            parallel_search=self.base_config.parallel_search,
            batch_size=self.base_config.batch_size,
            query_type_configs=self.base_config.query_type_configs
        )
        
        # Apply query-specific overrides
        if query_type in self.query_configs:
            query_config = self.query_configs[query_type]
            config.max_traversal_depth = query_config.max_depth
            
            # Adjust relationship weights based on priorities
            for i, rel_type in enumerate(query_config.relationship_priorities):
                if rel_type in config.path_scoring_config.relationship_weights:
                    # Boost priority relationships
                    weight_config = config.path_scoring_config.relationship_weights[rel_type]
                    priority_boost = 1.0 + (0.1 * (len(query_config.relationship_priorities) - i))
                    weight_config.weight *= priority_boost
                    weight_config.temporal_boost *= query_config.temporal_importance
        
        return config
    
    def update_relationship_weights(self, 
                                   relationship_weights: Dict[RelationshipType, RelationshipWeight]):
        """Update relationship weights in base configuration."""
        self.base_config.path_scoring_config.relationship_weights.update(relationship_weights)
    
    def set_performance_profile(self, profile: str):
        """Set performance profile (fast, balanced, comprehensive)."""
        if profile == "fast":
            self.base_config.max_vector_results = 10
            self.base_config.max_traversal_depth = TraversalDepth.DIRECT
            self.base_config.context_expansion.max_context_nodes = 20
            self.base_config.parallel_search = True
            
        elif profile == "balanced":
            self.base_config.max_vector_results = 20
            self.base_config.max_traversal_depth = TraversalDepth.LOCAL
            self.base_config.context_expansion.max_context_nodes = 50
            self.base_config.parallel_search = True
            
        elif profile == "comprehensive":
            self.base_config.max_vector_results = 50
            self.base_config.max_traversal_depth = TraversalDepth.COMPREHENSIVE
            self.base_config.context_expansion.max_context_nodes = 100
            self.base_config.parallel_search = True
            
        else:
            raise ValueError(f"Unknown performance profile: {profile}")