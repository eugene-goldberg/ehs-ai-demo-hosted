"""
EHS-specific graph traversal patterns and relationship templates.

This module provides graph traversal patterns, relationship templates, and
path scoring utilities for relationship-aware search in EHS data.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .vector_cypher_config import RelationshipType, TraversalDepth, RelationshipWeight
from .ehs_vector_config import QueryType, FacilityType, DocumentType

logger = logging.getLogger(__name__)


@dataclass
class GraphPath:
    """Represents a path through the graph."""
    nodes: List[str]  # Node IDs in the path
    relationships: List[RelationshipType]  # Relationship types
    scores: List[float]  # Score at each step
    total_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraversalPattern:
    """Template for graph traversal patterns."""
    name: str
    description: str
    start_node_types: List[str]
    relationship_sequence: List[RelationshipType]
    end_node_types: List[str]
    cypher_template: str
    max_depth: int = 3
    bidirectional: bool = False
    temporal_constraints: Optional[Dict[str, Any]] = None


@dataclass
class ContextAggregationStrategy:
    """Strategy for aggregating context from related nodes."""
    name: str
    node_weight_function: str  # "distance_decay", "relationship_weight", "combined"
    content_selection: str     # "all", "recent", "relevant"
    max_content_length: int = 2000
    include_metadata: bool = True
    temporal_weighting: bool = True


class EHSGraphPatterns:
    """EHS-specific graph traversal patterns."""
    
    @staticmethod
    def get_equipment_facility_pattern() -> TraversalPattern:
        """Pattern for finding equipment at specific facilities."""
        return TraversalPattern(
            name="equipment_facility",
            description="Find equipment located at facilities",
            start_node_types=["Equipment"],
            relationship_sequence=[RelationshipType.LOCATED_AT],
            end_node_types=["Facility"],
            cypher_template="""
            MATCH (e:Equipment)-[:LOCATED_AT]->(f:Facility)
            WHERE e.equipment_id IN $equipment_ids
            RETURN e, f, score
            """,
            max_depth=1,
            bidirectional=True
        )
    
    @staticmethod
    def get_permit_facility_equipment_pattern() -> TraversalPattern:
        """Pattern for finding equipment through permit-facility relationships."""
        return TraversalPattern(
            name="permit_facility_equipment",
            description="Find equipment via permits covering facilities",
            start_node_types=["Permit"],
            relationship_sequence=[
                RelationshipType.COVERS_FACILITY,
                RelationshipType.LOCATED_AT
            ],
            end_node_types=["Equipment"],
            cypher_template="""
            MATCH (p:Permit)-[:COVERS_FACILITY]->(f:Facility)<-[:LOCATED_AT]-(e:Equipment)
            WHERE p.permit_id IN $permit_ids
            AND p.status = 'active'
            RETURN p, f, e, score
            ORDER BY p.expiration_date ASC
            """,
            max_depth=2,
            bidirectional=False,
            temporal_constraints={"expiration_filter": True}
        )
    
    @staticmethod
    def get_incident_equipment_facility_pattern() -> TraversalPattern:
        """Pattern for finding incidents related to equipment and facilities."""
        return TraversalPattern(
            name="incident_equipment_facility",
            description="Find incidents involving equipment at facilities",
            start_node_types=["Incident"],
            relationship_sequence=[
                RelationshipType.HAS_INCIDENT,
                RelationshipType.LOCATED_AT
            ],
            end_node_types=["Facility"],
            cypher_template="""
            MATCH (i:Incident)<-[:HAS_INCIDENT]-(e:Equipment)-[:LOCATED_AT]->(f:Facility)
            WHERE i.incident_id IN $incident_ids
            AND i.date >= $start_date
            RETURN i, e, f, score
            ORDER BY i.date DESC
            """,
            max_depth=2,
            bidirectional=False,
            temporal_constraints={"date_range": True}
        )
    
    @staticmethod
    def get_document_context_pattern() -> TraversalPattern:
        """Pattern for finding documents and their context."""
        return TraversalPattern(
            name="document_context",
            description="Find documents and their related entities",
            start_node_types=["Document"],
            relationship_sequence=[
                RelationshipType.DOCUMENTS,
                RelationshipType.LOCATED_AT
            ],
            end_node_types=["Facility", "Equipment"],
            cypher_template="""
            MATCH (d:Document)-[:DOCUMENTS]->(entity)-[:LOCATED_AT*0..1]->(location)
            WHERE d.document_id IN $document_ids
            RETURN d, entity, location, score
            """,
            max_depth=2,
            bidirectional=True
        )
    
    @staticmethod
    def get_compliance_chain_pattern() -> TraversalPattern:
        """Pattern for tracing compliance relationships."""
        return TraversalPattern(
            name="compliance_chain",
            description="Trace compliance through permits and regulations",
            start_node_types=["Permit"],
            relationship_sequence=[
                RelationshipType.COVERS_FACILITY,
                RelationshipType.COMPLIES_WITH
            ],
            end_node_types=["Regulation"],
            cypher_template="""
            MATCH (p:Permit)-[:COVERS_FACILITY]->(f:Facility)-[:COMPLIES_WITH]->(r:Regulation)
            WHERE p.permit_id IN $permit_ids
            OPTIONAL MATCH (f)-[:VIOLATES]->(v:Regulation)
            RETURN p, f, r, v, score
            """,
            max_depth=2,
            bidirectional=False
        )


class PathScorer:
    """Scores graph paths based on relationship weights and context."""
    
    def __init__(self, relationship_weights: Dict[RelationshipType, RelationshipWeight]):
        """Initialize with relationship weights."""
        self.relationship_weights = relationship_weights
        
    def score_path(self, path: GraphPath, query_context: Optional[Dict[str, Any]] = None) -> float:
        """Score a graph traversal path."""
        if not path.relationships:
            return 0.0
            
        total_score = 1.0
        path_length = len(path.relationships)
        
        # Score each relationship in the path
        for i, rel_type in enumerate(path.relationships):
            if rel_type in self.relationship_weights:
                weight_config = self.relationship_weights[rel_type]
                
                # Base relationship weight
                rel_score = weight_config.weight
                
                # Apply decay based on position in path
                decay = weight_config.decay_factor ** i
                rel_score *= decay
                
                # Apply temporal boost if applicable
                if query_context and "temporal_focus" in query_context:
                    rel_score *= weight_config.temporal_boost
                
                total_score *= rel_score
            else:
                # Unknown relationship type - apply penalty
                total_score *= 0.5
        
        # Apply distance penalty
        distance_penalty = 0.9 ** (path_length - 1)
        total_score *= distance_penalty
        
        # Apply diversity bonus for varied relationship types
        unique_rels = len(set(path.relationships))
        if unique_rels > 1:
            diversity_bonus = 1.0 + (0.1 * (unique_rels - 1))
            total_score *= diversity_bonus
        
        return min(total_score, 1.0)  # Cap at 1.0
    
    def rank_paths(self, paths: List[GraphPath], 
                   query_context: Optional[Dict[str, Any]] = None) -> List[GraphPath]:
        """Rank paths by score."""
        for path in paths:
            path.total_score = self.score_path(path, query_context)
        
        return sorted(paths, key=lambda p: p.total_score, reverse=True)


class ContextAggregator:
    """Aggregates context from related nodes."""
    
    def __init__(self, strategy: ContextAggregationStrategy):
        """Initialize with aggregation strategy."""
        self.strategy = strategy
        
    def aggregate_context(self, paths: List[GraphPath], 
                         node_contents: Dict[str, str],
                         node_metadata: Dict[str, Dict[str, Any]]) -> str:
        """Aggregate context from paths."""
        context_parts = []
        total_length = 0
        
        # Sort paths by score
        sorted_paths = sorted(paths, key=lambda p: p.total_score, reverse=True)
        
        for path in sorted_paths:
            if total_length >= self.strategy.max_content_length:
                break
                
            # Process each node in the path
            for i, node_id in enumerate(path.nodes):
                if node_id not in node_contents:
                    continue
                    
                content = node_contents[node_id]
                metadata = node_metadata.get(node_id, {})
                
                # Apply content selection strategy
                if self._should_include_content(content, metadata, path, i):
                    # Apply weighting
                    weight = self._calculate_content_weight(path, i, metadata)
                    
                    # Format content with weight indicator
                    weighted_content = f"[Weight: {weight:.2f}] {content}"
                    
                    # Check length limit
                    if total_length + len(weighted_content) > self.strategy.max_content_length:
                        remaining = self.strategy.max_content_length - total_length
                        weighted_content = weighted_content[:remaining] + "..."
                        context_parts.append(weighted_content)
                        break
                    
                    context_parts.append(weighted_content)
                    total_length += len(weighted_content)
        
        return "\n\n".join(context_parts)
    
    def _should_include_content(self, content: str, metadata: Dict[str, Any], 
                               path: GraphPath, position: int) -> bool:
        """Determine if content should be included."""
        if self.strategy.content_selection == "all":
            return True
        elif self.strategy.content_selection == "recent":
            # Include if recent (within last year)
            if "created_date" in metadata:
                try:
                    created_date = datetime.fromisoformat(metadata["created_date"])
                    cutoff = datetime.now() - timedelta(days=365)
                    return created_date >= cutoff
                except (ValueError, TypeError):
                    return True  # Include if date parsing fails
            return True
        elif self.strategy.content_selection == "relevant":
            # Include if above relevance threshold
            return path.total_score > 0.3
        
        return True
    
    def _calculate_content_weight(self, path: GraphPath, position: int, 
                                 metadata: Dict[str, Any]) -> float:
        """Calculate weight for content based on position and metadata."""
        if self.strategy.node_weight_function == "distance_decay":
            return 1.0 / (position + 1)
        elif self.strategy.node_weight_function == "relationship_weight":
            if position < len(path.scores):
                return path.scores[position]
            return 0.5
        elif self.strategy.node_weight_function == "combined":
            distance_weight = 1.0 / (position + 1)
            relationship_weight = path.scores[position] if position < len(path.scores) else 0.5
            
            # Apply temporal weighting if enabled
            temporal_weight = 1.0
            if self.strategy.temporal_weighting and "created_date" in metadata:
                try:
                    created_date = datetime.fromisoformat(metadata["created_date"])
                    days_old = (datetime.now() - created_date).days
                    temporal_weight = max(0.1, 1.0 - (days_old / 365.0 * 0.5))
                except (ValueError, TypeError):
                    pass
            
            return distance_weight * relationship_weight * temporal_weight
        
        return 1.0


class TemporalPatternMatcher:
    """Handles temporal relationships and patterns."""
    
    @staticmethod
    def get_temporal_sequence_pattern(event_type: str) -> TraversalPattern:
        """Get pattern for temporal event sequences."""
        return TraversalPattern(
            name=f"temporal_sequence_{event_type}",
            description=f"Find temporal sequences of {event_type} events",
            start_node_types=[event_type],
            relationship_sequence=[RelationshipType.PRECEDES],
            end_node_types=[event_type],
            cypher_template=f"""
            MATCH (e1:{event_type})-[:PRECEDES*1..3]->(e2:{event_type})
            WHERE e1.event_id IN $event_ids
            AND e1.date <= e2.date
            RETURN e1, e2, score
            ORDER BY e1.date, e2.date
            """,
            max_depth=3,
            bidirectional=False,
            temporal_constraints={"sequence_order": True}
        )
    
    @staticmethod
    def get_concurrent_events_pattern(event_type: str) -> TraversalPattern:
        """Get pattern for concurrent events."""
        return TraversalPattern(
            name=f"concurrent_{event_type}",
            description=f"Find concurrent {event_type} events",
            start_node_types=[event_type],
            relationship_sequence=[RelationshipType.CONCURRENT_WITH],
            end_node_types=[event_type],
            cypher_template=f"""
            MATCH (e1:{event_type})-[:CONCURRENT_WITH]-(e2:{event_type})
            WHERE e1.event_id IN $event_ids
            AND abs(duration.between(e1.date, e2.date).days) <= 7
            RETURN e1, e2, score
            """,
            max_depth=1,
            bidirectional=True,
            temporal_constraints={"concurrency_window": 7}
        )


class EHSPatternLibrary:
    """Library of EHS-specific graph patterns."""
    
    def __init__(self):
        """Initialize pattern library."""
        self.patterns = self._load_patterns()
        self.scorers = {}
        self.aggregators = {}
        
    def _load_patterns(self) -> Dict[str, TraversalPattern]:
        """Load all EHS patterns."""
        patterns = {}
        
        # Equipment and facility patterns
        patterns["equipment_facility"] = EHSGraphPatterns.get_equipment_facility_pattern()
        patterns["permit_facility_equipment"] = EHSGraphPatterns.get_permit_facility_equipment_pattern()
        patterns["incident_equipment_facility"] = EHSGraphPatterns.get_incident_equipment_facility_pattern()
        patterns["document_context"] = EHSGraphPatterns.get_document_context_pattern()
        patterns["compliance_chain"] = EHSGraphPatterns.get_compliance_chain_pattern()
        
        # Temporal patterns
        patterns["incident_sequence"] = TemporalPatternMatcher.get_temporal_sequence_pattern("Incident")
        patterns["inspection_sequence"] = TemporalPatternMatcher.get_temporal_sequence_pattern("Inspection")
        patterns["concurrent_incidents"] = TemporalPatternMatcher.get_concurrent_events_pattern("Incident")
        
        return patterns
    
    def get_patterns_for_query_type(self, query_type: QueryType) -> List[TraversalPattern]:
        """Get relevant patterns for a query type."""
        pattern_mapping = {
            QueryType.EQUIPMENT_STATUS: [
                "equipment_facility",
                "document_context"
            ],
            QueryType.PERMIT_COMPLIANCE: [
                "permit_facility_equipment",
                "compliance_chain"
            ],
            QueryType.INCIDENT_ANALYSIS: [
                "incident_equipment_facility",
                "incident_sequence",
                "concurrent_incidents"
            ],
            QueryType.SAFETY_ANALYSIS: [
                "incident_equipment_facility",
                "incident_sequence",
                "equipment_facility"
            ],
            QueryType.ENVIRONMENTAL_IMPACT: [
                "permit_facility_equipment",
                "compliance_chain",
                "document_context"
            ],
            QueryType.COST_ANALYSIS: [
                "equipment_facility",
                "document_context"
            ],
            QueryType.PREDICTIVE_ANALYSIS: [
                "incident_sequence",
                "inspection_sequence",
                "concurrent_incidents"
            ]
        }
        
        pattern_names = pattern_mapping.get(query_type, [])
        return [self.patterns[name] for name in pattern_names if name in self.patterns]
    
    def get_pattern(self, pattern_name: str) -> Optional[TraversalPattern]:
        """Get a specific pattern by name."""
        return self.patterns.get(pattern_name)
    
    def create_path_scorer(self, relationship_weights: Dict[RelationshipType, RelationshipWeight]) -> PathScorer:
        """Create a path scorer with given weights."""
        return PathScorer(relationship_weights)
    
    def create_context_aggregator(self, strategy_name: str = "balanced") -> ContextAggregator:
        """Create a context aggregator with predefined strategy."""
        strategies = {
            "fast": ContextAggregationStrategy(
                name="fast",
                node_weight_function="distance_decay",
                content_selection="relevant",
                max_content_length=1000,
                include_metadata=False,
                temporal_weighting=False
            ),
            "balanced": ContextAggregationStrategy(
                name="balanced",
                node_weight_function="combined",
                content_selection="recent",
                max_content_length=2000,
                include_metadata=True,
                temporal_weighting=True
            ),
            "comprehensive": ContextAggregationStrategy(
                name="comprehensive",
                node_weight_function="combined",
                content_selection="all",
                max_content_length=4000,
                include_metadata=True,
                temporal_weighting=True
            )
        }
        
        strategy = strategies.get(strategy_name, strategies["balanced"])
        return ContextAggregator(strategy)