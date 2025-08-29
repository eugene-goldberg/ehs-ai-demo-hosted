"""
Neo4j Index Definitions for EHS AI Platform

This module provides comprehensive index definitions for optimal query performance
across all enhanced node types and common query patterns in the EHS AI system.

The indexes are organized by:
1. Performance indexes for individual node properties
2. Composite indexes for common query patterns  
3. Full-text search indexes for text-based queries
4. Vector indexes for embeddings and similarity search
5. Unique constraints for data integrity
6. Index management utilities

Author: AI Assistant
Created: 2025-08-28
Version: 1.0.0
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class IndexType(Enum):
    """Enumeration of Neo4j index types."""
    BTREE = "BTREE"
    RANGE = "RANGE" 
    TEXT = "TEXT"
    FULLTEXT = "FULLTEXT"
    VECTOR = "VECTOR"
    LOOKUP = "LOOKUP"
    POINT = "POINT"


class IndexStatus(Enum):
    """Enumeration of index statuses."""
    ONLINE = "ONLINE"
    POPULATING = "POPULATING"
    FAILED = "FAILED"
    DROPPED = "DROPPED"


@dataclass
class IndexDefinition:
    """Definition of a Neo4j index."""
    name: str
    node_label: str
    properties: List[str]
    index_type: IndexType
    description: str
    is_unique: bool = False
    composite: bool = False
    estimated_size: Optional[str] = None
    creation_priority: int = 1  # 1=high, 2=medium, 3=low
    options: Dict[str, Any] = field(default_factory=dict)
    
    def get_cypher_create(self) -> str:
        """Generate the Cypher statement to create this index."""
        props_str = ", ".join([f"n.{prop}" for prop in self.properties])
        
        if self.is_unique:
            return f"CREATE CONSTRAINT {self.name} IF NOT EXISTS FOR (n:{self.node_label}) REQUIRE ({props_str}) IS UNIQUE"
        
        elif self.index_type == IndexType.FULLTEXT:
            # Full-text indexes have different syntax
            node_labels = [self.node_label] if isinstance(self.node_label, str) else self.node_label
            labels_str = ", ".join([f'"{label}"' for label in node_labels])
            props_str = ", ".join([f'"{prop}"' for prop in self.properties])
            return f'CREATE FULLTEXT INDEX {self.name} IF NOT EXISTS FOR (n:{self.node_label}) ON EACH [{props_str}]'
        
        elif self.index_type == IndexType.VECTOR:
            # Vector indexes need specific configuration
            dimension = self.options.get('dimension', 1536)
            similarity = self.options.get('similarity', 'cosine')
            return f'CREATE VECTOR INDEX {self.name} IF NOT EXISTS FOR (n:{self.node_label}) ON (n.{self.properties[0]}) OPTIONS {{indexConfig: {{`vector.dimensions`: {dimension}, `vector.similarity_function`: "{similarity}"}}}}'
        
        else:
            # Regular BTREE/RANGE indexes
            return f"CREATE INDEX {self.name} IF NOT EXISTS FOR (n:{self.node_label}) ON ({props_str})"
    
    def get_cypher_drop(self) -> str:
        """Generate the Cypher statement to drop this index."""
        return f"DROP INDEX {self.name} IF EXISTS"


@dataclass
class ConstraintDefinition:
    """Definition of a Neo4j constraint."""
    name: str
    node_label: str
    properties: List[str]
    constraint_type: str  # UNIQUE, NODE_KEY, EXIST
    description: str
    
    def get_cypher_create(self) -> str:
        """Generate the Cypher statement to create this constraint."""
        props_str = ", ".join([f"n.{prop}" for prop in self.properties])
        
        if self.constraint_type == "UNIQUE":
            return f"CREATE CONSTRAINT {self.name} IF NOT EXISTS FOR (n:{self.node_label}) REQUIRE ({props_str}) IS UNIQUE"
        elif self.constraint_type == "NODE_KEY":
            return f"CREATE CONSTRAINT {self.name} IF NOT EXISTS FOR (n:{self.node_label}) REQUIRE ({props_str}) IS NODE KEY"
        elif self.constraint_type == "EXIST":
            return f"CREATE CONSTRAINT {self.name} IF NOT EXISTS FOR (n:{self.node_label}) REQUIRE n.{self.properties[0]} IS NOT NULL"
        
        return ""
    
    def get_cypher_drop(self) -> str:
        """Generate the Cypher statement to drop this constraint."""
        return f"DROP CONSTRAINT {self.name} IF EXISTS"


class EHSIndexDefinitions:
    """Comprehensive index definitions for the EHS AI platform."""
    
    @classmethod
    def get_all_indexes(cls) -> List[IndexDefinition]:
        """Get all index definitions for the EHS system."""
        indexes = []
        
        # Add indexes by category
        indexes.extend(cls.get_enhanced_historical_metric_indexes())
        indexes.extend(cls.get_goal_indexes())
        indexes.extend(cls.get_target_indexes())
        indexes.extend(cls.get_trend_analysis_indexes())
        indexes.extend(cls.get_recommendation_indexes())
        indexes.extend(cls.get_forecast_indexes())
        indexes.extend(cls.get_trend_period_indexes())
        indexes.extend(cls.get_legacy_node_indexes())
        indexes.extend(cls.get_full_text_indexes())
        indexes.extend(cls.get_vector_indexes())
        
        return indexes
    
    @classmethod
    def get_enhanced_historical_metric_indexes(cls) -> List[IndexDefinition]:
        """Indexes for EnhancedHistoricalMetric nodes."""
        return [
            # Primary identifier index
            IndexDefinition(
                name="idx_enhanced_metric_id",
                node_label="EnhancedHistoricalMetric",
                properties=["metric_id"],
                index_type=IndexType.BTREE,
                description="Primary key index for metric identification",
                is_unique=True,
                creation_priority=1
            ),
            
            # Facility and department lookup
            IndexDefinition(
                name="idx_enhanced_metric_facility",
                node_label="EnhancedHistoricalMetric",
                properties=["facility_name"],
                index_type=IndexType.BTREE,
                description="Index for facility-based metric queries",
                creation_priority=1
            ),
            
            # Composite index for facility + department queries
            IndexDefinition(
                name="idx_enhanced_metric_facility_dept",
                node_label="EnhancedHistoricalMetric",
                properties=["facility_name", "department"],
                index_type=IndexType.BTREE,
                description="Composite index for facility and department filtering",
                composite=True,
                creation_priority=1
            ),
            
            # Metric type filtering
            IndexDefinition(
                name="idx_enhanced_metric_type",
                node_label="EnhancedHistoricalMetric",
                properties=["metric_type"],
                index_type=IndexType.BTREE,
                description="Index for filtering by metric type",
                creation_priority=1
            ),
            
            # Time-based queries
            IndexDefinition(
                name="idx_enhanced_metric_reporting_period",
                node_label="EnhancedHistoricalMetric",
                properties=["reporting_period"],
                index_type=IndexType.BTREE,
                description="Index for time-based metric queries",
                creation_priority=1
            ),
            
            # Composite index for metric type + time range queries
            IndexDefinition(
                name="idx_enhanced_metric_type_time",
                node_label="EnhancedHistoricalMetric",
                properties=["metric_type", "reporting_period"],
                index_type=IndexType.BTREE,
                description="Composite index for metric type and time filtering",
                composite=True,
                creation_priority=1
            ),
            
            # Quality score filtering for data reliability queries
            IndexDefinition(
                name="idx_enhanced_metric_quality",
                node_label="EnhancedHistoricalMetric",
                properties=["data_quality_score"],
                index_type=IndexType.BTREE,
                description="Index for data quality filtering",
                creation_priority=2
            ),
            
            # Validation status for data integrity queries
            IndexDefinition(
                name="idx_enhanced_metric_validation",
                node_label="EnhancedHistoricalMetric",
                properties=["validation_status"],
                index_type=IndexType.BTREE,
                description="Index for validation status filtering",
                creation_priority=2
            ),
            
            # Composite index for comprehensive metric searches
            IndexDefinition(
                name="idx_enhanced_metric_comprehensive",
                node_label="EnhancedHistoricalMetric",
                properties=["facility_name", "metric_type", "reporting_period"],
                index_type=IndexType.BTREE,
                description="Comprehensive composite index for common query patterns",
                composite=True,
                creation_priority=1
            )
        ]
    
    @classmethod
    def get_goal_indexes(cls) -> List[IndexDefinition]:
        """Indexes for Goal nodes."""
        return [
            # Primary identifier
            IndexDefinition(
                name="idx_goal_id",
                node_label="Goal",
                properties=["goal_id"],
                index_type=IndexType.BTREE,
                description="Primary key index for goal identification",
                is_unique=True,
                creation_priority=1
            ),
            
            # Organization unit filtering
            IndexDefinition(
                name="idx_goal_org_unit",
                node_label="Goal",
                properties=["organization_unit"],
                index_type=IndexType.BTREE,
                description="Index for organization unit filtering",
                creation_priority=1
            ),
            
            # Goal level hierarchy
            IndexDefinition(
                name="idx_goal_level",
                node_label="Goal",
                properties=["goal_level"],
                index_type=IndexType.BTREE,
                description="Index for goal level filtering",
                creation_priority=1
            ),
            
            # Status filtering
            IndexDefinition(
                name="idx_goal_status",
                node_label="Goal",
                properties=["status"],
                index_type=IndexType.BTREE,
                description="Index for goal status filtering",
                creation_priority=1
            ),
            
            # Time-based queries
            IndexDefinition(
                name="idx_goal_dates",
                node_label="Goal",
                properties=["target_date"],
                index_type=IndexType.BTREE,
                description="Index for goal target date queries",
                creation_priority=2
            ),
            
            # Owner and stakeholder queries
            IndexDefinition(
                name="idx_goal_owner",
                node_label="Goal",
                properties=["owner"],
                index_type=IndexType.BTREE,
                description="Index for goal owner queries",
                creation_priority=2
            ),
            
            # Composite index for hierarchy and status
            IndexDefinition(
                name="idx_goal_level_status",
                node_label="Goal",
                properties=["goal_level", "status"],
                index_type=IndexType.BTREE,
                description="Composite index for goal level and status filtering",
                composite=True,
                creation_priority=1
            ),
            
            # Strategic priority filtering
            IndexDefinition(
                name="idx_goal_priority",
                node_label="Goal",
                properties=["strategic_priority"],
                index_type=IndexType.BTREE,
                description="Index for strategic priority filtering",
                creation_priority=2
            )
        ]
    
    @classmethod
    def get_target_indexes(cls) -> List[IndexDefinition]:
        """Indexes for Target nodes."""
        return [
            # Primary identifier
            IndexDefinition(
                name="idx_target_id",
                node_label="Target",
                properties=["target_id"],
                index_type=IndexType.BTREE,
                description="Primary key index for target identification",
                is_unique=True,
                creation_priority=1
            ),
            
            # Goal relationship lookup
            IndexDefinition(
                name="idx_target_goal_id",
                node_label="Target",
                properties=["goal_id"],
                index_type=IndexType.BTREE,
                description="Index for goal-target relationship queries",
                creation_priority=1
            ),
            
            # Metric type filtering
            IndexDefinition(
                name="idx_target_metric_type",
                node_label="Target",
                properties=["metric_type"],
                index_type=IndexType.BTREE,
                description="Index for target metric type filtering",
                creation_priority=1
            ),
            
            # Target date filtering
            IndexDefinition(
                name="idx_target_date",
                node_label="Target",
                properties=["target_date"],
                index_type=IndexType.BTREE,
                description="Index for target date queries",
                creation_priority=2
            ),
            
            # Performance tracking
            IndexDefinition(
                name="idx_target_status",
                node_label="Target",
                properties=["on_track_status"],
                index_type=IndexType.BTREE,
                description="Index for target performance status",
                creation_priority=1
            ),
            
            # Owner queries
            IndexDefinition(
                name="idx_target_owner",
                node_label="Target",
                properties=["owner"],
                index_type=IndexType.BTREE,
                description="Index for target owner queries",
                creation_priority=2
            ),
            
            # Composite index for goal + metric type
            IndexDefinition(
                name="idx_target_goal_metric",
                node_label="Target",
                properties=["goal_id", "metric_type"],
                index_type=IndexType.BTREE,
                description="Composite index for goal and metric type filtering",
                composite=True,
                creation_priority=1
            )
        ]
    
    @classmethod
    def get_trend_analysis_indexes(cls) -> List[IndexDefinition]:
        """Indexes for TrendAnalysis nodes."""
        return [
            # Primary identifier
            IndexDefinition(
                name="idx_trend_analysis_id",
                node_label="TrendAnalysis",
                properties=["analysis_id"],
                index_type=IndexType.BTREE,
                description="Primary key index for trend analysis identification",
                is_unique=True,
                creation_priority=1
            ),
            
            # Metric type filtering
            IndexDefinition(
                name="idx_trend_metric_type",
                node_label="TrendAnalysis",
                properties=["metric_type"],
                index_type=IndexType.BTREE,
                description="Index for trend analysis metric type filtering",
                creation_priority=1
            ),
            
            # Facility and department
            IndexDefinition(
                name="idx_trend_facility",
                node_label="TrendAnalysis",
                properties=["facility_name"],
                index_type=IndexType.BTREE,
                description="Index for facility-based trend queries",
                creation_priority=1
            ),
            
            # Analysis date range queries
            IndexDefinition(
                name="idx_trend_analysis_date",
                node_label="TrendAnalysis",
                properties=["analysis_date"],
                index_type=IndexType.BTREE,
                description="Index for analysis date queries",
                creation_priority=1
            ),
            
            # Trend characteristics
            IndexDefinition(
                name="idx_trend_direction",
                node_label="TrendAnalysis",
                properties=["trend_direction"],
                index_type=IndexType.BTREE,
                description="Index for trend direction filtering",
                creation_priority=2
            ),
            
            # Confidence filtering
            IndexDefinition(
                name="idx_trend_confidence",
                node_label="TrendAnalysis",
                properties=["trend_confidence"],
                index_type=IndexType.BTREE,
                description="Index for trend confidence filtering",
                creation_priority=2
            ),
            
            # Composite index for comprehensive trend queries
            IndexDefinition(
                name="idx_trend_comprehensive",
                node_label="TrendAnalysis",
                properties=["metric_type", "facility_name", "analysis_date"],
                index_type=IndexType.BTREE,
                description="Composite index for comprehensive trend analysis queries",
                composite=True,
                creation_priority=1
            )
        ]
    
    @classmethod
    def get_recommendation_indexes(cls) -> List[IndexDefinition]:
        """Indexes for Recommendation nodes."""
        return [
            # Primary identifier
            IndexDefinition(
                name="idx_recommendation_id",
                node_label="Recommendation",
                properties=["recommendation_id"],
                index_type=IndexType.BTREE,
                description="Primary key index for recommendation identification",
                is_unique=True,
                creation_priority=1
            ),
            
            # Source analysis relationship
            IndexDefinition(
                name="idx_recommendation_source",
                node_label="Recommendation",
                properties=["source_analysis_id"],
                index_type=IndexType.BTREE,
                description="Index for recommendation source analysis queries",
                creation_priority=1
            ),
            
            # Priority filtering
            IndexDefinition(
                name="idx_recommendation_priority",
                node_label="Recommendation",
                properties=["priority"],
                index_type=IndexType.BTREE,
                description="Index for recommendation priority filtering",
                creation_priority=1
            ),
            
            # Status tracking
            IndexDefinition(
                name="idx_recommendation_status",
                node_label="Recommendation",
                properties=["status"],
                index_type=IndexType.BTREE,
                description="Index for recommendation status filtering",
                creation_priority=1
            ),
            
            # Assignment queries
            IndexDefinition(
                name="idx_recommendation_assigned",
                node_label="Recommendation",
                properties=["assigned_to"],
                index_type=IndexType.BTREE,
                description="Index for recommendation assignment queries",
                creation_priority=1
            ),
            
            # Facility and department
            IndexDefinition(
                name="idx_recommendation_facility",
                node_label="Recommendation",
                properties=["facility_name"],
                index_type=IndexType.BTREE,
                description="Index for facility-based recommendation queries",
                creation_priority=1
            ),
            
            # Business impact filtering
            IndexDefinition(
                name="idx_recommendation_impact",
                node_label="Recommendation",
                properties=["business_impact"],
                index_type=IndexType.BTREE,
                description="Index for business impact filtering",
                creation_priority=2
            ),
            
            # Composite index for priority + status
            IndexDefinition(
                name="idx_recommendation_priority_status",
                node_label="Recommendation",
                properties=["priority", "status"],
                index_type=IndexType.BTREE,
                description="Composite index for priority and status filtering",
                composite=True,
                creation_priority=1
            )
        ]
    
    @classmethod
    def get_forecast_indexes(cls) -> List[IndexDefinition]:
        """Indexes for Forecast nodes."""
        return [
            # Primary identifier
            IndexDefinition(
                name="idx_forecast_id",
                node_label="Forecast",
                properties=["forecast_id"],
                index_type=IndexType.BTREE,
                description="Primary key index for forecast identification",
                is_unique=True,
                creation_priority=1
            ),
            
            # Metric type filtering
            IndexDefinition(
                name="idx_forecast_metric_type",
                node_label="Forecast",
                properties=["metric_type"],
                index_type=IndexType.BTREE,
                description="Index for forecast metric type filtering",
                creation_priority=1
            ),
            
            # Facility filtering
            IndexDefinition(
                name="idx_forecast_facility",
                node_label="Forecast",
                properties=["facility_name"],
                index_type=IndexType.BTREE,
                description="Index for facility-based forecast queries",
                creation_priority=1
            ),
            
            # Time horizon queries
            IndexDefinition(
                name="idx_forecast_dates",
                node_label="Forecast",
                properties=["forecast_start_date", "forecast_end_date"],
                index_type=IndexType.BTREE,
                description="Index for forecast time horizon queries",
                creation_priority=1
            ),
            
            # Model type filtering
            IndexDefinition(
                name="idx_forecast_model",
                node_label="Forecast",
                properties=["model_type"],
                index_type=IndexType.BTREE,
                description="Index for forecast model type filtering",
                creation_priority=2
            ),
            
            # Accuracy filtering for model performance queries
            IndexDefinition(
                name="idx_forecast_accuracy",
                node_label="Forecast",
                properties=["model_accuracy"],
                index_type=IndexType.BTREE,
                description="Index for model accuracy filtering",
                creation_priority=2
            ),
            
            # Composite index for comprehensive forecast queries
            IndexDefinition(
                name="idx_forecast_comprehensive",
                node_label="Forecast",
                properties=["metric_type", "facility_name", "forecast_start_date"],
                index_type=IndexType.BTREE,
                description="Composite index for comprehensive forecast queries",
                composite=True,
                creation_priority=1
            )
        ]
    
    @classmethod
    def get_trend_period_indexes(cls) -> List[IndexDefinition]:
        """Indexes for TrendPeriod nodes."""
        return [
            # Primary identifier
            IndexDefinition(
                name="idx_trend_period_id",
                node_label="TrendPeriod",
                properties=["period_id"],
                index_type=IndexType.BTREE,
                description="Primary key index for trend period identification",
                is_unique=True,
                creation_priority=1
            ),
            
            # Period type filtering
            IndexDefinition(
                name="idx_trend_period_type",
                node_label="TrendPeriod",
                properties=["period_type"],
                index_type=IndexType.BTREE,
                description="Index for period type filtering",
                creation_priority=1
            ),
            
            # Date range queries
            IndexDefinition(
                name="idx_trend_period_dates",
                node_label="TrendPeriod",
                properties=["start_date", "end_date"],
                index_type=IndexType.BTREE,
                description="Index for period date range queries",
                creation_priority=1
            ),
            
            # Hierarchy queries
            IndexDefinition(
                name="idx_trend_period_parent",
                node_label="TrendPeriod",
                properties=["parent_period_id"],
                index_type=IndexType.BTREE,
                description="Index for period hierarchy queries",
                creation_priority=1
            ),
            
            # Status filtering
            IndexDefinition(
                name="idx_trend_period_status",
                node_label="TrendPeriod",
                properties=["status"],
                index_type=IndexType.BTREE,
                description="Index for period status filtering",
                creation_priority=2
            )
        ]
    
    @classmethod
    def get_legacy_node_indexes(cls) -> List[IndexDefinition]:
        """Indexes for existing/legacy node types."""
        return [
            # Document node indexes
            IndexDefinition(
                name="idx_document_filename",
                node_label="Document",
                properties=["fileName"],
                index_type=IndexType.BTREE,
                description="Index for document filename queries",
                creation_priority=1
            ),
            
            IndexDefinition(
                name="idx_document_status",
                node_label="Document",
                properties=["status"],
                index_type=IndexType.BTREE,
                description="Index for document status filtering",
                creation_priority=1
            ),
            
            # Chunk node indexes
            IndexDefinition(
                name="idx_chunk_chunkid",
                node_label="Chunk",
                properties=["chunkId"],
                index_type=IndexType.BTREE,
                description="Index for chunk ID queries",
                creation_priority=1
            ),
            
            # Entity node indexes
            IndexDefinition(
                name="idx_entity_id",
                node_label="__Entity__",
                properties=["id"],
                index_type=IndexType.BTREE,
                description="Index for entity ID queries",
                creation_priority=1
            ),
            
            # Community node indexes
            IndexDefinition(
                name="idx_community_id",
                node_label="__Community__",
                properties=["id"],
                index_type=IndexType.BTREE,
                description="Index for community ID queries",
                creation_priority=2
            ),
            
            # Facility-related indexes for existing data
            IndexDefinition(
                name="idx_facility_name",
                node_label="Facility",
                properties=["name"],
                index_type=IndexType.BTREE,
                description="Index for facility name queries",
                creation_priority=1
            )
        ]
    
    @classmethod
    def get_full_text_indexes(cls) -> List[IndexDefinition]:
        """Full-text search indexes for text content."""
        return [
            # Enhanced metric full-text search
            IndexDefinition(
                name="idx_fulltext_enhanced_metric",
                node_label="EnhancedHistoricalMetric",
                properties=["metric_name", "business_context"],
                index_type=IndexType.FULLTEXT,
                description="Full-text search for enhanced metric names and context",
                creation_priority=2
            ),
            
            # Goal full-text search
            IndexDefinition(
                name="idx_fulltext_goal",
                node_label="Goal",
                properties=["goal_name", "description", "business_impact"],
                index_type=IndexType.FULLTEXT,
                description="Full-text search for goals",
                creation_priority=2
            ),
            
            # Target full-text search
            IndexDefinition(
                name="idx_fulltext_target",
                node_label="Target",
                properties=["target_name", "business_justification"],
                index_type=IndexType.FULLTEXT,
                description="Full-text search for targets",
                creation_priority=2
            ),
            
            # Recommendation full-text search
            IndexDefinition(
                name="idx_fulltext_recommendation",
                node_label="Recommendation",
                properties=["recommendation_title", "description"],
                index_type=IndexType.FULLTEXT,
                description="Full-text search for recommendations",
                creation_priority=2
            ),
            
            # Trend analysis insights search
            IndexDefinition(
                name="idx_fulltext_trend_insights",
                node_label="TrendAnalysis",
                properties=["analysis_name", "key_insights"],
                index_type=IndexType.FULLTEXT,
                description="Full-text search for trend analysis insights",
                creation_priority=3
            ),
            
            # Legacy content search
            IndexDefinition(
                name="idx_fulltext_chunk_content",
                node_label="Chunk",
                properties=["text"],
                index_type=IndexType.FULLTEXT,
                description="Full-text search for chunk content",
                creation_priority=2
            )
        ]
    
    @classmethod
    def get_vector_indexes(cls) -> List[IndexDefinition]:
        """Vector indexes for embedding-based similarity search."""
        return [
            # Enhanced metric embeddings
            IndexDefinition(
                name="idx_vector_enhanced_metric",
                node_label="EnhancedHistoricalMetric",
                properties=["embedding"],
                index_type=IndexType.VECTOR,
                description="Vector index for enhanced metric similarity search",
                creation_priority=3,
                options={"dimension": 1536, "similarity": "cosine"}
            ),
            
            # Goal embeddings for similarity matching
            IndexDefinition(
                name="idx_vector_goal",
                node_label="Goal",
                properties=["embedding"],
                index_type=IndexType.VECTOR,
                description="Vector index for goal similarity search",
                creation_priority=3,
                options={"dimension": 1536, "similarity": "cosine"}
            ),
            
            # Recommendation embeddings
            IndexDefinition(
                name="idx_vector_recommendation",
                node_label="Recommendation",
                properties=["embedding"],
                index_type=IndexType.VECTOR,
                description="Vector index for recommendation similarity search",
                creation_priority=3,
                options={"dimension": 1536, "similarity": "cosine"}
            ),
            
            # Legacy chunk embeddings
            IndexDefinition(
                name="idx_vector_chunk",
                node_label="Chunk",
                properties=["embedding"],
                index_type=IndexType.VECTOR,
                description="Vector index for chunk similarity search",
                creation_priority=3,
                options={"dimension": 1536, "similarity": "cosine"}
            )
        ]
    
    @classmethod
    def get_all_constraints(cls) -> List[ConstraintDefinition]:
        """Get all constraint definitions for data integrity."""
        return [
            # Enhanced Historical Metric constraints
            ConstraintDefinition(
                name="constr_enhanced_metric_id_unique",
                node_label="EnhancedHistoricalMetric",
                properties=["metric_id"],
                constraint_type="UNIQUE",
                description="Ensure metric_id uniqueness"
            ),
            
            ConstraintDefinition(
                name="constr_enhanced_metric_required_fields",
                node_label="EnhancedHistoricalMetric", 
                properties=["facility_name"],
                constraint_type="EXIST",
                description="Ensure facility_name is not null"
            ),
            
            # Goal constraints
            ConstraintDefinition(
                name="constr_goal_id_unique",
                node_label="Goal",
                properties=["goal_id"],
                constraint_type="UNIQUE",
                description="Ensure goal_id uniqueness"
            ),
            
            # Target constraints
            ConstraintDefinition(
                name="constr_target_id_unique",
                node_label="Target", 
                properties=["target_id"],
                constraint_type="UNIQUE",
                description="Ensure target_id uniqueness"
            ),
            
            # Trend Analysis constraints
            ConstraintDefinition(
                name="constr_trend_analysis_id_unique",
                node_label="TrendAnalysis",
                properties=["analysis_id"],
                constraint_type="UNIQUE",
                description="Ensure analysis_id uniqueness"
            ),
            
            # Recommendation constraints
            ConstraintDefinition(
                name="constr_recommendation_id_unique",
                node_label="Recommendation",
                properties=["recommendation_id"],
                constraint_type="UNIQUE",
                description="Ensure recommendation_id uniqueness"
            ),
            
            # Forecast constraints
            ConstraintDefinition(
                name="constr_forecast_id_unique",
                node_label="Forecast",
                properties=["forecast_id"],
                constraint_type="UNIQUE",
                description="Ensure forecast_id uniqueness"
            ),
            
            # Trend Period constraints
            ConstraintDefinition(
                name="constr_trend_period_id_unique",
                node_label="TrendPeriod",
                properties=["period_id"],
                constraint_type="UNIQUE",
                description="Ensure period_id uniqueness"
            )
        ]


class IndexManager:
    """Utility class for managing Neo4j indexes."""
    
    def __init__(self, graph_db):
        """Initialize with Neo4j graph database connection."""
        self.graph_db = graph_db
        self.logger = logging.getLogger(__name__)
    
    def create_all_indexes(self, priority_filter: Optional[int] = None) -> Dict[str, Any]:
        """Create all indexes with optional priority filtering.
        
        Args:
            priority_filter: Only create indexes with this priority level (1=high, 2=medium, 3=low)
            
        Returns:
            Dictionary with creation results
        """
        results = {
            "created": [],
            "failed": [],
            "skipped": []
        }
        
        indexes = EHSIndexDefinitions.get_all_indexes()
        
        # Filter by priority if specified
        if priority_filter:
            indexes = [idx for idx in indexes if idx.creation_priority == priority_filter]
        
        # Sort by priority (1=highest priority first)
        indexes.sort(key=lambda x: x.creation_priority)
        
        for index_def in indexes:
            try:
                cypher = index_def.get_cypher_create()
                self.graph_db.query(cypher)
                results["created"].append({
                    "name": index_def.name,
                    "type": index_def.index_type.value,
                    "priority": index_def.creation_priority
                })
                self.logger.info(f"Created index: {index_def.name}")
                
            except Exception as e:
                error_msg = str(e)
                if "already exists" in error_msg.lower():
                    results["skipped"].append({
                        "name": index_def.name,
                        "reason": "Already exists"
                    })
                else:
                    results["failed"].append({
                        "name": index_def.name,
                        "error": error_msg
                    })
                    self.logger.error(f"Failed to create index {index_def.name}: {error_msg}")
        
        return results
    
    def create_all_constraints(self) -> Dict[str, Any]:
        """Create all constraints for data integrity.
        
        Returns:
            Dictionary with creation results
        """
        results = {
            "created": [],
            "failed": [],
            "skipped": []
        }
        
        constraints = EHSIndexDefinitions.get_all_constraints()
        
        for constraint_def in constraints:
            try:
                cypher = constraint_def.get_cypher_create()
                self.graph_db.query(cypher)
                results["created"].append({
                    "name": constraint_def.name,
                    "type": constraint_def.constraint_type
                })
                self.logger.info(f"Created constraint: {constraint_def.name}")
                
            except Exception as e:
                error_msg = str(e)
                if "already exists" in error_msg.lower():
                    results["skipped"].append({
                        "name": constraint_def.name,
                        "reason": "Already exists"
                    })
                else:
                    results["failed"].append({
                        "name": constraint_def.name,
                        "error": error_msg
                    })
                    self.logger.error(f"Failed to create constraint {constraint_def.name}: {error_msg}")
        
        return results
    
    def get_index_status(self) -> List[Dict[str, Any]]:
        """Get the status of all indexes in the database.
        
        Returns:
            List of index status information
        """
        try:
            cypher = """
            SHOW INDEXES
            YIELD name, type, labelsOrTypes, properties, state, populationPercent
            RETURN name, type, labelsOrTypes, properties, state, populationPercent
            ORDER BY name
            """
            result = self.graph_db.query(cypher)
            return result
        except Exception as e:
            self.logger.error(f"Failed to get index status: {e}")
            return []
    
    def get_constraint_status(self) -> List[Dict[str, Any]]:
        """Get the status of all constraints in the database.
        
        Returns:
            List of constraint status information
        """
        try:
            cypher = """
            SHOW CONSTRAINTS
            YIELD name, type, labelsOrTypes, properties
            RETURN name, type, labelsOrTypes, properties
            ORDER BY name
            """
            result = self.graph_db.query(cypher)
            return result
        except Exception as e:
            self.logger.error(f"Failed to get constraint status: {e}")
            return []
    
    def drop_all_indexes(self, confirm: bool = False) -> Dict[str, Any]:
        """Drop all indexes (use with caution).
        
        Args:
            confirm: Must be True to actually drop indexes
            
        Returns:
            Dictionary with drop results
        """
        if not confirm:
            raise ValueError("Must set confirm=True to drop indexes")
        
        results = {
            "dropped": [],
            "failed": []
        }
        
        indexes = EHSIndexDefinitions.get_all_indexes()
        
        for index_def in indexes:
            try:
                cypher = index_def.get_cypher_drop()
                self.graph_db.query(cypher)
                results["dropped"].append(index_def.name)
                self.logger.info(f"Dropped index: {index_def.name}")
                
            except Exception as e:
                results["failed"].append({
                    "name": index_def.name,
                    "error": str(e)
                })
                self.logger.error(f"Failed to drop index {index_def.name}: {e}")
        
        return results
    
    def analyze_query_performance(self, cypher_query: str) -> Dict[str, Any]:
        """Analyze query performance and suggest indexes.
        
        Args:
            cypher_query: The Cypher query to analyze
            
        Returns:
            Performance analysis and index suggestions
        """
        try:
            # Use EXPLAIN to get query plan
            explain_query = f"EXPLAIN {cypher_query}"
            explain_result = self.graph_db.query(explain_query)
            
            # Use PROFILE for actual performance metrics (be careful with this on large datasets)
            profile_query = f"PROFILE {cypher_query}"
            profile_result = self.graph_db.query(profile_query)
            
            return {
                "query": cypher_query,
                "explain_plan": explain_result,
                "profile_stats": profile_result,
                "suggestions": self._generate_index_suggestions(cypher_query)
            }
        except Exception as e:
            self.logger.error(f"Failed to analyze query performance: {e}")
            return {"error": str(e)}
    
    def _generate_index_suggestions(self, cypher_query: str) -> List[str]:
        """Generate index suggestions based on query patterns."""
        suggestions = []
        
        # Basic pattern matching for common index needs
        if "WHERE" in cypher_query.upper():
            suggestions.append("Consider indexes on properties used in WHERE clauses")
        
        if "ORDER BY" in cypher_query.upper():
            suggestions.append("Consider indexes on properties used in ORDER BY clauses")
        
        if "MATCH" in cypher_query.upper():
            suggestions.append("Consider indexes on node labels and relationship types used in MATCH patterns")
        
        return suggestions


# Utility functions for easy access

def get_high_priority_indexes() -> List[IndexDefinition]:
    """Get only high priority indexes (priority = 1)."""
    all_indexes = EHSIndexDefinitions.get_all_indexes()
    return [idx for idx in all_indexes if idx.creation_priority == 1]


def get_indexes_by_node_type(node_label: str) -> List[IndexDefinition]:
    """Get all indexes for a specific node type."""
    all_indexes = EHSIndexDefinitions.get_all_indexes()
    return [idx for idx in all_indexes if idx.node_label == node_label]


def generate_index_creation_script() -> str:
    """Generate a complete Cypher script for creating all indexes and constraints."""
    lines = [
        "// Neo4j Index and Constraint Creation Script",
        "// Generated automatically for EHS AI Platform",
        f"// Generated on: {datetime.now().isoformat()}",
        "",
        "// ============================================",
        "// CONSTRAINTS (Create first for data integrity)",
        "// ============================================",
        ""
    ]
    
    # Add constraints first
    constraints = EHSIndexDefinitions.get_all_constraints()
    for constraint in constraints:
        lines.extend([
            f"// {constraint.description}",
            constraint.get_cypher_create() + ";",
            ""
        ])
    
    lines.extend([
        "// ============================================", 
        "// INDEXES (Create after constraints)",
        "// ============================================",
        ""
    ])
    
    # Add indexes grouped by priority
    all_indexes = EHSIndexDefinitions.get_all_indexes()
    
    for priority in [1, 2, 3]:
        priority_indexes = [idx for idx in all_indexes if idx.creation_priority == priority]
        if priority_indexes:
            priority_name = {1: "HIGH", 2: "MEDIUM", 3: "LOW"}[priority]
            lines.extend([
                f"// {priority_name} PRIORITY INDEXES",
                f"// Priority {priority} - Create these {'first' if priority == 1 else 'after higher priority indexes'}",
                ""
            ])
            
            for index_def in priority_indexes:
                lines.extend([
                    f"// {index_def.description}",
                    index_def.get_cypher_create() + ";",
                    ""
                ])
    
    return "\n".join(lines)


# Export main classes and functions
__all__ = [
    "IndexType",
    "IndexStatus", 
    "IndexDefinition",
    "ConstraintDefinition",
    "EHSIndexDefinitions",
    "IndexManager",
    "get_high_priority_indexes",
    "get_indexes_by_node_type",
    "generate_index_creation_script"
]