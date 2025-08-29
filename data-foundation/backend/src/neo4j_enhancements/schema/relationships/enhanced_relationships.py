"""
Enhanced relationship definitions for the Neo4j EHS AI system.

This module defines all enhanced relationship types with their properties,
validation rules, and helper functions for advanced EHS analytics.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RelationshipType(Enum):
    """Enumeration of all enhanced relationship types."""
    
    # Goal and Target relationships
    HAS_GOAL = "HAS_GOAL"
    HAS_TARGET = "HAS_TARGET"
    ACHIEVES_TARGET = "ACHIEVES_TARGET"
    
    # Analysis relationships
    ANALYZED_BY = "ANALYZED_BY"
    RECOMMENDS = "RECOMMENDS"
    FORECASTS = "FORECASTS"
    PREDICTS = "PREDICTS"
    
    # Temporal relationships
    BELONGS_TO_PERIOD = "BELONGS_TO_PERIOD"
    PARENT_PERIOD = "PARENT_PERIOD"
    FOLLOWS = "FOLLOWS"
    PRECEDES = "PRECEDES"
    
    # Performance relationships
    MEASURES = "MEASURES"
    BENCHMARKS_AGAINST = "BENCHMARKS_AGAINST"
    COMPARES_TO = "COMPARES_TO"
    
    # Action relationships
    TRIGGERS_ACTION = "TRIGGERS_ACTION"
    REQUIRES_ACTION = "REQUIRES_ACTION"
    IMPLEMENTS = "IMPLEMENTS"
    
    # Data lineage relationships
    DERIVED_FROM = "DERIVED_FROM"
    AGGREGATES = "AGGREGATES"
    TRANSFORMS = "TRANSFORMS"
    
    # Hierarchy relationships
    PARENT_OF = "PARENT_OF"
    CHILD_OF = "CHILD_OF"
    MEMBER_OF = "MEMBER_OF"


class RelationshipProperties:
    """Standard properties for enhanced relationships."""
    
    @staticmethod
    def get_base_properties() -> Dict[str, Any]:
        """Get base properties for all relationships."""
        return {
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "active": True
        }
    
    @staticmethod
    def get_temporal_properties() -> Dict[str, Any]:
        """Get temporal properties for time-based relationships."""
        return {
            "valid_from": None,
            "valid_to": None,
            "period_type": None,  # daily, weekly, monthly, quarterly, yearly
            "sequence_order": None
        }
    
    @staticmethod
    def get_analysis_properties() -> Dict[str, Any]:
        """Get properties for analysis relationships."""
        return {
            "analysis_type": None,
            "confidence_score": None,
            "analysis_date": None,
            "method": None,
            "parameters": {}
        }
    
    @staticmethod
    def get_performance_properties() -> Dict[str, Any]:
        """Get properties for performance relationships."""
        return {
            "metric_type": None,
            "calculation_method": None,
            "baseline_value": None,
            "target_value": None,
            "actual_value": None,
            "variance": None,
            "performance_rating": None
        }


class EnhancedRelationshipDefinitions:
    """Enhanced relationship definitions with validation and helper functions."""
    
    @classmethod
    def get_relationship_schema(cls, relationship_type: RelationshipType) -> Dict[str, Any]:
        """Get the schema definition for a relationship type."""
        schemas = {
            RelationshipType.HAS_GOAL: cls._get_has_goal_schema(),
            RelationshipType.HAS_TARGET: cls._get_has_target_schema(),
            RelationshipType.ACHIEVES_TARGET: cls._get_achieves_target_schema(),
            RelationshipType.ANALYZED_BY: cls._get_analyzed_by_schema(),
            RelationshipType.RECOMMENDS: cls._get_recommends_schema(),
            RelationshipType.FORECASTS: cls._get_forecasts_schema(),
            RelationshipType.PREDICTS: cls._get_predicts_schema(),
            RelationshipType.BELONGS_TO_PERIOD: cls._get_belongs_to_period_schema(),
            RelationshipType.PARENT_PERIOD: cls._get_parent_period_schema(),
            RelationshipType.FOLLOWS: cls._get_follows_schema(),
            RelationshipType.PRECEDES: cls._get_precedes_schema(),
            RelationshipType.MEASURES: cls._get_measures_schema(),
            RelationshipType.BENCHMARKS_AGAINST: cls._get_benchmarks_against_schema(),
            RelationshipType.COMPARES_TO: cls._get_compares_to_schema(),
            RelationshipType.TRIGGERS_ACTION: cls._get_triggers_action_schema(),
            RelationshipType.REQUIRES_ACTION: cls._get_requires_action_schema(),
            RelationshipType.IMPLEMENTS: cls._get_implements_schema(),
            RelationshipType.DERIVED_FROM: cls._get_derived_from_schema(),
            RelationshipType.AGGREGATES: cls._get_aggregates_schema(),
            RelationshipType.TRANSFORMS: cls._get_transforms_schema(),
            RelationshipType.PARENT_OF: cls._get_parent_of_schema(),
            RelationshipType.CHILD_OF: cls._get_child_of_schema(),
            RelationshipType.MEMBER_OF: cls._get_member_of_schema()
        }
        return schemas.get(relationship_type, {})
    
    @staticmethod
    def _get_has_goal_schema() -> Dict[str, Any]:
        """Schema for HAS_GOAL relationship."""
        return {
            "type": RelationshipType.HAS_GOAL.value,
            "description": "Facility has a specific goal",
            "from_node_types": ["Facility", "Department", "Organization"],
            "to_node_types": ["Goal"],
            "required_properties": ["goal_type", "priority", "status"],
            "optional_properties": ["assigned_date", "target_date", "owner", "description"],
            "default_properties": {
                **RelationshipProperties.get_base_properties(),
                "goal_type": None,
                "priority": "medium",
                "status": "active",
                "assigned_date": None,
                "target_date": None,
                "owner": None,
                "description": None
            }
        }
    
    @staticmethod
    def _get_has_target_schema() -> Dict[str, Any]:
        """Schema for HAS_TARGET relationship."""
        return {
            "type": RelationshipType.HAS_TARGET.value,
            "description": "Goal has specific targets",
            "from_node_types": ["Goal"],
            "to_node_types": ["Target"],
            "required_properties": ["target_type", "measurement_unit", "target_value"],
            "optional_properties": ["baseline_value", "deadline", "measurement_method"],
            "default_properties": {
                **RelationshipProperties.get_base_properties(),
                "target_type": None,
                "measurement_unit": None,
                "target_value": None,
                "baseline_value": None,
                "deadline": None,
                "measurement_method": None
            }
        }
    
    @staticmethod
    def _get_achieves_target_schema() -> Dict[str, Any]:
        """Schema for ACHIEVES_TARGET relationship."""
        return {
            "type": RelationshipType.ACHIEVES_TARGET.value,
            "description": "Metric achieves a target",
            "from_node_types": ["Metric", "KPI"],
            "to_node_types": ["Target"],
            "required_properties": ["achievement_percentage", "measurement_date"],
            "optional_properties": ["variance", "performance_rating", "notes"],
            "default_properties": {
                **RelationshipProperties.get_base_properties(),
                **RelationshipProperties.get_performance_properties(),
                "achievement_percentage": None,
                "measurement_date": None,
                "variance": None,
                "performance_rating": None,
                "notes": None
            }
        }
    
    @staticmethod
    def _get_analyzed_by_schema() -> Dict[str, Any]:
        """Schema for ANALYZED_BY relationship."""
        return {
            "type": RelationshipType.ANALYZED_BY.value,
            "description": "Metric is analyzed by trend analysis",
            "from_node_types": ["Metric", "KPI", "Incident"],
            "to_node_types": ["TrendAnalysis", "Analysis"],
            "required_properties": ["analysis_type", "analysis_date"],
            "optional_properties": ["confidence_score", "method", "parameters"],
            "default_properties": {
                **RelationshipProperties.get_base_properties(),
                **RelationshipProperties.get_analysis_properties(),
                "analysis_type": None,
                "analysis_date": None
            }
        }
    
    @staticmethod
    def _get_recommends_schema() -> Dict[str, Any]:
        """Schema for RECOMMENDS relationship."""
        return {
            "type": RelationshipType.RECOMMENDS.value,
            "description": "Analysis recommends specific actions",
            "from_node_types": ["Analysis", "TrendAnalysis", "RiskAssessment"],
            "to_node_types": ["Action", "Recommendation"],
            "required_properties": ["recommendation_type", "priority", "confidence_score"],
            "optional_properties": ["rationale", "expected_impact", "implementation_timeframe"],
            "default_properties": {
                **RelationshipProperties.get_base_properties(),
                "recommendation_type": None,
                "priority": "medium",
                "confidence_score": None,
                "rationale": None,
                "expected_impact": None,
                "implementation_timeframe": None
            }
        }
    
    @staticmethod
    def _get_forecasts_schema() -> Dict[str, Any]:
        """Schema for FORECASTS relationship."""
        return {
            "type": RelationshipType.FORECASTS.value,
            "description": "Forecast predicts future metrics",
            "from_node_types": ["Forecast", "PredictiveModel"],
            "to_node_types": ["Metric", "KPI"],
            "required_properties": ["forecast_period", "predicted_value", "confidence_interval"],
            "optional_properties": ["model_accuracy", "forecast_method", "assumptions"],
            "default_properties": {
                **RelationshipProperties.get_base_properties(),
                **RelationshipProperties.get_analysis_properties(),
                "forecast_period": None,
                "predicted_value": None,
                "confidence_interval": None,
                "model_accuracy": None,
                "forecast_method": None,
                "assumptions": {}
            }
        }
    
    @staticmethod
    def _get_predicts_schema() -> Dict[str, Any]:
        """Schema for PREDICTS relationship."""
        return {
            "type": RelationshipType.PREDICTS.value,
            "description": "Model predicts outcomes",
            "from_node_types": ["PredictiveModel", "MLModel"],
            "to_node_types": ["Outcome", "Risk", "Incident"],
            "required_properties": ["prediction_probability", "prediction_date"],
            "optional_properties": ["model_version", "input_features", "prediction_horizon"],
            "default_properties": {
                **RelationshipProperties.get_base_properties(),
                "prediction_probability": None,
                "prediction_date": None,
                "model_version": None,
                "input_features": [],
                "prediction_horizon": None
            }
        }
    
    @staticmethod
    def _get_belongs_to_period_schema() -> Dict[str, Any]:
        """Schema for BELONGS_TO_PERIOD relationship."""
        return {
            "type": RelationshipType.BELONGS_TO_PERIOD.value,
            "description": "Metric belongs to a specific time period",
            "from_node_types": ["Metric", "KPI", "Incident"],
            "to_node_types": ["TimePeriod", "ReportingPeriod"],
            "required_properties": ["period_type", "period_start", "period_end"],
            "optional_properties": ["aggregation_method", "period_label"],
            "default_properties": {
                **RelationshipProperties.get_base_properties(),
                **RelationshipProperties.get_temporal_properties(),
                "period_type": None,
                "period_start": None,
                "period_end": None,
                "aggregation_method": None,
                "period_label": None
            }
        }
    
    @staticmethod
    def _get_parent_period_schema() -> Dict[str, Any]:
        """Schema for PARENT_PERIOD relationship."""
        return {
            "type": RelationshipType.PARENT_PERIOD.value,
            "description": "Hierarchical relationship between time periods",
            "from_node_types": ["TimePeriod", "ReportingPeriod"],
            "to_node_types": ["TimePeriod", "ReportingPeriod"],
            "required_properties": ["hierarchy_level", "aggregation_type"],
            "optional_properties": ["rollup_method", "weight_factor"],
            "default_properties": {
                **RelationshipProperties.get_base_properties(),
                "hierarchy_level": None,
                "aggregation_type": None,
                "rollup_method": "sum",
                "weight_factor": 1.0
            }
        }
    
    @staticmethod
    def _get_follows_schema() -> Dict[str, Any]:
        """Schema for FOLLOWS relationship."""
        return {
            "type": RelationshipType.FOLLOWS.value,
            "description": "Sequential relationship between entities",
            "from_node_types": ["TimePeriod", "Phase", "Action"],
            "to_node_types": ["TimePeriod", "Phase", "Action"],
            "required_properties": ["sequence_order"],
            "optional_properties": ["time_gap", "dependency_type"],
            "default_properties": {
                **RelationshipProperties.get_base_properties(),
                "sequence_order": None,
                "time_gap": None,
                "dependency_type": "sequential"
            }
        }
    
    @staticmethod
    def _get_precedes_schema() -> Dict[str, Any]:
        """Schema for PRECEDES relationship."""
        return {
            "type": RelationshipType.PRECEDES.value,
            "description": "Entity comes before another in sequence",
            "from_node_types": ["TimePeriod", "Phase", "Action"],
            "to_node_types": ["TimePeriod", "Phase", "Action"],
            "required_properties": ["sequence_order"],
            "optional_properties": ["time_gap", "dependency_type"],
            "default_properties": {
                **RelationshipProperties.get_base_properties(),
                "sequence_order": None,
                "time_gap": None,
                "dependency_type": "sequential"
            }
        }
    
    @staticmethod
    def _get_measures_schema() -> Dict[str, Any]:
        """Schema for MEASURES relationship."""
        return {
            "type": RelationshipType.MEASURES.value,
            "description": "Metric measures a specific aspect",
            "from_node_types": ["Metric", "KPI"],
            "to_node_types": ["Performance", "Risk", "Compliance"],
            "required_properties": ["measurement_type", "unit_of_measure"],
            "optional_properties": ["measurement_method", "frequency", "accuracy"],
            "default_properties": {
                **RelationshipProperties.get_base_properties(),
                "measurement_type": None,
                "unit_of_measure": None,
                "measurement_method": None,
                "frequency": None,
                "accuracy": None
            }
        }
    
    @staticmethod
    def _get_benchmarks_against_schema() -> Dict[str, Any]:
        """Schema for BENCHMARKS_AGAINST relationship."""
        return {
            "type": RelationshipType.BENCHMARKS_AGAINST.value,
            "description": "Performance benchmarked against standard",
            "from_node_types": ["Performance", "Metric"],
            "to_node_types": ["Benchmark", "Standard"],
            "required_properties": ["benchmark_value", "comparison_result"],
            "optional_properties": ["variance_percentage", "benchmark_source"],
            "default_properties": {
                **RelationshipProperties.get_base_properties(),
                **RelationshipProperties.get_performance_properties(),
                "benchmark_value": None,
                "comparison_result": None,
                "variance_percentage": None,
                "benchmark_source": None
            }
        }
    
    @staticmethod
    def _get_compares_to_schema() -> Dict[str, Any]:
        """Schema for COMPARES_TO relationship."""
        return {
            "type": RelationshipType.COMPARES_TO.value,
            "description": "Direct comparison between entities",
            "from_node_types": ["Metric", "Performance", "Facility"],
            "to_node_types": ["Metric", "Performance", "Facility"],
            "required_properties": ["comparison_type", "comparison_result"],
            "optional_properties": ["comparison_date", "comparison_method"],
            "default_properties": {
                **RelationshipProperties.get_base_properties(),
                "comparison_type": None,
                "comparison_result": None,
                "comparison_date": None,
                "comparison_method": None
            }
        }
    
    @staticmethod
    def _get_triggers_action_schema() -> Dict[str, Any]:
        """Schema for TRIGGERS_ACTION relationship."""
        return {
            "type": RelationshipType.TRIGGERS_ACTION.value,
            "description": "Event or condition triggers an action",
            "from_node_types": ["Alert", "Threshold", "Condition"],
            "to_node_types": ["Action", "Response"],
            "required_properties": ["trigger_condition", "trigger_date"],
            "optional_properties": ["trigger_value", "urgency_level"],
            "default_properties": {
                **RelationshipProperties.get_base_properties(),
                "trigger_condition": None,
                "trigger_date": None,
                "trigger_value": None,
                "urgency_level": "medium"
            }
        }
    
    @staticmethod
    def _get_requires_action_schema() -> Dict[str, Any]:
        """Schema for REQUIRES_ACTION relationship."""
        return {
            "type": RelationshipType.REQUIRES_ACTION.value,
            "description": "Situation requires specific action",
            "from_node_types": ["Risk", "NonCompliance", "Incident"],
            "to_node_types": ["Action", "CorrectiveAction"],
            "required_properties": ["action_type", "required_by_date"],
            "optional_properties": ["priority_level", "responsibility"],
            "default_properties": {
                **RelationshipProperties.get_base_properties(),
                "action_type": None,
                "required_by_date": None,
                "priority_level": "medium",
                "responsibility": None
            }
        }
    
    @staticmethod
    def _get_implements_schema() -> Dict[str, Any]:
        """Schema for IMPLEMENTS relationship."""
        return {
            "type": RelationshipType.IMPLEMENTS.value,
            "description": "Action implements a recommendation or requirement",
            "from_node_types": ["Action", "CorrectiveAction"],
            "to_node_types": ["Recommendation", "Requirement", "Policy"],
            "required_properties": ["implementation_status", "start_date"],
            "optional_properties": ["completion_date", "implementation_method"],
            "default_properties": {
                **RelationshipProperties.get_base_properties(),
                "implementation_status": "planned",
                "start_date": None,
                "completion_date": None,
                "implementation_method": None
            }
        }
    
    @staticmethod
    def _get_derived_from_schema() -> Dict[str, Any]:
        """Schema for DERIVED_FROM relationship."""
        return {
            "type": RelationshipType.DERIVED_FROM.value,
            "description": "Data derived from source data",
            "from_node_types": ["Metric", "KPI", "Analysis"],
            "to_node_types": ["Metric", "RawData", "Source"],
            "required_properties": ["derivation_method", "transformation_date"],
            "optional_properties": ["transformation_rules", "data_quality_score"],
            "default_properties": {
                **RelationshipProperties.get_base_properties(),
                "derivation_method": None,
                "transformation_date": None,
                "transformation_rules": {},
                "data_quality_score": None
            }
        }
    
    @staticmethod
    def _get_aggregates_schema() -> Dict[str, Any]:
        """Schema for AGGREGATES relationship."""
        return {
            "type": RelationshipType.AGGREGATES.value,
            "description": "Metric aggregates other metrics",
            "from_node_types": ["Metric", "KPI"],
            "to_node_types": ["Metric", "KPI"],
            "required_properties": ["aggregation_method", "aggregation_level"],
            "optional_properties": ["weight_factor", "aggregation_period"],
            "default_properties": {
                **RelationshipProperties.get_base_properties(),
                "aggregation_method": None,
                "aggregation_level": None,
                "weight_factor": 1.0,
                "aggregation_period": None
            }
        }
    
    @staticmethod
    def _get_transforms_schema() -> Dict[str, Any]:
        """Schema for TRANSFORMS relationship."""
        return {
            "type": RelationshipType.TRANSFORMS.value,
            "description": "Process transforms data",
            "from_node_types": ["Process", "Transformation"],
            "to_node_types": ["Data", "Metric"],
            "required_properties": ["transformation_type", "transformation_date"],
            "optional_properties": ["transformation_parameters", "quality_check"],
            "default_properties": {
                **RelationshipProperties.get_base_properties(),
                "transformation_type": None,
                "transformation_date": None,
                "transformation_parameters": {},
                "quality_check": None
            }
        }
    
    @staticmethod
    def _get_parent_of_schema() -> Dict[str, Any]:
        """Schema for PARENT_OF relationship."""
        return {
            "type": RelationshipType.PARENT_OF.value,
            "description": "Hierarchical parent relationship",
            "from_node_types": ["Organization", "Department", "Facility"],
            "to_node_types": ["Department", "Facility", "Unit"],
            "required_properties": ["hierarchy_level"],
            "optional_properties": ["relationship_type", "authority_level"],
            "default_properties": {
                **RelationshipProperties.get_base_properties(),
                "hierarchy_level": None,
                "relationship_type": "organizational",
                "authority_level": None
            }
        }
    
    @staticmethod
    def _get_child_of_schema() -> Dict[str, Any]:
        """Schema for CHILD_OF relationship."""
        return {
            "type": RelationshipType.CHILD_OF.value,
            "description": "Hierarchical child relationship",
            "from_node_types": ["Department", "Facility", "Unit"],
            "to_node_types": ["Organization", "Department", "Facility"],
            "required_properties": ["hierarchy_level"],
            "optional_properties": ["relationship_type", "reporting_structure"],
            "default_properties": {
                **RelationshipProperties.get_base_properties(),
                "hierarchy_level": None,
                "relationship_type": "organizational",
                "reporting_structure": None
            }
        }
    
    @staticmethod
    def _get_member_of_schema() -> Dict[str, Any]:
        """Schema for MEMBER_OF relationship."""
        return {
            "type": RelationshipType.MEMBER_OF.value,
            "description": "Membership relationship",
            "from_node_types": ["Person", "Asset", "Resource"],
            "to_node_types": ["Team", "Group", "Category"],
            "required_properties": ["membership_type", "membership_date"],
            "optional_properties": ["role", "status", "end_date"],
            "default_properties": {
                **RelationshipProperties.get_base_properties(),
                "membership_type": None,
                "membership_date": None,
                "role": None,
                "status": "active",
                "end_date": None
            }
        }


class RelationshipValidator:
    """Validator for enhanced relationships."""
    
    @staticmethod
    def validate_relationship(
        relationship_type: RelationshipType,
        from_node_type: str,
        to_node_type: str,
        properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate a relationship definition.
        
        Args:
            relationship_type: The type of relationship
            from_node_type: Source node type
            to_node_type: Target node type
            properties: Relationship properties
            
        Returns:
            Validation result with errors if any
        """
        result = {"valid": True, "errors": [], "warnings": []}
        
        try:
            schema = EnhancedRelationshipDefinitions.get_relationship_schema(relationship_type)
            
            if not schema:
                result["valid"] = False
                result["errors"].append(f"Unknown relationship type: {relationship_type.value}")
                return result
            
            # Validate node types
            if from_node_type not in schema.get("from_node_types", []):
                result["errors"].append(
                    f"Invalid from_node_type '{from_node_type}' for relationship '{relationship_type.value}'")
            
            if to_node_type not in schema.get("to_node_types", []):
                result["errors"].append(
                    f"Invalid to_node_type '{to_node_type}' for relationship '{relationship_type.value}'")
            
            # Validate required properties
            required_props = schema.get("required_properties", [])
            for prop in required_props:
                if prop not in properties or properties[prop] is None:
                    result["errors"].append(f"Missing required property: {prop}")
            
            # Validate property types based on relationship type
            if relationship_type in [RelationshipType.ANALYZED_BY, RelationshipType.FORECASTS]:
                if "confidence_score" in properties:
                    score = properties["confidence_score"]
                    if score is not None and (not isinstance(score, (int, float)) or score < 0 or score > 1):
                        result["errors"].append("confidence_score must be a number between 0 and 1")
            
            if relationship_type == RelationshipType.ACHIEVES_TARGET:
                if "achievement_percentage" in properties:
                    perc = properties["achievement_percentage"]
                    if perc is not None and (not isinstance(perc, (int, float)) or perc < 0):
                        result["errors"].append("achievement_percentage must be a non-negative number")
            
            if result["errors"]:
                result["valid"] = False
            
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Validation error: {str(e)}")
            logger.error(f"Relationship validation error: {e}")
        
        return result
    
    @staticmethod
    def get_relationship_constraints(relationship_type: RelationshipType) -> Dict[str, Any]:
        """Get constraints for a relationship type.
        
        Args:
            relationship_type: The relationship type
            
        Returns:
            Dictionary of constraints
        """
        constraints = {
            RelationshipType.HAS_GOAL: {
                "cardinality": "one_to_many",
                "uniqueness": "allow_multiple",
                "direction": "outgoing"
            },
            RelationshipType.HAS_TARGET: {
                "cardinality": "one_to_many",
                "uniqueness": "allow_multiple",
                "direction": "outgoing"
            },
            RelationshipType.ANALYZED_BY: {
                "cardinality": "many_to_many",
                "uniqueness": "allow_multiple",
                "direction": "incoming"
            },
            RelationshipType.BELONGS_TO_PERIOD: {
                "cardinality": "many_to_one",
                "uniqueness": "unique_per_period_type",
                "direction": "outgoing"
            },
            RelationshipType.PARENT_PERIOD: {
                "cardinality": "many_to_one",
                "uniqueness": "unique_parent",
                "direction": "outgoing"
            }
        }
        
        return constraints.get(relationship_type, {
            "cardinality": "many_to_many",
            "uniqueness": "allow_multiple",
            "direction": "bidirectional"
        })


class RelationshipHelper:
    """Helper functions for enhanced relationships."""
    
    @staticmethod
    def create_relationship_properties(
        relationship_type: RelationshipType,
        custom_properties: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create relationship properties with defaults.
        
        Args:
            relationship_type: The relationship type
            custom_properties: Custom properties to merge
            
        Returns:
            Complete properties dictionary
        """
        schema = EnhancedRelationshipDefinitions.get_relationship_schema(relationship_type)
        properties = schema.get("default_properties", {}).copy()
        
        if custom_properties:
            properties.update(custom_properties)
        
        # Update timestamp
        properties["updated_at"] = datetime.utcnow().isoformat()
        
        return properties
    
    @staticmethod
    def get_related_relationship_types(relationship_type: RelationshipType) -> List[RelationshipType]:
        """Get relationship types that are commonly used together.
        
        Args:
            relationship_type: The primary relationship type
            
        Returns:
            List of related relationship types
        """
        related_types = {
            RelationshipType.HAS_GOAL: [RelationshipType.HAS_TARGET, RelationshipType.ACHIEVES_TARGET],
            RelationshipType.HAS_TARGET: [RelationshipType.HAS_GOAL, RelationshipType.ACHIEVES_TARGET],
            RelationshipType.ANALYZED_BY: [RelationshipType.RECOMMENDS, RelationshipType.FORECASTS],
            RelationshipType.BELONGS_TO_PERIOD: [RelationshipType.PARENT_PERIOD, RelationshipType.FOLLOWS],
            RelationshipType.FORECASTS: [RelationshipType.PREDICTS, RelationshipType.ANALYZED_BY],
            RelationshipType.TRIGGERS_ACTION: [RelationshipType.REQUIRES_ACTION, RelationshipType.IMPLEMENTS]
        }
        
        return related_types.get(relationship_type, [])
    
    @staticmethod
    def get_inverse_relationship_type(relationship_type: RelationshipType) -> Optional[RelationshipType]:
        """Get the inverse relationship type if it exists.
        
        Args:
            relationship_type: The relationship type
            
        Returns:
            Inverse relationship type or None
        """
        inverse_mappings = {
            RelationshipType.PARENT_OF: RelationshipType.CHILD_OF,
            RelationshipType.CHILD_OF: RelationshipType.PARENT_OF,
            RelationshipType.FOLLOWS: RelationshipType.PRECEDES,
            RelationshipType.PRECEDES: RelationshipType.FOLLOWS
        }
        
        return inverse_mappings.get(relationship_type)
    
    @staticmethod
    def suggest_properties_for_relationship(
        relationship_type: RelationshipType,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Suggest property values based on context.
        
        Args:
            relationship_type: The relationship type
            context: Context information for suggestions
            
        Returns:
            Suggested properties
        """
        suggestions = {}
        
        if context is None:
            context = {}
        
        # Add context-based suggestions
        if relationship_type == RelationshipType.HAS_GOAL:
            if context.get("facility_type") == "manufacturing":
                suggestions["goal_type"] = "production_efficiency"
            elif context.get("facility_type") == "office":
                suggestions["goal_type"] = "workplace_safety"
        
        elif relationship_type == RelationshipType.BELONGS_TO_PERIOD:
            if context.get("metric_frequency") == "daily":
                suggestions["period_type"] = "daily"
                suggestions["aggregation_method"] = "sum"
        
        return suggestions


# Export main classes and enums
__all__ = [
    "RelationshipType",
    "RelationshipProperties",
    "EnhancedRelationshipDefinitions",
    "RelationshipValidator",
    "RelationshipHelper"
]