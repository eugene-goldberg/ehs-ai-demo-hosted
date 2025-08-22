"""
Base retriever interface for EHS Analytics.

This module defines the abstract base class and common interfaces for all
retrieval strategies in the EHS Analytics system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel


class RetrievalStrategy(str, Enum):
    """Enumeration of available retrieval strategies."""
    
    TEXT2CYPHER = "text2cypher"
    VECTOR = "vector"
    HYBRID = "hybrid"
    VECTOR_CYPHER = "vector_cypher"
    HYBRID_CYPHER = "hybrid_cypher"


class QueryType(str, Enum):
    """Types of EHS queries that can be processed."""
    
    CONSUMPTION = "consumption"  # Utility consumption analysis
    EFFICIENCY = "efficiency"   # Equipment efficiency queries
    COMPLIANCE = "compliance"   # Permit compliance checks
    EMISSIONS = "emissions"     # Emission tracking and analysis
    RISK = "risk"              # Risk assessment queries
    RECOMMENDATION = "recommendation"  # Actionable recommendations
    GENERAL = "general"        # General information queries


@dataclass
class RetrievalMetadata:
    """Metadata associated with retrieval results."""
    
    strategy: RetrievalStrategy
    query_type: QueryType
    confidence_score: float
    execution_time_ms: float
    cypher_query: Optional[str] = None
    vector_similarity_scores: Optional[List[float]] = None
    nodes_retrieved: int = 0
    relationships_retrieved: int = 0
    error_message: Optional[str] = None


class RetrievalResult(BaseModel):
    """Result container for retrieval operations."""
    
    data: List[Dict[str, Any]]
    metadata: RetrievalMetadata
    success: bool
    message: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True


class BaseRetriever(ABC):
    """
    Abstract base class for all retrieval strategies in EHS Analytics.
    
    This class defines the common interface that all retrievers must implement,
    ensuring consistency across different retrieval strategies like Text2Cypher,
    Vector, and Hybrid approaches.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the retriever with configuration.
        
        Args:
            config: Configuration dictionary containing strategy-specific settings
        """
        self.config = config
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the retriever with necessary connections and resources.
        
        This method should set up database connections, load models,
        and perform any other initialization required by the specific strategy.
        """
        pass
    
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        query_type: QueryType = QueryType.GENERAL,
        limit: int = 10,
        **kwargs
    ) -> RetrievalResult:
        """
        Execute retrieval for the given query.
        
        Args:
            query: Natural language query from the user
            query_type: Type of EHS query being processed
            limit: Maximum number of results to return
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            RetrievalResult containing data, metadata, and execution information
        """
        pass
    
    @abstractmethod
    async def validate_query(self, query: str) -> bool:
        """
        Validate if the query can be processed by this retriever.
        
        Args:
            query: Natural language query to validate
            
        Returns:
            True if the query can be processed, False otherwise
        """
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the retriever.
        
        Returns:
            Dictionary containing health status information
        """
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "strategy": self.get_strategy().value,
            "initialized": self._initialized
        }
    
    @abstractmethod
    def get_strategy(self) -> RetrievalStrategy:
        """
        Get the retrieval strategy identifier.
        
        Returns:
            RetrievalStrategy enum value for this retriever
        """
        pass
    
    async def cleanup(self) -> None:
        """
        Clean up resources used by the retriever.
        
        This method should close database connections, release memory,
        and perform any other cleanup required.
        """
        self._initialized = False


class EHSSchemaAware:
    """
    Mixin class providing EHS-specific schema knowledge.
    
    This class contains the knowledge about EHS data model structure,
    node types, relationships, and common query patterns.
    """
    
    # EHS Node Types
    NODE_TYPES = {
        "Facility": {
            "properties": ["id", "name", "address", "type", "capacity", "created_at"],
            "description": "Physical facilities or locations"
        },
        "Equipment": {
            "properties": ["id", "name", "type", "manufacturer", "model", "efficiency_rating", "installation_date"],
            "description": "Equipment and machinery at facilities"
        },
        "Permit": {
            "properties": ["id", "permit_number", "type", "status", "issue_date", "expiry_date", "authority"],
            "description": "Environmental permits and licenses"
        },
        "UtilityBill": {
            "properties": ["id", "utility_type", "amount", "cost", "billing_period", "meter_reading"],
            "description": "Utility consumption records (water, electricity, gas)"
        },
        "Emission": {
            "properties": ["id", "emission_type", "amount", "unit", "measurement_date", "source"],
            "description": "Emission measurements and records"
        },
        "WasteRecord": {
            "properties": ["id", "waste_type", "amount", "disposal_method", "disposal_date", "cost"],
            "description": "Waste generation and disposal records"
        },
        "Incident": {
            "properties": ["id", "type", "severity", "date", "description", "status", "corrective_actions"],
            "description": "Safety and environmental incidents"
        }
    }
    
    # EHS Relationships
    RELATIONSHIPS = {
        "HAS_EQUIPMENT": "Facility has equipment",
        "HAS_PERMIT": "Facility has permits",
        "RECORDED_AT": "Records associated with facilities",
        "INVOLVES_EQUIPMENT": "Records involving specific equipment",
        "REQUIRES_PERMIT": "Activities requiring permits",
        "GENERATED_BY": "Waste/emissions generated by equipment",
        "OCCURRED_AT": "Incidents occurred at facilities",
        "MEASURED_BY": "Measurements taken by equipment"
    }
    
    # Common query patterns for each query type
    QUERY_PATTERNS = {
        QueryType.CONSUMPTION: [
            "utility consumption over time",
            "energy usage by facility",
            "water consumption trends",
            "monthly billing analysis"
        ],
        QueryType.EFFICIENCY: [
            "equipment efficiency ratings",
            "energy efficiency comparison",
            "performance optimization",
            "maintenance scheduling"
        ],
        QueryType.COMPLIANCE: [
            "permit status checks",
            "compliance deadlines",
            "regulatory requirements",
            "permit renewals"
        ],
        QueryType.EMISSIONS: [
            "emission levels tracking",
            "carbon footprint analysis",
            "pollution monitoring",
            "emission source identification"
        ],
        QueryType.RISK: [
            "risk assessment analysis",
            "incident history",
            "safety metrics",
            "hazard identification"
        ],
        QueryType.RECOMMENDATION: [
            "cost reduction opportunities",
            "efficiency improvements",
            "compliance recommendations",
            "best practices"
        ]
    }
    
    def get_relevant_nodes(self, query_type: QueryType) -> List[str]:
        """
        Get relevant node types for a specific query type.
        
        Args:
            query_type: Type of EHS query
            
        Returns:
            List of relevant node type names
        """
        type_mapping = {
            QueryType.CONSUMPTION: ["UtilityBill", "Facility", "Equipment"],
            QueryType.EFFICIENCY: ["Equipment", "Facility", "UtilityBill"],
            QueryType.COMPLIANCE: ["Permit", "Facility", "Incident"],
            QueryType.EMISSIONS: ["Emission", "Equipment", "Facility"],
            QueryType.RISK: ["Incident", "Facility", "Equipment", "Emission"],
            QueryType.RECOMMENDATION: ["Facility", "Equipment", "UtilityBill", "Permit"],
            QueryType.GENERAL: list(self.NODE_TYPES.keys())
        }
        return type_mapping.get(query_type, list(self.NODE_TYPES.keys()))
    
    def get_schema_context(self, query_type: QueryType) -> str:
        """
        Generate schema context string for LLM prompts.
        
        Args:
            query_type: Type of EHS query
            
        Returns:
            Formatted schema context string
        """
        relevant_nodes = self.get_relevant_nodes(query_type)
        
        context = "EHS Database Schema:\n\n"
        context += "Node Types:\n"
        
        for node_type in relevant_nodes:
            if node_type in self.NODE_TYPES:
                node_info = self.NODE_TYPES[node_type]
                context += f"- {node_type}: {node_info['description']}\n"
                context += f"  Properties: {', '.join(node_info['properties'])}\n\n"
        
        context += "Relationships:\n"
        for rel, desc in self.RELATIONSHIPS.items():
            context += f"- {rel}: {desc}\n"
        
        return context