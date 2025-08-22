"""
EHS-specific vector search configuration module.

This module provides configuration settings, document type mappings, chunk size optimization,
and similarity thresholds specifically designed for EHS (Environmental, Health, Safety)
document retrieval and analysis.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """EHS document types with specific handling requirements."""
    UTILITY_BILL = "utility_bill"
    ENVIRONMENTAL_PERMIT = "environmental_permit"
    COMPLIANCE_REPORT = "compliance_report"
    INCIDENT_REPORT = "incident_report"
    SAFETY_INSPECTION = "safety_inspection"
    WASTE_MANIFEST = "waste_manifest"
    EMISSION_REPORT = "emission_report"
    TRAINING_RECORD = "training_record"
    MAINTENANCE_LOG = "maintenance_log"
    AUDIT_REPORT = "audit_report"
    UNKNOWN = "unknown"


class QueryType(Enum):
    """Types of queries for EHS vector search."""
    EQUIPMENT_STATUS = "equipment_status"
    PERMIT_COMPLIANCE = "permit_compliance"
    CONSUMPTION_ANALYSIS = "consumption_analysis"
    INCIDENT_LOOKUP = "incident_lookup"
    SAFETY_ANALYSIS = "safety_analysis"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    COST_ANALYSIS = "cost_analysis"
    TREND_ANALYSIS = "trend_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    GENERAL_SEARCH = "general_search"


class FacilityType(Enum):
    """Types of facilities for specialized configuration."""
    MANUFACTURING = "manufacturing"
    HEALTHCARE = "healthcare"
    OFFICE = "office"
    WAREHOUSE = "warehouse"
    LABORATORY = "laboratory"
    DATA_CENTER = "data_center"
    RETAIL = "retail"
    MIXED_USE = "mixed_use"


@dataclass
class DocumentTypeConfig:
    """Configuration for specific document types."""
    chunk_size: int
    chunk_overlap: int
    similarity_threshold: float
    max_results: int
    section_markers: List[str]
    metadata_priority: List[str]
    content_weight: float = 1.0
    recency_weight: float = 0.1
    facility_weight: float = 0.05
    
    # Document-specific processing flags
    preserve_tables: bool = True
    preserve_numerical_data: bool = True
    extract_dates: bool = True
    extract_facilities: bool = True
    extract_compliance_info: bool = True


@dataclass
class QueryTypeConfig:
    """Configuration for specific query types."""
    preferred_document_types: List[DocumentType]
    similarity_threshold: float
    max_results: int
    date_range_priority: bool = False
    facility_priority: bool = False
    rerank_by_recency: bool = False
    boost_keywords: List[str] = field(default_factory=list)
    filter_keywords: List[str] = field(default_factory=list)


@dataclass
class FacilityConfig:
    """Configuration for specific facility types."""
    priority_document_types: List[DocumentType]
    consumption_tracking: List[str]  # Types of consumption to track
    compliance_requirements: List[str]  # Regulatory requirements
    risk_factors: List[str]  # Facility-specific risk factors
    reporting_frequency: str  # Expected reporting frequency


class EHSVectorConfig:
    """
    Comprehensive configuration for EHS vector search operations.
    
    Provides optimized settings for different document types, query types,
    and facility configurations to improve retrieval accuracy and relevance.
    """
    
    def __init__(self):
        """Initialize EHS vector configuration with optimized defaults."""
        self.document_type_configs = self._init_document_type_configs()
        self.query_type_configs = self._init_query_type_configs()
        self.facility_configs = self._init_facility_configs()
        self.similarity_thresholds = self._init_similarity_thresholds()
        self.keyword_mappings = self._init_keyword_mappings()
        
        logger.info("Initialized EHS Vector Configuration")
    
    def _init_document_type_configs(self) -> Dict[DocumentType, DocumentTypeConfig]:
        """Initialize document type specific configurations."""
        configs = {}
        
        # Utility Bills - Precise data retrieval with smaller chunks
        configs[DocumentType.UTILITY_BILL] = DocumentTypeConfig(
            chunk_size=800,
            chunk_overlap=100,
            similarity_threshold=0.75,
            max_results=15,
            section_markers=["Account", "Usage", "Charges", "Summary", "Consumption", "Billing", "Meter"],
            metadata_priority=["facility", "utility_type", "billing_period", "consumption_data"],
            content_weight=1.0,
            recency_weight=0.2,  # Higher recency weight for bills
            preserve_numerical_data=True
        )
        
        # Environmental Permits - Context-rich retrieval with larger chunks
        configs[DocumentType.ENVIRONMENTAL_PERMIT] = DocumentTypeConfig(
            chunk_size=1500,
            chunk_overlap=300,
            similarity_threshold=0.7,
            max_results=10,
            section_markers=["Permit", "Conditions", "Requirements", "Limits", "Monitoring", "Compliance"],
            metadata_priority=["permit_number", "facility", "regulatory_agency", "permit_type"],
            content_weight=1.0,
            recency_weight=0.05,  # Lower recency weight for permits
            facility_weight=0.1
        )
        
        # Compliance Reports - Balanced retrieval for findings and recommendations
        configs[DocumentType.COMPLIANCE_REPORT] = DocumentTypeConfig(
            chunk_size=1200,
            chunk_overlap=200,
            similarity_threshold=0.72,
            max_results=12,
            section_markers=["Executive", "Summary", "Findings", "Compliance", "Recommendations", "Actions"],
            metadata_priority=["facility", "compliance_status", "regulations_referenced", "findings"],
            content_weight=1.0,
            recency_weight=0.15
        )
        
        # Incident Reports - Narrative preservation with medium chunks
        configs[DocumentType.INCIDENT_REPORT] = DocumentTypeConfig(
            chunk_size=1000,
            chunk_overlap=150,
            similarity_threshold=0.68,
            max_results=8,
            section_markers=["Incident", "Description", "Analysis", "Cause", "Actions", "Follow-up"],
            metadata_priority=["facility", "incident_date", "severity", "injury_types"],
            content_weight=1.0,
            recency_weight=0.3,  # High recency weight for incidents
            preserve_tables=False  # Focus on narrative content
        )
        
        # Safety Inspections - Detailed findings with structured data
        configs[DocumentType.SAFETY_INSPECTION] = DocumentTypeConfig(
            chunk_size=1100,
            chunk_overlap=180,
            similarity_threshold=0.7,
            max_results=10,
            section_markers=["Inspection", "Findings", "Deficiencies", "Corrective", "Recommendations"],
            metadata_priority=["facility", "inspection_date", "inspector", "deficiencies"],
            content_weight=1.0,
            recency_weight=0.2
        )
        
        # Waste Manifests - Precise tracking with small chunks
        configs[DocumentType.WASTE_MANIFEST] = DocumentTypeConfig(
            chunk_size=600,
            chunk_overlap=80,
            similarity_threshold=0.8,
            max_results=20,
            section_markers=["Generator", "Transporter", "Facility", "Waste", "Manifest"],
            metadata_priority=["facility", "waste_type", "quantity", "disposal_facility"],
            preserve_numerical_data=True
        )
        
        # Emission Reports - Technical data with medium chunks
        configs[DocumentType.EMISSION_REPORT] = DocumentTypeConfig(
            chunk_size=1000,
            chunk_overlap=150,
            similarity_threshold=0.75,
            max_results=12,
            section_markers=["Emissions", "Monitoring", "Calculations", "Sources", "Limits"],
            metadata_priority=["facility", "emission_type", "monitoring_period", "values"],
            preserve_numerical_data=True
        )
        
        # Training Records - Personnel focused with smaller chunks
        configs[DocumentType.TRAINING_RECORD] = DocumentTypeConfig(
            chunk_size=700,
            chunk_overlap=100,
            similarity_threshold=0.7,
            max_results=15,
            section_markers=["Training", "Attendees", "Curriculum", "Certification", "Completion"],
            metadata_priority=["facility", "training_type", "date", "attendees"],
            recency_weight=0.25
        )
        
        # Default configuration for unknown document types
        configs[DocumentType.UNKNOWN] = DocumentTypeConfig(
            chunk_size=1000,
            chunk_overlap=200,
            similarity_threshold=0.7,
            max_results=10,
            section_markers=["Summary", "Overview", "Details", "Analysis", "Conclusion"],
            metadata_priority=["facility", "document_type", "date"]
        )
        
        return configs
    
    def _init_query_type_configs(self) -> Dict[QueryType, QueryTypeConfig]:
        """Initialize query type specific configurations."""
        configs = {}
        
        # Equipment Status Queries
        configs[QueryType.EQUIPMENT_STATUS] = QueryTypeConfig(
            preferred_document_types=[
                DocumentType.MAINTENANCE_LOG,
                DocumentType.SAFETY_INSPECTION,
                DocumentType.COMPLIANCE_REPORT
            ],
            similarity_threshold=0.7,
            max_results=10,
            facility_priority=True,
            boost_keywords=["equipment", "machinery", "device", "system", "maintenance", "status"],
            filter_keywords=["repair", "replacement", "calibration", "inspection"]
        )
        
        # Permit Compliance Queries
        configs[QueryType.PERMIT_COMPLIANCE] = QueryTypeConfig(
            preferred_document_types=[
                DocumentType.ENVIRONMENTAL_PERMIT,
                DocumentType.COMPLIANCE_REPORT,
                DocumentType.AUDIT_REPORT
            ],
            similarity_threshold=0.72,
            max_results=8,
            date_range_priority=True,
            boost_keywords=["permit", "compliance", "regulation", "requirement", "condition"],
            filter_keywords=["violation", "non-compliance", "deviation", "corrective"]
        )
        
        # Consumption Analysis Queries
        configs[QueryType.CONSUMPTION_ANALYSIS] = QueryTypeConfig(
            preferred_document_types=[
                DocumentType.UTILITY_BILL,
                DocumentType.EMISSION_REPORT
            ],
            similarity_threshold=0.75,
            max_results=15,
            date_range_priority=True,
            rerank_by_recency=True,
            boost_keywords=["consumption", "usage", "energy", "water", "gas", "electricity"],
            filter_keywords=["kwh", "gallons", "therms", "ccf", "meter", "reading"]
        )
        
        # Incident Lookup Queries
        configs[QueryType.INCIDENT_LOOKUP] = QueryTypeConfig(
            preferred_document_types=[
                DocumentType.INCIDENT_REPORT,
                DocumentType.SAFETY_INSPECTION
            ],
            similarity_threshold=0.68,
            max_results=12,
            rerank_by_recency=True,
            boost_keywords=["incident", "accident", "injury", "near miss", "safety"],
            filter_keywords=["cause", "investigation", "corrective", "prevention"]
        )
        
        # Safety Analysis Queries
        configs[QueryType.SAFETY_ANALYSIS] = QueryTypeConfig(
            preferred_document_types=[
                DocumentType.INCIDENT_REPORT,
                DocumentType.SAFETY_INSPECTION,
                DocumentType.TRAINING_RECORD
            ],
            similarity_threshold=0.7,
            max_results=10,
            boost_keywords=["safety", "hazard", "risk", "protection", "training"],
            filter_keywords=["ppe", "procedure", "protocol", "emergency"]
        )
        
        # Regulatory Compliance Queries
        configs[QueryType.REGULATORY_COMPLIANCE] = QueryTypeConfig(
            preferred_document_types=[
                DocumentType.COMPLIANCE_REPORT,
                DocumentType.ENVIRONMENTAL_PERMIT,
                DocumentType.AUDIT_REPORT
            ],
            similarity_threshold=0.72,
            max_results=8,
            boost_keywords=["regulation", "compliance", "regulatory", "requirement", "standard"],
            filter_keywords=["cfr", "osha", "epa", "violation", "audit"]
        )
        
        # Cost Analysis Queries
        configs[QueryType.COST_ANALYSIS] = QueryTypeConfig(
            preferred_document_types=[
                DocumentType.UTILITY_BILL,
                DocumentType.COMPLIANCE_REPORT
            ],
            similarity_threshold=0.7,
            max_results=12,
            date_range_priority=True,
            boost_keywords=["cost", "expense", "budget", "savings", "financial"],
            filter_keywords=["rate", "charge", "fee", "penalty", "investment"]
        )
        
        # Trend Analysis Queries
        configs[QueryType.TREND_ANALYSIS] = QueryTypeConfig(
            preferred_document_types=[
                DocumentType.UTILITY_BILL,
                DocumentType.EMISSION_REPORT,
                DocumentType.COMPLIANCE_REPORT
            ],
            similarity_threshold=0.7,
            max_results=20,
            date_range_priority=True,
            rerank_by_recency=False,  # Need historical data
            boost_keywords=["trend", "pattern", "increase", "decrease", "change"],
            filter_keywords=["monthly", "quarterly", "annual", "historical"]
        )
        
        # Risk Assessment Queries
        configs[QueryType.RISK_ASSESSMENT] = QueryTypeConfig(
            preferred_document_types=[
                DocumentType.INCIDENT_REPORT,
                DocumentType.COMPLIANCE_REPORT,
                DocumentType.SAFETY_INSPECTION
            ],
            similarity_threshold=0.68,
            max_results=15,
            boost_keywords=["risk", "hazard", "danger", "threat", "vulnerability"],
            filter_keywords=["assessment", "mitigation", "control", "prevention"]
        )
        
        # General Search - Balanced approach
        configs[QueryType.GENERAL_SEARCH] = QueryTypeConfig(
            preferred_document_types=list(DocumentType),
            similarity_threshold=0.7,
            max_results=10,
            boost_keywords=[],
            filter_keywords=[]
        )
        
        return configs
    
    def _init_facility_configs(self) -> Dict[FacilityType, FacilityConfig]:
        """Initialize facility type specific configurations."""
        configs = {}
        
        # Manufacturing Facilities
        configs[FacilityType.MANUFACTURING] = FacilityConfig(
            priority_document_types=[
                DocumentType.UTILITY_BILL,
                DocumentType.ENVIRONMENTAL_PERMIT,
                DocumentType.EMISSION_REPORT,
                DocumentType.WASTE_MANIFEST
            ],
            consumption_tracking=["electricity", "natural_gas", "water", "compressed_air"],
            compliance_requirements=["air_quality", "water_discharge", "waste_management", "safety"],
            risk_factors=["chemical_exposure", "machinery_accidents", "fire_hazard"],
            reporting_frequency="monthly"
        )
        
        # Healthcare Facilities
        configs[FacilityType.HEALTHCARE] = FacilityConfig(
            priority_document_types=[
                DocumentType.UTILITY_BILL,
                DocumentType.WASTE_MANIFEST,
                DocumentType.SAFETY_INSPECTION,
                DocumentType.TRAINING_RECORD
            ],
            consumption_tracking=["electricity", "water", "medical_gas"],
            compliance_requirements=["medical_waste", "infection_control", "safety", "hvac"],
            risk_factors=["infection_risk", "chemical_exposure", "patient_safety"],
            reporting_frequency="monthly"
        )
        
        # Office Buildings
        configs[FacilityType.OFFICE] = FacilityConfig(
            priority_document_types=[
                DocumentType.UTILITY_BILL,
                DocumentType.SAFETY_INSPECTION,
                DocumentType.TRAINING_RECORD
            ],
            consumption_tracking=["electricity", "water", "hvac"],
            compliance_requirements=["building_safety", "fire_safety", "accessibility"],
            risk_factors=["fire_hazard", "workplace_injury", "indoor_air_quality"],
            reporting_frequency="quarterly"
        )
        
        # Warehouses
        configs[FacilityType.WAREHOUSE] = FacilityConfig(
            priority_document_types=[
                DocumentType.UTILITY_BILL,
                DocumentType.SAFETY_INSPECTION,
                DocumentType.INCIDENT_REPORT
            ],
            consumption_tracking=["electricity", "heating", "material_handling"],
            compliance_requirements=["safety", "fire_safety", "structural"],
            risk_factors=["forklift_accidents", "falling_objects", "fire_hazard"],
            reporting_frequency="quarterly"
        )
        
        # Laboratories
        configs[FacilityType.LABORATORY] = FacilityConfig(
            priority_document_types=[
                DocumentType.WASTE_MANIFEST,
                DocumentType.ENVIRONMENTAL_PERMIT,
                DocumentType.SAFETY_INSPECTION,
                DocumentType.TRAINING_RECORD
            ],
            consumption_tracking=["electricity", "water", "specialty_gases", "chemicals"],
            compliance_requirements=["chemical_safety", "waste_management", "ventilation", "training"],
            risk_factors=["chemical_exposure", "biological_hazards", "fire_explosion"],
            reporting_frequency="monthly"
        )
        
        return configs
    
    def _init_similarity_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize similarity thresholds for different scenarios."""
        return {
            "document_type": {
                # Precise thresholds for document types
                DocumentType.UTILITY_BILL.value: 0.75,
                DocumentType.WASTE_MANIFEST.value: 0.8,
                DocumentType.INCIDENT_REPORT.value: 0.68,
                DocumentType.ENVIRONMENTAL_PERMIT.value: 0.7,
                DocumentType.COMPLIANCE_REPORT.value: 0.72,
                "default": 0.7
            },
            "query_type": {
                # Adjusted thresholds based on query complexity
                QueryType.CONSUMPTION_ANALYSIS.value: 0.75,
                QueryType.PERMIT_COMPLIANCE.value: 0.72,
                QueryType.INCIDENT_LOOKUP.value: 0.68,
                QueryType.EQUIPMENT_STATUS.value: 0.7,
                "default": 0.7
            },
            "content_length": {
                # Thresholds based on chunk size
                "short": 0.75,  # < 500 chars
                "medium": 0.7,  # 500-1500 chars
                "long": 0.65    # > 1500 chars
            }
        }
    
    def _init_keyword_mappings(self) -> Dict[str, List[str]]:
        """Initialize keyword mappings for query enhancement."""
        return {
            "utility_consumption": [
                "electricity", "electric", "kwh", "kilowatt",
                "water", "gallon", "gal", "usage",
                "gas", "natural gas", "therm", "ccf",
                "consumption", "usage", "meter", "reading"
            ],
            "compliance": [
                "permit", "license", "regulation", "requirement",
                "compliance", "violation", "non-compliance",
                "cfr", "epa", "osha", "standard", "code"
            ],
            "safety": [
                "incident", "accident", "injury", "near miss",
                "safety", "hazard", "risk", "protection",
                "ppe", "training", "procedure", "emergency"
            ],
            "environmental": [
                "emission", "discharge", "waste", "pollution",
                "air quality", "water quality", "environmental",
                "monitoring", "sampling", "analysis"
            ],
            "facilities": [
                "facility", "building", "plant", "site",
                "location", "address", "campus", "complex"
            ]
        }
    
    def get_document_config(self, document_type: DocumentType) -> DocumentTypeConfig:
        """Get configuration for a specific document type."""
        return self.document_type_configs.get(document_type, self.document_type_configs[DocumentType.UNKNOWN])
    
    def get_query_config(self, query_type: QueryType) -> QueryTypeConfig:
        """Get configuration for a specific query type."""
        return self.query_type_configs.get(query_type, self.query_type_configs[QueryType.GENERAL_SEARCH])
    
    def get_facility_config(self, facility_type: FacilityType) -> FacilityConfig:
        """Get configuration for a specific facility type."""
        return self.facility_configs.get(facility_type, self.facility_configs[FacilityType.MIXED_USE])
    
    def get_similarity_threshold(
        self,
        document_type: Optional[DocumentType] = None,
        query_type: Optional[QueryType] = None,
        content_length: Optional[int] = None
    ) -> float:
        """
        Get appropriate similarity threshold based on context.
        
        Args:
            document_type: Type of document being searched
            query_type: Type of query being performed
            content_length: Length of content being compared
            
        Returns:
            Appropriate similarity threshold
        """
        thresholds = []
        
        # Document type threshold
        if document_type:
            dt_threshold = self.similarity_thresholds["document_type"].get(
                document_type.value,
                self.similarity_thresholds["document_type"]["default"]
            )
            thresholds.append(dt_threshold)
        
        # Query type threshold
        if query_type:
            qt_threshold = self.similarity_thresholds["query_type"].get(
                query_type.value,
                self.similarity_thresholds["query_type"]["default"]
            )
            thresholds.append(qt_threshold)
        
        # Content length threshold
        if content_length:
            if content_length < 500:
                cl_threshold = self.similarity_thresholds["content_length"]["short"]
            elif content_length < 1500:
                cl_threshold = self.similarity_thresholds["content_length"]["medium"]
            else:
                cl_threshold = self.similarity_thresholds["content_length"]["long"]
            thresholds.append(cl_threshold)
        
        # Return average of applicable thresholds, or default
        if thresholds:
            return sum(thresholds) / len(thresholds)
        else:
            return 0.7
    
    def get_document_type_priority(self, query_text: str) -> Dict[DocumentType, float]:
        """
        Determine document type priority based on query content.
        
        Args:
            query_text: Natural language query
            
        Returns:
            Dictionary mapping document types to priority scores
        """
        query_lower = query_text.lower()
        priorities = {}
        
        # Initialize all document types with base priority
        for doc_type in DocumentType:
            priorities[doc_type] = 0.0
        
        # Utility bill keywords
        if any(keyword in query_lower for keyword in self.keyword_mappings["utility_consumption"]):
            priorities[DocumentType.UTILITY_BILL] += 0.8
            priorities[DocumentType.EMISSION_REPORT] += 0.3
        
        # Compliance keywords
        if any(keyword in query_lower for keyword in self.keyword_mappings["compliance"]):
            priorities[DocumentType.COMPLIANCE_REPORT] += 0.8
            priorities[DocumentType.ENVIRONMENTAL_PERMIT] += 0.7
            priorities[DocumentType.AUDIT_REPORT] += 0.6
        
        # Safety keywords
        if any(keyword in query_lower for keyword in self.keyword_mappings["safety"]):
            priorities[DocumentType.INCIDENT_REPORT] += 0.8
            priorities[DocumentType.SAFETY_INSPECTION] += 0.7
            priorities[DocumentType.TRAINING_RECORD] += 0.5
        
        # Environmental keywords
        if any(keyword in query_lower for keyword in self.keyword_mappings["environmental"]):
            priorities[DocumentType.ENVIRONMENTAL_PERMIT] += 0.8
            priorities[DocumentType.EMISSION_REPORT] += 0.7
            priorities[DocumentType.WASTE_MANIFEST] += 0.6
        
        # Specific document indicators
        document_indicators = {
            "bill": DocumentType.UTILITY_BILL,
            "permit": DocumentType.ENVIRONMENTAL_PERMIT,
            "incident": DocumentType.INCIDENT_REPORT,
            "manifest": DocumentType.WASTE_MANIFEST,
            "inspection": DocumentType.SAFETY_INSPECTION,
            "training": DocumentType.TRAINING_RECORD,
            "audit": DocumentType.AUDIT_REPORT,
            "maintenance": DocumentType.MAINTENANCE_LOG
        }
        
        for indicator, doc_type in document_indicators.items():
            if indicator in query_lower:
                priorities[doc_type] += 0.5
        
        return priorities
    
    def enhance_query(self, query: str, query_type: Optional[QueryType] = None) -> str:
        """
        Enhance query with relevant keywords based on query type.
        
        Args:
            query: Original query string
            query_type: Type of query (optional)
            
        Returns:
            Enhanced query string
        """
        if not query_type:
            return query
        
        config = self.get_query_config(query_type)
        
        # Add boost keywords if not already present
        query_lower = query.lower()
        missing_keywords = []
        
        for keyword in config.boost_keywords:
            if keyword not in query_lower:
                missing_keywords.append(keyword)
        
        # Add up to 3 missing keywords to avoid query dilution
        if missing_keywords:
            enhanced_keywords = missing_keywords[:3]
            enhanced_query = f"{query} {' '.join(enhanced_keywords)}"
        else:
            enhanced_query = query
        
        return enhanced_query
    
    def get_optimal_chunk_size(
        self,
        document_type: DocumentType,
        query_type: Optional[QueryType] = None
    ) -> Tuple[int, int]:
        """
        Get optimal chunk size and overlap for given document and query types.
        
        Args:
            document_type: Type of document
            query_type: Type of query (optional)
            
        Returns:
            Tuple of (chunk_size, chunk_overlap)
        """
        doc_config = self.get_document_config(document_type)
        
        # Adjust based on query type if provided
        if query_type:
            query_config = self.get_query_config(query_type)
            
            # For detailed analysis queries, use larger chunks
            if query_type in [QueryType.TREND_ANALYSIS, QueryType.RISK_ASSESSMENT]:
                chunk_size = min(doc_config.chunk_size + 300, 2000)
                chunk_overlap = min(doc_config.chunk_overlap + 50, 400)
            
            # For precise lookups, use smaller chunks
            elif query_type in [QueryType.CONSUMPTION_ANALYSIS, QueryType.INCIDENT_LOOKUP]:
                chunk_size = max(doc_config.chunk_size - 200, 600)
                chunk_overlap = max(doc_config.chunk_overlap - 30, 50)
            
            else:
                chunk_size = doc_config.chunk_size
                chunk_overlap = doc_config.chunk_overlap
        else:
            chunk_size = doc_config.chunk_size
            chunk_overlap = doc_config.chunk_overlap
        
        return chunk_size, chunk_overlap
    
    def get_reranking_weights(
        self,
        query_type: Optional[QueryType] = None,
        facility_type: Optional[FacilityType] = None
    ) -> Dict[str, float]:
        """
        Get reranking weights for different factors.
        
        Args:
            query_type: Type of query
            facility_type: Type of facility
            
        Returns:
            Dictionary of reranking weights
        """
        weights = {
            "content_similarity": 1.0,
            "recency": 0.1,
            "facility_match": 0.05,
            "document_type": 0.1,
            "metadata_quality": 0.05
        }
        
        # Adjust based on query type
        if query_type:
            if query_type == QueryType.CONSUMPTION_ANALYSIS:
                weights["recency"] = 0.2
                weights["document_type"] = 0.15
            elif query_type == QueryType.INCIDENT_LOOKUP:
                weights["recency"] = 0.3
                weights["facility_match"] = 0.1
            elif query_type == QueryType.PERMIT_COMPLIANCE:
                weights["recency"] = 0.05
                weights["metadata_quality"] = 0.15
        
        # Adjust based on facility type
        if facility_type:
            facility_config = self.get_facility_config(facility_type)
            if facility_config.reporting_frequency == "monthly":
                weights["recency"] += 0.05
            elif facility_config.reporting_frequency == "quarterly":
                weights["recency"] -= 0.02
        
        return weights