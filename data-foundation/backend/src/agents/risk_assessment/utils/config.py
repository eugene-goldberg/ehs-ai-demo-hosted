"""
Risk Assessment Agent Configuration System

This module provides comprehensive configuration management for the risk assessment agent,
including environment variables, risk thresholds, LLM models, and industry-specific settings.
"""

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import os
from pydantic import BaseSettings, Field, validator, root_validator
from pydantic_settings import BaseSettings as PydanticSettings
import json
import logging

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Risk level enumeration"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


class DocumentType(str, Enum):
    """Document type enumeration for risk profiling"""
    SDS = "sds"  # Safety Data Sheet
    INCIDENT_REPORT = "incident_report"
    AUDIT_REPORT = "audit_report"
    INSPECTION_REPORT = "inspection_report"
    TRAINING_MATERIAL = "training_material"
    POLICY_DOCUMENT = "policy_document"
    PROCEDURE = "procedure"
    REGULATORY_FILING = "regulatory_filing"
    PERMIT = "permit"
    CERTIFICATE = "certificate"
    UNKNOWN = "unknown"


class Industry(str, Enum):
    """Industry enumeration for risk assessment customization"""
    MANUFACTURING = "manufacturing"
    CHEMICAL = "chemical"
    OIL_GAS = "oil_gas"
    MINING = "mining"
    CONSTRUCTION = "construction"
    HEALTHCARE = "healthcare"
    FOOD_BEVERAGE = "food_beverage"
    PHARMACEUTICALS = "pharmaceuticals"
    UTILITIES = "utilities"
    TRANSPORTATION = "transportation"
    GENERAL = "general"


class AssessmentType(str, Enum):
    """Types of risk assessments"""
    CHEMICAL_EXPOSURE = "chemical_exposure"
    PHYSICAL_HAZARD = "physical_hazard"
    ENVIRONMENTAL_IMPACT = "environmental_impact"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    OPERATIONAL_SAFETY = "operational_safety"
    PROCESS_SAFETY = "process_safety"
    WORKPLACE_SAFETY = "workplace_safety"
    COMPREHENSIVE = "comprehensive"


class RiskThresholds(BaseSettings):
    """Risk assessment threshold configurations"""
    
    # Chemical Risk Thresholds
    chemical_exposure_critical: float = Field(default=9.0, description="Critical chemical exposure risk threshold")
    chemical_exposure_high: float = Field(default=7.0, description="High chemical exposure risk threshold")
    chemical_exposure_medium: float = Field(default=5.0, description="Medium chemical exposure risk threshold")
    chemical_exposure_low: float = Field(default=3.0, description="Low chemical exposure risk threshold")
    
    # Physical Hazard Thresholds
    physical_hazard_critical: float = Field(default=8.5, description="Critical physical hazard threshold")
    physical_hazard_high: float = Field(default=6.5, description="High physical hazard threshold")
    physical_hazard_medium: float = Field(default=4.5, description="Medium physical hazard threshold")
    physical_hazard_low: float = Field(default=2.5, description="Low physical hazard threshold")
    
    # Environmental Impact Thresholds
    environmental_critical: float = Field(default=8.0, description="Critical environmental impact threshold")
    environmental_high: float = Field(default=6.0, description="High environmental impact threshold")
    environmental_medium: float = Field(default=4.0, description="Medium environmental impact threshold")
    environmental_low: float = Field(default=2.0, description="Low environmental impact threshold")
    
    # Regulatory Compliance Thresholds
    compliance_critical: float = Field(default=9.5, description="Critical compliance risk threshold")
    compliance_high: float = Field(default=7.5, description="High compliance risk threshold")
    compliance_medium: float = Field(default=5.5, description="Medium compliance risk threshold")
    compliance_low: float = Field(default=3.5, description="Low compliance risk threshold")
    
    # Operational Safety Thresholds
    operational_critical: float = Field(default=8.0, description="Critical operational safety threshold")
    operational_high: float = Field(default=6.0, description="High operational safety threshold")
    operational_medium: float = Field(default=4.0, description="Medium operational safety threshold")
    operational_low: float = Field(default=2.0, description="Low operational safety threshold")

    class Config:
        env_prefix = "RISK_THRESHOLD_"
        case_sensitive = False


class DocumentRiskProfile(BaseSettings):
    """Risk profiles for different document types"""
    
    # Base risk multipliers by document type
    sds_multiplier: float = Field(default=1.5, description="SDS document risk multiplier")
    incident_report_multiplier: float = Field(default=1.8, description="Incident report risk multiplier")
    audit_report_multiplier: float = Field(default=1.2, description="Audit report risk multiplier")
    inspection_report_multiplier: float = Field(default=1.3, description="Inspection report risk multiplier")
    training_material_multiplier: float = Field(default=0.8, description="Training material risk multiplier")
    policy_document_multiplier: float = Field(default=1.0, description="Policy document risk multiplier")
    procedure_multiplier: float = Field(default=1.1, description="Procedure risk multiplier")
    regulatory_filing_multiplier: float = Field(default=1.4, description="Regulatory filing risk multiplier")
    permit_multiplier: float = Field(default=1.3, description="Permit risk multiplier")
    certificate_multiplier: float = Field(default=0.9, description="Certificate risk multiplier")
    unknown_multiplier: float = Field(default=1.0, description="Unknown document type risk multiplier")
    
    # Confidence thresholds for document type classification
    classification_confidence_threshold: float = Field(default=0.7, description="Document classification confidence threshold")
    
    class Config:
        env_prefix = "DOC_RISK_"
        case_sensitive = False


class IndustrySettings(BaseSettings):
    """Industry-specific configuration settings"""
    
    # Industry risk adjustment factors
    manufacturing_adjustment: float = Field(default=1.2, description="Manufacturing industry risk adjustment")
    chemical_adjustment: float = Field(default=1.5, description="Chemical industry risk adjustment")
    oil_gas_adjustment: float = Field(default=1.4, description="Oil & Gas industry risk adjustment")
    mining_adjustment: float = Field(default=1.3, description="Mining industry risk adjustment")
    construction_adjustment: float = Field(default=1.2, description="Construction industry risk adjustment")
    healthcare_adjustment: float = Field(default=1.1, description="Healthcare industry risk adjustment")
    food_beverage_adjustment: float = Field(default=1.0, description="Food & Beverage industry risk adjustment")
    pharmaceuticals_adjustment: float = Field(default=1.3, description="Pharmaceuticals industry risk adjustment")
    utilities_adjustment: float = Field(default=1.1, description="Utilities industry risk adjustment")
    transportation_adjustment: float = Field(default=1.2, description="Transportation industry risk adjustment")
    general_adjustment: float = Field(default=1.0, description="General industry risk adjustment")
    
    # Industry-specific regulatory frameworks
    regulatory_frameworks: Dict[str, List[str]] = Field(
        default={
            "chemical": ["REACH", "CLP", "OSHA", "EPA", "DOT"],
            "oil_gas": ["OSHA", "EPA", "DOT", "PHMSA", "API"],
            "mining": ["MSHA", "OSHA", "EPA", "DOI"],
            "manufacturing": ["OSHA", "EPA", "ISO_14001", "ISO_45001"],
            "healthcare": ["OSHA", "FDA", "CDC", "HIPAA"],
            "food_beverage": ["FDA", "USDA", "OSHA", "EPA"],
            "pharmaceuticals": ["FDA", "DEA", "OSHA", "EPA", "GMP"],
            "construction": ["OSHA", "EPA", "DOT", "NFPA"],
            "utilities": ["OSHA", "EPA", "FERC", "NERC"],
            "transportation": ["DOT", "OSHA", "EPA", "FRA", "FTA"],
            "general": ["OSHA", "EPA"]
        },
        description="Regulatory frameworks by industry"
    )
    
    class Config:
        env_prefix = "INDUSTRY_"
        case_sensitive = False


class LLMModelConfig(BaseSettings):
    """LLM Model configuration settings"""
    
    # Primary model for risk assessment
    primary_model: str = Field(default="openai_gpt_4o", description="Primary LLM model for risk assessment")
    
    # Fallback model
    fallback_model: str = Field(default="openai_gpt_4o_mini", description="Fallback LLM model")
    
    # Model-specific parameters
    temperature: float = Field(default=0.3, description="LLM temperature for risk assessment")
    max_tokens: int = Field(default=2000, description="Maximum tokens per LLM response")
    max_retries: int = Field(default=3, description="Maximum retry attempts for LLM calls")
    retry_delay: float = Field(default=1.0, description="Delay between retry attempts in seconds")
    
    # Model selection by assessment type
    model_by_assessment: Dict[str, str] = Field(
        default={
            "chemical_exposure": "openai_gpt_4o",
            "physical_hazard": "openai_gpt_4o",
            "environmental_impact": "openai_gpt_4o",
            "regulatory_compliance": "openai_gpt_4o",
            "operational_safety": "openai_gpt_4o_mini",
            "process_safety": "openai_gpt_4o",
            "workplace_safety": "openai_gpt_4o_mini",
            "comprehensive": "openai_gpt_4o"
        },
        description="LLM model selection by assessment type"
    )
    
    # Context window sizes for different models
    context_windows: Dict[str, int] = Field(
        default={
            "openai_gpt_4o": 128000,
            "openai_gpt_4o_mini": 128000,
            "openai_gpt_3.5": 16385,
            "gemini_1.5_pro": 2097152,
            "gemini_1.5_flash": 1048576,
            "anthropic_claude_4_sonnet": 200000
        },
        description="Context window sizes for different models"
    )
    
    class Config:
        env_prefix = "LLM_"
        case_sensitive = False


class FeatureToggles(BaseSettings):
    """Feature toggle configuration for different assessment capabilities"""
    
    # Assessment type toggles
    enable_chemical_assessment: bool = Field(default=True, description="Enable chemical exposure assessment")
    enable_physical_hazard_assessment: bool = Field(default=True, description="Enable physical hazard assessment")
    enable_environmental_assessment: bool = Field(default=True, description="Enable environmental impact assessment")
    enable_compliance_assessment: bool = Field(default=True, description="Enable regulatory compliance assessment")
    enable_operational_assessment: bool = Field(default=True, description="Enable operational safety assessment")
    enable_process_safety_assessment: bool = Field(default=True, description="Enable process safety assessment")
    enable_workplace_safety_assessment: bool = Field(default=True, description="Enable workplace safety assessment")
    
    # Analysis features
    enable_severity_scoring: bool = Field(default=True, description="Enable severity scoring")
    enable_probability_analysis: bool = Field(default=True, description="Enable probability analysis")
    enable_impact_analysis: bool = Field(default=True, description="Enable impact analysis")
    enable_mitigation_suggestions: bool = Field(default=True, description="Enable mitigation suggestions")
    enable_regulatory_mapping: bool = Field(default=True, description="Enable regulatory requirement mapping")
    
    # Advanced features
    enable_trend_analysis: bool = Field(default=False, description="Enable trend analysis (experimental)")
    enable_predictive_modeling: bool = Field(default=False, description="Enable predictive risk modeling (experimental)")
    enable_benchmarking: bool = Field(default=True, description="Enable industry benchmarking")
    enable_multi_language: bool = Field(default=False, description="Enable multi-language support")
    
    # Integration features
    enable_graph_integration: bool = Field(default=True, description="Enable Neo4j graph integration")
    enable_embedding_similarity: bool = Field(default=True, description="Enable embedding-based similarity analysis")
    enable_external_apis: bool = Field(default=False, description="Enable external API integrations")
    enable_real_time_updates: bool = Field(default=False, description="Enable real-time risk updates")
    
    class Config:
        env_prefix = "FEATURE_"
        case_sensitive = False


class DatabaseConfig(BaseSettings):
    """Database configuration for risk assessment data storage"""
    
    # Neo4j configuration (inherited from main config)
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j database URI")
    neo4j_username: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="", description="Neo4j password")
    neo4j_database: str = Field(default="neo4j", description="Neo4j database name")
    
    # Risk assessment specific collections/labels
    risk_node_label: str = Field(default="RiskAssessment", description="Neo4j node label for risk assessments")
    hazard_node_label: str = Field(default="Hazard", description="Neo4j node label for hazards")
    mitigation_node_label: str = Field(default="Mitigation", description="Neo4j node label for mitigation measures")
    
    # Connection settings
    connection_timeout: int = Field(default=30, description="Database connection timeout in seconds")
    max_connection_pool_size: int = Field(default=50, description="Maximum connection pool size")
    
    class Config:
        env_prefix = "DB_"
        case_sensitive = False


class ProcessingConfig(BaseSettings):
    """Processing configuration for risk assessment workflows"""
    
    # Batch processing settings
    batch_size: int = Field(default=10, description="Batch size for document processing")
    max_concurrent_assessments: int = Field(default=5, description="Maximum concurrent risk assessments")
    processing_timeout: int = Field(default=300, description="Processing timeout per document in seconds")
    
    # Chunking settings
    max_chunk_size: int = Field(default=2000, description="Maximum chunk size for document processing")
    chunk_overlap: int = Field(default=200, description="Overlap between document chunks")
    
    # Quality settings
    min_confidence_score: float = Field(default=0.6, description="Minimum confidence score for risk assessments")
    require_human_review_threshold: float = Field(default=8.0, description="Risk score threshold requiring human review")
    
    # Caching settings
    enable_result_caching: bool = Field(default=True, description="Enable caching of assessment results")
    cache_ttl: int = Field(default=3600, description="Cache time-to-live in seconds")
    
    class Config:
        env_prefix = "PROCESSING_"
        case_sensitive = False


class RiskAssessmentConfig(PydanticSettings):
    """
    Comprehensive configuration class for the Risk Assessment Agent
    
    This class aggregates all configuration settings and provides validation,
    environment variable support, and configuration management capabilities.
    """
    
    # Sub-configurations
    risk_thresholds: RiskThresholds = Field(default_factory=RiskThresholds)
    document_profiles: DocumentRiskProfile = Field(default_factory=DocumentRiskProfile)
    industry_settings: IndustrySettings = Field(default_factory=IndustrySettings)
    llm_config: LLMModelConfig = Field(default_factory=LLMModelConfig)
    feature_toggles: FeatureToggles = Field(default_factory=FeatureToggles)
    database_config: DatabaseConfig = Field(default_factory=DatabaseConfig)
    processing_config: ProcessingConfig = Field(default_factory=ProcessingConfig)
    
    # Global settings
    environment: str = Field(default="development", description="Environment (development/staging/production)")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # API settings
    api_timeout: int = Field(default=60, description="API timeout in seconds")
    rate_limit_per_minute: int = Field(default=100, description="API rate limit per minute")
    
    # Default industry and document type
    default_industry: Industry = Field(default=Industry.GENERAL, description="Default industry for risk assessment")
    default_document_type: DocumentType = Field(default=DocumentType.UNKNOWN, description="Default document type")
    
    # Configuration file path
    config_file_path: Optional[str] = Field(default=None, description="Path to additional configuration file")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_assignment = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_additional_config()
        self._setup_logging()
    
    def _load_additional_config(self):
        """Load additional configuration from file if specified"""
        if self.config_file_path and os.path.exists(self.config_file_path):
            try:
                with open(self.config_file_path, 'r') as f:
                    additional_config = json.load(f)
                    # Update configuration with additional settings
                    for key, value in additional_config.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
            except Exception as e:
                logger.warning(f"Failed to load additional config from {self.config_file_path}: {e}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    @validator('environment')
    def validate_environment(cls, v):
        """Validate environment setting"""
        valid_environments = ['development', 'staging', 'production']
        if v.lower() not in valid_environments:
            raise ValueError(f'Environment must be one of {valid_environments}')
        return v.lower()
    
    @root_validator
    def validate_model_availability(cls, values):
        """Validate that required models are configured"""
        llm_config = values.get('llm_config', LLMModelConfig())
        primary_model = llm_config.primary_model
        
        # Check if primary model is configured in environment
        model_config_key = f"LLM_MODEL_CONFIG_{primary_model}"
        if model_config_key not in os.environ:
            logger.warning(f"Primary model {primary_model} not configured in environment")
        
        return values
    
    def get_risk_threshold(self, assessment_type: AssessmentType, risk_level: RiskLevel) -> float:
        """
        Get risk threshold for specific assessment type and risk level
        
        Args:
            assessment_type: Type of risk assessment
            risk_level: Risk level (critical, high, medium, low)
            
        Returns:
            Risk threshold value
        """
        threshold_map = {
            AssessmentType.CHEMICAL_EXPOSURE: {
                RiskLevel.CRITICAL: self.risk_thresholds.chemical_exposure_critical,
                RiskLevel.HIGH: self.risk_thresholds.chemical_exposure_high,
                RiskLevel.MEDIUM: self.risk_thresholds.chemical_exposure_medium,
                RiskLevel.LOW: self.risk_thresholds.chemical_exposure_low,
            },
            AssessmentType.PHYSICAL_HAZARD: {
                RiskLevel.CRITICAL: self.risk_thresholds.physical_hazard_critical,
                RiskLevel.HIGH: self.risk_thresholds.physical_hazard_high,
                RiskLevel.MEDIUM: self.risk_thresholds.physical_hazard_medium,
                RiskLevel.LOW: self.risk_thresholds.physical_hazard_low,
            },
            AssessmentType.ENVIRONMENTAL_IMPACT: {
                RiskLevel.CRITICAL: self.risk_thresholds.environmental_critical,
                RiskLevel.HIGH: self.risk_thresholds.environmental_high,
                RiskLevel.MEDIUM: self.risk_thresholds.environmental_medium,
                RiskLevel.LOW: self.risk_thresholds.environmental_low,
            },
            AssessmentType.REGULATORY_COMPLIANCE: {
                RiskLevel.CRITICAL: self.risk_thresholds.compliance_critical,
                RiskLevel.HIGH: self.risk_thresholds.compliance_high,
                RiskLevel.MEDIUM: self.risk_thresholds.compliance_medium,
                RiskLevel.LOW: self.risk_thresholds.compliance_low,
            },
        }
        
        # Default to operational safety thresholds for other assessment types
        default_thresholds = {
            RiskLevel.CRITICAL: self.risk_thresholds.operational_critical,
            RiskLevel.HIGH: self.risk_thresholds.operational_high,
            RiskLevel.MEDIUM: self.risk_thresholds.operational_medium,
            RiskLevel.LOW: self.risk_thresholds.operational_low,
        }
        
        return threshold_map.get(assessment_type, default_thresholds).get(risk_level, 5.0)
    
    def get_document_risk_multiplier(self, document_type: DocumentType) -> float:
        """
        Get risk multiplier for specific document type
        
        Args:
            document_type: Type of document
            
        Returns:
            Risk multiplier value
        """
        multiplier_map = {
            DocumentType.SDS: self.document_profiles.sds_multiplier,
            DocumentType.INCIDENT_REPORT: self.document_profiles.incident_report_multiplier,
            DocumentType.AUDIT_REPORT: self.document_profiles.audit_report_multiplier,
            DocumentType.INSPECTION_REPORT: self.document_profiles.inspection_report_multiplier,
            DocumentType.TRAINING_MATERIAL: self.document_profiles.training_material_multiplier,
            DocumentType.POLICY_DOCUMENT: self.document_profiles.policy_document_multiplier,
            DocumentType.PROCEDURE: self.document_profiles.procedure_multiplier,
            DocumentType.REGULATORY_FILING: self.document_profiles.regulatory_filing_multiplier,
            DocumentType.PERMIT: self.document_profiles.permit_multiplier,
            DocumentType.CERTIFICATE: self.document_profiles.certificate_multiplier,
            DocumentType.UNKNOWN: self.document_profiles.unknown_multiplier,
        }
        
        return multiplier_map.get(document_type, 1.0)
    
    def get_industry_adjustment(self, industry: Industry) -> float:
        """
        Get risk adjustment factor for specific industry
        
        Args:
            industry: Industry type
            
        Returns:
            Industry risk adjustment factor
        """
        adjustment_map = {
            Industry.MANUFACTURING: self.industry_settings.manufacturing_adjustment,
            Industry.CHEMICAL: self.industry_settings.chemical_adjustment,
            Industry.OIL_GAS: self.industry_settings.oil_gas_adjustment,
            Industry.MINING: self.industry_settings.mining_adjustment,
            Industry.CONSTRUCTION: self.industry_settings.construction_adjustment,
            Industry.HEALTHCARE: self.industry_settings.healthcare_adjustment,
            Industry.FOOD_BEVERAGE: self.industry_settings.food_beverage_adjustment,
            Industry.PHARMACEUTICALS: self.industry_settings.pharmaceuticals_adjustment,
            Industry.UTILITIES: self.industry_settings.utilities_adjustment,
            Industry.TRANSPORTATION: self.industry_settings.transportation_adjustment,
            Industry.GENERAL: self.industry_settings.general_adjustment,
        }
        
        return adjustment_map.get(industry, 1.0)
    
    def get_regulatory_frameworks(self, industry: Industry) -> List[str]:
        """
        Get applicable regulatory frameworks for specific industry
        
        Args:
            industry: Industry type
            
        Returns:
            List of applicable regulatory frameworks
        """
        return self.industry_settings.regulatory_frameworks.get(industry.value, ["OSHA", "EPA"])
    
    def get_model_for_assessment(self, assessment_type: AssessmentType) -> str:
        """
        Get appropriate LLM model for specific assessment type
        
        Args:
            assessment_type: Type of risk assessment
            
        Returns:
            Model identifier string
        """
        return self.llm_config.model_by_assessment.get(assessment_type.value, self.llm_config.primary_model)
    
    def is_feature_enabled(self, feature: str) -> bool:
        """
        Check if a specific feature is enabled
        
        Args:
            feature: Feature name
            
        Returns:
            True if feature is enabled, False otherwise
        """
        return getattr(self.feature_toggles, f"enable_{feature}", False)
    
    def requires_human_review(self, risk_score: float) -> bool:
        """
        Determine if a risk score requires human review
        
        Args:
            risk_score: Calculated risk score
            
        Returns:
            True if human review is required, False otherwise
        """
        return risk_score >= self.processing_config.require_human_review_threshold
    
    def get_context_window_size(self, model: str) -> int:
        """
        Get context window size for specific model
        
        Args:
            model: Model identifier
            
        Returns:
            Context window size in tokens
        """
        return self.llm_config.context_windows.get(model, 4096)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary
        
        Returns:
            Configuration as dictionary
        """
        return self.dict()
    
    def save_to_file(self, file_path: str):
        """
        Save configuration to JSON file
        
        Args:
            file_path: Path to save configuration file
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.dict(), f, indent=2, default=str)
            logger.info(f"Configuration saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {file_path}: {e}")
            raise
    
    @classmethod
    def from_file(cls, file_path: str) -> 'RiskAssessmentConfig':
        """
        Load configuration from JSON file
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            RiskAssessmentConfig instance
        """
        try:
            with open(file_path, 'r') as f:
                config_data = json.load(f)
            return cls(**config_data)
        except Exception as e:
            logger.error(f"Failed to load configuration from {file_path}: {e}")
            raise


# Global configuration instance
_config: Optional[RiskAssessmentConfig] = None


def get_config() -> RiskAssessmentConfig:
    """
    Get global configuration instance (singleton pattern)
    
    Returns:
        RiskAssessmentConfig instance
    """
    global _config
    if _config is None:
        _config = RiskAssessmentConfig()
    return _config


def reload_config():
    """Reload global configuration from environment and files"""
    global _config
    _config = RiskAssessmentConfig()
    return _config


def set_config(config: RiskAssessmentConfig):
    """
    Set global configuration instance
    
    Args:
        config: RiskAssessmentConfig instance to set as global
    """
    global _config
    _config = config


# Utility functions for common configuration operations
def get_risk_threshold(assessment_type: AssessmentType, risk_level: RiskLevel) -> float:
    """Convenience function to get risk threshold"""
    return get_config().get_risk_threshold(assessment_type, risk_level)


def get_document_multiplier(document_type: DocumentType) -> float:
    """Convenience function to get document risk multiplier"""
    return get_config().get_document_risk_multiplier(document_type)


def get_industry_adjustment(industry: Industry) -> float:
    """Convenience function to get industry adjustment"""
    return get_config().get_industry_adjustment(industry)


def is_feature_enabled(feature: str) -> bool:
    """Convenience function to check if feature is enabled"""
    return get_config().is_feature_enabled(feature)


def requires_human_review(risk_score: float) -> bool:
    """Convenience function to check if human review is required"""
    return get_config().requires_human_review(risk_score)