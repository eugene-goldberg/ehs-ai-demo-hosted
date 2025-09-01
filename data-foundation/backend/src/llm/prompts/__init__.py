"""
LLM prompt templates for environmental data analysis, risk assessment, and recommendations generation.
"""

from .environmental_prompts import (
    EnvironmentalPromptTemplates,
    AnalysisType,
    OutputFormat,
    PromptContext,
    environmental_prompts,
    create_electricity_analysis_prompt,
    create_water_analysis_prompt,
    create_waste_analysis_prompt,
    create_multi_domain_analysis_prompt
)

from .risk_assessment_prompts import (
    RiskAssessmentPromptTemplates,
    RiskType,
    RiskDomain,
    RiskSeverity,
    RiskProbability,
    RiskContext,
    risk_assessment_prompts,
    create_electricity_risk_assessment_prompt,
    create_water_risk_assessment_prompt,
    create_waste_risk_assessment_prompt,
    create_comprehensive_risk_assessment_prompt
)

from .recommendation_prompts import (
    RecommendationPromptTemplates,
    RecommendationType,
    ImplementationTimeframe,
    RecommendationPriority,
    ImplementationEffort,
    TechnologyCategory,
    RecommendationContext,
    recommendation_prompts,
    create_electricity_optimization_recommendations,
    create_water_conservation_recommendations,
    create_waste_reduction_recommendations,
    create_quick_wins_recommendations,
    create_comprehensive_sustainability_recommendations
)

__all__ = [
    # Environmental prompts
    "EnvironmentalPromptTemplates",
    "AnalysisType",
    "OutputFormat", 
    "PromptContext",
    "environmental_prompts",
    "create_electricity_analysis_prompt",
    "create_water_analysis_prompt",
    "create_waste_analysis_prompt",
    "create_multi_domain_analysis_prompt",
    # Risk assessment prompts
    "RiskAssessmentPromptTemplates",
    "RiskType",
    "RiskDomain",
    "RiskSeverity",
    "RiskProbability",
    "RiskContext",
    "risk_assessment_prompts",
    "create_electricity_risk_assessment_prompt",
    "create_water_risk_assessment_prompt",
    "create_waste_risk_assessment_prompt",
    "create_comprehensive_risk_assessment_prompt",
    # Recommendation prompts
    "RecommendationPromptTemplates",
    "RecommendationType",
    "ImplementationTimeframe",
    "RecommendationPriority",
    "ImplementationEffort",
    "TechnologyCategory",
    "RecommendationContext",
    "recommendation_prompts",
    "create_electricity_optimization_recommendations",
    "create_water_conservation_recommendations",
    "create_waste_reduction_recommendations",
    "create_quick_wins_recommendations",
    "create_comprehensive_sustainability_recommendations"
]