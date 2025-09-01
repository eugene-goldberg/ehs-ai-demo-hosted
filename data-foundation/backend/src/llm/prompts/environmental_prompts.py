"""
Comprehensive LLM prompt templates module for environmental facts analysis.
Provides structured prompts for analyzing electricity, water, and waste consumption data.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Types of environmental analysis."""
    ELECTRICITY = "electricity"
    WATER = "water"
    WASTE = "waste"
    CROSS_DOMAIN = "cross_domain"
    TEMPORAL = "temporal"
    BENCHMARKING = "benchmarking"


class OutputFormat(Enum):
    """Supported output formats."""
    JSON = "json"
    STRUCTURED_TEXT = "structured_text"
    MARKDOWN = "markdown"


@dataclass
class PromptContext:
    """Context information for prompt generation."""
    facility_name: Optional[str] = None
    time_period: Optional[str] = None
    data_types: List[str] = None
    analysis_goals: List[str] = None
    baseline_period: Optional[str] = None
    target_metrics: List[str] = None


class EnvironmentalPromptTemplates:
    """
    Comprehensive prompt templates for environmental data analysis.
    
    This class provides structured prompt templates for analyzing environmental
    consumption data across electricity, water, and waste domains using LLM models.
    """

    # System prompts that establish AI expertise
    SYSTEM_PROMPTS = {
        "base_environmental_expert": """You are an expert Environmental Sustainability Analyst with deep knowledge in:
- Energy management and consumption patterns
- Water resource management and conservation
- Waste management and circular economy principles
- Environmental regulations and compliance
- Sustainability metrics and KPI analysis
- Carbon footprint and emissions calculations
- Resource efficiency optimization

Your analysis should be:
- Data-driven and factually accurate
- Actionable with clear recommendations
- Aligned with industry best practices
- Compliant with environmental standards (EPA, ISO 14001, etc.)
- Focused on measurable sustainability outcomes

Always provide evidence-based insights and quantify environmental impacts when possible.""",

        "utility_data_expert": """You are a specialized Utility Data Analyst expert in:
- Electricity consumption patterns and demand forecasting
- Utility bill analysis and rate structure optimization
- Peak demand management and load balancing
- Renewable energy integration and grid efficiency
- Power quality analysis and energy auditing
- Demand response programs and energy storage
- Time-of-use pricing and cost optimization

Focus on identifying consumption trends, efficiency opportunities, and cost savings while maintaining operational requirements.""",

        "water_management_expert": """You are a Water Resource Management Specialist with expertise in:
- Water consumption analysis and conservation strategies
- Water quality monitoring and treatment processes
- Leak detection and infrastructure optimization
- Wastewater treatment and reuse systems
- Stormwater management and runoff control
- Water footprint calculation and reduction
- Regulatory compliance (Clean Water Act, local ordinances)

Emphasize water conservation, quality maintenance, and sustainable water use practices.""",

        "waste_management_expert": """You are a Waste Management and Circular Economy Expert specializing in:
- Waste stream analysis and characterization
- Waste minimization and source reduction
- Recycling program optimization
- Hazardous waste management and compliance
- Waste-to-energy and resource recovery
- Circular economy principles and implementation
- Waste disposal cost optimization and vendor management

Focus on waste reduction, diversion rates, compliance, and circular economy opportunities."""
    }

    # JSON Schema templates for structured outputs
    JSON_SCHEMAS = {
        "consumption_analysis": {
            "type": "object",
            "properties": {
                "analysis_summary": {
                    "type": "object",
                    "properties": {
                        "total_consumption": {"type": "number"},
                        "consumption_unit": {"type": "string"},
                        "time_period": {"type": "string"},
                        "consumption_trend": {"type": "string", "enum": ["increasing", "decreasing", "stable", "volatile"]},
                        "key_findings": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "consumption_breakdown": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string"},
                            "amount": {"type": "number"},
                            "unit": {"type": "string"},
                            "percentage_of_total": {"type": "number"}
                        }
                    }
                },
                "efficiency_metrics": {
                    "type": "object",
                    "properties": {
                        "consumption_per_unit": {"type": "number"},
                        "unit_description": {"type": "string"},
                        "efficiency_rating": {"type": "string", "enum": ["excellent", "good", "average", "poor", "critical"]},
                        "benchmark_comparison": {"type": "string"}
                    }
                },
                "cost_analysis": {
                    "type": "object",
                    "properties": {
                        "total_cost": {"type": "number"},
                        "cost_per_unit": {"type": "number"},
                        "cost_trends": {"type": "string"},
                        "cost_saving_opportunities": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "recommendations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string"},
                            "recommendation": {"type": "string"},
                            "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                            "estimated_savings": {"type": "string"},
                            "implementation_effort": {"type": "string", "enum": ["low", "medium", "high"]},
                            "timeframe": {"type": "string"}
                        }
                    }
                },
                "environmental_impact": {
                    "type": "object",
                    "properties": {
                        "carbon_emissions": {"type": "number"},
                        "emissions_unit": {"type": "string"},
                        "environmental_rating": {"type": "string"},
                        "sustainability_score": {"type": "number", "minimum": 0, "maximum": 100}
                    }
                }
            },
            "required": ["analysis_summary", "recommendations"]
        },

        "cross_domain_analysis": {
            "type": "object",
            "properties": {
                "correlation_analysis": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "domain_1": {"type": "string"},
                            "domain_2": {"type": "string"},
                            "correlation_strength": {"type": "string", "enum": ["strong", "moderate", "weak", "none"]},
                            "correlation_direction": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                            "statistical_significance": {"type": "number"},
                            "explanation": {"type": "string"}
                        }
                    }
                },
                "integrated_insights": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "insight": {"type": "string"},
                            "domains_involved": {"type": "array", "items": {"type": "string"}},
                            "impact_level": {"type": "string", "enum": ["high", "medium", "low"]},
                            "actionability": {"type": "string", "enum": ["immediate", "short-term", "long-term"]}
                        }
                    }
                },
                "optimization_opportunities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "opportunity": {"type": "string"},
                            "affected_domains": {"type": "array", "items": {"type": "string"}},
                            "potential_impact": {"type": "string"},
                            "implementation_complexity": {"type": "string", "enum": ["low", "medium", "high"]},
                            "estimated_roi": {"type": "string"}
                        }
                    }
                }
            },
            "required": ["correlation_analysis", "integrated_insights"]
        },

        "temporal_analysis": {
            "type": "object",
            "properties": {
                "trend_analysis": {
                    "type": "object",
                    "properties": {
                        "overall_trend": {"type": "string", "enum": ["improving", "declining", "stable", "volatile"]},
                        "trend_strength": {"type": "number", "minimum": 0, "maximum": 1},
                        "seasonality_detected": {"type": "boolean"},
                        "cyclical_patterns": {"type": "array", "items": {"type": "string"}},
                        "anomalies_detected": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "period_comparisons": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "period_1": {"type": "string"},
                            "period_2": {"type": "string"},
                            "change_percentage": {"type": "number"},
                            "change_direction": {"type": "string", "enum": ["increase", "decrease", "stable"]},
                            "significance": {"type": "string", "enum": ["significant", "moderate", "minimal"]},
                            "contributing_factors": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "forecasting": {
                    "type": "object",
                    "properties": {
                        "next_period_prediction": {"type": "number"},
                        "confidence_level": {"type": "number", "minimum": 0, "maximum": 1},
                        "prediction_basis": {"type": "string"},
                        "risk_factors": {"type": "array", "items": {"type": "string"}},
                        "recommended_actions": {"type": "array", "items": {"type": "string"}}
                    }
                }
            },
            "required": ["trend_analysis", "period_comparisons"]
        }
    }

    # Base prompt templates for different analysis types
    BASE_TEMPLATES = {
        "electricity_analysis": """Analyze the following electricity consumption data for {facility_name} during {time_period}.

DATA TO ANALYZE:
{consumption_data}

ANALYSIS REQUIREMENTS:
- Identify total consumption, peak demand patterns, and usage trends
- Calculate energy efficiency metrics and cost per kWh
- Detect any unusual consumption spikes or patterns
- Assess power quality indicators if available
- Evaluate demand charges vs consumption charges
- Identify opportunities for load shifting and peak demand reduction
- Consider renewable energy integration possibilities
- Analyze time-of-use patterns and rate optimization opportunities

{output_format_instruction}

Focus on actionable insights that can reduce energy costs and improve sustainability.""",

        "water_analysis": """Analyze the following water consumption data for {facility_name} during {time_period}.

DATA TO ANALYZE:
{consumption_data}

ANALYSIS REQUIREMENTS:
- Identify total water consumption and usage patterns
- Calculate water efficiency metrics and cost per gallon/cubic meter
- Detect potential leaks or unusual consumption patterns
- Assess seasonal variations and operational impacts
- Evaluate water quality parameters if available
- Identify water conservation opportunities
- Analyze wastewater generation and treatment costs
- Consider water reuse and recycling potential

{output_format_instruction}

Focus on water conservation strategies and cost-effective improvements.""",

        "waste_analysis": """Analyze the following waste generation and disposal data for {facility_name} during {time_period}.

DATA TO ANALYZE:
{consumption_data}

ANALYSIS REQUIREMENTS:
- Categorize waste streams by type and disposal method
- Calculate waste generation rates and diversion rates
- Identify recycling and composting opportunities
- Assess hazardous waste management compliance
- Evaluate waste disposal costs by category
- Analyze waste-to-landfill ratios and sustainability metrics
- Identify source reduction opportunities
- Consider circular economy and waste-to-energy options

{output_format_instruction}

Focus on waste reduction, increased diversion rates, and circular economy opportunities.""",

        "cross_domain_analysis": """Perform a comprehensive cross-domain analysis examining relationships between electricity, water, and waste consumption for {facility_name} during {time_period}.

MULTI-DOMAIN DATA:
{consumption_data}

ANALYSIS REQUIREMENTS:
- Identify correlations between different resource consumption patterns
- Analyze how changes in one domain affect others (e.g., energy for water treatment)
- Calculate integrated sustainability metrics
- Assess overall resource efficiency and circular economy potential
- Identify optimization opportunities that benefit multiple domains
- Evaluate trade-offs and synergies between resource management strategies
- Calculate total environmental footprint and carbon equivalent

{output_format_instruction}

Focus on integrated sustainability strategies that optimize across all domains.""",

        "temporal_analysis": """Conduct a temporal trend analysis of {data_types} consumption for {facility_name} comparing {time_period} with historical data.

TEMPORAL DATA:
{consumption_data}

ANALYSIS REQUIREMENTS:
- Identify long-term trends and patterns
- Detect seasonal variations and cyclical patterns
- Compare current performance to historical baselines
- Identify significant changes and their potential causes
- Assess the effectiveness of implemented conservation measures
- Predict future consumption patterns based on trends
- Identify critical periods requiring attention
- Evaluate progress toward sustainability goals

{output_format_instruction}

Focus on trend identification, performance changes, and forward-looking recommendations."""
    }

    # Fact extraction prompts for identifying key insights
    FACT_EXTRACTION_TEMPLATES = {
        "consumption_facts": """Extract key consumption facts from the following environmental data:

{data_content}

Extract the following types of facts:
1. QUANTITATIVE FACTS - Specific numbers, measurements, percentages
2. TEMPORAL FACTS - Time periods, dates, durations, trends over time
3. COMPARATIVE FACTS - Comparisons between periods, benchmarks, targets
4. EFFICIENCY FACTS - Efficiency metrics, ratios, performance indicators
5. COST FACTS - Financial data, cost per unit, budget impacts
6. COMPLIANCE FACTS - Regulatory compliance, permit requirements, violations
7. OPERATIONAL FACTS - Operational changes, equipment performance, usage patterns

Format each fact as:
- **Fact Type**: [QUANTITATIVE/TEMPORAL/COMPARATIVE/EFFICIENCY/COST/COMPLIANCE/OPERATIONAL]
- **Statement**: [Clear, specific fact statement]
- **Value**: [Numerical value if applicable]
- **Unit**: [Unit of measurement if applicable]
- **Context**: [Additional context or explanation]
- **Confidence**: [High/Medium/Low based on data quality]
- **Source**: [Reference to source data section]

Prioritize facts that are:
- Actionable for decision-making
- Relevant to sustainability goals
- Supported by clear data evidence
- Important for compliance or cost management""",

        "anomaly_detection": """Identify and analyze anomalies in the following environmental consumption data:

{data_content}

ANOMALY DETECTION FOCUS:
- Consumption spikes or drops exceeding normal variation
- Unexpected patterns or deviations from seasonal norms
- Cost anomalies that don't align with consumption patterns
- Efficiency degradations or improvements
- Compliance deviations or concerning trends

For each anomaly identified, provide:
1. **Anomaly Type**: [Consumption/Cost/Efficiency/Pattern/Compliance]
2. **Description**: Clear description of the anomaly
3. **Magnitude**: How significant is the deviation
4. **Time Period**: When the anomaly occurred
5. **Potential Causes**: Likely explanations for the anomaly
6. **Impact Assessment**: Operational, financial, or environmental impact
7. **Recommended Actions**: Steps to investigate or address the anomaly
8. **Priority Level**: [High/Medium/Low] for addressing this anomaly

Focus on anomalies that could indicate:
- Equipment malfunctions or inefficiencies
- Operational changes or process issues
- Data quality problems
- Compliance risks
- Cost optimization opportunities""",

        "benchmark_comparison": """Compare the following consumption data against industry benchmarks and best practices:

{data_content}

BENCHMARKING ANALYSIS:
- Compare consumption per unit (sq ft, employee, production unit, etc.)
- Assess efficiency metrics against industry standards
- Evaluate cost performance relative to regional averages
- Compare sustainability metrics to industry leaders
- Assess compliance performance against regulatory standards

BENCHMARK CATEGORIES:
1. **Industry Standards**: Compare to typical performance in the industry
2. **Best-in-Class**: Compare to top performers in the sector
3. **Regulatory Targets**: Compare to compliance thresholds and targets
4. **Historical Performance**: Compare to facility's own historical data
5. **Peer Facilities**: Compare to similar facilities if data available

For each benchmark comparison, provide:
- **Metric**: What is being compared
- **Current Performance**: Facility's current value
- **Benchmark Value**: Industry/regulatory/target value
- **Performance Gap**: Difference and percentage variance
- **Performance Rating**: [Excellent/Good/Average/Below Average/Poor]
- **Improvement Potential**: Estimated opportunity for improvement
- **Action Priority**: [High/Medium/Low] for addressing gaps"""
    }

    # Few-shot examples for better output quality
    FEW_SHOT_EXAMPLES = {
        "electricity_analysis_example": {
            "input": "Monthly electricity consumption: 45,000 kWh, Peak demand: 120 kW, Cost: $4,500, Previous month: 42,000 kWh",
            "output": {
                "analysis_summary": {
                    "total_consumption": 45000,
                    "consumption_unit": "kWh",
                    "time_period": "Current month",
                    "consumption_trend": "increasing",
                    "key_findings": [
                        "7.1% increase in consumption compared to previous month",
                        "Peak demand of 120 kW indicates potential for demand charge optimization",
                        "Average cost of $0.10 per kWh is within normal range"
                    ]
                },
                "recommendations": [
                    {
                        "category": "Peak Demand Management",
                        "recommendation": "Implement load scheduling to reduce peak demand during high-cost periods",
                        "priority": "high",
                        "estimated_savings": "$200-400/month",
                        "implementation_effort": "medium",
                        "timeframe": "30-60 days"
                    }
                ]
            }
        },

        "cross_domain_example": {
            "input": "Electricity: 45,000 kWh, Water: 15,000 gallons, Waste: 2.5 tons - manufacturing facility",
            "output": {
                "correlation_analysis": [
                    {
                        "domain_1": "electricity",
                        "domain_2": "water",
                        "correlation_strength": "strong",
                        "correlation_direction": "positive",
                        "statistical_significance": 0.85,
                        "explanation": "Higher electricity consumption correlates with increased water usage due to cooling systems and process equipment"
                    }
                ],
                "integrated_insights": [
                    {
                        "insight": "Energy-intensive processes drive both high electricity and water consumption",
                        "domains_involved": ["electricity", "water"],
                        "impact_level": "high",
                        "actionability": "short-term"
                    }
                ]
            }
        }
    }

    def __init__(self):
        """Initialize the Environmental Prompt Templates."""
        logger.info("Initializing Environmental Prompt Templates")

    def get_system_prompt(self, analysis_type: AnalysisType) -> str:
        """
        Get appropriate system prompt based on analysis type.
        
        Args:
            analysis_type: Type of environmental analysis
            
        Returns:
            System prompt string
        """
        prompt_mapping = {
            AnalysisType.ELECTRICITY: "utility_data_expert",
            AnalysisType.WATER: "water_management_expert",
            AnalysisType.WASTE: "waste_management_expert",
            AnalysisType.CROSS_DOMAIN: "base_environmental_expert",
            AnalysisType.TEMPORAL: "base_environmental_expert",
            AnalysisType.BENCHMARKING: "base_environmental_expert"
        }
        
        return self.SYSTEM_PROMPTS[prompt_mapping.get(analysis_type, "base_environmental_expert")]

    def get_output_format_instruction(self, output_format: OutputFormat, schema_name: str = None) -> str:
        """
        Get output format instructions for prompts.
        
        Args:
            output_format: Desired output format
            schema_name: Name of JSON schema to use if JSON format
            
        Returns:
            Format instruction string
        """
        if output_format == OutputFormat.JSON:
            if schema_name and schema_name in self.JSON_SCHEMAS:
                return f"""OUTPUT FORMAT: Provide your analysis as valid JSON following this exact schema:
{self.JSON_SCHEMAS[schema_name]}

Ensure all required fields are included and data types match the schema."""
            else:
                return """OUTPUT FORMAT: Provide your analysis as structured JSON with clear sections for findings, metrics, and recommendations."""
        
        elif output_format == OutputFormat.STRUCTURED_TEXT:
            return """OUTPUT FORMAT: Provide your analysis in structured text format with clear headers:
## EXECUTIVE SUMMARY
## KEY FINDINGS
## CONSUMPTION ANALYSIS
## EFFICIENCY METRICS
## COST ANALYSIS
## RECOMMENDATIONS
## IMPLEMENTATION PRIORITIES"""
        
        elif output_format == OutputFormat.MARKDOWN:
            return """OUTPUT FORMAT: Provide your analysis in well-formatted Markdown with:
- Clear section headers
- Bullet points for key findings
- Tables for data comparisons
- Bold text for important metrics
- Action items clearly highlighted"""
        
        return ""

    def create_consumption_analysis_prompt(
        self,
        analysis_type: AnalysisType,
        consumption_data: str,
        context: PromptContext,
        output_format: OutputFormat = OutputFormat.JSON,
        include_examples: bool = True
    ) -> Dict[str, str]:
        """
        Create a comprehensive consumption analysis prompt.
        
        Args:
            analysis_type: Type of analysis to perform
            consumption_data: Raw consumption data to analyze
            context: Context information for the analysis
            output_format: Desired output format
            include_examples: Whether to include few-shot examples
            
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        # Get base template
        template_key = f"{analysis_type.value}_analysis"
        if template_key not in self.BASE_TEMPLATES:
            template_key = "cross_domain_analysis"
        
        template = self.BASE_TEMPLATES[template_key]
        
        # Get output format instructions
        schema_name = "consumption_analysis" if analysis_type != AnalysisType.CROSS_DOMAIN else "cross_domain_analysis"
        format_instruction = self.get_output_format_instruction(output_format, schema_name)
        
        # Format the template
        user_prompt = template.format(
            facility_name=context.facility_name or "the facility",
            time_period=context.time_period or "the specified period",
            consumption_data=consumption_data,
            data_types=", ".join(context.data_types or []),
            output_format_instruction=format_instruction
        )
        
        # Add few-shot examples if requested
        if include_examples and analysis_type.value + "_analysis_example" in self.FEW_SHOT_EXAMPLES:
            example = self.FEW_SHOT_EXAMPLES[analysis_type.value + "_analysis_example"]
            user_prompt += f"""

EXAMPLE ANALYSIS:
Input: {example['input']}
Output: {example['output']}

Now analyze the provided data following the same structure and level of detail."""

        return {
            "system": self.get_system_prompt(analysis_type),
            "user": user_prompt
        }

    def create_fact_extraction_prompt(
        self,
        data_content: str,
        extraction_type: str = "consumption_facts",
        specific_focus: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Create a fact extraction prompt for identifying key insights.
        
        Args:
            data_content: Raw data content to extract facts from
            extraction_type: Type of fact extraction
            specific_focus: Specific areas to focus fact extraction on
            
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        template = self.FACT_EXTRACTION_TEMPLATES.get(extraction_type, self.FACT_EXTRACTION_TEMPLATES["consumption_facts"])
        
        user_prompt = template.format(data_content=data_content)
        
        if specific_focus:
            user_prompt += f"""

SPECIFIC FOCUS AREAS:
Pay special attention to extracting facts related to:
{chr(10).join(f"- {focus}" for focus in specific_focus)}"""

        return {
            "system": self.SYSTEM_PROMPTS["base_environmental_expert"],
            "user": user_prompt
        }

    def create_temporal_analysis_prompt(
        self,
        consumption_data: str,
        context: PromptContext,
        output_format: OutputFormat = OutputFormat.JSON
    ) -> Dict[str, str]:
        """
        Create a temporal trend analysis prompt.
        
        Args:
            consumption_data: Time-series consumption data
            context: Context information including time periods
            output_format: Desired output format
            
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        format_instruction = self.get_output_format_instruction(output_format, "temporal_analysis")
        
        user_prompt = self.BASE_TEMPLATES["temporal_analysis"].format(
            facility_name=context.facility_name or "the facility",
            time_period=context.time_period or "the analysis period",
            data_types=", ".join(context.data_types or ["consumption"]),
            consumption_data=consumption_data,
            output_format_instruction=format_instruction
        )
        
        if context.baseline_period:
            user_prompt += f"\n\nBASELINE PERIOD: Use {context.baseline_period} as the baseline for comparisons."
        
        if context.target_metrics:
            user_prompt += f"\n\nTARGET METRICS: Focus analysis on these specific metrics: {', '.join(context.target_metrics)}"

        return {
            "system": self.get_system_prompt(AnalysisType.TEMPORAL),
            "user": user_prompt
        }

    def create_benchmarking_prompt(
        self,
        consumption_data: str,
        context: PromptContext,
        benchmark_data: Optional[str] = None,
        output_format: OutputFormat = OutputFormat.JSON
    ) -> Dict[str, str]:
        """
        Create a benchmarking analysis prompt.
        
        Args:
            consumption_data: Facility consumption data
            context: Context information
            benchmark_data: Optional benchmark data to compare against
            output_format: Desired output format
            
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        template = self.FACT_EXTRACTION_TEMPLATES["benchmark_comparison"]
        format_instruction = self.get_output_format_instruction(output_format)
        
        user_prompt = template.format(data_content=consumption_data) + f"\n\n{format_instruction}"
        
        if benchmark_data:
            user_prompt += f"\n\nBENCHMARK DATA:\n{benchmark_data}"
        
        if context.target_metrics:
            user_prompt += f"\n\nFOCUS METRICS: Prioritize benchmarking for: {', '.join(context.target_metrics)}"

        return {
            "system": self.get_system_prompt(AnalysisType.BENCHMARKING),
            "user": user_prompt
        }

    def create_custom_prompt(
        self,
        custom_instructions: str,
        consumption_data: str,
        context: PromptContext,
        analysis_type: AnalysisType = AnalysisType.CROSS_DOMAIN,
        output_format: OutputFormat = OutputFormat.JSON
    ) -> Dict[str, str]:
        """
        Create a custom analysis prompt with specific instructions.
        
        Args:
            custom_instructions: Custom analysis instructions
            consumption_data: Data to analyze
            context: Context information
            analysis_type: Base analysis type for system prompt
            output_format: Desired output format
            
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        format_instruction = self.get_output_format_instruction(output_format)
        
        user_prompt = f"""Analyze the following environmental consumption data according to these specific instructions:

CUSTOM ANALYSIS INSTRUCTIONS:
{custom_instructions}

DATA TO ANALYZE:
{consumption_data}

CONTEXT:
- Facility: {context.facility_name or 'Not specified'}
- Time Period: {context.time_period or 'Not specified'}
- Analysis Goals: {', '.join(context.analysis_goals or ['General analysis'])}

{format_instruction}

Ensure your analysis addresses all aspects requested in the custom instructions while maintaining focus on actionable sustainability insights."""

        return {
            "system": self.get_system_prompt(analysis_type),
            "user": user_prompt
        }

    def get_available_templates(self) -> Dict[str, List[str]]:
        """
        Get a summary of all available prompt templates.
        
        Returns:
            Dictionary listing available templates by category
        """
        return {
            "system_prompts": list(self.SYSTEM_PROMPTS.keys()),
            "analysis_templates": list(self.BASE_TEMPLATES.keys()),
            "fact_extraction": list(self.FACT_EXTRACTION_TEMPLATES.keys()),
            "json_schemas": list(self.JSON_SCHEMAS.keys()),
            "examples": list(self.FEW_SHOT_EXAMPLES.keys())
        }

    def validate_schema_compliance(self, response: Dict[str, Any], schema_name: str) -> Dict[str, Any]:
        """
        Validate LLM response against expected JSON schema.
        
        Args:
            response: LLM response to validate
            schema_name: Name of schema to validate against
            
        Returns:
            Dictionary with validation results
        """
        if schema_name not in self.JSON_SCHEMAS:
            return {"valid": False, "error": f"Schema '{schema_name}' not found"}
        
        schema = self.JSON_SCHEMAS[schema_name]
        
        try:
            # Basic validation - check required fields
            required_fields = schema.get("required", [])
            missing_fields = [field for field in required_fields if field not in response]
            
            if missing_fields:
                return {
                    "valid": False,
                    "error": f"Missing required fields: {missing_fields}"
                }
            
            # Check field types for top-level properties
            properties = schema.get("properties", {})
            type_errors = []
            
            for field, field_schema in properties.items():
                if field in response:
                    expected_type = field_schema.get("type")
                    actual_value = response[field]
                    
                    if expected_type == "object" and not isinstance(actual_value, dict):
                        type_errors.append(f"{field} should be object, got {type(actual_value)}")
                    elif expected_type == "array" and not isinstance(actual_value, list):
                        type_errors.append(f"{field} should be array, got {type(actual_value)}")
                    elif expected_type == "string" and not isinstance(actual_value, str):
                        type_errors.append(f"{field} should be string, got {type(actual_value)}")
                    elif expected_type == "number" and not isinstance(actual_value, (int, float)):
                        type_errors.append(f"{field} should be number, got {type(actual_value)}")
            
            if type_errors:
                return {
                    "valid": False,
                    "error": f"Type validation errors: {type_errors}"
                }
            
            return {"valid": True, "message": "Response validates against schema"}
            
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}


# Convenience functions for common use cases
def create_electricity_analysis_prompt(
    consumption_data: str,
    facility_name: str = None,
    time_period: str = None,
    output_format: OutputFormat = OutputFormat.JSON
) -> Dict[str, str]:
    """
    Convenience function for creating electricity analysis prompts.
    
    Args:
        consumption_data: Electricity consumption data
        facility_name: Name of the facility
        time_period: Time period for analysis
        output_format: Desired output format
        
    Returns:
        Dictionary with 'system' and 'user' prompts
    """
    templates = EnvironmentalPromptTemplates()
    context = PromptContext(
        facility_name=facility_name,
        time_period=time_period,
        data_types=["electricity"]
    )
    
    return templates.create_consumption_analysis_prompt(
        AnalysisType.ELECTRICITY,
        consumption_data,
        context,
        output_format
    )


def create_water_analysis_prompt(
    consumption_data: str,
    facility_name: str = None,
    time_period: str = None,
    output_format: OutputFormat = OutputFormat.JSON
) -> Dict[str, str]:
    """
    Convenience function for creating water analysis prompts.
    
    Args:
        consumption_data: Water consumption data
        facility_name: Name of the facility
        time_period: Time period for analysis
        output_format: Desired output format
        
    Returns:
        Dictionary with 'system' and 'user' prompts
    """
    templates = EnvironmentalPromptTemplates()
    context = PromptContext(
        facility_name=facility_name,
        time_period=time_period,
        data_types=["water"]
    )
    
    return templates.create_consumption_analysis_prompt(
        AnalysisType.WATER,
        consumption_data,
        context,
        output_format
    )


def create_waste_analysis_prompt(
    consumption_data: str,
    facility_name: str = None,
    time_period: str = None,
    output_format: OutputFormat = OutputFormat.JSON
) -> Dict[str, str]:
    """
    Convenience function for creating waste analysis prompts.
    
    Args:
        consumption_data: Waste generation data
        facility_name: Name of the facility
        time_period: Time period for analysis
        output_format: Desired output format
        
    Returns:
        Dictionary with 'system' and 'user' prompts
    """
    templates = EnvironmentalPromptTemplates()
    context = PromptContext(
        facility_name=facility_name,
        time_period=time_period,
        data_types=["waste"]
    )
    
    return templates.create_consumption_analysis_prompt(
        AnalysisType.WASTE,
        consumption_data,
        context,
        output_format
    )


def create_multi_domain_analysis_prompt(
    consumption_data: str,
    facility_name: str = None,
    time_period: str = None,
    data_types: List[str] = None,
    output_format: OutputFormat = OutputFormat.JSON
) -> Dict[str, str]:
    """
    Convenience function for creating multi-domain cross-analysis prompts.
    
    Args:
        consumption_data: Multi-domain consumption data
        facility_name: Name of the facility
        time_period: Time period for analysis
        data_types: List of data types being analyzed
        output_format: Desired output format
        
    Returns:
        Dictionary with 'system' and 'user' prompts
    """
    templates = EnvironmentalPromptTemplates()
    context = PromptContext(
        facility_name=facility_name,
        time_period=time_period,
        data_types=data_types or ["electricity", "water", "waste"]
    )
    
    return templates.create_consumption_analysis_prompt(
        AnalysisType.CROSS_DOMAIN,
        consumption_data,
        context,
        output_format
    )


# Module-level instance for easy access
environmental_prompts = EnvironmentalPromptTemplates()