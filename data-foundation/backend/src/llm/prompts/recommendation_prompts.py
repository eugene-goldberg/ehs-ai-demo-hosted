"""
Comprehensive LLM prompt templates module for environmental recommendations generation.
Provides structured prompts for generating actionable sustainability recommendations across electricity, water, and waste domains.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RecommendationType(Enum):
    """Types of environmental recommendations."""
    ELECTRICITY_OPTIMIZATION = "electricity_optimization"
    WATER_CONSERVATION = "water_conservation"
    WASTE_REDUCTION = "waste_reduction"
    COST_SAVINGS = "cost_savings"
    TECHNOLOGY_INTEGRATION = "technology_integration"
    BEST_PRACTICES = "best_practices"
    QUICK_WINS = "quick_wins"
    STRATEGIC_INITIATIVES = "strategic_initiatives"
    CROSS_DOMAIN_OPTIMIZATION = "cross_domain_optimization"
    COMPLIANCE_IMPROVEMENT = "compliance_improvement"


class ImplementationTimeframe(Enum):
    """Implementation timeframes for recommendations."""
    IMMEDIATE = "immediate"  # 0-30 days
    SHORT_TERM = "short_term"  # 1-6 months
    MEDIUM_TERM = "medium_term"  # 6-18 months
    LONG_TERM = "long_term"  # 18+ months


class RecommendationPriority(Enum):
    """Priority levels for recommendations."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ImplementationEffort(Enum):
    """Implementation effort levels."""
    MINIMAL = "minimal"  # < 1 week effort
    LOW = "low"  # 1-4 weeks effort
    MEDIUM = "medium"  # 1-6 months effort
    HIGH = "high"  # 6+ months effort


class TechnologyCategory(Enum):
    """Technology recommendation categories."""
    IOT_SENSORS = "iot_sensors"
    AUTOMATION = "automation"
    MONITORING_SYSTEMS = "monitoring_systems"
    ENERGY_MANAGEMENT = "energy_management"
    WATER_MANAGEMENT = "water_management"
    WASTE_TRACKING = "waste_tracking"
    ANALYTICS_PLATFORMS = "analytics_platforms"
    RENEWABLE_ENERGY = "renewable_energy"


@dataclass
class RecommendationContext:
    """Context information for recommendation generation."""
    facility_name: Optional[str] = None
    industry_sector: Optional[str] = None
    facility_size: Optional[str] = None
    current_performance: Optional[Dict[str, Any]] = None
    budget_range: Optional[str] = None
    sustainability_goals: List[str] = None
    regulatory_requirements: List[str] = None
    existing_systems: List[str] = None
    constraints: List[str] = None
    time_horizon: Optional[str] = None


class RecommendationPromptTemplates:
    """
    Comprehensive prompt templates for environmental recommendation generation.

    This class provides structured prompt templates for generating actionable
    sustainability recommendations across electricity, water, and waste domains.
    """

    # Standard formatting instructions for consistent output formatting
    FORMATTING_INSTRUCTIONS = """
FORMAT YOUR RESPONSE USING PROFESSIONAL MARKDOWN STRUCTURE:

### Main Headers
Use ### for primary sections (Summary, Key Recommendations, Implementation Plan)

#### Subheaders
Use #### for subsections within main categories

**Bold Emphasis**
Use **bold text** for:
- Key metrics and quantified savings
- Important deadlines and targets
- Critical action items
- Cost figures and ROI percentages

*Bullet Points and Lists*
- Use bullet points (-) for recommendation lists
- Use numbered lists (1., 2., 3.) for sequential steps
- Indent sub-items with spaces for hierarchy


*Clear Section Spacing*
- Leave blank lines between major sections
- Use horizontal rules (---) to separate distinct topics when needed
- Group related information logically

*Citation and Reference Format*
- Include relevant standards: [ENERGY STAR], [ISO 14001], [LEED Guidelines]
- Reference industry benchmarks where applicable
- Link to specific regulations or compliance requirements

ENSURE ALL RESPONSES ARE:
- Well-structured with clear visual hierarchy
- Easy to scan with bold emphasis on key points
- Professional and actionable with specific implementation guidance
- Consistently formatted across all recommendation types
"""

    # System prompts that establish AI expertise as a sustainability consultant
    SYSTEM_PROMPTS = {
        "sustainability_consultant": """You are a senior Environmental Sustainability Consultant with 15+ years of expertise in:
- Energy efficiency optimization and renewable energy integration
- Water conservation strategies and water management systems
- Waste reduction, recycling optimization, and circular economy implementation
- Sustainability technology selection and implementation
- Cost-benefit analysis for environmental initiatives
- ROI calculation and business case development for sustainability projects
- Industry best practices across manufacturing, commercial, and institutional sectors
- Environmental compliance and regulatory requirements
- Change management for sustainability initiatives
- Performance monitoring and measurement systems

Your recommendations should be:
- Practical and immediately actionable with clear implementation steps
- Financially justified with detailed cost-benefit analysis
- Technologically sound and feasible with current market solutions
- Aligned with industry best practices and proven methodologies
- Compliant with relevant environmental regulations and standards
- Scalable and adaptable to different facility sizes and sectors
- Measurable with specific KPIs and success metrics
- Risk-assessed with mitigation strategies for implementation challenges

FORMATTING REQUIREMENTS:
Structure your response using professional markdown formatting:
- ### for main section headers (Executive Summary, Key Recommendations, Implementation Plan)
- #### for subsections within each main category
- **Bold text** for key metrics, cost savings, and critical action items
- Bullet points (-) for recommendation lists and action items
- Use bullet points for comparing metrics, costs, and timelines
- Clear spacing between sections for readability
- Include specific quantified benefits with **bold emphasis** on savings figures

Always provide evidence-based recommendations with clear business cases and quantified benefits.""",

        "energy_efficiency_expert": """You are a Certified Energy Manager (CEM) and Energy Efficiency Expert specializing in:
- Comprehensive energy auditing and assessment methodologies
- HVAC optimization and building automation systems
- Lighting efficiency upgrades (LED, smart controls, daylight harvesting)
- Motor and drive efficiency improvements
- Power factor correction and electrical system optimization
- Peak demand management and load shifting strategies
- Renewable energy integration (solar, wind, geothermal)
- Energy storage solutions and grid optimization
- Utility rate analysis and tariff optimization
- Energy monitoring and management systems (EMS/BMS)
- ENERGY STAR certification and green building standards
- Demand response programs and grid interaction

Focus on energy cost reduction, demand management, and carbon footprint minimization while maintaining operational requirements.

ENERGY-SPECIFIC FORMATTING REQUIREMENTS:
Structure your energy recommendations using:
- ### Energy Analysis Summary with **total kWh savings** and **cost reductions**
- #### Equipment-Specific Recommendations (HVAC, Lighting, Motors, etc.)
- **Bold emphasis** on energy savings (kWh, kW demand reduction, % efficiency gains)
- Use bullet points for energy metrics comparison with clear formatting
- #### Implementation Phases with timeline and **payback periods**
- Reference relevant standards: [ENERGY STAR], [ASHRAE 90.1], [IEEE 519]
- Include utility rate impact analysis with **demand charge savings**""",

        "water_conservation_expert": """You are a Water Conservation Specialist and Water Systems Engineer with expertise in:
- Water audit methodologies and leak detection systems
- High-efficiency fixtures and water-saving technologies
- Water recycling and reuse system design
- Rainwater harvesting and stormwater management
- Irrigation optimization and smart water management
- Water treatment and purification technologies
- Greywater and blackwater treatment systems
- Water quality monitoring and management
- Drought contingency planning and water security
- Water footprint reduction strategies
- Smart meter implementation and water analytics
- Regulatory compliance (Clean Water Act, local water ordinances)

Emphasize water conservation, quality maintenance, cost reduction, and sustainable water use practices.

WATER-SPECIFIC FORMATTING REQUIREMENTS:
Structure your water conservation recommendations using:
- ### Water Usage Analysis with **total gallons saved** and **cost reductions**
- #### Conservation Strategy Categories (Fixtures, Systems, Behavioral)
- **Bold emphasis** on water savings (gallons/day, % reduction, leak elimination)
- Use bullet points for water consumption data with clear categorization
- #### Implementation Timeline with **payback periods** for efficiency upgrades
- Reference compliance standards: [Clean Water Act], [Local Water Ordinances], [EPA WaterSense]
- Include water quality parameters and **maintenance cost reductions**""",

        "waste_optimization_expert": """You are a Waste Management Optimization Specialist and Circular Economy Expert with expertise in:
- Waste stream analysis and characterization studies
- Zero waste program development and implementation
- Recycling program optimization and contamination reduction
- Organic waste composting and anaerobic digestion
- Hazardous waste minimization and proper disposal
- Packaging reduction and sustainable procurement
- Waste-to-energy feasibility and implementation
- Circular economy principles and material flow optimization
- Waste tracking and reporting systems
- Vendor management and waste service optimization
- Employee training and behavior change programs
- Regulatory compliance and waste auditing

Focus on waste minimization, diversion rate improvement, cost reduction, and circular economy opportunities.

WASTE-SPECIFIC FORMATTING REQUIREMENTS:
Structure your waste management recommendations using:
- ### Waste Stream Analysis with **total waste diverted** and **disposal cost savings**
- #### Waste Category Strategies (Recyclables, Organics, Hazardous, General)
- **Bold emphasis** on diversion rates (%, tons diverted, cost per ton savings)
- Use bullet points for waste disposal data and cost comparisons
- #### Implementation Methods with **ROI calculations** and disposal alternatives
- Reference regulations: [RCRA], [Local Waste Ordinances], [Zero Waste Standards]
- Include contamination reduction strategies and **vendor cost optimizations**""",

        "technology_integration_expert": """You are a Sustainability Technology Integration Expert specializing in:
- IoT sensor networks for environmental monitoring
- Building automation and smart facility systems
- Data analytics platforms for sustainability metrics
- Energy management software and optimization tools
- Water monitoring and leak detection technologies
- Waste tracking and analytics systems
- Predictive maintenance and equipment optimization
- Dashboard development and visualization tools
- System integration and interoperability
- Cybersecurity for sustainability technologies
- ROI analysis for technology investments
- Change management for technology adoption

Focus on technology solutions that provide measurable environmental and financial benefits with clear implementation pathways.

TECHNOLOGY-SPECIFIC FORMATTING REQUIREMENTS:
Structure your technology recommendations using:
- ### Technology Assessment Summary with **total implementation costs** and **projected savings**
- #### Technology Categories (Monitoring, Analytics, Automation, Integration)
- **Bold emphasis** on system specifications, data accuracy improvements, and **cost-benefit ratios**
- Use bullet points for technology comparison and cost analysis
- #### Integration Requirements with **compatibility assessments** and **security considerations**
- Reference standards: [IoT Security], [Data Privacy], [System Interoperability]
- Include vendor evaluation criteria and **ongoing maintenance costs**"""
    }

    # JSON Schema templates for structured recommendation outputs
    JSON_SCHEMAS = {
        "recommendation_analysis": {
            "type": "object",
            "properties": {
                "executive_summary": {
                    "type": "object",
                    "properties": {
                        "total_recommendations": {"type": "number"},
                        "estimated_annual_savings": {"type": "number"},
                        "total_investment_required": {"type": "number"},
                        "average_payback_period": {"type": "number"},
                        "key_focus_areas": {"type": "array", "items": {"type": "string"}},
                        "implementation_timeframe": {"type": "string"}
                    }
                },
                "quick_wins": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "recommendation": {"type": "string"},
                            "domain": {"type": "string", "enum": ["electricity", "water", "waste", "cross-domain"]},
                            "estimated_savings": {"type": "number"},
                            "savings_unit": {"type": "string"},
                            "implementation_cost": {"type": "number"},
                            "payback_period": {"type": "number"},
                            "effort_level": {"type": "string", "enum": ["minimal", "low", "medium", "high"]},
                            "timeframe": {"type": "string", "enum": ["immediate", "short_term", "medium_term", "long_term"]},
                            "implementation_steps": {"type": "array", "items": {"type": "string"}},
                            "success_metrics": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "strategic_recommendations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string"},
                            "recommendation": {"type": "string"},
                            "business_case": {"type": "string"},
                            "domains_affected": {"type": "array", "items": {"type": "string"}},
                            "priority": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
                            "estimated_savings": {"type": "number"},
                            "implementation_cost": {"type": "number"},
                            "roi_percentage": {"type": "number"},
                            "payback_period": {"type": "number"},
                            "implementation_effort": {"type": "string", "enum": ["minimal", "low", "medium", "high"]},
                            "timeframe": {"type": "string", "enum": ["immediate", "short_term", "medium_term", "long_term"]},
                            "risks": {"type": "array", "items": {"type": "string"}},
                            "mitigation_strategies": {"type": "array", "items": {"type": "string"}},
                            "implementation_roadmap": {"type": "array", "items": {"type": "string"}},
                            "success_metrics": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "technology_recommendations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "technology_category": {"type": "string"},
                            "solution_name": {"type": "string"},
                            "description": {"type": "string"},
                            "domains": {"type": "array", "items": {"type": "string"}},
                            "benefits": {"type": "array", "items": {"type": "string"}},
                            "estimated_cost": {"type": "number"},
                            "implementation_complexity": {"type": "string", "enum": ["low", "medium", "high"]},
                            "vendor_considerations": {"type": "array", "items": {"type": "string"}},
                            "integration_requirements": {"type": "array", "items": {"type": "string"}},
                            "expected_roi": {"type": "number"},
                            "maintenance_requirements": {"type": "string"}
                        }
                    }
                },
                "cost_benefit_analysis": {
                    "type": "object",
                    "properties": {
                        "total_investment": {"type": "number"},
                        "annual_operating_savings": {"type": "number"},
                        "one_time_savings": {"type": "number"},
                        "payback_period_years": {"type": "number"},
                        "net_present_value": {"type": "number"},
                        "internal_rate_of_return": {"type": "number"},
                        "cost_breakdown": {
                            "type": "object",
                            "properties": {
                                "equipment_costs": {"type": "number"},
                                "installation_costs": {"type": "number"},
                                "training_costs": {"type": "number"},
                                "ongoing_maintenance": {"type": "number"}
                            }
                        },
                        "savings_breakdown": {
                            "type": "object",
                            "properties": {
                                "energy_savings": {"type": "number"},
                                "water_savings": {"type": "number"},
                                "waste_disposal_savings": {"type": "number"},
                                "operational_efficiency": {"type": "number"},
                                "avoided_costs": {"type": "number"}
                            }
                        }
                    }
                },
                "implementation_roadmap": {
                    "type": "object",
                    "properties": {
                        "phase_1_immediate": {
                            "type": "object",
                            "properties": {
                                "duration": {"type": "string"},
                                "focus_areas": {"type": "array", "items": {"type": "string"}},
                                "key_actions": {"type": "array", "items": {"type": "string"}},
                                "expected_outcomes": {"type": "array", "items": {"type": "string"}},
                                "budget_required": {"type": "number"}
                            }
                        },
                        "phase_2_short_term": {
                            "type": "object",
                            "properties": {
                                "duration": {"type": "string"},
                                "focus_areas": {"type": "array", "items": {"type": "string"}},
                                "key_actions": {"type": "array", "items": {"type": "string"}},
                                "expected_outcomes": {"type": "array", "items": {"type": "string"}},
                                "budget_required": {"type": "number"}
                            }
                        },
                        "phase_3_medium_term": {
                            "type": "object",
                            "properties": {
                                "duration": {"type": "string"},
                                "focus_areas": {"type": "array", "items": {"type": "string"}},
                                "key_actions": {"type": "array", "items": {"type": "string"}},
                                "expected_outcomes": {"type": "array", "items": {"type": "string"}},
                                "budget_required": {"type": "number"}
                            }
                        },
                        "phase_4_long_term": {
                            "type": "object",
                            "properties": {
                                "duration": {"type": "string"},
                                "focus_areas": {"type": "array", "items": {"type": "string"}},
                                "key_actions": {"type": "array", "items": {"type": "string"}},
                                "expected_outcomes": {"type": "array", "items": {"type": "string"}},
                                "budget_required": {"type": "number"}
                            }
                        }
                    }
                }
            },
            "required": ["executive_summary", "quick_wins", "strategic_recommendations"]
        },

        "cost_benefit_analysis": {
            "type": "object",
            "properties": {
                "recommendation_summary": {
                    "type": "object",
                    "properties": {
                        "recommendation_name": {"type": "string"},
                        "category": {"type": "string"},
                        "domains_affected": {"type": "array", "items": {"type": "string"}},
                        "priority_level": {"type": "string", "enum": ["critical", "high", "medium", "low"]}
                    }
                },
                "financial_analysis": {
                    "type": "object",
                    "properties": {
                        "initial_investment": {"type": "number"},
                        "annual_operating_costs": {"type": "number"},
                        "annual_savings": {"type": "number"},
                        "net_annual_benefit": {"type": "number"},
                        "payback_period_years": {"type": "number"},
                        "roi_percentage": {"type": "number"},
                        "npv_10_years": {"type": "number"},
                        "irr_percentage": {"type": "number"}
                    }
                },
                "cost_breakdown": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "cost_category": {"type": "string"},
                            "amount": {"type": "number"},
                            "description": {"type": "string"},
                            "timing": {"type": "string"}
                        }
                    }
                },
                "savings_analysis": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "savings_category": {"type": "string"},
                            "annual_amount": {"type": "number"},
                            "calculation_method": {"type": "string"},
                            "assumptions": {"type": "array", "items": {"type": "string"}},
                            "confidence_level": {"type": "string", "enum": ["high", "medium", "low"]}
                        }
                    }
                },
                "sensitivity_analysis": {
                    "type": "object",
                    "properties": {
                        "optimistic_scenario": {"type": "object", "properties": {"roi": {"type": "number"}, "payback": {"type": "number"}}},
                        "base_case_scenario": {"type": "object", "properties": {"roi": {"type": "number"}, "payback": {"type": "number"}}},
                        "pessimistic_scenario": {"type": "object", "properties": {"roi": {"type": "number"}, "payback": {"type": "number"}}},
                        "key_risk_factors": {"type": "array", "items": {"type": "string"}}
                    }
                }
            },
            "required": ["recommendation_summary", "financial_analysis", "cost_breakdown", "savings_analysis"]
        },

        "implementation_roadmap": {
            "type": "object",
            "properties": {
                "project_overview": {
                    "type": "object",
                    "properties": {
                        "project_name": {"type": "string"},
                        "total_duration": {"type": "string"},
                        "total_budget": {"type": "number"},
                        "key_stakeholders": {"type": "array", "items": {"type": "string"}},
                        "success_criteria": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "implementation_phases": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "phase_name": {"type": "string"},
                            "duration": {"type": "string"},
                            "start_date": {"type": "string"},
                            "end_date": {"type": "string"},
                            "budget": {"type": "number"},
                            "key_deliverables": {"type": "array", "items": {"type": "string"}},
                            "milestones": {"type": "array", "items": {"type": "string"}},
                            "resources_required": {"type": "array", "items": {"type": "string"}},
                            "risks": {"type": "array", "items": {"type": "string"}},
                            "dependencies": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "resource_requirements": {
                    "type": "object",
                    "properties": {
                        "internal_resources": {"type": "array", "items": {"type": "string"}},
                        "external_vendors": {"type": "array", "items": {"type": "string"}},
                        "equipment_needs": {"type": "array", "items": {"type": "string"}},
                        "training_requirements": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "risk_mitigation": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "risk": {"type": "string"},
                            "probability": {"type": "string", "enum": ["high", "medium", "low"]},
                            "impact": {"type": "string", "enum": ["high", "medium", "low"]},
                            "mitigation_strategy": {"type": "string"},
                            "contingency_plan": {"type": "string"}
                        }
                    }
                },
                "monitoring_plan": {
                    "type": "object",
                    "properties": {
                        "kpi_tracking": {"type": "array", "items": {"type": "string"}},
                        "reporting_frequency": {"type": "string"},
                        "review_checkpoints": {"type": "array", "items": {"type": "string"}},
                        "success_metrics": {"type": "array", "items": {"type": "string"}}
                    }
                }
            },
            "required": ["project_overview", "implementation_phases", "resource_requirements"]
        }
    }

    # Base prompt templates for different recommendation types
    BASE_TEMPLATES = {
        "electricity_optimization": """Generate comprehensive electricity optimization recommendations for {facility_name} based on the following consumption analysis:

CONSUMPTION DATA AND ANALYSIS:
{consumption_data}

OPTIMIZATION FOCUS AREAS:
- Energy efficiency improvements and equipment upgrades
- Peak demand management and load shifting opportunities
- Utility rate optimization and time-of-use strategies
- Power factor correction and electrical system improvements
- HVAC optimization and building automation enhancements
- Lighting efficiency upgrades and smart controls
- Motor and drive efficiency improvements
- Renewable energy integration opportunities
- Energy storage and grid optimization solutions
- Monitoring and management system implementations

RECOMMENDATION REQUIREMENTS:
- Provide specific, actionable recommendations with clear implementation steps
- Include detailed cost-benefit analysis with payback periods
- Prioritize recommendations by ROI and implementation effort
- Consider facility constraints and operational requirements
- Address both immediate quick wins and long-term strategic initiatives
- Include technology recommendations with vendor considerations
- Ensure compliance with electrical codes and utility requirements

REQUIRED RESPONSE STRUCTURE:
### Energy Efficiency Analysis Summary
- **Total potential kWh savings** and **annual cost reductions**
- **Peak demand reduction** opportunities and **utility rate optimization**

#### Quick Win Opportunities (0-6 months)
- Equipment adjustments and **immediate savings** potential
- **Low-cost improvements** with high ROI

#### Strategic Initiatives (6+ months)
- Major equipment upgrades and **long-term savings**
- **Capital investment** requirements and **payback analysis**

#### Implementation Timeline

{output_format_instruction}

Focus on measurable energy cost reduction and sustainability improvements with proven implementation strategies.""",

        "water_conservation": """Generate comprehensive water conservation recommendations for {facility_name} based on the following consumption analysis:

CONSUMPTION DATA AND ANALYSIS:
{consumption_data}

CONSERVATION FOCUS AREAS:
- Water efficiency improvements and fixture upgrades
- Leak detection and infrastructure optimization
- Water recycling and reuse system implementation
- Rainwater harvesting and stormwater management
- Irrigation optimization and smart water management
- Water quality monitoring and treatment improvements
- Greywater and blackwater treatment systems
- Smart meter implementation and water analytics
- Drought contingency planning and water security
- Employee training and behavior change programs

RECOMMENDATION REQUIREMENTS:
- Provide specific, actionable water conservation strategies
- Include detailed cost-benefit analysis with water and cost savings
- Consider water quality requirements and regulatory compliance
- Prioritize recommendations by water savings potential and ROI
- Address both immediate conservation measures and long-term systems
- Include technology recommendations for monitoring and control
- Ensure compliance with water regulations and local ordinances

REQUIRED RESPONSE STRUCTURE:
### Water Conservation Analysis Summary
- **Total potential gallons saved** per month/year and **annual cost reductions**
- **Peak usage reduction** opportunities and **utility rate optimization**

#### Immediate Conservation Measures (0-3 months)
- Fixture upgrades and **immediate water savings** potential
- **Low-cost efficiency improvements** with quick ROI

#### System Improvements (3-12 months)
- Infrastructure upgrades and **long-term water savings**
- **Capital investment** requirements and **payback analysis**

#### Water Usage Optimization

{output_format_instruction}

Focus on measurable water reduction, cost savings, and sustainable water management practices.""",

        "waste_reduction": """Generate comprehensive waste reduction and optimization recommendations for {facility_name} based on the following waste analysis:

WASTE DATA AND ANALYSIS:
{consumption_data}

WASTE OPTIMIZATION FOCUS AREAS:
- Source reduction strategies and waste minimization
- Recycling program optimization and contamination reduction
- Composting and organic waste management programs
- Hazardous waste minimization and proper disposal
- Packaging reduction and sustainable procurement
- Waste-to-energy feasibility and implementation
- Circular economy principles and material flow optimization
- Waste tracking and reporting system improvements
- Vendor management and service optimization
- Employee training and behavior change initiatives

RECOMMENDATION REQUIREMENTS:
- Provide specific, actionable waste reduction strategies
- Include detailed cost-benefit analysis with disposal cost savings
- Consider waste stream characteristics and facility operations
- Prioritize recommendations by diversion potential and cost savings
- Address regulatory compliance and reporting requirements
- Include technology recommendations for tracking and monitoring
- Ensure alignment with circular economy principles

REQUIRED RESPONSE STRUCTURE:
### Waste Stream Analysis Summary
- **Total waste diverted** from landfill and **annual disposal cost savings**
- **Diversion rate improvements** and **contamination reduction** targets

#### Source Reduction Opportunities (0-6 months)
- Process improvements and **immediate waste reduction** potential
- **Low-cost prevention strategies** with high impact

#### Diversion Programs (3-12 months)
- Recycling and composting programs with **long-term savings**
- **Capital investment** requirements and **payback analysis**

#### Waste Management Optimization

{output_format_instruction}

Focus on measurable waste reduction, increased diversion rates, and cost-effective waste management solutions.""",

        "quick_wins_identification": """Identify high-impact, low-effort quick win opportunities for {facility_name} based on the following environmental data:

ENVIRONMENTAL DATA:
{consumption_data}

QUICK WIN CRITERIA:
- Implementation timeframe: 0-90 days maximum
- Implementation effort: Minimal to low resource requirements
- Payback period: Less than 2 years preferred
- No major capital expenditure required
- Minimal operational disruption
- High probability of success with existing resources

QUICK WIN CATEGORIES TO EVALUATE:
- Operational adjustments and behavior changes
- Equipment setting optimizations
- Maintenance improvements and scheduling changes
- Low-cost technology implementations
- Process modifications and efficiency improvements
- Employee training and awareness programs
- Vendor renegotiations and service optimizations
- Simple monitoring and tracking implementations

ANALYSIS REQUIREMENTS:
- Identify specific opportunities in electricity, water, and waste domains
- Quantify potential savings (cost and environmental impact)
- Assess implementation effort and resource requirements
- Provide step-by-step implementation guidance
- Include measurement and verification strategies
- Consider potential barriers and mitigation approaches

{output_format_instruction}

Focus on immediately actionable opportunities that provide measurable benefits with minimal investment and effort.""",

        "strategic_initiatives": """Develop long-term strategic sustainability initiatives for {facility_name} based on comprehensive environmental analysis:

COMPREHENSIVE ENVIRONMENTAL DATA:
{consumption_data}

STRATEGIC PLANNING HORIZON: {time_horizon}

STRATEGIC INITIATIVE CATEGORIES:
- Major technology implementations and system overhauls
- Infrastructure improvements and facility modifications
- Renewable energy systems and energy independence strategies
- Advanced automation and smart building technologies
- Comprehensive sustainability program development
- Circular economy and zero waste initiatives
- Carbon neutrality and emissions reduction programs
- Water independence and closed-loop systems
- Advanced analytics and predictive optimization
- Organizational change and culture transformation

STRATEGIC ANALYSIS REQUIREMENTS:
- Develop multi-year implementation roadmaps
- Conduct comprehensive business case analysis
- Assess organizational readiness and change management needs
- Identify funding sources and financing options
- Consider regulatory trends and future compliance requirements
- Evaluate technology trends and emerging solutions
- Address scalability and future expansion considerations
- Include risk assessment and mitigation strategies

{output_format_instruction}

Focus on transformational initiatives that position the facility as a sustainability leader with significant long-term benefits.""",

        "cross_domain_optimization": """Develop integrated optimization strategies for {facility_name} that address electricity, water, and waste domains holistically:

MULTI-DOMAIN ENVIRONMENTAL DATA:
{consumption_data}

INTEGRATED OPTIMIZATION FOCUS:
- Cross-domain correlations and optimization opportunities
- Synergistic solutions that benefit multiple domains simultaneously
- Resource recovery and circular economy implementations
- Integrated monitoring and management systems
- Energy-water nexus optimization strategies
- Waste-to-energy and resource recovery systems
- Combined heat and power with water treatment integration
- Smart building systems with multi-domain optimization
- Integrated sustainability metrics and reporting
- Holistic cost optimization across all domains

CROSS-DOMAIN ANALYSIS REQUIREMENTS:
- Identify interdependencies between electricity, water, and waste systems
- Develop solutions that optimize total resource efficiency
- Consider trade-offs and balance optimization across domains
- Assess cumulative environmental and financial impacts
- Prioritize integrated solutions over single-domain approaches
- Include system-level thinking and whole-facility optimization
- Address operational complexity and management considerations

{output_format_instruction}

Focus on integrated solutions that maximize overall sustainability performance and resource efficiency across all environmental domains.""",

        "technology_recommendations": """Generate comprehensive technology recommendations for environmental optimization at {facility_name}:

CURRENT PERFORMANCE DATA:
{consumption_data}

EXISTING SYSTEMS: {existing_systems}

TECHNOLOGY CATEGORIES TO EVALUATE:
- IoT sensors and monitoring systems for real-time data collection
- Building automation and control systems (BAS/BMS)
- Energy management software and optimization platforms
- Water monitoring and leak detection technologies
- Waste tracking and analytics systems
- Predictive maintenance and equipment optimization tools
- Dashboard development and data visualization platforms
- Smart meters and sub-metering solutions
- Renewable energy systems and energy storage
- Water treatment and recycling technologies

TECHNOLOGY ASSESSMENT CRITERIA:
- Technical feasibility and compatibility with existing systems
- Cost-effectiveness and return on investment analysis
- Implementation complexity and resource requirements
- Scalability and future expansion capabilities
- Vendor reliability and long-term support
- Integration requirements and interoperability
- Cybersecurity considerations and data protection
- Maintenance requirements and operational impact

{output_format_instruction}

Provide specific technology solutions with vendor considerations, implementation roadmaps, and business case justification.""",

        "cost_benefit_analysis": """Conduct detailed cost-benefit analysis for the following sustainability recommendation:

RECOMMENDATION DETAILS:
{recommendation_details}

FACILITY CONTEXT:
- Facility: {facility_name}
- Industry: {industry_sector}
- Current Performance: {current_performance}
- Budget Range: {budget_range}

COST-BENEFIT ANALYSIS REQUIREMENTS:
- Initial capital investment breakdown (equipment, installation, training)
- Ongoing operational costs (maintenance, monitoring, management)
- Annual cost savings by category (energy, water, waste, operational)
- Non-monetary benefits quantification (environmental, compliance, productivity)
- Financial metrics calculation (ROI, NPV, IRR, payback period)
- Sensitivity analysis for key variables and assumptions
- Risk assessment and potential cost variations
- Financing options and cash flow analysis

FINANCIAL MODELING ASSUMPTIONS:
- Analysis period: 10 years unless specified otherwise
- Discount rate: 8% for NPV calculations
- Inflation assumptions: 3% annual for costs, 2% for savings
- Utility rate escalation: 4% annually
- Equipment life assumptions: Per manufacturer specifications

{output_format_instruction}

Provide comprehensive financial justification with clear business case documentation and decision-making support.""",

        "best_practices_implementation": """Recommend industry best practices for environmental management at {facility_name}:

CURRENT FACILITY PROFILE:
- Industry Sector: {industry_sector}
- Facility Size: {facility_size}
- Current Environmental Performance: {current_performance}
- Existing Programs: {existing_systems}

BEST PRACTICES CATEGORIES:
- Energy management and efficiency programs
- Water conservation and management strategies
- Waste minimization and circular economy practices
- Environmental monitoring and reporting systems
- Employee engagement and training programs
- Vendor and supply chain sustainability integration
- Compliance management and regulatory alignment
- Performance measurement and continuous improvement
- Technology adoption and digital transformation
- Sustainability communication and transparency

INDUSTRY BENCHMARKING FOCUS:
- Compare current practices to industry leaders
- Identify gap areas and improvement opportunities
- Recommend proven methodologies and frameworks
- Consider certification programs (ENERGY STAR, LEED, ISO 14001)
- Address regulatory compliance and reporting requirements
- Include change management and implementation strategies

{output_format_instruction}

Focus on proven, industry-tested practices that can be adapted to the specific facility context and operational requirements."""
    }

    # Specialized templates for prioritization and scoring
    PRIORITIZATION_TEMPLATES = {
        "recommendation_scoring": """Score and prioritize the following sustainability recommendations using a comprehensive evaluation framework:

RECOMMENDATIONS TO EVALUATE:
{recommendations_list}

SCORING CRITERIA (1-10 scale for each):
1. **Financial Impact** - Cost savings potential and ROI strength
2. **Implementation Feasibility** - Ease of implementation and resource requirements
3. **Environmental Benefit** - Sustainability impact and emission reductions
4. **Risk Level** - Implementation risks and probability of success
5. **Strategic Alignment** - Alignment with organizational goals and priorities
6. **Compliance Value** - Regulatory compliance and risk mitigation benefits
7. **Scalability** - Potential for expansion and broader application
8. **Innovation Factor** - Technology advancement and competitive advantage

PRIORITIZATION METHODOLOGY:
- Calculate weighted scores based on facility priorities
- Group recommendations by implementation timeframe
- Consider interdependencies and sequencing requirements
- Assess cumulative impact of recommendation packages
- Identify optimal implementation sequences
- Address resource constraints and capacity limitations

OUTPUT REQUIREMENTS:
For each recommendation provide:
- Individual criterion scores (1-10) with justification
- Weighted total score and priority ranking
- Implementation sequence recommendation
- Resource requirement assessment
- Risk-adjusted priority considering success probability
- Synergy opportunities with other recommendations

{output_format_instruction}

Focus on creating a clear implementation roadmap with justified prioritization and resource optimization.""",

        "portfolio_optimization": """Optimize the portfolio of sustainability recommendations for {facility_name} considering budget constraints and resource limitations:

AVAILABLE RECOMMENDATIONS:
{recommendations_portfolio}

CONSTRAINTS:
- Total Budget Available: {budget_constraint}
- Implementation Timeline: {timeline_constraint}
- Resource Constraints: {resource_constraints}
- Priority Focus Areas: {priority_areas}

PORTFOLIO OPTIMIZATION OBJECTIVES:
- Maximize total environmental impact within budget constraints
- Optimize total financial return (NPV) over planning horizon
- Balance quick wins with long-term strategic initiatives
- Ensure implementation feasibility with available resources
- Consider recommendation interdependencies and synergies
- Minimize implementation risks and complexity

OPTIMIZATION APPROACH:
- Evaluate individual recommendation metrics and constraints
- Identify synergistic recommendation combinations
- Consider alternative implementation sequences and timing
- Assess cumulative impact and resource requirements
- Optimize for multiple objectives (cost, environment, risk)
- Include contingency planning and alternative scenarios

OUTPUT REQUIREMENTS:
- Recommended implementation portfolio with justification
- Alternative portfolio scenarios (aggressive, conservative, balanced)
- Implementation timeline and resource allocation plan
- Expected outcomes and performance metrics
- Risk assessment and mitigation strategies
- Monitoring and adjustment recommendations

{output_format_instruction}

Provide a strategically optimized implementation plan that maximizes sustainability impact within operational and financial constraints."""
    }

    def __init__(self):
        """Initialize the Recommendation Prompt Templates."""
        logger.info("Initializing Recommendation Prompt Templates")

    def get_system_prompt(self, recommendation_type: RecommendationType) -> str:
        """
        Get appropriate system prompt based on recommendation type.
        
        Args:
            recommendation_type: Type of recommendation to generate
            
        Returns:
            System prompt string
        """
        prompt_mapping = {
            RecommendationType.ELECTRICITY_OPTIMIZATION: "energy_efficiency_expert",
            RecommendationType.WATER_CONSERVATION: "water_conservation_expert",
            RecommendationType.WASTE_REDUCTION: "waste_optimization_expert",
            RecommendationType.TECHNOLOGY_INTEGRATION: "technology_integration_expert",
            RecommendationType.QUICK_WINS: "sustainability_consultant",
            RecommendationType.STRATEGIC_INITIATIVES: "sustainability_consultant",
            RecommendationType.CROSS_DOMAIN_OPTIMIZATION: "sustainability_consultant",
            RecommendationType.BEST_PRACTICES: "sustainability_consultant",
            RecommendationType.COST_SAVINGS: "sustainability_consultant",
            RecommendationType.COMPLIANCE_IMPROVEMENT: "sustainability_consultant"
        }
        
        return self.SYSTEM_PROMPTS[prompt_mapping.get(recommendation_type, "sustainability_consultant")]

    def get_output_format_instruction(self, output_format: str, schema_name: str = None) -> str:
        """
        Get output format instructions for prompts.

        Args:
            output_format: Desired output format ('json', 'structured_text', 'markdown')
            schema_name: Name of JSON schema to use if JSON format

        Returns:
            Format instruction string
        """
        if output_format == "json":
            if schema_name and schema_name in self.JSON_SCHEMAS:
                return f"""OUTPUT FORMAT: Provide your recommendations as valid JSON following this exact schema:
{self.JSON_SCHEMAS[schema_name]}

Ensure all required fields are included and data types match the schema.

{self.FORMATTING_INSTRUCTIONS}"""
            else:
                return f"""OUTPUT FORMAT: Provide your recommendations as structured JSON with clear sections for analysis, recommendations, costs, and implementation details.

{self.FORMATTING_INSTRUCTIONS}"""

        elif output_format == "structured_text":
            return f"""OUTPUT FORMAT: Provide your recommendations in structured text format with clear headers:
### EXECUTIVE SUMMARY
### QUICK WIN OPPORTUNITIES
### STRATEGIC RECOMMENDATIONS
### TECHNOLOGY SOLUTIONS
### COST-BENEFIT ANALYSIS
### IMPLEMENTATION ROADMAP
### SUCCESS METRICS

{self.FORMATTING_INSTRUCTIONS}"""

        elif output_format == "markdown":
            return f"""OUTPUT FORMAT: Provide your recommendations in well-formatted Markdown with:
- ### for main section headers (Executive Summary, Key Recommendations, Implementation Plan)
- #### for subsections within each main category
- **Bold text** for key metrics, cost savings, and critical action items
- Bullet points (-) for recommendation lists and action items
- Use bullet points for cost-benefit comparisons and metrics
- Clear spacing between sections for readability
- Links to relevant standards and resources where applicable

{self.FORMATTING_INSTRUCTIONS}"""

        return f"{self.FORMATTING_INSTRUCTIONS}"

    def create_comprehensive_recommendation_prompt(
        self,
        consumption_data: str,
        context: RecommendationContext,
        recommendation_types: List[RecommendationType] = None,
        output_format: str = "json",
        include_cost_analysis: bool = True,
        include_implementation_roadmap: bool = True
    ) -> Dict[str, str]:
        """
        Create a comprehensive recommendation prompt covering multiple domains.
        
        Args:
            consumption_data: Environmental consumption and performance data
            context: Context information for recommendations
            recommendation_types: Specific types of recommendations to focus on
            output_format: Desired output format
            include_cost_analysis: Whether to include detailed cost-benefit analysis
            include_implementation_roadmap: Whether to include implementation roadmap
            
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        # Default to comprehensive analysis if no specific types requested
        if not recommendation_types:
            recommendation_types = [
                RecommendationType.ELECTRICITY_OPTIMIZATION,
                RecommendationType.WATER_CONSERVATION,
                RecommendationType.WASTE_REDUCTION,
                RecommendationType.QUICK_WINS,
                RecommendationType.STRATEGIC_INITIATIVES
            ]

        # Build comprehensive prompt
        format_instruction = self.get_output_format_instruction(output_format, "recommendation_analysis")
        
        user_prompt = f"""Generate comprehensive sustainability recommendations for {context.facility_name or 'the facility'} based on detailed environmental performance analysis.

FACILITY PROFILE:
- Industry Sector: {context.industry_sector or 'Not specified'}
- Facility Size: {context.facility_size or 'Not specified'}
- Budget Range: {context.budget_range or 'To be determined'}
- Time Horizon: {context.time_horizon or '3-5 years'}

CURRENT ENVIRONMENTAL PERFORMANCE DATA:
{consumption_data}

EXISTING SYSTEMS AND INFRASTRUCTURE:
{', '.join(context.existing_systems) if context.existing_systems else 'To be assessed during implementation'}

SUSTAINABILITY GOALS:
{chr(10).join(f'- {goal}' for goal in context.sustainability_goals) if context.sustainability_goals else '- Reduce environmental impact and operating costs'}

REGULATORY REQUIREMENTS:
{chr(10).join(f'- {req}' for req in context.regulatory_requirements) if context.regulatory_requirements else '- Ensure compliance with all applicable environmental regulations'}

OPERATIONAL CONSTRAINTS:
{chr(10).join(f'- {constraint}' for constraint in context.constraints) if context.constraints else '- Minimize operational disruption during implementation'}

RECOMMENDATION FOCUS AREAS:
{chr(10).join(f'- {rec_type.value.replace("_", " ").title()}' for rec_type in recommendation_types)}

ANALYSIS REQUIREMENTS:

1. **QUICK WIN IDENTIFICATION**
   - Identify high-impact, low-effort opportunities (0-6 months implementation)
   - Focus on operational changes and low-cost improvements
   - Provide detailed implementation steps and expected savings
   - Prioritize by ROI and implementation ease

2. **STRATEGIC RECOMMENDATIONS**
   - Develop medium to long-term sustainability initiatives
   - Include major technology implementations and system upgrades
   - Address infrastructure improvements and facility modifications
   - Consider regulatory trends and future requirements

3. **TECHNOLOGY INTEGRATION**
   - Recommend specific technologies for monitoring and optimization
   - Include IoT sensors, automation systems, and analytics platforms
   - Assess integration requirements and vendor considerations
   - Provide implementation complexity and cost estimates

4. **CROSS-DOMAIN OPTIMIZATION**
   - Identify synergies between electricity, water, and waste management
   - Recommend integrated solutions that optimize multiple domains
   - Consider system-level thinking and whole-facility optimization
   - Address operational complexity and management requirements"""

        if include_cost_analysis:
            user_prompt += """

5. **COMPREHENSIVE COST-BENEFIT ANALYSIS**
   - Provide detailed financial analysis for all major recommendations
   - Include initial investment, ongoing costs, and projected savings
   - Calculate ROI, NPV, IRR, and payback periods
   - Include sensitivity analysis and risk assessment"""

        if include_implementation_roadmap:
            user_prompt += """

6. **IMPLEMENTATION ROADMAP**
   - Develop phased implementation plan with clear timelines
   - Identify resource requirements and key stakeholders
   - Address potential risks and mitigation strategies
   - Include monitoring and measurement plans"""

        user_prompt += f"""

DELIVERABLE REQUIREMENTS:
- Prioritize recommendations by impact, feasibility, and ROI
- Provide specific, actionable implementation guidance
- Include vendor recommendations and technology specifications where appropriate
- Address change management and organizational readiness
- Ensure compliance with industry standards and regulations
- Include performance metrics and success measurement criteria

{format_instruction}

Focus on creating a comprehensive, actionable sustainability strategy that delivers measurable environmental and financial benefits."""

        return {
            "system": self.get_system_prompt(RecommendationType.CROSS_DOMAIN_OPTIMIZATION),
            "user": user_prompt
        }

    def create_domain_specific_recommendation_prompt(
        self,
        domain: str,
        consumption_data: str,
        context: RecommendationContext,
        output_format: str = "json"
    ) -> Dict[str, str]:
        """
        Create domain-specific recommendation prompt (electricity, water, or waste).
        
        Args:
            domain: Environmental domain ('electricity', 'water', 'waste')
            consumption_data: Domain-specific consumption data
            context: Context information for recommendations
            output_format: Desired output format
            
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        domain_mapping = {
            "electricity": ("electricity_optimization", RecommendationType.ELECTRICITY_OPTIMIZATION),
            "water": ("water_conservation", RecommendationType.WATER_CONSERVATION),
            "waste": ("waste_reduction", RecommendationType.WASTE_REDUCTION)
        }
        
        if domain not in domain_mapping:
            raise ValueError(f"Unsupported domain: {domain}. Must be one of: electricity, water, waste")
        
        template_key, recommendation_type = domain_mapping[domain]
        template = self.BASE_TEMPLATES[template_key]
        
        format_instruction = self.get_output_format_instruction(output_format, "recommendation_analysis")
        
        user_prompt = template.format(
            facility_name=context.facility_name or "the facility",
            consumption_data=consumption_data,
            output_format_instruction=format_instruction
        )
        
        # Add context-specific information
        if context.sustainability_goals:
            user_prompt += f"\n\nSUSTAINABILITY GOALS:\n{chr(10).join(f'- {goal}' for goal in context.sustainability_goals)}"
        
        if context.budget_range:
            user_prompt += f"\n\nBUDGET CONSIDERATIONS: {context.budget_range}"
        
        if context.constraints:
            user_prompt += f"\n\nOPERATIONAL CONSTRAINTS:\n{chr(10).join(f'- {constraint}' for constraint in context.constraints)}"

        return {
            "system": self.get_system_prompt(recommendation_type),
            "user": user_prompt
        }

    def create_quick_wins_prompt(
        self,
        consumption_data: str,
        context: RecommendationContext,
        max_implementation_days: int = 90,
        max_payback_years: float = 2.0,
        output_format: str = "json"
    ) -> Dict[str, str]:
        """
        Create prompt specifically for identifying quick win opportunities.
        
        Args:
            consumption_data: Environmental performance data
            context: Context information
            max_implementation_days: Maximum implementation timeframe in days
            max_payback_years: Maximum acceptable payback period in years
            output_format: Desired output format
            
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        template = self.BASE_TEMPLATES["quick_wins_identification"]
        format_instruction = self.get_output_format_instruction(output_format, "recommendation_analysis")
        
        user_prompt = template.format(
            facility_name=context.facility_name or "the facility",
            consumption_data=consumption_data,
            output_format_instruction=format_instruction
        )
        
        user_prompt += f"""

SPECIFIC QUICK WIN CRITERIA:
- Maximum implementation timeframe: {max_implementation_days} days
- Maximum payback period: {max_payback_years} years
- Focus on operational and behavioral changes requiring minimal capital investment
- Prioritize solutions that can be implemented with existing staff and resources

EXPECTED DELIVERABLES:
- Ranked list of quick win opportunities by ROI and ease of implementation
- Specific implementation steps for each recommendation
- Resource requirements and timeline estimates
- Expected savings quantification (cost and environmental)
- Risk assessment and success probability
- Measurement and verification strategies"""

        return {
            "system": self.get_system_prompt(RecommendationType.QUICK_WINS),
            "user": user_prompt
        }

    def create_technology_recommendation_prompt(
        self,
        consumption_data: str,
        context: RecommendationContext,
        technology_categories: List[TechnologyCategory] = None,
        output_format: str = "json"
    ) -> Dict[str, str]:
        """
        Create prompt for technology-specific recommendations.
        
        Args:
            consumption_data: Environmental performance data
            context: Context information
            technology_categories: Specific technology categories to focus on
            output_format: Desired output format
            
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        template = self.BASE_TEMPLATES["technology_recommendations"]
        format_instruction = self.get_output_format_instruction(output_format, "recommendation_analysis")
        
        existing_systems = ', '.join(context.existing_systems) if context.existing_systems else 'To be assessed'
        
        user_prompt = template.format(
            facility_name=context.facility_name or "the facility",
            consumption_data=consumption_data,
            existing_systems=existing_systems,
            output_format_instruction=format_instruction
        )
        
        if technology_categories:
            tech_focus = '\n'.join(f'- {tech.value.replace("_", " ").title()}' for tech in technology_categories)
            user_prompt += f"""

SPECIFIC TECHNOLOGY FOCUS AREAS:
{tech_focus}"""
        
        if context.budget_range:
            user_prompt += f"\n\nBUDGET RANGE: {context.budget_range}"
        
        user_prompt += """

TECHNOLOGY EVALUATION REQUIREMENTS:
- Provide specific vendor recommendations and product specifications
- Include implementation complexity assessment and timeline estimates
- Address cybersecurity considerations and data protection requirements
- Consider scalability and future expansion capabilities
- Include training requirements and change management considerations
- Provide total cost of ownership (TCO) analysis including maintenance costs"""

        return {
            "system": self.get_system_prompt(RecommendationType.TECHNOLOGY_INTEGRATION),
            "user": user_prompt
        }

    def create_cost_benefit_analysis_prompt(
        self,
        recommendation_details: str,
        context: RecommendationContext,
        analysis_period_years: int = 10,
        discount_rate: float = 0.08,
        output_format: str = "json"
    ) -> Dict[str, str]:
        """
        Create detailed cost-benefit analysis prompt for specific recommendations.
        
        Args:
            recommendation_details: Detailed description of the recommendation
            context: Context information
            analysis_period_years: Analysis period for financial calculations
            discount_rate: Discount rate for NPV calculations
            output_format: Desired output format
            
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        template = self.BASE_TEMPLATES["cost_benefit_analysis"]
        format_instruction = self.get_output_format_instruction(output_format, "cost_benefit_analysis")
        
        current_performance = str(context.current_performance) if context.current_performance else 'To be determined based on baseline assessment'
        
        user_prompt = template.format(
            recommendation_details=recommendation_details,
            facility_name=context.facility_name or "the facility",
            industry_sector=context.industry_sector or "Not specified",
            current_performance=current_performance,
            budget_range=context.budget_range or "To be determined",
            output_format_instruction=format_instruction
        )
        
        user_prompt += f"""

FINANCIAL ANALYSIS PARAMETERS:
- Analysis Period: {analysis_period_years} years
- Discount Rate: {discount_rate:.1%} for NPV calculations
- Include sensitivity analysis for key variables
- Consider financing options and cash flow implications

ADDITIONAL ANALYSIS REQUIREMENTS:
- Include non-monetary benefits quantification where possible
- Address potential risks and cost variations
- Consider regulatory compliance benefits and avoided costs
- Include benchmarking against industry standards
- Provide recommendation on implementation timing and sequencing"""

        return {
            "system": self.SYSTEM_PROMPTS["sustainability_consultant"],
            "user": user_prompt
        }

    def create_implementation_roadmap_prompt(
        self,
        recommendations_list: str,
        context: RecommendationContext,
        total_implementation_period: str = "24 months",
        output_format: str = "json"
    ) -> Dict[str, str]:
        """
        Create implementation roadmap prompt for a set of recommendations.
        
        Args:
            recommendations_list: List of recommendations to implement
            context: Context information
            total_implementation_period: Total period for implementation
            output_format: Desired output format
            
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        format_instruction = self.get_output_format_instruction(output_format, "implementation_roadmap")
        
        user_prompt = f"""Develop a comprehensive implementation roadmap for sustainability recommendations at {context.facility_name or 'the facility'}.

RECOMMENDATIONS TO IMPLEMENT:
{recommendations_list}

IMPLEMENTATION CONTEXT:
- Total Implementation Period: {total_implementation_period}
- Budget Range: {context.budget_range or 'To be determined'}
- Existing Systems: {', '.join(context.existing_systems) if context.existing_systems else 'To be assessed'}
- Operational Constraints: {', '.join(context.constraints) if context.constraints else 'Minimize disruption to operations'}

ROADMAP DEVELOPMENT REQUIREMENTS:

1. **PHASE PLANNING**
   - Divide implementation into logical phases (immediate, short-term, medium-term, long-term)
   - Sequence recommendations based on dependencies and prerequisites
   - Balance quick wins with strategic initiatives
   - Consider resource availability and capacity constraints

2. **RESOURCE PLANNING**
   - Identify internal resource requirements (staff, time, expertise)
   - Determine external vendor and contractor needs
   - Plan equipment procurement and installation schedules
   - Address training and capability development requirements

3. **RISK MANAGEMENT**
   - Identify implementation risks and potential obstacles
   - Develop mitigation strategies and contingency plans
   - Address change management and organizational readiness
   - Plan for potential delays and budget variations

4. **MONITORING AND CONTROL**
   - Define key milestones and success criteria
   - Establish monitoring and reporting frameworks
   - Plan performance measurement and verification protocols
   - Include continuous improvement and optimization strategies

5. **STAKEHOLDER ENGAGEMENT**
   - Identify key stakeholders and their roles
   - Plan communication and engagement strategies
   - Address training and change management needs
   - Ensure leadership support and organizational alignment

{format_instruction}

Focus on creating a practical, actionable roadmap that ensures successful implementation while minimizing risks and operational disruption."""

        return {
            "system": self.SYSTEM_PROMPTS["sustainability_consultant"],
            "user": user_prompt
        }

    def create_prioritization_scoring_prompt(
        self,
        recommendations_list: str,
        context: RecommendationContext,
        scoring_weights: Dict[str, float] = None,
        output_format: str = "json"
    ) -> Dict[str, str]:
        """
        Create prompt for scoring and prioritizing recommendations.
        
        Args:
            recommendations_list: List of recommendations to score and prioritize
            context: Context information
            scoring_weights: Custom weights for scoring criteria
            output_format: Desired output format
            
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        template = self.PRIORITIZATION_TEMPLATES["recommendation_scoring"]
        format_instruction = self.get_output_format_instruction(output_format)
        
        user_prompt = template.format(
            recommendations_list=recommendations_list,
            output_format_instruction=format_instruction
        )
        
        if scoring_weights:
            weights_text = '\n'.join(f'- {criterion}: {weight:.1%}' for criterion, weight in scoring_weights.items())
            user_prompt += f"""

CUSTOM SCORING WEIGHTS:
{weights_text}"""
        else:
            user_prompt += """

DEFAULT SCORING WEIGHTS:
- Financial Impact: 25%
- Implementation Feasibility: 20%
- Environmental Benefit: 20%
- Risk Level: 15%
- Strategic Alignment: 10%
- Compliance Value: 5%
- Scalability: 3%
- Innovation Factor: 2%"""
        
        if context.sustainability_goals:
            user_prompt += f"""

FACILITY SUSTAINABILITY PRIORITIES:
{chr(10).join(f'- {goal}' for goal in context.sustainability_goals)}"""
        
        if context.constraints:
            user_prompt += f"""

IMPLEMENTATION CONSTRAINTS:
{chr(10).join(f'- {constraint}' for constraint in context.constraints)}"""

        return {
            "system": self.SYSTEM_PROMPTS["sustainability_consultant"],
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
            "base_templates": list(self.BASE_TEMPLATES.keys()),
            "prioritization_templates": list(self.PRIORITIZATION_TEMPLATES.keys()),
            "json_schemas": list(self.JSON_SCHEMAS.keys()),
            "recommendation_types": [rt.value for rt in RecommendationType],
            "technology_categories": [tc.value for tc in TechnologyCategory],
            "implementation_timeframes": [tf.value for tf in ImplementationTimeframe],
            "priority_levels": [pl.value for pl in RecommendationPriority]
        }


# Convenience functions for common use cases
def create_electricity_optimization_recommendations(
    consumption_data: str,
    facility_name: str = None,
    industry_sector: str = None,
    budget_range: str = None,
    output_format: str = "json"
) -> Dict[str, str]:
    """
    Convenience function for creating electricity optimization recommendations.
    
    Args:
        consumption_data: Electricity consumption and performance data
        facility_name: Name of the facility
        industry_sector: Industry sector for context
        budget_range: Available budget range
        output_format: Desired output format
        
    Returns:
        Dictionary with 'system' and 'user' prompts
    """
    templates = RecommendationPromptTemplates()
    context = RecommendationContext(
        facility_name=facility_name,
        industry_sector=industry_sector,
        budget_range=budget_range
    )
    
    return templates.create_domain_specific_recommendation_prompt(
        "electricity",
        consumption_data,
        context,
        output_format
    )


def create_water_conservation_recommendations(
    consumption_data: str,
    facility_name: str = None,
    industry_sector: str = None,
    budget_range: str = None,
    output_format: str = "json"
) -> Dict[str, str]:
    """
    Convenience function for creating water conservation recommendations.
    
    Args:
        consumption_data: Water consumption and performance data
        facility_name: Name of the facility
        industry_sector: Industry sector for context
        budget_range: Available budget range
        output_format: Desired output format
        
    Returns:
        Dictionary with 'system' and 'user' prompts
    """
    templates = RecommendationPromptTemplates()
    context = RecommendationContext(
        facility_name=facility_name,
        industry_sector=industry_sector,
        budget_range=budget_range
    )
    
    return templates.create_domain_specific_recommendation_prompt(
        "water",
        consumption_data,
        context,
        output_format
    )


def create_waste_reduction_recommendations(
    consumption_data: str,
    facility_name: str = None,
    industry_sector: str = None,
    budget_range: str = None,
    output_format: str = "json"
) -> Dict[str, str]:
    """
    Convenience function for creating waste reduction recommendations.
    
    Args:
        consumption_data: Waste generation and management data
        facility_name: Name of the facility
        industry_sector: Industry sector for context
        budget_range: Available budget range
        output_format: Desired output format
        
    Returns:
        Dictionary with 'system' and 'user' prompts
    """
    templates = RecommendationPromptTemplates()
    context = RecommendationContext(
        facility_name=facility_name,
        industry_sector=industry_sector,
        budget_range=budget_range
    )
    
    return templates.create_domain_specific_recommendation_prompt(
        "waste",
        consumption_data,
        context,
        output_format
    )


def create_quick_wins_recommendations(
    consumption_data: str,
    facility_name: str = None,
    max_payback_years: float = 2.0,
    output_format: str = "json"
) -> Dict[str, str]:
    """
    Convenience function for creating quick win recommendations.
    
    Args:
        consumption_data: Environmental performance data
        facility_name: Name of the facility
        max_payback_years: Maximum acceptable payback period
        output_format: Desired output format
        
    Returns:
        Dictionary with 'system' and 'user' prompts
    """
    templates = RecommendationPromptTemplates()
    context = RecommendationContext(facility_name=facility_name)
    
    return templates.create_quick_wins_prompt(
        consumption_data,
        context,
        max_payback_years=max_payback_years,
        output_format=output_format
    )


def create_comprehensive_sustainability_recommendations(
    consumption_data: str,
    facility_name: str = None,
    industry_sector: str = None,
    budget_range: str = None,
    sustainability_goals: List[str] = None,
    output_format: str = "json"
) -> Dict[str, str]:
    """
    Convenience function for creating comprehensive sustainability recommendations.
    
    Args:
        consumption_data: Comprehensive environmental performance data
        facility_name: Name of the facility
        industry_sector: Industry sector for context
        budget_range: Available budget range
        sustainability_goals: Specific sustainability goals
        output_format: Desired output format
        
    Returns:
        Dictionary with 'system' and 'user' prompts
    """
    templates = RecommendationPromptTemplates()
    context = RecommendationContext(
        facility_name=facility_name,
        industry_sector=industry_sector,
        budget_range=budget_range,
        sustainability_goals=sustainability_goals or ["Reduce environmental impact", "Optimize operational costs"]
    )
    
    return templates.create_comprehensive_recommendation_prompt(
        consumption_data,
        context,
        output_format=output_format
    )


# Module-level instance for easy access
recommendation_prompts = RecommendationPromptTemplates()