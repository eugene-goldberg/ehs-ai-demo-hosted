"""
Comprehensive LLM prompt templates module for environmental risk assessment.
Provides structured prompts for analyzing risks across electricity, water, and waste domains.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RiskType(Enum):
    """Types of environmental risk assessment."""
    OPERATIONAL = "operational"
    COMPLIANCE = "compliance"
    FINANCIAL = "financial"
    ENVIRONMENTAL = "environmental"
    CROSS_DOMAIN = "cross_domain"
    SUPPLY_CHAIN = "supply_chain"
    INFRASTRUCTURE = "infrastructure"


class RiskDomain(Enum):
    """Environmental domains for risk assessment."""
    ELECTRICITY = "electricity"
    WATER = "water"
    WASTE = "waste"
    ALL_DOMAINS = "all_domains"


class RiskSeverity(Enum):
    """Risk severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


class RiskProbability(Enum):
    """Risk probability levels."""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class RiskContext:
    """Context information for risk assessment."""
    facility_name: Optional[str] = None
    time_period: Optional[str] = None
    risk_domains: List[str] = None
    assessment_scope: Optional[str] = None
    regulatory_framework: Optional[str] = None
    risk_tolerance: Optional[str] = None
    baseline_period: Optional[str] = None
    target_metrics: List[str] = None


class RiskAssessmentPromptTemplates:
    """
    Comprehensive prompt templates for environmental risk assessment.
    
    This class provides structured prompt templates for analyzing environmental
    risks across electricity, water, and waste domains using LLM models.
    """

    # System prompts that establish AI expertise in risk assessment
    SYSTEM_PROMPTS = {
        "risk_assessment_expert": """You are a senior Environmental Risk Assessment Expert with deep expertise in:
- Environmental risk identification and quantification
- Operational risk assessment for utilities and waste management
- Regulatory compliance risk analysis (EPA, OSHA, local regulations)
- Financial risk evaluation for environmental liabilities
- Environmental impact and sustainability risk assessment
- Supply chain and infrastructure risk analysis
- Cross-domain risk correlation and cascade analysis
- Risk mitigation strategy development and implementation
- Enterprise risk management frameworks and methodologies

Your risk assessments should be:
- Systematic and methodology-driven using industry standards
- Quantitative when possible with clear scoring metrics
- Compliant with risk management frameworks (ISO 31000, COSO ERM)
- Actionable with specific mitigation strategies
- Prioritized based on risk severity and probability
- Forward-looking considering emerging risks and trends

Always provide evidence-based risk evaluations with clear rationale and measurable risk metrics.""",

        "operational_risk_expert": """You are an Operational Risk Management Specialist focused on:
- Equipment failure and infrastructure risk assessment
- Process safety and operational continuity risks
- Resource availability and supply disruption risks
- Performance degradation and efficiency loss risks
- Human factor and operational error risks
- Technology and system reliability risks
- Maintenance and asset management risks
- Capacity and demand mismatch risks

Emphasize operational continuity, safety, and performance optimization in your risk assessments.""",

        "compliance_risk_expert": """You are a Regulatory Compliance Risk Expert specializing in:
- Environmental regulation compliance assessment (EPA, state, local)
- Permit and licensing risk evaluation
- Reporting and documentation compliance risks
- Audit and inspection preparation and risks
- Penalty and enforcement action risks
- Regulatory change and adaptation risks
- Industry standard compliance (ISO 14001, etc.)
- Legal liability and litigation risks

Focus on compliance gaps, regulatory exposure, and proactive compliance strategies.""",

        "financial_risk_expert": """You are a Financial Risk Assessment Specialist with expertise in:
- Environmental liability and remediation cost risks
- Utility cost volatility and budget impact risks
- Capital expenditure and investment risks
- Operational cost escalation risks
- Insurance and coverage gap risks
- Stranded asset and obsolescence risks
- Market and commodity price risks
- Cash flow and liquidity risks

Emphasize quantifiable financial impacts and cost-effective risk mitigation strategies.""",

        "environmental_impact_expert": """You are an Environmental Impact Risk Assessor focused on:
- Ecosystem and biodiversity impact risks
- Air quality and emissions risks
- Water quality and contamination risks
- Soil and groundwater impact risks
- Carbon footprint and climate change risks
- Waste disposal and contamination risks
- Resource depletion and sustainability risks
- Community and stakeholder impact risks

Focus on environmental stewardship, sustainability, and long-term environmental health."""
    }

    # JSON Schema templates for structured risk assessment outputs
    JSON_SCHEMAS = {
        "risk_assessment": {
            "type": "object",
            "properties": {
                "risk_summary": {
                    "type": "object",
                    "properties": {
                        "total_risks_identified": {"type": "number"},
                        "risk_assessment_date": {"type": "string"},
                        "assessment_scope": {"type": "string"},
                        "overall_risk_rating": {"type": "string", "enum": ["critical", "high", "medium", "low", "negligible"]},
                        "key_risk_areas": {"type": "array", "items": {"type": "string"}},
                        "immediate_actions_required": {"type": "boolean"}
                    }
                },
                "identified_risks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "risk_id": {"type": "string"},
                            "risk_name": {"type": "string"},
                            "risk_category": {"type": "string"},
                            "domain": {"type": "string", "enum": ["electricity", "water", "waste", "cross_domain"]},
                            "risk_description": {"type": "string"},
                            "potential_causes": {"type": "array", "items": {"type": "string"}},
                            "potential_impacts": {"type": "array", "items": {"type": "string"}},
                            "current_controls": {"type": "array", "items": {"type": "string"}},
                            "control_effectiveness": {"type": "string", "enum": ["effective", "partially_effective", "ineffective", "unknown"]}
                        }
                    }
                },
                "risk_scoring": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "risk_id": {"type": "string"},
                            "probability_score": {"type": "number", "minimum": 1, "maximum": 5},
                            "probability_level": {"type": "string", "enum": ["very_high", "high", "medium", "low", "very_low"]},
                            "severity_score": {"type": "number", "minimum": 1, "maximum": 5},
                            "severity_level": {"type": "string", "enum": ["critical", "high", "medium", "low", "negligible"]},
                            "overall_risk_score": {"type": "number", "minimum": 1, "maximum": 25},
                            "risk_rating": {"type": "string", "enum": ["critical", "high", "medium", "low", "negligible"]},
                            "scoring_rationale": {"type": "string"}
                        }
                    }
                },
                "mitigation_strategies": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "risk_id": {"type": "string"},
                            "mitigation_approach": {"type": "string", "enum": ["avoid", "mitigate", "transfer", "accept"]},
                            "recommended_actions": {"type": "array", "items": {"type": "string"}},
                            "implementation_priority": {"type": "string", "enum": ["immediate", "high", "medium", "low"]},
                            "estimated_cost": {"type": "string"},
                            "expected_risk_reduction": {"type": "string"},
                            "implementation_timeframe": {"type": "string"},
                            "responsible_party": {"type": "string"},
                            "success_metrics": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "risk_monitoring": {
                    "type": "object",
                    "properties": {
                        "key_risk_indicators": {"type": "array", "items": {"type": "string"}},
                        "monitoring_frequency": {"type": "string"},
                        "escalation_criteria": {"type": "array", "items": {"type": "string"}},
                        "review_schedule": {"type": "string"},
                        "reporting_requirements": {"type": "array", "items": {"type": "string"}}
                    }
                }
            },
            "required": ["risk_summary", "identified_risks", "risk_scoring", "mitigation_strategies"]
        },

        "risk_correlation_analysis": {
            "type": "object",
            "properties": {
                "correlation_summary": {
                    "type": "object",
                    "properties": {
                        "analysis_scope": {"type": "string"},
                        "domains_analyzed": {"type": "array", "items": {"type": "string"}},
                        "correlation_strength_overall": {"type": "string", "enum": ["strong", "moderate", "weak", "none"]},
                        "cascade_risk_potential": {"type": "string", "enum": ["high", "medium", "low"]},
                        "key_correlations": {"type": "number"}
                    }
                },
                "risk_correlations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "risk_1": {"type": "string"},
                            "risk_2": {"type": "string"},
                            "correlation_type": {"type": "string", "enum": ["causal", "contributory", "coincidental", "competitive"]},
                            "correlation_strength": {"type": "number", "minimum": 0, "maximum": 1},
                            "correlation_direction": {"type": "string", "enum": ["positive", "negative", "bidirectional"]},
                            "domain_1": {"type": "string"},
                            "domain_2": {"type": "string"},
                            "correlation_explanation": {"type": "string"},
                            "cascade_potential": {"type": "string", "enum": ["high", "medium", "low", "none"]}
                        }
                    }
                },
                "cascade_scenarios": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "scenario_name": {"type": "string"},
                            "trigger_risk": {"type": "string"},
                            "cascade_sequence": {"type": "array", "items": {"type": "string"}},
                            "total_impact_potential": {"type": "string", "enum": ["catastrophic", "major", "moderate", "minor"]},
                            "probability_of_cascade": {"type": "string", "enum": ["high", "medium", "low"]},
                            "prevention_strategies": {"type": "array", "items": {"type": "string"}},
                            "containment_measures": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "integrated_mitigation": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "strategy_name": {"type": "string"},
                            "target_risks": {"type": "array", "items": {"type": "string"}},
                            "affected_domains": {"type": "array", "items": {"type": "string"}},
                            "strategy_description": {"type": "string"},
                            "implementation_approach": {"type": "string"},
                            "expected_benefits": {"type": "array", "items": {"type": "string"}},
                            "resource_requirements": {"type": "string"},
                            "success_indicators": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                }
            },
            "required": ["correlation_summary", "risk_correlations"]
        }
    }

    # Risk identification prompt templates
    RISK_IDENTIFICATION_TEMPLATES = {
        "electricity_risk_identification": """Conduct a comprehensive risk identification analysis for electricity-related operations using the following data:

{consumption_data}

ELECTRICITY RISK IDENTIFICATION FOCUS AREAS:

1. **Supply and Availability Risks**
   - Grid reliability and power outages
   - Utility supplier stability and performance
   - Backup power and emergency generation
   - Peak demand and capacity constraints

2. **Infrastructure and Equipment Risks**
   - Electrical system failures and equipment breakdown
   - Aging infrastructure and maintenance requirements
   - Power quality issues and voltage fluctuations
   - Electrical safety and fire hazards

3. **Cost and Financial Risks**
   - Energy price volatility and rate increases
   - Demand charge escalation
   - Budget overruns and cost control
   - Contract and procurement risks

4. **Operational and Performance Risks**
   - Load management and demand response
   - Energy efficiency degradation
   - Process interruption due to power issues
   - Technology obsolescence and upgrade needs

5. **Regulatory and Compliance Risks**
   - Energy efficiency standards compliance
   - Environmental regulations (emissions, renewable requirements)
   - Safety and electrical code compliance
   - Reporting and documentation requirements

For each identified risk, provide:
- Risk name and category
- Detailed risk description
- Potential causes and triggers
- Possible impacts and consequences
- Current control measures (if any)
- Risk indicators and warning signs

{output_format_instruction}

Focus on risks that could significantly impact operations, costs, compliance, or safety.""",

        "water_risk_identification": """Conduct a comprehensive risk identification analysis for water-related operations using the following data:

{consumption_data}

WATER RISK IDENTIFICATION FOCUS AREAS:

1. **Supply and Availability Risks**
   - Water supply interruption and scarcity
   - Source water quality degradation
   - Drought and seasonal availability
   - Supplier reliability and capacity

2. **Infrastructure and System Risks**
   - Pipe leaks and distribution system failures
   - Water treatment system breakdowns
   - Storage tank and reservoir issues
   - Cross-connection and contamination risks

3. **Quality and Safety Risks**
   - Water contamination and health hazards
   - Chemical and biological contaminants
   - Treatment process failures
   - Testing and monitoring gaps

4. **Cost and Financial Risks**
   - Water rate increases and pricing volatility
   - Infrastructure repair and replacement costs
   - Regulatory penalties and fines
   - Emergency response and remediation costs

5. **Regulatory and Compliance Risks**
   - Safe Drinking Water Act compliance
   - Wastewater discharge permit violations
   - Water quality monitoring requirements
   - Environmental impact assessments

6. **Environmental and Sustainability Risks**
   - Groundwater depletion and sustainability
   - Ecosystem impact and habitat disruption
   - Climate change and extreme weather
   - Water footprint and conservation targets

For each identified risk, provide:
- Risk name and category
- Detailed risk description
- Potential causes and triggers
- Possible impacts and consequences
- Current control measures (if any)
- Risk indicators and warning signs

{output_format_instruction}

Focus on risks that could affect water availability, quality, compliance, or environmental sustainability.""",

        "waste_risk_identification": """Conduct a comprehensive risk identification analysis for waste management operations using the following data:

{consumption_data}

WASTE RISK IDENTIFICATION FOCUS AREAS:

1. **Disposal and Treatment Risks**
   - Landfill capacity and availability
   - Treatment facility breakdowns
   - Waste stream contamination
   - Disposal method restrictions

2. **Hazardous Waste Risks**
   - Improper hazardous waste handling
   - Storage and containment failures
   - Transportation and shipping risks
   - Regulatory non-compliance penalties

3. **Environmental Impact Risks**
   - Soil and groundwater contamination
   - Air emissions and odor issues
   - Wildlife and ecosystem impact
   - Community health and safety concerns

4. **Cost and Financial Risks**
   - Disposal cost escalation
   - Regulatory penalty and fine exposure
   - Cleanup and remediation liabilities
   - Insurance and coverage gaps

5. **Operational and Process Risks**
   - Waste segregation and sorting errors
   - Collection and transportation disruptions
   - Recycling market volatility
   - Volume and capacity management

6. **Regulatory and Compliance Risks**
   - RCRA and hazardous waste regulations
   - Manifest and tracking requirements
   - Permit and licensing compliance
   - Reporting and documentation gaps

7. **Supply Chain and Vendor Risks**
   - Waste hauler and vendor reliability
   - Treatment facility compliance
   - Contract and service interruptions
   - Vendor financial stability

For each identified risk, provide:
- Risk name and category
- Detailed risk description
- Potential causes and triggers
- Possible impacts and consequences
- Current control measures (if any)
- Risk indicators and warning signs

{output_format_instruction}

Focus on risks that could result in regulatory violations, environmental damage, or significant cost impacts."""
    }

    # Risk severity assessment templates with scoring criteria
    SEVERITY_ASSESSMENT_TEMPLATES = {
        "severity_scoring": """Evaluate the potential severity of the following identified environmental risks using a standardized scoring methodology:

{risk_data}

SEVERITY SCORING CRITERIA (1-5 scale):

**SCORE 5 - CRITICAL SEVERITY:**
- Catastrophic operational impact (>24 hour shutdown)
- Severe environmental damage or contamination
- Major regulatory violations with criminal liability
- Financial impact >$1M or >10% annual budget
- Serious injury or fatality potential
- Permanent reputation damage

**SCORE 4 - HIGH SEVERITY:**
- Major operational disruption (4-24 hours)
- Significant environmental impact
- Major regulatory penalties ($100K+)
- Financial impact $100K-$1M or 2-10% budget
- Potential for serious injury
- Significant reputation damage

**SCORE 3 - MEDIUM SEVERITY:**
- Moderate operational impact (1-4 hours)
- Localized environmental impact
- Moderate regulatory penalties ($10K-$100K)
- Financial impact $10K-$100K or 0.5-2% budget
- Minor injury potential
- Moderate reputation impact

**SCORE 2 - LOW SEVERITY:**
- Minor operational inconvenience (<1 hour)
- Minimal environmental impact
- Minor regulatory issues ($1K-$10K)
- Financial impact $1K-$10K or <0.5% budget
- No injury potential
- Minimal reputation impact

**SCORE 1 - NEGLIGIBLE SEVERITY:**
- No operational impact
- No environmental impact
- No regulatory consequences
- Minimal financial impact (<$1K)
- No safety concerns
- No reputation impact

ASSESSMENT REQUIREMENTS:
For each risk, provide:
1. **Risk ID and Name**
2. **Severity Score** (1-5)
3. **Severity Level** (Critical/High/Medium/Low/Negligible)
4. **Impact Categories Affected** (Operational/Environmental/Financial/Regulatory/Safety/Reputation)
5. **Detailed Severity Rationale** explaining the score
6. **Worst-Case Scenario Description**
7. **Impact Quantification** (where possible)
8. **Supporting Evidence** from the data

{output_format_instruction}

Consider both direct impacts and secondary/cascading effects in your severity assessment.""",

        "multi_domain_severity": """Assess the severity of cross-domain environmental risks considering interconnected impacts across electricity, water, and waste systems:

{risk_data}

CROSS-DOMAIN SEVERITY FACTORS:

**AMPLIFICATION EFFECTS:**
- How risks in one domain amplify impacts in others
- Cascade potential across multiple systems
- Cumulative environmental and operational impacts
- Resource interdependency vulnerabilities

**DOMAIN-SPECIFIC IMPACT AREAS:**

*Electricity Domain Impacts:*
- Power outages and grid instability
- Energy cost escalation and budget impacts
- Equipment failures and safety hazards
- Compliance with energy regulations

*Water Domain Impacts:*
- Water supply interruption or contamination
- Treatment system failures
- Wastewater discharge violations
- Water conservation target failures

*Waste Domain Impacts:*
- Waste disposal service disruption
- Hazardous waste compliance violations
- Environmental contamination events
- Recycling and diversion target failures

**INTEGRATED SEVERITY SCORING:**
Consider cumulative severity when risks affect multiple domains:
- Single domain impact: Use standard severity criteria
- Two domain impact: Increase severity by 0.5-1.0 points
- Three+ domain impact: Increase severity by 1.0-1.5 points
- System-wide cascade risk: Maximum severity consideration

For each cross-domain risk, provide:
1. **Risk identification and affected domains**
2. **Individual domain severity scores**
3. **Cross-domain amplification factor**
4. **Integrated severity score and level**
5. **Cascade impact analysis**
6. **Cumulative impact description**
7. **Multi-domain mitigation complexity assessment**

{output_format_instruction}

Focus on risks that create significant vulnerabilities across multiple environmental systems."""
    }

    # Risk probability evaluation prompts
    PROBABILITY_ASSESSMENT_TEMPLATES = {
        "probability_scoring": """Evaluate the probability of occurrence for the following identified environmental risks using historical data, current conditions, and predictive indicators:

{risk_data}

PROBABILITY SCORING CRITERIA (1-5 scale):

**SCORE 5 - VERY HIGH PROBABILITY (>80% likelihood):**
- Risk factors are currently present and active
- Historical data shows frequent occurrence
- Contributing conditions are worsening
- Early warning indicators are triggered
- Expert consensus indicates imminent risk

**SCORE 4 - HIGH PROBABILITY (60-80% likelihood):**
- Most risk factors are present
- Historical data shows regular occurrence
- Current trends support risk materialization
- Multiple risk indicators are elevated
- Preventive controls are strained or failing

**SCORE 3 - MEDIUM PROBABILITY (30-60% likelihood):**
- Some risk factors are present
- Historical occurrence is periodic/seasonal
- Current conditions are mixed
- Some risk indicators show concern
- Control systems are functioning but stressed

**SCORE 2 - LOW PROBABILITY (10-30% likelihood):**
- Few risk factors are present
- Historical occurrence is rare
- Current conditions are generally favorable
- Risk indicators are mostly normal
- Control systems are effective

**SCORE 1 - VERY LOW PROBABILITY (<10% likelihood):**
- Risk factors are minimal or absent
- No historical occurrence or extremely rare
- Current conditions strongly favor prevention
- All risk indicators are favorable
- Strong control systems are in place

PROBABILITY ASSESSMENT FACTORS:

**Historical Analysis:**
- Frequency of similar incidents in the past
- Seasonal or cyclical patterns
- Trend analysis and pattern recognition
- Industry benchmarking data

**Current Conditions:**
- Present state of risk factors
- Environmental and operational conditions
- Equipment age and condition
- Resource constraints and pressures

**Predictive Indicators:**
- Leading indicators and early warning signs
- Predictive maintenance and monitoring data
- Environmental and regulatory changes
- Market and external factor trends

**Control Effectiveness:**
- Current risk control adequacy
- Control system reliability and performance
- Human factor and training effectiveness
- Preventive maintenance and inspection results

For each risk, provide:
1. **Risk ID and Name**
2. **Probability Score** (1-5)
3. **Probability Level** (Very High/High/Medium/Low/Very Low)
4. **Historical Evidence** supporting the assessment
5. **Current Risk Factors** present or absent
6. **Predictive Indicators** and their status
7. **Control System Assessment** and effectiveness
8. **Probability Rationale** with supporting data
9. **Time Horizon** for the probability assessment
10. **Confidence Level** in the assessment

{output_format_instruction}

Base your assessment on available data and clearly identify assumptions or data gaps."""
    }

    # Compliance risk detection templates
    COMPLIANCE_RISK_TEMPLATES = {
        "regulatory_compliance_assessment": """Conduct a comprehensive regulatory compliance risk assessment for environmental operations across electricity, water, and waste domains:

{compliance_data}

REGULATORY FRAMEWORK ANALYSIS:

**Federal Regulations:**
- EPA environmental regulations (Clean Air Act, Clean Water Act, RCRA)
- OSHA safety and health requirements
- DOE energy efficiency standards
- DOT hazardous material transportation

**State and Local Regulations:**
- State environmental protection requirements
- Local water and sewer ordinances
- State waste management regulations
- Building and electrical codes

**Industry Standards:**
- ISO 14001 Environmental Management
- ISO 45001 Occupational Health and Safety
- Industry-specific standards and best practices
- Voluntary certification programs

COMPLIANCE RISK CATEGORIES:

**1. Permit and Licensing Risks:**
- Permit expiration and renewal risks
- License condition violations
- Operating outside permit limits
- New permit requirements

**2. Reporting and Documentation Risks:**
- Required report submission failures
- Data accuracy and completeness issues
- Record keeping and documentation gaps
- Third-party certification requirements

**3. Monitoring and Testing Risks:**
- Required monitoring program failures
- Testing frequency and method compliance
- Calibration and quality assurance gaps
- Data validation and verification issues

**4. Operational Compliance Risks:**
- Process and procedure violations
- Equipment and technology standard compliance
- Training and competency requirements
- Emergency response and preparedness

**5. Financial and Penalty Risks:**
- Regulatory penalty and fine exposure
- Cleanup and remediation liabilities
- Legal costs and litigation risks
- Insurance and bonding requirements

COMPLIANCE RISK ASSESSMENT:
For each compliance area, evaluate:
1. **Regulatory Requirement** description
2. **Current Compliance Status** (Compliant/At Risk/Non-Compliant)
3. **Compliance Gap Analysis** and specific issues
4. **Risk of Detection** by regulators
5. **Potential Penalties** and consequences
6. **Probability of Violation** occurrence
7. **Impact Severity** of non-compliance
8. **Corrective Actions Required** to achieve compliance
9. **Timeline and Resources** needed
10. **Monitoring and Assurance** measures

{output_format_instruction}

Prioritize compliance risks based on regulatory scrutiny, penalty severity, and likelihood of violation.""",

        "audit_readiness_assessment": """Evaluate audit readiness and compliance verification risks for environmental management systems:

{audit_data}

AUDIT READINESS EVALUATION:

**Documentation Review:**
- Policy and procedure completeness
- Record keeping and data management
- Training and competency documentation
- Incident and corrective action records

**System Implementation:**
- Environmental management system effectiveness
- Monitoring and measurement systems
- Internal audit and review processes
- Management review and oversight

**Performance Verification:**
- Compliance with legal requirements
- Achievement of environmental objectives
- Effectiveness of control measures
- Continuous improvement demonstration

**Audit Risk Factors:**
- Previous audit findings and patterns
- Regulatory enforcement trends
- Industry scrutiny and focus areas
- Stakeholder complaints and concerns

For each audit area, assess:
1. **Audit Topic** and regulatory focus
2. **Documentation Adequacy** (Complete/Adequate/Inadequate)
3. **System Implementation Status** (Effective/Partial/Ineffective)
4. **Evidence Quality** and availability
5. **Potential Audit Findings** and deficiencies
6. **Corrective Action Requirements** and timeline
7. **Audit Risk Rating** (High/Medium/Low)
8. **Preparation Priority** and resource needs

{output_format_instruction}

Focus on audit areas with highest risk of findings and regulatory consequences."""
    }

    def __init__(self):
        """Initialize the Risk Assessment Prompt Templates."""
        logger.info("Initializing Risk Assessment Prompt Templates")

    def get_system_prompt(self, risk_type: RiskType) -> str:
        """
        Get appropriate system prompt based on risk assessment type.
        
        Args:
            risk_type: Type of risk assessment
            
        Returns:
            System prompt string
        """
        prompt_mapping = {
            RiskType.OPERATIONAL: "operational_risk_expert",
            RiskType.COMPLIANCE: "compliance_risk_expert",
            RiskType.FINANCIAL: "financial_risk_expert",
            RiskType.ENVIRONMENTAL: "environmental_impact_expert",
            RiskType.CROSS_DOMAIN: "risk_assessment_expert",
            RiskType.SUPPLY_CHAIN: "risk_assessment_expert",
            RiskType.INFRASTRUCTURE: "operational_risk_expert"
        }
        
        return self.SYSTEM_PROMPTS[prompt_mapping.get(risk_type, "risk_assessment_expert")]

    def get_output_format_instruction(self, output_format: str, schema_name: str = None) -> str:
        """
        Get output format instructions for risk assessment prompts.
        
        Args:
            output_format: Desired output format
            schema_name: Name of JSON schema to use if JSON format
            
        Returns:
            Format instruction string
        """
        if output_format.lower() == "json":
            if schema_name and schema_name in self.JSON_SCHEMAS:
                return f"""OUTPUT FORMAT: Provide your risk assessment as valid JSON following this exact schema:
{self.JSON_SCHEMAS[schema_name]}

Ensure all required fields are included and data types match the schema."""
            else:
                return """OUTPUT FORMAT: Provide your risk assessment as structured JSON with clear sections for risk identification, scoring, and mitigation strategies."""
        
        elif output_format.lower() == "structured_text":
            return """OUTPUT FORMAT: Provide your risk assessment in structured text format with clear headers:
## RISK ASSESSMENT SUMMARY
## IDENTIFIED RISKS
## RISK SCORING AND PRIORITIZATION
## SEVERITY AND PROBABILITY ANALYSIS
## MITIGATION STRATEGIES
## MONITORING AND CONTROL RECOMMENDATIONS
## IMPLEMENTATION PRIORITIES"""
        
        elif output_format.lower() == "markdown":
            return """OUTPUT FORMAT: Provide your risk assessment in well-formatted Markdown with:
- Clear section headers and subsections
- Risk matrices and scoring tables
- Bullet points for risk factors and mitigation actions
- Bold text for critical risks and priorities
- Structured lists for recommendations and action items"""
        
        return ""

    def create_risk_identification_prompt(
        self,
        domain: RiskDomain,
        consumption_data: str,
        context: RiskContext,
        output_format: str = "json"
    ) -> Dict[str, str]:
        """
        Create a risk identification prompt for specific environmental domain.
        
        Args:
            domain: Environmental domain for risk assessment
            consumption_data: Consumption and operational data
            context: Context information for the assessment
            output_format: Desired output format
            
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        # Select appropriate template
        template_key = f"{domain.value}_risk_identification"
        if template_key not in self.RISK_IDENTIFICATION_TEMPLATES:
            # Fall back to general cross-domain template
            template_key = "electricity_risk_identification"  # Use as base template
        
        template = self.RISK_IDENTIFICATION_TEMPLATES[template_key]
        
        # Get output format instructions
        format_instruction = self.get_output_format_instruction(output_format, "risk_assessment")
        
        # Format the template
        user_prompt = template.format(
            consumption_data=consumption_data,
            output_format_instruction=format_instruction
        )
        
        # Add context information
        if context.regulatory_framework:
            user_prompt += f"\n\nREGULATORY CONTEXT: Consider compliance requirements under {context.regulatory_framework}."
        
        if context.risk_tolerance:
            user_prompt += f"\n\nRISK TOLERANCE: Organization risk tolerance level is {context.risk_tolerance}."
        
        if context.assessment_scope:
            user_prompt += f"\n\nASSESSMENT SCOPE: Focus the risk assessment on {context.assessment_scope}."

        return {
            "system": self.get_system_prompt(RiskType.OPERATIONAL),
            "user": user_prompt
        }

    def create_severity_assessment_prompt(
        self,
        risk_data: str,
        context: RiskContext,
        multi_domain: bool = False,
        output_format: str = "json"
    ) -> Dict[str, str]:
        """
        Create a risk severity assessment prompt with scoring criteria.
        
        Args:
            risk_data: Identified risks data
            context: Context information
            multi_domain: Whether to assess cross-domain impacts
            output_format: Desired output format
            
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        template_key = "multi_domain_severity" if multi_domain else "severity_scoring"
        template = self.SEVERITY_ASSESSMENT_TEMPLATES[template_key]
        
        format_instruction = self.get_output_format_instruction(output_format, "risk_assessment")
        
        user_prompt = template.format(
            risk_data=risk_data,
            output_format_instruction=format_instruction
        )
        
        if context.facility_name:
            user_prompt += f"\n\nFACILITY CONTEXT: Assessment is for {context.facility_name}."
        
        if context.target_metrics:
            user_prompt += f"\n\nFOCUS METRICS: Prioritize severity assessment for impacts on: {', '.join(context.target_metrics)}."

        return {
            "system": self.get_system_prompt(RiskType.OPERATIONAL),
            "user": user_prompt
        }

    def create_probability_assessment_prompt(
        self,
        risk_data: str,
        context: RiskContext,
        output_format: str = "json"
    ) -> Dict[str, str]:
        """
        Create a risk probability assessment prompt.
        
        Args:
            risk_data: Risk information for probability assessment
            context: Context information
            output_format: Desired output format
            
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        template = self.PROBABILITY_ASSESSMENT_TEMPLATES["probability_scoring"]
        format_instruction = self.get_output_format_instruction(output_format, "risk_assessment")
        
        user_prompt = template.format(
            risk_data=risk_data,
            output_format_instruction=format_instruction
        )
        
        if context.baseline_period:
            user_prompt += f"\n\nHISTORICAL BASELINE: Use {context.baseline_period} as the baseline period for historical analysis."
        
        if context.time_period:
            user_prompt += f"\n\nASSESSMENT TIMEFRAME: Evaluate probability for {context.time_period}."

        return {
            "system": self.get_system_prompt(RiskType.OPERATIONAL),
            "user": user_prompt
        }

    def create_compliance_risk_prompt(
        self,
        compliance_data: str,
        context: RiskContext,
        assessment_type: str = "regulatory_compliance",
        output_format: str = "json"
    ) -> Dict[str, str]:
        """
        Create a compliance risk assessment prompt.
        
        Args:
            compliance_data: Compliance-related data
            context: Context information
            assessment_type: Type of compliance assessment
            output_format: Desired output format
            
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        template = self.COMPLIANCE_RISK_TEMPLATES.get(
            f"{assessment_type}_assessment",
            self.COMPLIANCE_RISK_TEMPLATES["regulatory_compliance_assessment"]
        )
        
        format_instruction = self.get_output_format_instruction(output_format, "risk_assessment")
        
        user_prompt = template.format(
            compliance_data=compliance_data,
            output_format_instruction=format_instruction
        )
        
        if context.regulatory_framework:
            user_prompt += f"\n\nREGULATORY FRAMEWORK: Focus on compliance with {context.regulatory_framework}."

        return {
            "system": self.get_system_prompt(RiskType.COMPLIANCE),
            "user": user_prompt
        }

    def create_cross_domain_risk_analysis_prompt(
        self,
        multi_domain_data: str,
        context: RiskContext,
        output_format: str = "json"
    ) -> Dict[str, str]:
        """
        Create a cross-domain risk correlation analysis prompt.
        
        Args:
            multi_domain_data: Data across multiple environmental domains
            context: Context information
            output_format: Desired output format
            
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        user_prompt = f"""Perform a comprehensive cross-domain risk correlation analysis examining relationships and cascade potential between electricity, water, and waste management risks:

{multi_domain_data}

CROSS-DOMAIN RISK ANALYSIS REQUIREMENTS:

**1. Risk Correlation Analysis:**
- Identify risks that are correlated across domains
- Assess correlation strength and statistical significance
- Determine causal vs. coincidental relationships
- Evaluate bidirectional risk influences

**2. Cascade Risk Assessment:**
- Map potential cascade scenarios where risks in one domain trigger risks in others
- Assess cascade probability and potential magnitude
- Identify critical failure points and vulnerabilities
- Evaluate containment and prevention strategies

**3. Integrated Risk Impact:**
- Calculate cumulative risk exposure across domains
- Assess system-wide vulnerabilities and resilience
- Identify single points of failure affecting multiple domains
- Evaluate resource competition and constraint risks

**4. Optimization Opportunities:**
- Identify integrated risk mitigation strategies
- Assess cross-domain risk management efficiency
- Recommend portfolio-level risk optimization
- Prioritize integrated risk reduction investments

**5. Monitoring and Early Warning:**
- Develop cross-domain risk indicators
- Design integrated monitoring systems
- Establish cascade risk triggers and alerts
- Create coordinated response protocols

{self.get_output_format_instruction(output_format, "risk_correlation_analysis")}

Focus on risks that create significant vulnerabilities or opportunities when considered across all environmental domains."""
        
        if context.risk_domains:
            user_prompt += f"\n\nDOMAIN FOCUS: Prioritize analysis for: {', '.join(context.risk_domains)}."

        return {
            "system": self.get_system_prompt(RiskType.CROSS_DOMAIN),
            "user": user_prompt
        }

    def create_mitigation_strategy_prompt(
        self,
        risk_assessment_data: str,
        context: RiskContext,
        output_format: str = "json"
    ) -> Dict[str, str]:
        """
        Create a risk mitigation strategy generation prompt.
        
        Args:
            risk_assessment_data: Completed risk assessment data
            context: Context information
            output_format: Desired output format
            
        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        user_prompt = f"""Develop comprehensive risk mitigation strategies for the identified environmental risks:

{risk_assessment_data}

MITIGATION STRATEGY DEVELOPMENT:

**Strategy Categories:**
- **AVOID**: Eliminate the risk source or activity
- **MITIGATE**: Reduce risk probability or impact
- **TRANSFER**: Share or transfer risk to third parties
- **ACCEPT**: Accept risk with monitoring and contingency plans

**Mitigation Planning Requirements:**

**1. Risk-Specific Mitigation:**
- Develop targeted strategies for each identified risk
- Consider cost-effectiveness and implementation feasibility
- Prioritize based on risk rating and resource availability
- Include both preventive and responsive measures

**2. Implementation Planning:**
- Define specific actions and deliverables
- Establish timelines and milestones
- Identify required resources and budget
- Assign responsibilities and accountability

**3. Effectiveness Measurement:**
- Define success metrics and KPIs
- Establish monitoring and review processes
- Create feedback loops and continuous improvement
- Plan for strategy adjustment and optimization

**4. Integration and Coordination:**
- Ensure strategies work together effectively
- Avoid conflicting or competing approaches
- Leverage synergies and shared resources
- Coordinate with existing management systems

**5. Contingency and Response:**
- Develop backup plans and alternatives
- Create emergency response protocols
- Establish escalation criteria and procedures
- Plan for business continuity and recovery

For each mitigation strategy, provide:
- Target risk(s) and mitigation approach
- Specific recommended actions
- Implementation priority and timeline
- Resource requirements and estimated costs
- Expected risk reduction and benefits
- Success metrics and monitoring plan
- Responsible parties and governance
- Dependencies and critical success factors

{self.get_output_format_instruction(output_format, "risk_assessment")}

Prioritize strategies that provide the greatest risk reduction for the available resources."""
        
        if context.risk_tolerance:
            user_prompt += f"\n\nRISK TOLERANCE: Develop strategies appropriate for {context.risk_tolerance} risk tolerance."

        return {
            "system": self.get_system_prompt(RiskType.OPERATIONAL),
            "user": user_prompt
        }

    def get_available_templates(self) -> Dict[str, List[str]]:
        """
        Get a summary of all available risk assessment prompt templates.
        
        Returns:
            Dictionary listing available templates by category
        """
        return {
            "system_prompts": list(self.SYSTEM_PROMPTS.keys()),
            "risk_identification": list(self.RISK_IDENTIFICATION_TEMPLATES.keys()),
            "severity_assessment": list(self.SEVERITY_ASSESSMENT_TEMPLATES.keys()),
            "probability_assessment": list(self.PROBABILITY_ASSESSMENT_TEMPLATES.keys()),
            "compliance_assessment": list(self.COMPLIANCE_RISK_TEMPLATES.keys()),
            "json_schemas": list(self.JSON_SCHEMAS.keys())
        }


# Convenience functions for common risk assessment use cases
def create_electricity_risk_assessment_prompt(
    consumption_data: str,
    facility_name: str = None,
    assessment_scope: str = None,
    output_format: str = "json"
) -> Dict[str, str]:
    """
    Convenience function for creating electricity risk assessment prompts.
    
    Args:
        consumption_data: Electricity consumption and operational data
        facility_name: Name of the facility
        assessment_scope: Scope of the risk assessment
        output_format: Desired output format
        
    Returns:
        Dictionary with 'system' and 'user' prompts
    """
    templates = RiskAssessmentPromptTemplates()
    context = RiskContext(
        facility_name=facility_name,
        assessment_scope=assessment_scope,
        risk_domains=["electricity"]
    )
    
    return templates.create_risk_identification_prompt(
        RiskDomain.ELECTRICITY,
        consumption_data,
        context,
        output_format
    )


def create_water_risk_assessment_prompt(
    consumption_data: str,
    facility_name: str = None,
    assessment_scope: str = None,
    output_format: str = "json"
) -> Dict[str, str]:
    """
    Convenience function for creating water risk assessment prompts.
    
    Args:
        consumption_data: Water consumption and operational data
        facility_name: Name of the facility
        assessment_scope: Scope of the risk assessment
        output_format: Desired output format
        
    Returns:
        Dictionary with 'system' and 'user' prompts
    """
    templates = RiskAssessmentPromptTemplates()
    context = RiskContext(
        facility_name=facility_name,
        assessment_scope=assessment_scope,
        risk_domains=["water"]
    )
    
    return templates.create_risk_identification_prompt(
        RiskDomain.WATER,
        consumption_data,
        context,
        output_format
    )


def create_waste_risk_assessment_prompt(
    consumption_data: str,
    facility_name: str = None,
    assessment_scope: str = None,
    output_format: str = "json"
) -> Dict[str, str]:
    """
    Convenience function for creating waste risk assessment prompts.
    
    Args:
        consumption_data: Waste generation and disposal data
        facility_name: Name of the facility
        assessment_scope: Scope of the risk assessment
        output_format: Desired output format
        
    Returns:
        Dictionary with 'system' and 'user' prompts
    """
    templates = RiskAssessmentPromptTemplates()
    context = RiskContext(
        facility_name=facility_name,
        assessment_scope=assessment_scope,
        risk_domains=["waste"]
    )
    
    return templates.create_risk_identification_prompt(
        RiskDomain.WASTE,
        consumption_data,
        context,
        output_format
    )


def create_comprehensive_risk_assessment_prompt(
    multi_domain_data: str,
    facility_name: str = None,
    regulatory_framework: str = None,
    risk_tolerance: str = "medium",
    output_format: str = "json"
) -> Dict[str, str]:
    """
    Convenience function for creating comprehensive cross-domain risk assessment prompts.
    
    Args:
        multi_domain_data: Data across electricity, water, and waste domains
        facility_name: Name of the facility
        regulatory_framework: Applicable regulatory framework
        risk_tolerance: Organization's risk tolerance level
        output_format: Desired output format
        
    Returns:
        Dictionary with 'system' and 'user' prompts
    """
    templates = RiskAssessmentPromptTemplates()
    context = RiskContext(
        facility_name=facility_name,
        regulatory_framework=regulatory_framework,
        risk_tolerance=risk_tolerance,
        risk_domains=["electricity", "water", "waste"]
    )
    
    return templates.create_cross_domain_risk_analysis_prompt(
        multi_domain_data,
        context,
        output_format
    )


# Module-level instance for easy access
risk_assessment_prompts = RiskAssessmentPromptTemplates()