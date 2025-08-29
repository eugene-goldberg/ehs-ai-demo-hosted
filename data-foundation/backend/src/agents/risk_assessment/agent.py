"""
LangGraph-based Risk Assessment Agent for EHS AI Platform.

This module implements a comprehensive risk assessment workflow using LangGraph that:
1. Analyzes environmental, health, and safety data from Neo4j
2. Performs risk assessment using structured methodologies
3. Generates actionable recommendations and mitigation strategies
4. Integrates with LangSmith for tracing and monitoring
5. Provides comprehensive error handling and retry logic
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional, TypedDict, Union
from datetime import datetime, timedelta, date
from enum import Enum
import asyncio

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field, validator
from neo4j import GraphDatabase, Transaction
from neo4j.time import Date as Neo4jDate, DateTime as Neo4jDateTime, Time as Neo4jTime

# Local imports
from ...langsmith_config import config as langsmith_config, tracing_context, tag_ingestion_trace
from ...shared.common_fn import create_graph_database_connection

logger = logging.getLogger(__name__)


def neo4j_json_serializer(obj):
    """
    Custom JSON serializer that handles Neo4j Date and DateTime objects.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-serializable representation
    """
    if isinstance(obj, (Neo4jDate, date)):
        return obj.isoformat()
    elif isinstance(obj, (Neo4jDateTime, datetime)):
        return obj.isoformat()
    elif isinstance(obj, Neo4jTime):
        return str(obj)
    elif hasattr(obj, 'isoformat'):  # Catch any other datetime-like objects
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj)


def safe_json_dumps(data, **kwargs):
    """
    Safe JSON dumps function that handles Neo4j types and other edge cases.
    
    Args:
        data: Data to serialize
        **kwargs: Additional arguments for json.dumps
        
    Returns:
        JSON string
    """
    kwargs.setdefault('default', neo4j_json_serializer)
    try:
        return json.dumps(data, **kwargs)
    except TypeError as e:
        logger.warning(f"JSON serialization failed, falling back to string representation: {str(e)}")
        # Fallback: convert all problematic objects to strings
        return json.dumps(data, default=str, **kwargs)


# Risk Assessment Enums
class RiskLevel(str, Enum):
    """Risk level classification."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


class RiskCategory(str, Enum):
    """Risk category classification."""
    ENVIRONMENTAL = "environmental"
    HEALTH = "health"
    SAFETY = "safety"
    COMPLIANCE = "compliance"
    OPERATIONAL = "operational"


class AssessmentStatus(str, Enum):
    """Risk assessment processing status."""
    PENDING = "pending"
    ANALYZING = "analyzing"
    ASSESSING = "assessing"
    RECOMMENDING = "recommending"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"


# Pydantic Models
class RiskFactor(BaseModel):
    """Individual risk factor identified in the analysis."""
    id: str = Field(description="Unique identifier for the risk factor")
    name: str = Field(description="Human-readable name of the risk factor")
    category: RiskCategory = Field(description="Category of risk")
    description: str = Field(description="Detailed description of the risk factor")
    source_data: List[str] = Field(description="Source data points that contribute to this risk")
    severity: float = Field(ge=0.0, le=10.0, description="Severity score (0-10)")
    probability: float = Field(ge=0.0, le=1.0, description="Probability of occurrence (0-1)")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in assessment (0-1)")


class RiskAssessment(BaseModel):
    """Complete risk assessment for a facility or operation."""
    facility_id: str = Field(description="Facility identifier")
    assessment_date: datetime = Field(description="Date of assessment")
    overall_risk_level: RiskLevel = Field(description="Overall risk classification")
    risk_score: float = Field(ge=0.0, le=100.0, description="Composite risk score (0-100)")
    risk_factors: List[RiskFactor] = Field(description="Individual risk factors identified")
    methodology: str = Field(description="Assessment methodology used")
    confidence_level: float = Field(ge=0.0, le=1.0, description="Overall confidence in assessment")


class RiskRecommendation(BaseModel):
    """Risk mitigation recommendation."""
    id: str = Field(description="Unique identifier for the recommendation")
    title: str = Field(description="Brief title of the recommendation")
    description: str = Field(description="Detailed description of recommended action")
    priority: str = Field(description="Priority level (critical, high, medium, low)")
    target_risk_factors: List[str] = Field(description="Risk factor IDs this recommendation addresses")
    estimated_impact: float = Field(ge=0.0, le=10.0, description="Expected impact on risk reduction (0-10)")
    implementation_timeline: str = Field(description="Recommended implementation timeframe")
    cost_estimate: Optional[str] = Field(description="Cost estimate category")
    responsible_party: str = Field(description="Recommended responsible party")


class RecommendationSet(BaseModel):
    """Complete set of recommendations for risk mitigation."""
    facility_id: str = Field(description="Facility identifier")
    assessment_id: str = Field(description="Associated risk assessment ID")
    recommendations: List[RiskRecommendation] = Field(description="List of recommendations")
    implementation_plan: str = Field(description="High-level implementation plan")
    estimated_risk_reduction: float = Field(ge=0.0, le=100.0, description="Expected overall risk reduction percentage")


# State Definition
class RiskAssessmentState(TypedDict):
    """State for risk assessment workflow."""
    # Input parameters
    facility_id: str
    assessment_id: str
    assessment_scope: Dict[str, Any]  # Date ranges, specific areas, etc.
    request_metadata: Dict[str, Any]
    
    # Data collection phase
    environmental_data: Optional[List[Dict[str, Any]]]
    health_data: Optional[List[Dict[str, Any]]]
    safety_data: Optional[List[Dict[str, Any]]]
    compliance_data: Optional[List[Dict[str, Any]]]
    facility_info: Optional[Dict[str, Any]]
    
    # Analysis phase
    risk_factors: Optional[List[RiskFactor]]
    analysis_results: Optional[Dict[str, Any]]
    
    # Assessment phase
    risk_assessment: Optional[RiskAssessment]
    assessment_methodology: str
    
    # Recommendation phase  
    recommendations: Optional[RecommendationSet]
    
    # Execution tracking
    status: str
    current_step: str
    errors: List[str]
    retry_count: int
    step_retry_count: int  # Track retries per step
    processing_time: Optional[float]
    
    # LangSmith tracing
    langsmith_session: Optional[str]
    trace_metadata: Dict[str, Any]
    
    # Neo4j data
    neo4j_results: Optional[Dict[str, Any]]
    
    # Output
    final_report: Optional[Dict[str, Any]]


class RiskAssessmentAgent:
    """
    LangGraph-based risk assessment agent that performs comprehensive EHS risk analysis.
    
    This agent orchestrates a multi-step workflow:
    1. Data Collection: Retrieve relevant EHS data from Neo4j
    2. Risk Analysis: Identify and analyze individual risk factors
    3. Risk Assessment: Calculate overall risk levels and scores
    4. Recommendation Generation: Create actionable mitigation strategies
    """
    
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_username: str,
        neo4j_password: str,
        neo4j_database: str = "neo4j",
        llm_model: str = "gpt-4o",
        max_retries: int = 3,
        max_step_retries: int = 2,  # Maximum retries per individual step
        enable_langsmith: bool = True,
        risk_assessment_methodology: str = "comprehensive"
    ):
        """
        Initialize the Risk Assessment Agent.
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            neo4j_database: Neo4j database name
            llm_model: LLM model to use for analysis
            max_retries: Maximum retry attempts for failed operations
            max_step_retries: Maximum retries per individual step
            enable_langsmith: Enable LangSmith tracing
            risk_assessment_methodology: Risk assessment methodology to use
        """
        self.max_retries = max_retries
        self.max_step_retries = max_step_retries
        self.enable_langsmith = enable_langsmith and langsmith_config.is_available
        self.methodology = risk_assessment_methodology
        
        # Initialize Neo4j connection
        try:
            self.graph = create_graph_database_connection(
                neo4j_uri, neo4j_username, neo4j_password, neo4j_database
            )
            logger.info("Neo4j connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise
        
        # Initialize LLM
        try:
            if "claude" in llm_model.lower():
                self.llm = ChatAnthropic(model=llm_model, temperature=0)
            else:
                self.llm = ChatOpenAI(model=llm_model, temperature=0)
            logger.info(f"LLM initialized: {llm_model}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise
        
        # Initialize output parsers
        self.risk_factor_parser = PydanticOutputParser(pydantic_object=RiskFactor)
        self.risk_assessment_parser = PydanticOutputParser(pydantic_object=RiskAssessment)
        self.recommendation_parser = PydanticOutputParser(pydantic_object=RiskRecommendation)
        
        # Initialize LLM chains
        self._initialize_llm_chains()
        
        # Build workflow graph
        self.workflow = self._build_workflow()
        
        if self.enable_langsmith:
            logger.info("LangSmith tracing enabled for risk assessment agent")
        
    def _initialize_llm_chains(self):
        """Initialize LLM chains for different assessment phases."""
        
        # Data Analysis Chain
        self.data_analysis_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """You are an expert EHS (Environmental, Health, Safety) data analyst. 
                Your task is to analyze facility data and identify potential risk factors.
                Focus on:
                1. Environmental risks (emissions, waste, water usage, compliance violations)
                2. Health risks (exposure levels, safety incidents, occupational hazards)
                3. Safety risks (equipment failures, process deviations, near misses)
                4. Compliance risks (permit violations, regulatory non-compliance)
                For each risk factor identified, provide:
                - Clear description and context
                - Potential severity and probability
                - Supporting data points
                - Category classification
                
                IMPORTANT: If no significant risk factors are found in the data, still provide:
                - A summary of data reviewed
                - Indication of data completeness
                - Any baseline risk factors typical for this facility type
                - Recommendations for additional data collection if needed
                
                Output your analysis as a structured list of risk factors."""
            ),
            HumanMessagePromptTemplate.from_template(
                """Analyze the following EHS data for facility {facility_id}:
                Environmental Data: {environmental_data}
                Health Data: {health_data}  
                Safety Data: {safety_data}
                Compliance Data: {compliance_data}
                Facility Information: {facility_info}
                Assessment Scope: {assessment_scope}
                Identify and analyze all significant risk factors present in this data.
                """
            )
        ])
        
        self.data_analysis_chain = (
            self.data_analysis_prompt 
            | self.llm 
            | RunnableLambda(lambda x: x.content)
        )
        
        # Risk Assessment Chain
        self.risk_assessment_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """You are an expert risk assessor specializing in EHS risk evaluation.
                Your task is to perform a comprehensive risk assessment based on identified risk factors.
                Use the following methodology: {methodology}
                For risk assessment:
                1. Evaluate each risk factor for severity (0-10) and probability (0-1)
                2. Calculate composite risk scores
                3. Determine overall risk level (critical/high/medium/low/negligible)
                4. Assess confidence levels based on data quality and completeness
                5. Consider interdependencies between risk factors
                
                If no risk factors were identified, create a baseline assessment with:
                - Overall risk level: NEGLIGIBLE or LOW (appropriate for facility type)
                - Risk score: 0-20 (baseline score for facility with no significant findings)
                - At least one baseline risk factor representing general operational risks
                - Clear indication that assessment is based on limited data
                
                Output format must be valid JSON matching the RiskAssessment schema.
                {format_instructions}"""
            ),
            HumanMessagePromptTemplate.from_template(
                """Assess the following risk factors for facility {facility_id}:
                Risk Factors: {risk_factors}
                Assessment Context:
                - Facility Type: {facility_type}
                - Assessment Scope: {assessment_scope}
                - Data Quality: {data_quality}
                Perform comprehensive risk assessment and provide structured output.
                """
            )
        ])
        
        self.risk_assessment_chain = (
            self.risk_assessment_prompt 
            | self.llm 
            | self.risk_assessment_parser
        )
        
        # Recommendation Generation Chain
        self.recommendation_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """You are an expert EHS consultant specializing in risk mitigation strategies.
                Your task is to generate actionable recommendations to address identified risks.
                For each recommendation:
                1. Provide specific, actionable steps
                2. Prioritize based on risk severity and implementation feasibility
                3. Estimate impact on risk reduction
                4. Suggest realistic implementation timelines
                5. Identify responsible parties
                6. Consider cost-benefit analysis
                Focus on:
                - Immediate actions for critical risks
                - Long-term strategic improvements
                - Compliance and regulatory requirements
                - Best practices and industry standards
                
                If assessment shows low/negligible risks, provide:
                - Recommendations for maintaining current status
                - Preventive measures to avoid future risks
                - Data collection improvements
                - Periodic review schedules
                
                Output format must be valid JSON matching the RecommendationSet schema.
                {format_instructions}"""
            ),
            HumanMessagePromptTemplate.from_template(
                """Generate recommendations for the following risk assessment:
                Facility ID: {facility_id}
                Risk Assessment: {risk_assessment}
                Risk Factors Summary:
                {risk_factors_summary}
                Facility Context: {facility_context}
                Generate comprehensive, prioritized recommendations for risk mitigation.
                """
            )
        ])
        
        self.recommendation_chain = (
            self.recommendation_prompt 
            | self.llm 
            | JsonOutputParser()
        )
    
    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow for risk assessment.
        
        Returns:
            Compiled workflow graph
        """
        # Create workflow
        workflow = StateGraph(RiskAssessmentState)
        
        # Add nodes
        workflow.add_node("initialize", self.initialize_assessment)
        workflow.add_node("collect_environmental_data", self.collect_environmental_data)
        workflow.add_node("collect_health_data", self.collect_health_data)
        workflow.add_node("collect_safety_data", self.collect_safety_data)
        workflow.add_node("collect_compliance_data", self.collect_compliance_data)
        workflow.add_node("collect_facility_info", self.collect_facility_info)
        workflow.add_node("analyze_risks", self.analyze_risks)
        workflow.add_node("assess_risks", self.assess_risks)
        workflow.add_node("generate_recommendations", self.generate_recommendations)
        workflow.add_node("compile_report", self.compile_report)
        workflow.add_node("handle_error", self.handle_error)
        workflow.add_node("complete_assessment", self.complete_assessment)
        
        # Add edges
        workflow.add_edge("initialize", "collect_environmental_data")
        workflow.add_edge("collect_environmental_data", "collect_health_data")
        workflow.add_edge("collect_health_data", "collect_safety_data")
        workflow.add_edge("collect_safety_data", "collect_compliance_data")
        workflow.add_edge("collect_compliance_data", "collect_facility_info")
        workflow.add_edge("collect_facility_info", "analyze_risks")
        
        # Conditional edges for analysis
        workflow.add_conditional_edges(
            "analyze_risks",
            self.check_analysis_results,
            {
                "continue": "assess_risks",
                "retry": "analyze_risks",
                "error": "handle_error"
            }
        )
        
        # Conditional edges for assessment
        workflow.add_conditional_edges(
            "assess_risks",
            self.check_assessment_results,
            {
                "continue": "generate_recommendations",
                "retry": "assess_risks", 
                "error": "handle_error"
            }
        )
        
        # Conditional edges for recommendations
        workflow.add_conditional_edges(
            "generate_recommendations",
            self.check_recommendation_results,
            {
                "continue": "compile_report",
                "retry": "generate_recommendations",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("compile_report", "complete_assessment")
        
        # Error handling
        workflow.add_conditional_edges(
            "handle_error",
            self.check_retry_needed,
            {
                "retry": "initialize",
                "fail": END
            }
        )
        
        workflow.add_edge("complete_assessment", END)
        
        # Set entry point
        workflow.set_entry_point("initialize")
        
        # Compile and return
        return workflow.compile()
    
    def initialize_assessment(self, state: RiskAssessmentState) -> RiskAssessmentState:
        """
        Initialize the risk assessment workflow.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        logger.info(f"Initializing risk assessment for facility: {state['facility_id']}")
        
        # Set initial status
        state["status"] = AssessmentStatus.ANALYZING
        state["current_step"] = "initialization"
        state["errors"] = state.get("errors", [])
        state["retry_count"] = state.get("retry_count", 0)
        state["step_retry_count"] = 0  # Reset step retry counter
        state["assessment_methodology"] = self.methodology
        
        # Initialize trace metadata
        state["trace_metadata"] = {
            "facility_id": state["facility_id"],
            "assessment_id": state["assessment_id"],
            "methodology": self.methodology,
            "start_time": datetime.utcnow().isoformat()
        }
        
        # Set up LangSmith session if enabled
        if self.enable_langsmith:
            try:
                session_name = f"risk_assessment_{state['facility_id']}_{state['assessment_id']}"
                langsmith_config.enable_tracing(f"ehs-risk-assessment-{session_name}")
                state["langsmith_session"] = session_name
                logger.info(f"LangSmith session initialized: {session_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize LangSmith session: {str(e)}")
                state["langsmith_session"] = None
        
        logger.info("Risk assessment initialization completed")
        return state
    
    def collect_environmental_data(self, state: RiskAssessmentState) -> RiskAssessmentState:
        """
        Collect environmental data from Neo4j using actual schema.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        logger.info("Collecting environmental data")
        state["current_step"] = "environmental_data_collection"
        
        try:
            # Neo4j query using actual schema - collect documents and incidents related to environmental data
            query = """
            MATCH (f:Facility {name: $facility_id})
            OPTIONAL MATCH (f)-[:BELONGS_TO*]-(d:Document)
            WHERE d.type IN ['utility_bill', 'water_bill', 'waste_manifest', 'environmental_permit', 'emission_report']
            OPTIONAL MATCH (f)-[:HAS_INCIDENT]->(i:Incident)
            WHERE i.type IN ['environmental', 'spill', 'emission_exceedance', 'waste_violation']
            WITH f, collect(DISTINCT d) as environmental_docs, collect(DISTINCT i) as environmental_incidents
            
            RETURN {
                facility_id: f.name,
                environmental_documents: [doc IN environmental_docs | {
                    id: doc.id,
                    type: doc.type,
                    title: doc.title,
                    date: doc.date,
                    content: doc.content,
                    source: doc.source
                }],
                environmental_incidents: [inc IN environmental_incidents | {
                    id: inc.id,
                    type: inc.type,
                    date: inc.date,
                    severity: inc.severity,
                    description: inc.description,
                    status: inc.status
                }]
            } as environmental_data
            """
            
            result = self.graph.query(query, {"facility_id": state["facility_id"]})
            record = result[0] if result else None
            if record:
                env_data = record["environmental_data"]
                state["environmental_data"] = [env_data] if env_data else []
                logger.info(f"Collected environmental data with {len(env_data.get('environmental_documents', []))} documents and {len(env_data.get('environmental_incidents', []))} incidents")
            else:
                state["environmental_data"] = []
                logger.warning("No environmental data found")
                    
        except Exception as e:
            error_msg = f"Environmental data collection failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            
        return state
    
    def collect_health_data(self, state: RiskAssessmentState) -> RiskAssessmentState:
        """
        Collect health and safety data from Neo4j using actual schema.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        logger.info("Collecting health and safety data")
        state["current_step"] = "health_data_collection"
        
        try:
            # Neo4j query using actual schema - collect health-related documents and incidents
            query = """
            MATCH (f:Facility {name: $facility_id})
            OPTIONAL MATCH (f)-[:BELONGS_TO*]-(d:Document)
            WHERE d.type IN ['health_report', 'medical_surveillance', 'exposure_assessment', 'health_inspection']
            OPTIONAL MATCH (f)-[:HAS_INCIDENT]->(i:Incident)
            WHERE i.type IN ['health', 'occupational_injury', 'occupational_illness', 'exposure_incident']
            WITH f, collect(DISTINCT d) as health_docs, collect(DISTINCT i) as health_incidents
            
            RETURN {
                facility_id: f.name,
                health_documents: [doc IN health_docs | {
                    id: doc.id,
                    type: doc.type,
                    title: doc.title,
                    date: doc.date,
                    content: doc.content,
                    source: doc.source
                }],
                health_incidents: [inc IN health_incidents | {
                    id: inc.id,
                    type: inc.type,
                    date: inc.date,
                    severity: inc.severity,
                    description: inc.description,
                    status: inc.status,
                    affected_personnel: inc.affected_personnel
                }]
            } as health_data
            """
            
            result = self.graph.query(query, {"facility_id": state["facility_id"]})
            record = result[0] if result else None
            if record:
                health_data = record["health_data"]
                state["health_data"] = [health_data] if health_data else []
                logger.info(f"Collected health data with {len(health_data.get('health_documents', []))} documents and {len(health_data.get('health_incidents', []))} incidents")
            else:
                state["health_data"] = []
                logger.warning("No health data found")
                    
        except Exception as e:
            error_msg = f"Health data collection failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            
        return state
    
    def collect_safety_data(self, state: RiskAssessmentState) -> RiskAssessmentState:
        """
        Collect safety-specific data from Neo4j using actual schema.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        logger.info("Collecting safety data")
        state["current_step"] = "safety_data_collection"
        
        try:
            # Neo4j query using actual schema - collect safety-related documents and incidents
            query = """
            MATCH (f:Facility {name: $facility_id})
            OPTIONAL MATCH (f)-[:BELONGS_TO*]-(d:Document)
            WHERE d.type IN ['safety_report', 'equipment_inspection', 'maintenance_log', 'safety_training', 'hazard_assessment']
            OPTIONAL MATCH (f)-[:HAS_INCIDENT]->(i:Incident)
            WHERE i.type IN ['safety', 'equipment_failure', 'near_miss', 'process_deviation', 'fire', 'explosion']
            WITH f, collect(DISTINCT d) as safety_docs, collect(DISTINCT i) as safety_incidents
            
            RETURN {
                facility_id: f.name,
                safety_documents: [doc IN safety_docs | {
                    id: doc.id,
                    type: doc.type,
                    title: doc.title,
                    date: doc.date,
                    content: doc.content,
                    source: doc.source,
                    equipment_id: doc.equipment_id
                }],
                safety_incidents: [inc IN safety_incidents | {
                    id: inc.id,
                    type: inc.type,
                    date: inc.date,
                    severity: inc.severity,
                    description: inc.description,
                    status: inc.status,
                    equipment_involved: inc.equipment_involved,
                    root_cause: inc.root_cause
                }]
            } as safety_data
            """
            
            result = self.graph.query(query, {"facility_id": state["facility_id"]})
            record = result[0] if result else None
            if record:
                safety_data = record["safety_data"]
                state["safety_data"] = [safety_data] if safety_data else []
                logger.info(f"Collected safety data with {len(safety_data.get('safety_documents', []))} documents and {len(safety_data.get('safety_incidents', []))} incidents")
            else:
                state["safety_data"] = []
                logger.warning("No safety data found")
                    
        except Exception as e:
            error_msg = f"Safety data collection failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            
        return state
    
    def collect_compliance_data(self, state: RiskAssessmentState) -> RiskAssessmentState:
        """
        Collect compliance and regulatory data from Neo4j using actual schema.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        logger.info("Collecting compliance data")
        state["current_step"] = "compliance_data_collection"
        
        try:
            # Neo4j query using actual schema - collect compliance-related documents and incidents
            query = """
            MATCH (f:Facility {name: $facility_id})
            OPTIONAL MATCH (f)-[:BELONGS_TO*]-(d:Document)
            WHERE d.type IN ['compliance_report', 'audit_report', 'permit', 'regulation_update', 'violation_notice', 'corrective_action_plan']
            OPTIONAL MATCH (f)-[:HAS_INCIDENT]->(i:Incident)
            WHERE i.type IN ['compliance_violation', 'regulatory_breach', 'permit_exceedance', 'audit_finding']
            WITH f, collect(DISTINCT d) as compliance_docs, collect(DISTINCT i) as compliance_incidents
            
            RETURN {
                facility_id: f.name,
                compliance_documents: [doc IN compliance_docs | {
                    id: doc.id,
                    type: doc.type,
                    title: doc.title,
                    date: doc.date,
                    content: doc.content,
                    source: doc.source,
                    regulation_reference: doc.regulation_reference,
                    compliance_status: doc.compliance_status
                }],
                compliance_incidents: [inc IN compliance_incidents | {
                    id: inc.id,
                    type: inc.type,
                    date: inc.date,
                    severity: inc.severity,
                    description: inc.description,
                    status: inc.status,
                    regulatory_reference: inc.regulatory_reference,
                    fine_amount: inc.fine_amount,
                    corrective_actions: inc.corrective_actions
                }]
            } as compliance_data
            """
            
            result = self.graph.query(query, {"facility_id": state["facility_id"]})
            record = result[0] if result else None
            if record:
                compliance_data = record["compliance_data"]
                state["compliance_data"] = [compliance_data] if compliance_data else []
                logger.info(f"Collected compliance data with {len(compliance_data.get('compliance_documents', []))} documents and {len(compliance_data.get('compliance_incidents', []))} incidents")
            else:
                state["compliance_data"] = []
                logger.warning("No compliance data found")
                    
        except Exception as e:
            error_msg = f"Compliance data collection failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            
        return state
    
    def collect_facility_info(self, state: RiskAssessmentState) -> RiskAssessmentState:
        """
        Collect facility information and context from Neo4j using actual schema.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        logger.info("Collecting facility information")
        state["current_step"] = "facility_info_collection"
        
        try:
            # Neo4j query using actual schema - get facility properties and relationships
            query = """
            MATCH (f:Facility {name: $facility_id})
            OPTIONAL MATCH (f)-[:BELONGS_TO*]-(d:Document)
            WHERE d.type IN ['facility_profile', 'operations_manual', 'process_description']
            WITH f, collect(DISTINCT d) as facility_docs
            
            RETURN {
                facility: {
                    name: f.name,
                    id: f.id,
                    type: f.type,
                    address: f.address,
                    coordinates: f.coordinates,
                    industry_sector: f.industry_sector,
                    operational_status: f.operational_status,
                    established_date: f.established_date,
                    employee_count: f.employee_count,
                    annual_revenue: f.annual_revenue
                },
                facility_documents: [doc IN facility_docs | {
                    id: doc.id,
                    type: doc.type,
                    title: doc.title,
                    date: doc.date,
                    content: doc.content
                }]
            } as facility_info
            """
            
            result = self.graph.query(query, {"facility_id": state["facility_id"]})
            record = result[0] if result else None
            if record:
                state["facility_info"] = record["facility_info"]
                logger.info(f"Collected facility information with {len(state['facility_info'].get('facility_documents', []))} supporting documents")
            else:
                state["facility_info"] = {}
                logger.warning("No facility information found")
                    
        except Exception as e:
            error_msg = f"Facility info collection failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            
        return state
    
    def analyze_risks(self, state: RiskAssessmentState) -> RiskAssessmentState:
        """
        Analyze collected data to identify risk factors.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        logger.info("Analyzing risks from collected data")
        state["current_step"] = "risk_analysis"
        
        try:
            # Prepare data for analysis using the safe JSON serializer
            analysis_input = {
                "facility_id": state["facility_id"],
                "environmental_data": safe_json_dumps(state.get("environmental_data", [])),
                "health_data": safe_json_dumps(state.get("health_data", [])),
                "safety_data": safe_json_dumps(state.get("safety_data", [])),
                "compliance_data": safe_json_dumps(state.get("compliance_data", [])),
                "facility_info": safe_json_dumps(state.get("facility_info", {})),
                "assessment_scope": safe_json_dumps(state.get("assessment_scope", {}))
            }
            
            # Run analysis chain
            analysis_result = self.data_analysis_chain.invoke(analysis_input)
            
            # Process analysis results
            state["analysis_results"] = {
                "raw_analysis": analysis_result,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
            # Parse risk factors from analysis
            # Note: In a production system, you would implement proper parsing
            # of the LLM output to extract structured risk factors
            state["risk_factors"] = []  # Placeholder - implement parsing logic
            
            logger.info(f"Risk analysis completed. Raw analysis length: {len(str(analysis_result))}")
            
        except Exception as e:
            error_msg = f"Risk analysis failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            
        return state
    
    def assess_risks(self, state: RiskAssessmentState) -> RiskAssessmentState:
        """
        Perform formal risk assessment based on identified risk factors.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        logger.info("Performing risk assessment")
        state["current_step"] = "risk_assessment"
        
        try:
            # Prepare assessment input
            assessment_input = {
                "facility_id": state["facility_id"],
                "risk_factors": safe_json_dumps([rf.dict() if hasattr(rf, 'dict') else rf for rf in state.get("risk_factors", [])]),
                "methodology": self.methodology,
                "facility_type": state.get("facility_info", {}).get("facility", {}).get("type", "unknown"),
                "assessment_scope": safe_json_dumps(state.get("assessment_scope", {})),
                "data_quality": "high",  # This would be calculated based on data completeness
                "format_instructions": self.risk_assessment_parser.get_format_instructions()
            }
            
            # Run assessment chain
            assessment_result = self.risk_assessment_chain.invoke(assessment_input)
            
            state["risk_assessment"] = assessment_result
            logger.info(f"Risk assessment completed. Overall risk level: {assessment_result.overall_risk_level}")
            
        except Exception as e:
            error_msg = f"Risk assessment failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            
        return state
    
    def generate_recommendations(self, state: RiskAssessmentState) -> RiskAssessmentState:
        """
        Generate risk mitigation recommendations.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        logger.info("Generating risk mitigation recommendations")
        state["current_step"] = "recommendation_generation"
        
        try:
            # Prepare recommendation input
            risk_assessment = state.get("risk_assessment")
            if not risk_assessment:
                raise ValueError("No risk assessment available for recommendation generation")
            
            # Safely serialize the risk assessment data
            risk_assessment_data = risk_assessment.dict() if hasattr(risk_assessment, 'dict') else str(risk_assessment)
            
            recommendation_input = {
                "facility_id": state["facility_id"],
                "risk_assessment": safe_json_dumps(risk_assessment_data),
                "risk_factors_summary": safe_json_dumps([rf.dict() if hasattr(rf, 'dict') else rf for rf in state.get("risk_factors", [])]),
                "facility_context": safe_json_dumps(state.get("facility_info", {})),
                "format_instructions": "Return valid JSON matching RecommendationSet schema"
            }
            
            # Run recommendation chain
            recommendation_result = self.recommendation_chain.invoke(recommendation_input)
            
            state["recommendations"] = recommendation_result
            logger.info(f"Generated {len(recommendation_result.get('recommendations', []))} recommendations")
            
        except Exception as e:
            error_msg = f"Recommendation generation failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            
        return state
    
    def compile_report(self, state: RiskAssessmentState) -> RiskAssessmentState:
        """
        Compile final risk assessment report.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        logger.info("Compiling final risk assessment report")
        state["current_step"] = "report_compilation"
        
        try:
            # Compile comprehensive report
            report = {
                "assessment_id": state["assessment_id"],
                "facility_id": state["facility_id"],
                "assessment_date": datetime.utcnow().isoformat(),
                "methodology": self.methodology,
                "status": "completed",
                "executive_summary": {
                    "overall_risk_level": state.get("risk_assessment", {}).get("overall_risk_level"),
                    "risk_score": state.get("risk_assessment", {}).get("risk_score"),
                    "total_risk_factors": len(state.get("risk_factors", [])),
                    "total_recommendations": len(state.get("recommendations", {}).get("recommendations", []))
                },
                "risk_assessment": state.get("risk_assessment"),
                "risk_factors": state.get("risk_factors"),
                "recommendations": state.get("recommendations"),
                "data_sources": {
                    "environmental_records": len(state.get("environmental_data", [])),
                    "health_records": len(state.get("health_data", [])),
                    "safety_records": len(state.get("safety_data", [])),
                    "compliance_records": len(state.get("compliance_data", []))
                },
                "metadata": {
                    "processing_time": state.get("processing_time"),
                    "langsmith_session": state.get("langsmith_session"),
                    "methodology": self.methodology,
                    "generated_by": "EHS AI Risk Assessment Agent",
                    "version": "1.0.0"
                }
            }
            
            state["final_report"] = report
            logger.info("Risk assessment report compiled successfully")
            
        except Exception as e:
            error_msg = f"Report compilation failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            
        return state
    
    def complete_assessment(self, state: RiskAssessmentState) -> RiskAssessmentState:
        """
        Complete the risk assessment workflow.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        logger.info("Completing risk assessment")
        
        state["status"] = AssessmentStatus.COMPLETED
        state["current_step"] = "completed"
        state["processing_time"] = datetime.utcnow().timestamp() - state.get("start_time", datetime.utcnow().timestamp())
        
        # Update trace metadata
        if state.get("trace_metadata"):
            state["trace_metadata"]["completion_time"] = datetime.utcnow().isoformat()
            state["trace_metadata"]["total_processing_time"] = state["processing_time"]
            state["trace_metadata"]["final_status"] = state["status"]
        
        logger.info(f"Risk assessment completed in {state['processing_time']:.2f} seconds")
        return state
    
    def handle_error(self, state: RiskAssessmentState) -> RiskAssessmentState:
        """
        Handle errors during risk assessment with improved retry logic.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        state["retry_count"] += 1
        
        logger.error(f"Error in risk assessment. Step: {state.get('current_step')}. Total retries: {state['retry_count']}")
        logger.error(f"Errors: {state['errors']}")
        
        if state["retry_count"] >= self.max_retries:
            state["status"] = AssessmentStatus.FAILED
            logger.error("Maximum retries exceeded. Risk assessment failed.")
        else:
            state["status"] = AssessmentStatus.RETRY
            # Clear step-specific errors for retry but keep history
            state["errors"] = state.get("errors", [])
        
        return state
    
    # Conditional edge check methods with improved retry logic
    def check_analysis_results(self, state: RiskAssessmentState) -> str:
        """
        Check if risk analysis completed successfully.
        
        Improved logic:
        - Allows completion even if no risk factors found (normal case)
        - Limits retries per step
        - Better error handling for missing data vs actual failures
        """
        current_step = "analyze_risks"
        step_retry_count = state.get("step_retry_count", 0)
        
        # Check for actual errors first
        if state.get("errors"):
            if step_retry_count < self.max_step_retries:
                state["step_retry_count"] = step_retry_count + 1
                logger.warning(f"Retrying {current_step}, attempt {step_retry_count + 1}/{self.max_step_retries}")
                return "retry"
            else:
                logger.error(f"Max step retries exceeded for {current_step}")
                return "error"
        
        # Check if we have analysis results (even if empty)
        analysis_results = state.get("analysis_results")
        if not analysis_results:
            if step_retry_count < self.max_step_retries:
                state["step_retry_count"] = step_retry_count + 1
                logger.warning(f"No analysis results, retrying {current_step}, attempt {step_retry_count + 1}/{self.max_step_retries}")
                return "retry"
            else:
                logger.error(f"Max step retries exceeded for {current_step} - no analysis results")
                return "error"
        
        # Reset step retry counter for next step
        state["step_retry_count"] = 0
        logger.info(f"{current_step} completed successfully")
        return "continue"
    
    def check_assessment_results(self, state: RiskAssessmentState) -> str:
        """
        Check if risk assessment completed successfully.
        
        Improved logic with step-specific retry limits.
        """
        current_step = "assess_risks"
        step_retry_count = state.get("step_retry_count", 0)
        
        # Check for actual errors first
        if state.get("errors"):
            if step_retry_count < self.max_step_retries:
                state["step_retry_count"] = step_retry_count + 1
                logger.warning(f"Retrying {current_step}, attempt {step_retry_count + 1}/{self.max_step_retries}")
                return "retry"
            else:
                logger.error(f"Max step retries exceeded for {current_step}")
                return "error"
        
        # Check if we have risk assessment results
        risk_assessment = state.get("risk_assessment")
        if not risk_assessment:
            if step_retry_count < self.max_step_retries:
                state["step_retry_count"] = step_retry_count + 1
                logger.warning(f"No risk assessment results, retrying {current_step}, attempt {step_retry_count + 1}/{self.max_step_retries}")
                return "retry"
            else:
                logger.error(f"Max step retries exceeded for {current_step} - no risk assessment results")
                return "error"
        
        # Reset step retry counter for next step
        state["step_retry_count"] = 0
        logger.info(f"{current_step} completed successfully")
        return "continue"
    
    def check_recommendation_results(self, state: RiskAssessmentState) -> str:
        """
        Check if recommendation generation completed successfully.
        
        Improved logic with step-specific retry limits.
        """
        current_step = "generate_recommendations"
        step_retry_count = state.get("step_retry_count", 0)
        
        # Check for actual errors first
        if state.get("errors"):
            if step_retry_count < self.max_step_retries:
                state["step_retry_count"] = step_retry_count + 1
                logger.warning(f"Retrying {current_step}, attempt {step_retry_count + 1}/{self.max_step_retries}")
                return "retry"
            else:
                logger.error(f"Max step retries exceeded for {current_step}")
                return "error"
        
        # Check if we have recommendations
        recommendations = state.get("recommendations")
        if not recommendations:
            if step_retry_count < self.max_step_retries:
                state["step_retry_count"] = step_retry_count + 1
                logger.warning(f"No recommendations generated, retrying {current_step}, attempt {step_retry_count + 1}/{self.max_step_retries}")
                return "retry"
            else:
                logger.error(f"Max step retries exceeded for {current_step} - no recommendations generated")
                return "error"
        
        # Reset step retry counter for next step
        state["step_retry_count"] = 0
        logger.info(f"{current_step} completed successfully")
        return "continue"
    
    def check_retry_needed(self, state: RiskAssessmentState) -> str:
        """Check if retry is needed or assessment should fail."""
        if state["status"] == AssessmentStatus.RETRY and state["retry_count"] < self.max_retries:
            logger.info(f"Retrying assessment, attempt {state['retry_count']}/{self.max_retries}")
            return "retry"
        else:
            logger.error("Assessment failed - maximum retries exceeded")
            return "fail"
    
    def assess_facility_risk(
        self,
        facility_id: str,
        assessment_scope: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        max_recursion_depth: int = 5  # Add recursion depth limit
    ) -> RiskAssessmentState:
        """
        Perform comprehensive risk assessment for a facility with improved retry handling.
        
        Args:
            facility_id: Unique facility identifier
            assessment_scope: Scope parameters (date ranges, areas, etc.)
            metadata: Additional metadata for the assessment
            max_recursion_depth: Maximum recursion depth to prevent infinite loops
            
        Returns:
            Final assessment state with results
        """
        # Check recursion depth
        recursion_depth = metadata.get("recursion_depth", 0) if metadata else 0
        if recursion_depth >= max_recursion_depth:
            logger.error(f"Maximum recursion depth ({max_recursion_depth}) exceeded for facility {facility_id}")
            error_state = self._create_error_state(facility_id, "Maximum recursion depth exceeded")
            return error_state
        
        # Generate unique assessment ID
        assessment_id = f"risk_assessment_{facility_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Update metadata with recursion tracking
        if not metadata:
            metadata = {}
        metadata["recursion_depth"] = recursion_depth + 1
        
        # Initialize state
        initial_state: RiskAssessmentState = {
            "facility_id": facility_id,
            "assessment_id": assessment_id,
            "assessment_scope": assessment_scope or {},
            "request_metadata": metadata,
            "environmental_data": None,
            "health_data": None,
            "safety_data": None,
            "compliance_data": None,
            "facility_info": None,
            "risk_factors": None,
            "analysis_results": None,
            "risk_assessment": None,
            "assessment_methodology": self.methodology,
            "recommendations": None,
            "status": AssessmentStatus.PENDING,
            "current_step": "initialization",
            "errors": [],
            "retry_count": 0,
            "step_retry_count": 0,  # Initialize step retry counter
            "processing_time": None,
            "langsmith_session": None,
            "trace_metadata": {},
            "neo4j_results": None,
            "final_report": None,
            "start_time": datetime.utcnow().timestamp()
        }
        
        # Execute workflow with error handling
        try:
            logger.info(f"Starting risk assessment for facility {facility_id} (depth: {recursion_depth})")
            
            if self.enable_langsmith:
                with tracing_context(ingestion_id=assessment_id):
                    final_state = self.workflow.invoke(initial_state)
            else:
                final_state = self.workflow.invoke(initial_state)
            
            # Log final status
            status = final_state.get("status", "unknown")
            logger.info(f"Risk assessment completed for {facility_id}: {status}")
            
            return final_state
            
        except Exception as e:
            logger.error(f"Risk assessment workflow failed for {facility_id}: {str(e)}")
            error_state = self._create_error_state(facility_id, f"Workflow execution failed: {str(e)}")
            return error_state
    
    def _create_error_state(self, facility_id: str, error_message: str) -> RiskAssessmentState:
        """
        Create an error state for failed assessments.
        
        Args:
            facility_id: Facility identifier
            error_message: Error description
            
        Returns:
            Error state
        """
        assessment_id = f"error_assessment_{facility_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        error_state: RiskAssessmentState = {
            "facility_id": facility_id,
            "assessment_id": assessment_id,
            "assessment_scope": {},
            "request_metadata": {},
            "environmental_data": None,
            "health_data": None,
            "safety_data": None,
            "compliance_data": None,
            "facility_info": None,
            "risk_factors": None,
            "analysis_results": None,
            "risk_assessment": None,
            "assessment_methodology": self.methodology,
            "recommendations": None,
            "status": AssessmentStatus.FAILED,
            "current_step": "error",
            "errors": [error_message],
            "retry_count": self.max_retries,  # Mark as max retries reached
            "step_retry_count": self.max_step_retries,
            "processing_time": 0.0,
            "langsmith_session": None,
            "trace_metadata": {
                "error_time": datetime.utcnow().isoformat(),
                "error_message": error_message
            },
            "neo4j_results": None,
            "final_report": None,
            "start_time": datetime.utcnow().timestamp()
        }
        
        return error_state
    
    def assess_multiple_facilities(
        self,
        facility_ids: List[str],
        assessment_scope: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[RiskAssessmentState]:
        """
        Perform risk assessments for multiple facilities.
        
        Args:
            facility_ids: List of facility identifiers
            assessment_scope: Scope parameters for all assessments
            metadata: Additional metadata
            
        Returns:
            List of assessment results
        """
        results = []
        
        logger.info(f"Starting risk assessment for {len(facility_ids)} facilities")
        
        for facility_id in facility_ids:
            logger.info(f"Starting risk assessment for facility: {facility_id}")
            
            result = self.assess_facility_risk(
                facility_id=facility_id,
                assessment_scope=assessment_scope,
                metadata=metadata
            )
            
            results.append(result)
            
            # Log completion status
            status = result.get("status", "unknown")
            logger.info(f"Risk assessment completed for {facility_id}: {status}")
            
        logger.info(f"Completed risk assessments for {len(facility_ids)} facilities")
        return results
    
    def close(self):
        """Close the risk assessment agent and cleanup resources."""
        try:
            if self.graph:
                self.graph.close()
                logger.info("Neo4j connection closed")
            
            if self.enable_langsmith:
                langsmith_config.disable_tracing()
                logger.info("LangSmith tracing disabled")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


# Utility functions for integration
def create_risk_assessment_agent(
    neo4j_uri: str = None,
    neo4j_username: str = None,
    neo4j_password: str = None,
    neo4j_database: str = "neo4j",
    llm_model: str = "gpt-4o",
    **kwargs
) -> RiskAssessmentAgent:
    """
    Factory function to create a risk assessment agent with environment-based configuration.
    
    Args:
        neo4j_uri: Neo4j URI (falls back to NEO4J_URI env var)
        neo4j_username: Neo4j username (falls back to NEO4J_USERNAME env var)
        neo4j_password: Neo4j password (falls back to NEO4J_PASSWORD env var)
        neo4j_database: Neo4j database name
        llm_model: LLM model to use
        **kwargs: Additional arguments for RiskAssessmentAgent
        
    Returns:
        Configured RiskAssessmentAgent instance
    """
    # Use environment variables as fallback
    neo4j_uri = neo4j_uri or os.getenv('NEO4J_URI')
    neo4j_username = neo4j_username or os.getenv('NEO4J_USERNAME')
    neo4j_password = neo4j_password or os.getenv('NEO4J_PASSWORD')
    
    if not all([neo4j_uri, neo4j_username, neo4j_password]):
        raise ValueError("Neo4j connection parameters must be provided either as arguments or environment variables")
    
    return RiskAssessmentAgent(
        neo4j_uri=neo4j_uri,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password,
        neo4j_database=neo4j_database,
        llm_model=llm_model,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create agent
        agent = create_risk_assessment_agent()
        
        # Example assessment
        if len(sys.argv) > 1:
            facility_id = sys.argv[1]
            result = agent.assess_facility_risk(facility_id)
            
            print(f"Assessment Status: {result['status']}")
            print(f"Risk Level: {result.get('risk_assessment', {}).get('overall_risk_level', 'N/A')}")
            print(f"Processing Time: {result.get('processing_time', 0):.2f} seconds")
            
            if result.get('final_report'):
                print("\nExecutive Summary:")
                summary = result['final_report'].get('executive_summary', {})
                for key, value in summary.items():
                    print(f"  {key}: {value}")
        else:
            print("Risk Assessment Agent initialized successfully")
            print("Usage: python agent.py <facility_id>")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)