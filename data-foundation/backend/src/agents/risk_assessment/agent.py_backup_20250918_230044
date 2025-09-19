"""
Risk Assessment Agent for Environmental, Health & Safety (EHS) evaluation.

This agent performs comprehensive risk assessments using a workflow-based approach.
It integrates multiple data sources, employs AI-powered analysis, and generates
actionable recommendations for risk mitigation.

Key Features:
1. Multi-source data integration (environmental, health, safety, compliance)
2. AI-powered risk analysis using LangChain and OpenAI
3. Generates actionable recommendations and mitigation strategies
4. Configurable assessment methodologies and risk scoring
5. Comprehensive reporting and audit trail capabilities
6. Error handling and retry mechanisms for reliability
"""

import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

# Add the parent directory to sys.path to import from src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session

from config.database import get_db
from models.facility import Facility
from models.risk_assessment import RiskAssessmentResult
from services.data_collector import DataCollector
from utils.json_utils import safe_json_dumps

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskFactor(BaseModel):
    """Individual risk factor identified in the assessment."""
    
    factor_id: str = Field(description="Unique identifier for the risk factor")
    category: str = Field(description="Category (environmental, health, safety, compliance)")
    name: str = Field(description="Name/title of the risk factor")
    description: str = Field(description="Detailed description of the risk")
    likelihood: float = Field(description="Likelihood score (0-10)")
    impact: float = Field(description="Impact score (0-10)")
    risk_score: float = Field(description="Calculated risk score (likelihood * impact)")
    severity_level: str = Field(description="Risk severity level (Low, Medium, High, Critical)")
    data_sources: List[str] = Field(description="List of data sources that identified this risk")
    location: Optional[str] = Field(description="Specific location or area affected")
    affected_populations: List[str] = Field(default=[], description="Groups/populations affected")
    regulatory_concerns: List[str] = Field(default=[], description="Relevant regulations or compliance issues")
    
    @validator('likelihood', 'impact')
    def validate_scores(cls, v):
        if not 0 <= v <= 10:
            raise ValueError('Score must be between 0 and 10')
        return v
    
    @validator('severity_level')
    def validate_severity(cls, v):
        if v not in ['Low', 'Medium', 'High', 'Critical']:
            raise ValueError('Severity level must be Low, Medium, High, or Critical')
        return v


class RiskAssessment(BaseModel):
    """Overall risk assessment results."""
    
    assessment_id: str = Field(description="Unique assessment identifier")
    overall_risk_level: str = Field(description="Overall risk level (Low, Medium, High, Critical)")
    risk_score: float = Field(description="Overall risk score (0-100)")
    confidence_level: float = Field(description="Confidence in assessment (0-1)")
    methodology: str = Field(description="Assessment methodology used")
    assessment_date: str = Field(description="ISO format date of assessment")
    key_findings: List[str] = Field(description="Key findings summary")
    data_quality_score: float = Field(description="Quality score of input data (0-1)")
    limitations: List[str] = Field(default=[], description="Assessment limitations")
    
    @validator('overall_risk_level')
    def validate_risk_level(cls, v):
        if v not in ['Low', 'Medium', 'High', 'Critical']:
            raise ValueError('Risk level must be Low, Medium, High, or Critical')
        return v
    
    @validator('risk_score')
    def validate_risk_score(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Risk score must be between 0 and 100')
        return v
    
    @validator('confidence_level', 'data_quality_score')
    def validate_scores_0_to_1(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Score must be between 0 and 1')
        return v


class RiskRecommendation(BaseModel):
    """Individual recommendation for risk mitigation."""
    
    recommendation_id: str = Field(description="Unique recommendation identifier")
    category: str = Field(description="Category (preventive, corrective, monitoring)")
    title: str = Field(description="Brief title of the recommendation")
    description: str = Field(description="Detailed description of the recommendation")
    priority: str = Field(description="Priority level (Low, Medium, High, Critical)")
    estimated_cost: Optional[str] = Field(description="Estimated implementation cost range")
    timeline: str = Field(description="Recommended implementation timeline")
    responsible_party: str = Field(description="Who should implement this recommendation")
    success_metrics: List[str] = Field(description="How to measure success")
    related_risks: List[str] = Field(description="Risk factor IDs this addresses")
    implementation_steps: List[str] = Field(description="Step-by-step implementation guide")
    regulatory_compliance: List[str] = Field(default=[], description="Regulatory requirements addressed")
    
    @validator('priority')
    def validate_priority(cls, v):
        if v not in ['Low', 'Medium', 'High', 'Critical']:
            raise ValueError('Priority must be Low, Medium, High, or Critical')
        return v


class RecommendationSet(BaseModel):
    """Complete set of recommendations for risk mitigation."""
    
    assessment_id: str = Field(description="Related assessment identifier")
    recommendations: List[RiskRecommendation] = Field(description="List of recommendations")
    total_estimated_cost: Optional[str] = Field(description="Total estimated cost for all recommendations")
    implementation_phases: List[str] = Field(default=[], description="Suggested implementation phases")
    quick_wins: List[str] = Field(default=[], description="Quick win recommendation IDs")
    long_term_goals: List[str] = Field(default=[], description="Long-term recommendation IDs")


class FacilityInfo(BaseModel):
    """Facility information for context."""
    
    facility_id: str
    name: str
    location: str
    industry_type: str
    size: str
    operations: List[str]


class RiskAssessmentState(TypedDict):
    """State maintained throughout the risk assessment workflow."""
    
    # Assessment metadata
    assessment_id: str
    facility_id: str
    assessment_scope: Dict[str, Any]
    methodology: str
    
    # Data collection
    environmental_data: List[Dict[str, Any]]
    health_data: List[Dict[str, Any]]
    safety_data: List[Dict[str, Any]]
    compliance_data: List[Dict[str, Any]]
    facility_info: Dict[str, Any]
    
    # Analysis results
    analysis_results: Optional[Dict[str, Any]]
    risk_factors: List[RiskFactor]
    risk_assessment: Optional[RiskAssessment]
    recommendations: Optional[RecommendationSet]
    
    # Workflow control
    current_step: str
    step_retry_counts: Dict[str, int]
    errors: List[str]
    warnings: List[str]
    
    # Status tracking
    status: str
    progress: float
    start_time: str
    end_time: Optional[str]


class RiskAssessmentAgent:
    """
    AI-powered Risk Assessment Agent for Environmental, Health & Safety evaluation.
    
    This agent uses a workflow-based approach to conduct comprehensive risk assessments
    by integrating multiple data sources, performing AI-powered analysis, and generating
    actionable recommendations.
    """
    
    def __init__(
        self,
        methodology: str = "comprehensive_ehs",
        max_retries: int = 3,
        max_step_retries: int = 2,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-4-turbo-preview"
    ):
        """
        Initialize the Risk Assessment Agent.
        
        Args:
            methodology: Assessment methodology to use
            max_retries: Maximum number of workflow retries
            max_step_retries: Maximum retries for individual steps
            openai_api_key: OpenAI API key (uses env var if not provided)
            model_name: OpenAI model to use for analysis
        """
        self.methodology = methodology
        self.max_retries = max_retries
        self.max_step_retries = max_step_retries
        
        # Initialize OpenAI client
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required")
            
        self.llm = ChatOpenAI(
            api_key=api_key,
            model=model_name,
            temperature=0.1,
            max_tokens=4000
        )
        
        # Initialize data collector
        self.data_collector = DataCollector()
        
        # Initialize analysis chains
        self._setup_analysis_chains()
        
        # Build workflow
        self.workflow = self._build_workflow()
        
        logger.info(f"RiskAssessmentAgent initialized with methodology: {methodology}")
    
    def _setup_analysis_chains(self):
        """Set up LangChain chains for different analysis tasks."""
        
        # Data analysis chain
        data_analysis_prompt = ChatPromptTemplate.from_template(
            """You are an expert Environmental, Health & Safety (EHS) risk analyst.
            
            Analyze the following facility data and identify potential risk factors:
            
            Facility ID: {facility_id}
            Facility Information: {facility_info}
            Assessment Scope: {assessment_scope}
            
            Environmental Data: {environmental_data}
            Health Data: {health_data}
            Safety Data: {safety_data}
            Compliance Data: {compliance_data}
            
            Provide a comprehensive analysis identifying:
            1. Key risk patterns and trends
            2. Data quality issues or gaps
            3. Potential correlations between different data sources
            4. Areas requiring immediate attention
            5. Regulatory compliance concerns
            
            Focus on actionable insights that will inform risk factor identification."""
        )
        
        self.data_analysis_chain = data_analysis_prompt | self.llm | StrOutputParser()
        
        # Risk assessment chain
        risk_assessment_prompt = ChatPromptTemplate.from_template(
            """You are an expert EHS risk assessor. Based on the following analysis results
            and risk factors, provide a comprehensive risk assessment.
            
            Analysis Results: {analysis_results}
            Risk Factors: {risk_factors}
            Facility Context: {facility_context}
            
            {format_instructions}
            
            Provide a thorough risk assessment following the RiskAssessment schema.
            Consider:
            1. Overall risk level based on individual risk factors
            2. Cumulative and interactive effects of multiple risks
            3. Confidence level based on data quality and completeness
            4. Key findings that decision-makers need to know
            5. Limitations of the assessment
            
            Be precise with scoring and provide clear justification."""
        )
        
        risk_assessment_parser = PydanticOutputParser(pydantic_object=RiskAssessment)
        self.risk_assessment_chain = (
            risk_assessment_prompt.partial(
                format_instructions=risk_assessment_parser.get_format_instructions()
            ) | self.llm | risk_assessment_parser
        )
        
        # Recommendation generation chain  
        recommendation_prompt = ChatPromptTemplate.from_template(
            """You are an expert EHS risk management consultant.
            
            Based on the risk assessment results, generate comprehensive, actionable
            recommendations for risk mitigation and management.
            
            Facility ID: {facility_id}
            Risk Assessment: {risk_assessment}
            Risk Factors Summary: {risk_factors_summary}
            Facility Context: {facility_context}
            
            {format_instructions}
            
            Generate recommendations that are:
            1. Specific and actionable
            2. Prioritized by risk level and feasibility
            3. Cost-effective and practical to implement
            4. Compliant with relevant regulations
            5. Measurable with clear success metrics
            
            Include both immediate actions and long-term strategies.
            Consider resource constraints and implementation challenges."""
        )
        
        # Use a simpler approach for recommendations to handle parsing issues
        if hasattr(self, '_get_recommendation_format_instructions'):
            format_instructions = self._get_recommendation_format_instructions()
        else:
            format_instructions = """Return valid JSON matching RecommendationSet schema"""
            
        self.recommendation_chain = (
            recommendation_prompt.partial(format_instructions=format_instructions) 
            | self.llm 
            | StrOutputParser()
        )
    
    def _get_recommendation_format_instructions(self):
        """Get format instructions for recommendation generation."""
        try:
            recommendation_parser = PydanticOutputParser(pydantic_object=RecommendationSet)
            return recommendation_parser.get_format_instructions()
        except Exception as e:
            logger.warning(f"Could not create recommendation parser: {e}")
            return """Generate recommendations for the following risk assessment:
            
            Return a JSON object with:
            - assessment_id: string
            - recommendations: array of recommendation objects
            - Each recommendation should have: recommendation_id, category, title, description, priority, timeline, responsible_party, success_metrics, related_risks, implementation_steps
            
            Generate comprehensive, prioritized recommendations for risk mitigation."""
    
    def _build_workflow(self):
        """Build the LangGraph workflow for risk assessment."""
        
        # Create workflow graph
        workflow = StateGraph(RiskAssessmentState)
        
        # Add nodes for each step
        workflow.add_node("initialize_assessment", self.initialize_assessment)
        workflow.add_node("collect_facility_data", self.collect_facility_data)
        workflow.add_node("collect_environmental_data", self.collect_environmental_data)
        workflow.add_node("collect_health_data", self.collect_health_data)
        workflow.add_node("collect_safety_data", self.collect_safety_data)
        workflow.add_node("collect_compliance_data", self.collect_compliance_data)
        workflow.add_node("analyze_data", self.analyze_data)
        workflow.add_node("assess_risks", self.assess_risks)
        workflow.add_node("generate_recommendations", self.generate_recommendations)
        workflow.add_node("compile_report", self.compile_report)
        workflow.add_node("handle_error", self.handle_error)
        
        # Set entry point
        workflow.set_entry_point("initialize_assessment")
        
        # Add conditional edges for data collection flow
        workflow.add_conditional_edges(
            "initialize_assessment",
            self.should_continue_or_error,
            {
                "continue": "collect_facility_data",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "collect_facility_data",
            self.should_continue_or_retry,
            {
                "continue": "collect_environmental_data",
                "retry": "collect_facility_data",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "collect_environmental_data", 
            self.should_continue_or_retry,
            {
                "continue": "collect_health_data",
                "retry": "collect_environmental_data",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "collect_health_data",
            self.should_continue_or_retry,
            {
                "continue": "collect_safety_data", 
                "retry": "collect_health_data",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "collect_safety_data",
            self.should_continue_or_retry,
            {
                "continue": "collect_compliance_data",
                "retry": "collect_safety_data", 
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "collect_compliance_data",
            self.should_continue_or_retry,
            {
                "continue": "analyze_data",
                "retry": "collect_compliance_data",
                "error": "handle_error"
            }
        )
        
        # Analysis and assessment flow
        workflow.add_conditional_edges(
            "analyze_data",
            self.should_continue_or_retry,
            {
                "continue": "assess_risks",
                "retry": "analyze_data",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "assess_risks",
            self.should_continue_or_retry,
            {
                "continue": "generate_recommendations",
                "retry": "assess_risks",
                "error": "handle_error"
            }
        )
        
        # Conditional edges for recommendations
        workflow.add_conditional_edges(
            "generate_recommendations",
            self.should_continue_or_retry,
            {
                "continue": "compile_report",
                "retry": "generate_recommendations", 
                "error": "handle_error"
            }
        )
        
        # Final steps
        workflow.add_edge("compile_report", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    # Workflow node implementations
    
    def initialize_assessment(self, state: RiskAssessmentState) -> RiskAssessmentState:
        """
        Initialize the risk assessment workflow.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with initialization data
        """
        logger.info(f"Initializing risk assessment for facility {state['facility_id']}")
        
        state["current_step"] = "initialization"
        state["start_time"] = datetime.utcnow().isoformat()
        state["status"] = "running"
        state["progress"] = 0.0
        state["errors"] = []
        state["warnings"] = []
        state["step_retry_counts"] = {}
        
        # Validate required inputs
        if not state.get("facility_id"):
            state["errors"].append("facility_id is required")
        if not state.get("assessment_id"):
            state["errors"].append("assessment_id is required")
            
        logger.info("Risk assessment initialized")
        return state
    
    def collect_facility_data(self, state: RiskAssessmentState) -> RiskAssessmentState:
        """
        Collect basic facility information and context.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with facility data
        """
        logger.info("Collecting facility data")
        state["current_step"] = "facility_data_collection"
        
        try:
            # Get facility information from database
            facility_info = self.data_collector.get_facility_info(state["facility_id"])
            state["facility_info"] = facility_info
            state["progress"] = 10.0
            logger.info(f"Collected facility data for {facility_info.get('name', 'Unknown')}")
            
        except Exception as e:
            error_msg = f"Facility data collection failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            
        return state
    
    def collect_environmental_data(self, state: RiskAssessmentState) -> RiskAssessmentState:
        """
        Collect environmental monitoring and compliance data.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with environmental data
        """
        logger.info("Collecting environmental data")
        state["current_step"] = "environmental_data_collection"
        
        try:
            env_data = self.data_collector.get_environmental_data(
                state["facility_id"],
                state.get("assessment_scope", {})
            )
            state["environmental_data"] = env_data
            state["progress"] = 25.0
            logger.info(f"Collected {len(env_data)} environmental data records")
            
        except Exception as e:
            error_msg = f"Environmental data collection failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            
        return state
    
    def collect_health_data(self, state: RiskAssessmentState) -> RiskAssessmentState:
        """
        Collect occupational health and safety data.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with health data
        """
        logger.info("Collecting health data")
        state["current_step"] = "health_data_collection"
        
        try:
            health_data = self.data_collector.get_health_data(
                state["facility_id"],
                state.get("assessment_scope", {})
            )
            state["health_data"] = health_data
            state["progress"] = 40.0
            logger.info(f"Collected {len(health_data)} health data records")
            
        except Exception as e:
            error_msg = f"Health data collection failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            
        return state
    
    def collect_safety_data(self, state: RiskAssessmentState) -> RiskAssessmentState:
        """
        Collect safety incident and compliance data.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with safety data
        """
        logger.info("Collecting safety data")
        state["current_step"] = "safety_data_collection"
        
        try:
            safety_data = self.data_collector.get_safety_data(
                state["facility_id"],
                state.get("assessment_scope", {})
            )
            state["safety_data"] = safety_data
            state["progress"] = 55.0
            logger.info(f"Collected {len(safety_data)} safety data records")
            
        except Exception as e:
            error_msg = f"Safety data collection failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            
        return state
    
    def collect_compliance_data(self, state: RiskAssessmentState) -> RiskAssessmentState:
        """
        Collect regulatory compliance and audit data.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with compliance data
        """
        logger.info("Collecting compliance data")
        state["current_step"] = "compliance_data_collection"
        
        try:
            compliance_data = self.data_collector.get_compliance_data(
                state["facility_id"],
                state.get("assessment_scope", {})
            )
            state["compliance_data"] = compliance_data
            state["progress"] = 70.0
            logger.info(f"Collected {len(compliance_data)} compliance data records")
            
        except Exception as e:
            error_msg = f"Compliance data collection failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            
        return state
    
    def analyze_data(self, state: RiskAssessmentState) -> RiskAssessmentState:
        """
        Analyze collected data to identify risk patterns and factors.
        
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
            Updated state with risk assessment
        """
        logger.info("Performing formal risk assessment")
        state["current_step"] = "risk_assessment"
        
        try:
            # Prepare risk assessment input
            analysis_results = state.get("analysis_results", {})
            risk_factors = state.get("risk_factors", [])
            
            assessment_input = {
                "analysis_results": safe_json_dumps(analysis_results),
                "risk_factors": safe_json_dumps([rf.dict() if hasattr(rf, 'dict') else rf for rf in risk_factors]),
                "facility_context": safe_json_dumps(state.get("facility_info", {}))
            }
            
            # Run risk assessment chain
            risk_assessment = self.risk_assessment_chain.invoke(assessment_input)
            
            state["risk_assessment"] = risk_assessment
            state["progress"] = 85.0
            logger.info(f"Risk assessment completed with overall level: {risk_assessment.overall_risk_level}")
            
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
            Updated state with final report
        """
        logger.info("Compiling final risk assessment report")
        state["current_step"] = "report_compilation"
        
        try:
            # Get the recommendations and safely extract the count
            recommendations = state.get("recommendations")
            total_recommendations = 0
            
            if recommendations:
                if isinstance(recommendations, dict):
                    # Handle case where recommendations is a dict with a "recommendations" key
                    recommendation_list = recommendations.get("recommendations", [])
                    total_recommendations = len(recommendation_list)
                elif hasattr(recommendations, 'recommendations'):
                    # Handle case where recommendations is a Pydantic object
                    total_recommendations = len(recommendations.recommendations)
                elif isinstance(recommendations, list):
                    # Handle case where recommendations is directly a list
                    total_recommendations = len(recommendations)
            
            # Compile comprehensive report
            report = {
                "assessment_id": state["assessment_id"],
                "facility_id": state["facility_id"],
                "assessment_date": datetime.utcnow().isoformat(),
                "methodology": self.methodology,
                "status": "completed",
                "executive_summary": {
                    "overall_risk_level": state.get("risk_assessment", {}).get("overall_risk_level") if state.get("risk_assessment") else None,
                    "risk_score": state.get("risk_assessment", {}).get("risk_score") if state.get("risk_assessment") else None,
                    "total_risk_factors": len(state.get("risk_factors", [])),
                    "total_recommendations": total_recommendations
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
                "workflow_metadata": {
                    "start_time": state["start_time"],
                    "end_time": datetime.utcnow().isoformat(),
                    "total_steps": len([k for k in state["step_retry_counts"].keys()]),
                    "errors": state["errors"],
                    "warnings": state["warnings"]
                }
            }
            
            state["final_report"] = report
            state["status"] = "completed"
            state["progress"] = 100.0
            state["end_time"] = datetime.utcnow().isoformat()
            
            logger.info("Risk assessment report compilation completed successfully")
            
        except Exception as e:
            error_msg = f"Report compilation failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            
        return state
    
    def handle_error(self, state: RiskAssessmentState) -> RiskAssessmentState:
        """
        Handle workflow errors and cleanup.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with error handling
        """
        logger.error(f"Handling workflow error in step: {state.get('current_step', 'unknown')}")
        
        state["status"] = "failed"
        state["end_time"] = datetime.utcnow().isoformat()
        
        # Log all errors
        for error in state.get("errors", []):
            logger.error(f"Workflow error: {error}")
        
        # Create error report
        error_report = {
            "assessment_id": state.get("assessment_id", "unknown"),
            "facility_id": state.get("facility_id", "unknown"),
            "status": "failed",
            "errors": state.get("errors", []),
            "warnings": state.get("warnings", []),
            "failed_step": state.get("current_step", "unknown"),
            "completion_time": datetime.utcnow().isoformat()
        }
        
        state["error_report"] = error_report
        
        return state
    
    # Workflow control methods
    
    def should_continue_or_error(self, state: RiskAssessmentState) -> str:
        """
        Determine if workflow should continue or handle error.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next action: "continue" or "error"
        """
        if state.get("errors"):
            return "error"
        return "continue"
    
    def should_continue_or_retry(self, state: RiskAssessmentState) -> str:
        """
        Determine if workflow should continue, retry, or handle error.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next action: "continue", "retry", or "error"
        """
        current_step = state.get("current_step", "unknown")
        
        # Check for errors
        if state.get("errors"):
            # Check if we should retry this step
            step_retry_count = state["step_retry_counts"].get(current_step, 0)
            if step_retry_count < self.max_step_retries:
                state["step_retry_counts"][current_step] = step_retry_count + 1
                logger.warning(f"Retrying {current_step}, attempt {step_retry_count + 1}/{self.max_step_retries}")
                # Clear the last error for retry
                state["errors"] = state["errors"][:-1] if state["errors"] else []
                return "retry"
            else:
                logger.error(f"Max step retries exceeded for {current_step}")
                return "error"
        
        # Step-specific validation
        if current_step == "facility_data_collection":
            if not state.get("facility_info"):
                return "error"
                
        elif current_step == "risk_assessment":
            if not state.get("risk_assessment"):
                return "error"
                
        elif current_step == "recommendation_generation":
            # Check if we have recommendations
            recommendations = state.get("recommendations")
            if not recommendations:
                step_retry_count = state["step_retry_counts"].get(current_step, 0)
                if step_retry_count < self.max_step_retries:
                    state["step_retry_counts"][current_step] = step_retry_count + 1
                    logger.warning(f"No recommendations generated, retrying {current_step}, attempt {step_retry_count + 1}/{self.max_step_retries}")
                    return "retry"
                else:
                    logger.error(f"Max step retries exceeded for {current_step} - no recommendations generated")
                    return "error"
        
        return "continue"
    
    # Public API methods
    
    def run_assessment(
        self,
        facility_id: str,
        assessment_scope: Optional[Dict[str, Any]] = None,
        assessment_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a complete risk assessment for a facility.
        
        Args:
            facility_id: Unique facility identifier
            assessment_scope: Scope and parameters for the assessment
            assessment_id: Unique assessment identifier (generated if not provided)
            
        Returns:
            Complete assessment results or error report
        """
        # Generate assessment ID if not provided
        if not assessment_id:
            assessment_id = f"ASSESS_{facility_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize state
        initial_state: RiskAssessmentState = {
            "assessment_id": assessment_id,
            "facility_id": facility_id,
            "assessment_scope": assessment_scope or {},
            "methodology": self.methodology,
            
            # Data placeholders
            "environmental_data": [],
            "health_data": [],
            "safety_data": [], 
            "compliance_data": [],
            "facility_info": {},
            
            # Analysis placeholders
            "analysis_results": None,
            "risk_factors": [],
            "risk_assessment": None,
            "recommendations": None,
            
            # Workflow control
            "current_step": "",
            "step_retry_counts": {},
            "errors": [],
            "warnings": [],
            
            # Status
            "status": "initialized",
            "progress": 0.0,
            "start_time": "",
            "end_time": None
        }
        
        logger.info(f"Starting risk assessment {assessment_id} for facility {facility_id}")
        
        try:
            # Run the workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Return appropriate result
            if final_state.get("status") == "completed":
                logger.info(f"Risk assessment {assessment_id} completed successfully")
                return final_state.get("final_report", final_state)
            else:
                logger.error(f"Risk assessment {assessment_id} failed")
                return final_state.get("error_report", final_state)
                
        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            logger.error(error_msg)
            return {
                "assessment_id": assessment_id,
                "facility_id": facility_id,
                "status": "failed",
                "error": error_msg,
                "completion_time": datetime.utcnow().isoformat()
            }
    
    def get_assessment_status(self, assessment_id: str) -> Dict[str, Any]:
        """
        Get the current status of an ongoing assessment.
        
        Args:
            assessment_id: Assessment identifier
            
        Returns:
            Assessment status information
        """
        # In a production system, this would query a database or cache
        # For now, return a placeholder
        return {
            "assessment_id": assessment_id,
            "status": "not_implemented",
            "message": "Status tracking not yet implemented"
        }
    
    def save_assessment_results(self, results: Dict[str, Any], db_session: Session) -> bool:
        """
        Save assessment results to the database.
        
        Args:
            results: Assessment results to save
            db_session: Database session
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Create database record
            assessment_record = RiskAssessmentResult(
                assessment_id=results["assessment_id"],
                facility_id=results["facility_id"],
                methodology=results.get("methodology", self.methodology),
                status=results["status"],
                results_data=results,  # Store full results as JSON
                created_at=datetime.utcnow()
            )
            
            db_session.add(assessment_record)
            db_session.commit()
            
            logger.info(f"Assessment results saved to database: {results['assessment_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save assessment results: {str(e)}")
            db_session.rollback()
            return False


# Example usage and testing functions
def create_test_assessment_agent() -> RiskAssessmentAgent:
    """Create a test instance of the Risk Assessment Agent."""
    return RiskAssessmentAgent(
        methodology="test_comprehensive_ehs",
        max_retries=2,
        max_step_retries=1
    )


def run_sample_assessment():
    """Run a sample risk assessment for testing."""
    try:
        agent = create_test_assessment_agent()
        
        # Sample assessment parameters
        facility_id = "FACILITY_001"
        assessment_scope = {
            "include_environmental": True,
            "include_health": True,
            "include_safety": True,
            "include_compliance": True,
            "time_range_days": 365
        }
        
        # Run assessment
        results = agent.run_assessment(
            facility_id=facility_id,
            assessment_scope=assessment_scope
        )
        
        print(f"Assessment completed with status: {results.get('status', 'unknown')}")
        print(f"Assessment ID: {results.get('assessment_id', 'unknown')}")
        
        if results.get("status") == "completed":
            summary = results.get("executive_summary", {})
            print(f"Overall Risk Level: {summary.get('overall_risk_level', 'unknown')}")
            print(f"Risk Score: {summary.get('risk_score', 'unknown')}")
            print(f"Total Recommendations: {summary.get('total_recommendations', 0)}")
        else:
            print(f"Errors: {results.get('errors', [])}")
            
        return results
        
    except Exception as e:
        print(f"Sample assessment failed: {str(e)}")
        return None


if __name__ == "__main__":
    # Run sample assessment if called directly
    print("Running sample risk assessment...")
    results = run_sample_assessment()
    
    if results:
        print("\nAssessment completed. Full results available in the returned dictionary.")
    else:
        print("\nAssessment failed. Check logs for details.")