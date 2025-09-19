"""
Comprehensive Environmental Assessment Orchestrator using LangChain and LangGraph.

This module implements a sophisticated workflow orchestrator that coordinates multiple analysis stages:
1. Data Collection - Gathers electricity, water, and waste consumption data
2. Facts Analysis - Extracts key insights from each domain using specialized prompts
3. Risk Assessment - Evaluates environmental, compliance, and operational risks
4. Recommendations Generation - Creates actionable sustainability recommendations
5. Cross-Domain Correlation - Analyzes interdependencies between domains

The orchestrator uses LangGraph StateGraph for workflow management and integrates with
existing prompt templates and LLM configurations.
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, TypedDict, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field, validator
from neo4j import GraphDatabase, Transaction

# Import existing components
from src.llm import get_llm, get_llm_model_name
from src.llm.prompts.environmental_prompts import (
    EnvironmentalPromptTemplates, 
    AnalysisType, 
    OutputFormat, 
    PromptContext,
    environmental_prompts
)
from src.llm.prompts.risk_assessment_prompts import RiskAssessmentPromptTemplates
from src.llm.prompts.recommendation_prompts import RecommendationPromptTemplates

logger = logging.getLogger(__name__)


class AssessmentStatus(str, Enum):
    """Assessment processing status."""
    PENDING = "pending"
    COLLECTING_DATA = "collecting_data"
    ANALYZING_FACTS = "analyzing_facts"
    ASSESSING_RISKS = "assessing_risks"
    GENERATING_RECOMMENDATIONS = "generating_recommendations"
    CORRELATING_DOMAINS = "correlating_domains"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"


class AssessmentDomain(str, Enum):
    """Environmental assessment domains."""
    ELECTRICITY = "electricity"
    WATER = "water"
    WASTE = "waste"
    CROSS_DOMAIN = "cross_domain"


class ProcessingMode(str, Enum):
    """Processing mode for analysis stages."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    MIXED = "mixed"


# Pydantic models for structured outputs
class DomainAnalysis(BaseModel):
    """Analysis results for a specific environmental domain."""
    domain: str = Field(description="Environmental domain (electricity, water, waste)")
    total_consumption: Optional[float] = Field(description="Total consumption amount")
    consumption_unit: Optional[str] = Field(description="Unit of measurement")
    consumption_trend: Optional[str] = Field(description="Trend direction")
    key_findings: List[str] = Field(description="Key insights from analysis")
    efficiency_metrics: Dict[str, Any] = Field(default_factory=dict, description="Efficiency indicators")
    cost_analysis: Dict[str, Any] = Field(default_factory=dict, description="Cost-related insights")
    environmental_impact: Dict[str, Any] = Field(default_factory=dict, description="Environmental impact metrics")
    data_quality_score: float = Field(ge=0.0, le=1.0, description="Quality of source data")
    processing_time: Optional[float] = Field(description="Time taken for analysis")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")


class RiskFactor(BaseModel):
    """Individual risk factor identified in the analysis."""
    risk_id: str = Field(description="Unique identifier for the risk")
    risk_name: str = Field(description="Human-readable name")
    risk_category: str = Field(description="Category (operational, compliance, financial, environmental)")
    domain: str = Field(description="Associated domain")
    description: str = Field(description="Detailed description")
    potential_causes: List[str] = Field(description="Likely causes")
    potential_impacts: List[str] = Field(description="Potential consequences")
    probability_score: float = Field(ge=1.0, le=5.0, description="Probability score (1-5)")
    severity_score: float = Field(ge=1.0, le=5.0, description="Severity score (1-5)")
    overall_risk_score: float = Field(description="Overall risk score")
    current_controls: List[str] = Field(default_factory=list, description="Existing controls")
    control_effectiveness: str = Field(description="Effectiveness of current controls")


class Recommendation(BaseModel):
    """Environmental improvement recommendation."""
    recommendation_id: str = Field(description="Unique identifier")
    title: str = Field(description="Brief title")
    category: str = Field(description="Recommendation category")
    domain: str = Field(description="Primary domain affected")
    description: str = Field(description="Detailed description")
    implementation_steps: List[str] = Field(description="Step-by-step implementation guide")
    priority: str = Field(description="Priority level (critical, high, medium, low)")
    timeframe: str = Field(description="Implementation timeframe")
    estimated_cost: Optional[str] = Field(description="Cost estimate")
    estimated_savings: Optional[str] = Field(description="Expected savings")
    implementation_effort: str = Field(description="Required effort level")
    success_metrics: List[str] = Field(description="KPIs to measure success")
    risk_factors_addressed: List[str] = Field(description="Risk factors this addresses")


class CrossDomainCorrelation(BaseModel):
    """Cross-domain correlation analysis result."""
    domain_1: str = Field(description="First domain")
    domain_2: str = Field(description="Second domain")
    correlation_strength: str = Field(description="Correlation strength")
    correlation_direction: str = Field(description="Positive or negative correlation")
    statistical_significance: float = Field(description="Statistical significance")
    explanation: str = Field(description="Explanation of correlation")
    optimization_opportunities: List[str] = Field(description="Optimization opportunities")


class EnvironmentalAssessmentState(TypedDict):
    """State for environmental assessment workflow."""
    # Input parameters
    facility_id: str
    assessment_id: str
    assessment_scope: Dict[str, Any]  # Date ranges, specific domains, etc.
    processing_mode: str  # sequential, parallel, mixed
    output_format: str  # json, markdown, pdf
    
    # Data collection
    raw_data: Dict[str, Any]  # Raw data from Neo4j by domain
    data_quality_metrics: Dict[str, Any]  # Data quality assessment
    
    # Domain-specific analysis
    electricity_analysis: Optional[DomainAnalysis]
    water_analysis: Optional[DomainAnalysis] 
    waste_analysis: Optional[DomainAnalysis]
    
    # Risk assessment
    identified_risks: List[RiskFactor]
    overall_risk_rating: Optional[str]
    risk_assessment_summary: Dict[str, Any]
    
    # Recommendations
    recommendations: List[Recommendation]
    implementation_plan: Dict[str, Any]
    
    # Cross-domain analysis
    domain_correlations: List[CrossDomainCorrelation]
    integrated_insights: List[Dict[str, Any]]
    
    # Processing state
    status: str
    current_step: str
    errors: List[str]
    retry_count: int
    step_retry_count: int
    processing_time: Optional[float]
    
    # Configuration
    llm_model: str
    max_retries: int
    parallel_processing: bool
    
    # Output
    final_report: Optional[Dict[str, Any]]


class EnvironmentalAssessmentOrchestrator:
    """
    Comprehensive Environmental Assessment Orchestrator using LangChain and LangGraph.
    
    This orchestrator coordinates multiple analysis stages across electricity, water, and waste
    domains using specialized prompt templates and LLM models. It implements parallel processing
    for efficiency and provides comprehensive error handling with retry logic.
    """
    
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_username: str,
        neo4j_password: str,
        neo4j_database: str = "neo4j",
        llm_model: str = "gpt-4o",
        max_retries: int = 3,
        max_step_retries: int = 2,
        parallel_processing: bool = True,
        output_dir: str = "./assessments"
    ):
        """
        Initialize the Environmental Assessment Orchestrator.
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            neo4j_database: Neo4j database name
            llm_model: LLM model to use for analysis
            max_retries: Maximum retry attempts for failed operations
            max_step_retries: Maximum retries per individual step
            parallel_processing: Enable parallel processing for domain analysis
            output_dir: Directory for output reports
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.neo4j_database = neo4j_database
        self.llm_model = llm_model
        self.max_retries = max_retries
        self.max_step_retries = max_step_retries
        self.parallel_processing = parallel_processing
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize Neo4j connection
        try:
            self.driver = GraphDatabase.driver(
                neo4j_uri,
                auth=(neo4j_username, neo4j_password)
            )
            logger.info("Neo4j connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise
        
        # Initialize LLM
        try:
            self.llm, self.llm_model_name = get_llm(llm_model)
            logger.info(f"LLM initialized: {self.llm_model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise
        
        # Initialize prompt templates
        self.env_prompts = EnvironmentalPromptTemplates()
        self.risk_prompts = RiskAssessmentPromptTemplates()
        self.rec_prompts = RecommendationPromptTemplates()
        
        # Initialize LLM chains
        self._initialize_llm_chains()
        
        # Build workflow graph
        self.workflow = self._build_workflow()
        
        logger.info("Environmental Assessment Orchestrator initialized successfully")
    
    def _initialize_llm_chains(self):
        """Initialize LLM chains for different analysis phases."""
        
        # Domain analysis chains
        self.electricity_chain = self._create_domain_analysis_chain(AnalysisType.ELECTRICITY)
        self.water_chain = self._create_domain_analysis_chain(AnalysisType.WATER)
        self.waste_chain = self._create_domain_analysis_chain(AnalysisType.WASTE)
        
        # Cross-domain analysis chain
        self.cross_domain_chain = self._create_domain_analysis_chain(AnalysisType.CROSS_DOMAIN)
        
        # Risk assessment chain
        self.risk_assessment_chain = self._create_risk_assessment_chain()
        
        # Recommendations chain
        self.recommendations_chain = self._create_recommendations_chain()
        
        logger.info("LLM chains initialized successfully")
    
    def _create_domain_analysis_chain(self, analysis_type: AnalysisType):
        """Create LLM chain for domain-specific analysis."""
        
        def create_prompt(input_data: Dict[str, Any]) -> Dict[str, str]:
            context = PromptContext(
                facility_name=input_data.get('facility_id'),
                time_period=input_data.get('time_period'),
                data_types=[analysis_type.value]
            )
            
            return self.env_prompts.create_consumption_analysis_prompt(
                analysis_type=analysis_type,
                consumption_data=input_data.get('consumption_data', ''),
                context=context,
                output_format=OutputFormat.JSON
            )
        
        def process_response(response) -> Dict[str, Any]:
            try:
                if hasattr(response, 'content'):
                    content = response.content
                else:
                    content = str(response)
                
                # Attempt to parse JSON response
                parsed = json.loads(content)
                return parsed
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON response for {analysis_type.value}, returning raw content")
                return {"raw_response": content, "parsing_error": True}
            except Exception as e:
                logger.error(f"Error processing response for {analysis_type.value}: {str(e)}")
                return {"error": str(e), "raw_response": str(response)}
        
        # Create the chain
        prompt_creator = RunnableLambda(create_prompt)
        llm_caller = RunnableLambda(lambda prompts: self.llm.invoke([
            SystemMessage(content=prompts["system"]),
            HumanMessage(content=prompts["user"])
        ]))
        response_processor = RunnableLambda(process_response)
        
        return prompt_creator | llm_caller | response_processor
    
    def _create_risk_assessment_chain(self):
        """Create LLM chain for risk assessment."""
        
        def create_risk_prompt(input_data: Dict[str, Any]) -> Dict[str, str]:
            # Create risk assessment prompt using risk assessment templates
            system_prompt = self.risk_prompts.SYSTEM_PROMPTS["risk_assessment_expert"]
            
            user_prompt = f"""Perform a comprehensive environmental risk assessment based on the following analysis results:

ELECTRICITY ANALYSIS:
{json.dumps(input_data.get('electricity_analysis', {}), indent=2)}

WATER ANALYSIS: 
{json.dumps(input_data.get('water_analysis', {}), indent=2)}

WASTE ANALYSIS:
{json.dumps(input_data.get('waste_analysis', {}), indent=2)}

FACILITY CONTEXT:
- Facility ID: {input_data.get('facility_id', 'Unknown')}
- Assessment Scope: {json.dumps(input_data.get('assessment_scope', {}), indent=2)}

RISK ASSESSMENT REQUIREMENTS:
1. Identify specific risk factors across all environmental domains
2. Assess operational, compliance, financial, and environmental risks
3. Evaluate risk probability and severity on a 1-5 scale
4. Calculate overall risk scores and ratings
5. Assess effectiveness of current controls
6. Provide evidence-based risk rationale

OUTPUT FORMAT: Provide structured JSON response with:
- risk_summary: Overall risk assessment summary
- identified_risks: Array of risk factors with detailed analysis
- risk_scoring: Quantitative risk scores and ratings
- mitigation_priorities: Prioritized list of risks requiring immediate attention

Focus on actionable risk insights that support decision-making and risk mitigation planning."""

            return {"system": system_prompt, "user": user_prompt}
        
        def process_risk_response(response) -> Dict[str, Any]:
            try:
                content = response.content if hasattr(response, 'content') else str(response)
                parsed = json.loads(content)
                
                # Validate and structure risk factors
                risk_factors = []
                for risk_data in parsed.get('identified_risks', []):
                    try:
                        risk_factor = RiskFactor(**risk_data)
                        risk_factors.append(risk_factor)
                    except Exception as e:
                        logger.warning(f"Failed to validate risk factor: {str(e)}")
                        continue
                
                return {
                    "risk_summary": parsed.get('risk_summary', {}),
                    "risk_factors": risk_factors,
                    "risk_scoring": parsed.get('risk_scoring', []),
                    "mitigation_priorities": parsed.get('mitigation_priorities', []),
                    "raw_response": parsed
                }
                
            except json.JSONDecodeError:
                logger.error("Failed to parse risk assessment JSON response")
                return {"error": "JSON parsing failed", "raw_response": str(response)}
            except Exception as e:
                logger.error(f"Error processing risk response: {str(e)}")
                return {"error": str(e), "raw_response": str(response)}
        
        prompt_creator = RunnableLambda(create_risk_prompt)
        llm_caller = RunnableLambda(lambda prompts: self.llm.invoke([
            SystemMessage(content=prompts["system"]),
            HumanMessage(content=prompts["user"])
        ]))
        response_processor = RunnableLambda(process_risk_response)
        
        return prompt_creator | llm_caller | response_processor
    
    def _create_recommendations_chain(self):
        """Create LLM chain for recommendations generation."""
        
        def create_rec_prompt(input_data: Dict[str, Any]) -> Dict[str, str]:
            system_prompt = self.rec_prompts.SYSTEM_PROMPTS["sustainability_consultant"]
            
            user_prompt = f"""Generate comprehensive environmental sustainability recommendations based on the following assessment results:

DOMAIN ANALYSIS RESULTS:
Electricity: {json.dumps(input_data.get('electricity_analysis', {}), indent=2)}
Water: {json.dumps(input_data.get('water_analysis', {}), indent=2)}  
Waste: {json.dumps(input_data.get('waste_analysis', {}), indent=2)}

IDENTIFIED RISKS:
{json.dumps([risk.dict() if hasattr(risk, 'dict') else risk for risk in input_data.get('risk_factors', [])], indent=2)}

RISK ASSESSMENT SUMMARY:
{json.dumps(input_data.get('risk_assessment_summary', {}), indent=2)}

FACILITY CONTEXT:
- Facility ID: {input_data.get('facility_id', 'Unknown')}
- Assessment Scope: {json.dumps(input_data.get('assessment_scope', {}), indent=2)}

RECOMMENDATION REQUIREMENTS:
1. Generate specific, actionable recommendations for each domain
2. Address identified risk factors with targeted mitigation strategies
3. Prioritize recommendations by impact and implementation feasibility
4. Include cost-benefit analysis and ROI estimates where possible
5. Provide implementation timelines and effort estimates
6. Define success metrics and monitoring approaches
7. Consider cross-domain optimization opportunities

OUTPUT FORMAT: Provide structured JSON response with:
- executive_summary: High-level summary of recommendations
- quick_wins: Immediate actions with high impact/low effort
- strategic_recommendations: Long-term strategic initiatives
- implementation_plan: Phased implementation approach
- success_metrics: KPIs for measuring progress

Focus on practical, cost-effective recommendations that deliver measurable environmental and financial benefits."""

            return {"system": system_prompt, "user": user_prompt}
        
        def process_rec_response(response) -> Dict[str, Any]:
            try:
                content = response.content if hasattr(response, 'content') else str(response)
                parsed = json.loads(content)
                
                # Validate and structure recommendations
                recommendations = []
                for rec_data in parsed.get('strategic_recommendations', []):
                    try:
                        # Add required fields if missing
                        rec_data.setdefault('recommendation_id', f"rec_{len(recommendations)+1}")
                        rec_data.setdefault('risk_factors_addressed', [])
                        rec_data.setdefault('implementation_steps', [])
                        rec_data.setdefault('success_metrics', [])
                        
                        recommendation = Recommendation(**rec_data)
                        recommendations.append(recommendation)
                    except Exception as e:
                        logger.warning(f"Failed to validate recommendation: {str(e)}")
                        continue
                
                return {
                    "executive_summary": parsed.get('executive_summary', {}),
                    "quick_wins": parsed.get('quick_wins', []),
                    "recommendations": recommendations,
                    "implementation_plan": parsed.get('implementation_plan', {}),
                    "success_metrics": parsed.get('success_metrics', []),
                    "raw_response": parsed
                }
                
            except json.JSONDecodeError:
                logger.error("Failed to parse recommendations JSON response")
                return {"error": "JSON parsing failed", "raw_response": str(response)}
            except Exception as e:
                logger.error(f"Error processing recommendations response: {str(e)}")
                return {"error": str(e), "raw_response": str(response)}
        
        prompt_creator = RunnableLambda(create_rec_prompt)
        llm_caller = RunnableLambda(lambda prompts: self.llm.invoke([
            SystemMessage(content=prompts["system"]),
            HumanMessage(content=prompts["user"])
        ]))
        response_processor = RunnableLambda(process_rec_response)
        
        return prompt_creator | llm_caller | response_processor
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for environmental assessment."""
        
        # Create workflow
        workflow = StateGraph(EnvironmentalAssessmentState)
        
        # Add nodes
        workflow.add_node("initialize", self.initialize_assessment)
        workflow.add_node("collect_data", self.collect_environmental_data)
        workflow.add_node("analyze_electricity", self.analyze_electricity)
        workflow.add_node("analyze_water", self.analyze_water)
        workflow.add_node("analyze_waste", self.analyze_waste)
        workflow.add_node("analyze_parallel", self.analyze_domains_parallel)
        workflow.add_node("assess_risks", self.assess_environmental_risks)
        workflow.add_node("generate_recommendations", self.generate_recommendations)
        workflow.add_node("correlate_domains", self.correlate_domains)
        workflow.add_node("compile_report", self.compile_final_report)
        workflow.add_node("handle_error", self.handle_error)
        workflow.add_node("complete_assessment", self.complete_assessment)
        
        # Add edges - conditional based on processing mode
        workflow.add_edge("initialize", "collect_data")
        
        # Conditional edges for processing mode
        workflow.add_conditional_edges(
            "collect_data",
            self.check_processing_mode,
            {
                "sequential": "analyze_electricity",
                "parallel": "analyze_parallel",
                "error": "handle_error"
            }
        )
        
        # Sequential processing path
        workflow.add_edge("analyze_electricity", "analyze_water")
        workflow.add_edge("analyze_water", "analyze_waste")
        workflow.add_edge("analyze_waste", "assess_risks")
        
        # Parallel processing path
        workflow.add_edge("analyze_parallel", "assess_risks")
        
        # Common path after domain analysis
        workflow.add_conditional_edges(
            "assess_risks",
            self.check_risks_complete,
            {
                "continue": "generate_recommendations",
                "retry": "assess_risks",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "generate_recommendations",
            self.check_recommendations_complete,
            {
                "continue": "correlate_domains",
                "retry": "generate_recommendations", 
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("correlate_domains", "compile_report")
        workflow.add_edge("compile_report", "complete_assessment")
        
        # Error handling
        workflow.add_conditional_edges(
            "handle_error",
            self.check_retry_needed,
            {
                "retry": "collect_data",
                "fail": END
            }
        )
        
        workflow.add_edge("complete_assessment", END)
        
        # Set entry point
        workflow.set_entry_point("initialize")
        
        return workflow.compile()
    
    def initialize_assessment(self, state: EnvironmentalAssessmentState) -> EnvironmentalAssessmentState:
        """Initialize the environmental assessment workflow."""
        logger.info(f"Initializing environmental assessment for facility: {state['facility_id']}")
        
        # Set initial status
        state["status"] = AssessmentStatus.COLLECTING_DATA
        state["current_step"] = "initialization"
        state["errors"] = state.get("errors", [])
        state["retry_count"] = state.get("retry_count", 0)
        state["step_retry_count"] = 0
        
        # Initialize analysis results
        state["electricity_analysis"] = None
        state["water_analysis"] = None
        state["waste_analysis"] = None
        state["identified_risks"] = []
        state["recommendations"] = []
        state["domain_correlations"] = []
        state["integrated_insights"] = []
        
        # Set processing configuration
        state["parallel_processing"] = self.parallel_processing
        state["max_retries"] = self.max_retries
        state["llm_model"] = self.llm_model
        
        logger.info("Environmental assessment initialization completed")
        return state
    
    def collect_environmental_data(self, state: EnvironmentalAssessmentState) -> EnvironmentalAssessmentState:
        """Collect environmental data from Neo4j across all domains."""
        logger.info("Collecting environmental data from Neo4j")
        state["current_step"] = "data_collection"
        
        try:
            facility_id = state["facility_id"]
            assessment_scope = state.get("assessment_scope", {})
            
            # Get date range from scope
            start_date = assessment_scope.get("start_date")
            end_date = assessment_scope.get("end_date")
            
            raw_data = {}
            data_quality_metrics = {}
            
            with self.driver.session() as session:
                # Collect electricity data
                electricity_data = self._collect_electricity_data(session, facility_id, start_date, end_date)
                raw_data["electricity"] = electricity_data
                data_quality_metrics["electricity"] = self._assess_data_quality(electricity_data, "electricity")
                
                # Collect water data  
                water_data = self._collect_water_data(session, facility_id, start_date, end_date)
                raw_data["water"] = water_data
                data_quality_metrics["water"] = self._assess_data_quality(water_data, "water")
                
                # Collect waste data
                waste_data = self._collect_waste_data(session, facility_id, start_date, end_date)
                raw_data["waste"] = waste_data
                data_quality_metrics["waste"] = self._assess_data_quality(waste_data, "waste")
            
            state["raw_data"] = raw_data
            state["data_quality_metrics"] = data_quality_metrics
            
            logger.info(f"Data collection completed. Electricity: {len(electricity_data)} records, "
                       f"Water: {len(water_data)} records, Waste: {len(waste_data)} records")
            
        except Exception as e:
            error_msg = f"Data collection failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
        
        return state
    
    def _collect_electricity_data(self, session, facility_id: str, start_date: str = None, end_date: str = None) -> List[Dict]:
        """Collect electricity consumption data from Neo4j."""
        query = """
        MATCH (f:Facility {name: $facility_id})
        OPTIONAL MATCH (f)<-[:BILLED_TO]-(b:UtilityBill)
        WHERE ($start_date IS NULL OR b.billing_period_start >= date($start_date))
          AND ($end_date IS NULL OR b.billing_period_end <= date($end_date))
        OPTIONAL MATCH (b)-[:RESULTED_IN]->(e:Emission)
        RETURN f, collect(DISTINCT b) as utility_bills, collect(DISTINCT e) as emissions
        """
        
        result = session.run(query, {
            "facility_id": facility_id,
            "start_date": start_date,
            "end_date": end_date
        })
        
        records = []
        for record in result:
            facility = record["f"]
            bills = record["utility_bills"]
            emissions = record["emissions"]
            
            for bill in bills:
                if bill:  # Skip None values
                    bill_data = dict(bill)
                    bill_data["facility"] = dict(facility) if facility else {}
                    bill_data["associated_emissions"] = [dict(e) for e in emissions if e]
                    records.append(bill_data)
        
        return records
    
    def _collect_water_data(self, session, facility_id: str, start_date: str = None, end_date: str = None) -> List[Dict]:
        """Collect water consumption data from Neo4j."""
        query = """
        MATCH (f:Facility {name: $facility_id})
        OPTIONAL MATCH (f)<-[:BILLED_TO]-(w:WaterBill)
        WHERE ($start_date IS NULL OR w.billing_period_start >= date($start_date))
          AND ($end_date IS NULL OR w.billing_period_end <= date($end_date))
        OPTIONAL MATCH (w)-[:PROVIDED_BY]->(p:UtilityProvider)
        OPTIONAL MATCH (w)-[:RESULTED_IN]->(e:Emission)
        OPTIONAL MATCH (m:Meter)-[:RECORDED_IN]->(w)
        RETURN f, collect(DISTINCT w) as water_bills, 
               collect(DISTINCT p) as providers,
               collect(DISTINCT e) as emissions,
               collect(DISTINCT m) as meters
        """
        
        result = session.run(query, {
            "facility_id": facility_id,
            "start_date": start_date,
            "end_date": end_date
        })
        
        records = []
        for record in result:
            facility = record["f"]
            bills = record["water_bills"]
            providers = record["providers"]
            emissions = record["emissions"]
            meters = record["meters"]
            
            for bill in bills:
                if bill:  # Skip None values
                    bill_data = dict(bill)
                    bill_data["facility"] = dict(facility) if facility else {}
                    bill_data["providers"] = [dict(p) for p in providers if p]
                    bill_data["associated_emissions"] = [dict(e) for e in emissions if e]
                    bill_data["meters"] = [dict(m) for m in meters if m]
                    records.append(bill_data)
        
        return records
    
    def _collect_waste_data(self, session, facility_id: str, start_date: str = None, end_date: str = None) -> List[Dict]:
        """Collect waste generation data from Neo4j."""
        query = """
        MATCH (f:Facility {name: $facility_id})
        OPTIONAL MATCH (f)-[:BELONGS_TO*]-(d:Document)-[:TRACKS]->(wm:WasteManifest)
        OPTIONAL MATCH (wm)-[:DOCUMENTS]->(ws:WasteShipment)
        WHERE ($start_date IS NULL OR ws.shipment_date >= date($start_date))
          AND ($end_date IS NULL OR ws.shipment_date <= date($end_date))
        OPTIONAL MATCH (ws)-[:GENERATED_BY]->(g:WasteGenerator)
        OPTIONAL MATCH (ws)-[:TRANSPORTED_BY]->(t:Transporter)
        OPTIONAL MATCH (ws)-[:DISPOSED_AT]->(df:DisposalFacility)
        OPTIONAL MATCH (ws)-[:CONTAINS_WASTE]->(wi:WasteItem)
        OPTIONAL MATCH (ws)-[:RESULTED_IN]->(e:Emission)
        RETURN f, collect(DISTINCT wm) as manifests,
               collect(DISTINCT ws) as shipments,
               collect(DISTINCT g) as generators,
               collect(DISTINCT t) as transporters,
               collect(DISTINCT df) as disposal_facilities,
               collect(DISTINCT wi) as waste_items,
               collect(DISTINCT e) as emissions
        """
        
        result = session.run(query, {
            "facility_id": facility_id,
            "start_date": start_date,
            "end_date": end_date
        })
        
        records = []
        for record in result:
            facility = record["f"]
            manifests = record["manifests"]
            shipments = record["shipments"]
            generators = record["generators"]
            transporters = record["transporters"]
            disposal_facilities = record["disposal_facilities"]
            waste_items = record["waste_items"]
            emissions = record["emissions"]
            
            for manifest in manifests:
                if manifest:  # Skip None values
                    manifest_data = dict(manifest)
                    manifest_data["facility"] = dict(facility) if facility else {}
                    manifest_data["shipments"] = [dict(s) for s in shipments if s]
                    manifest_data["generators"] = [dict(g) for g in generators if g]
                    manifest_data["transporters"] = [dict(t) for t in transporters if t]
                    manifest_data["disposal_facilities"] = [dict(df) for df in disposal_facilities if df]
                    manifest_data["waste_items"] = [dict(wi) for wi in waste_items if wi]
                    manifest_data["associated_emissions"] = [dict(e) for e in emissions if e]
                    records.append(manifest_data)
        
        return records
    
    def _assess_data_quality(self, data: List[Dict], domain: str) -> Dict[str, Any]:
        """Assess quality of collected data."""
        if not data:
            return {
                "quality_score": 0.0,
                "completeness": 0.0,
                "record_count": 0,
                "issues": ["No data available"],
                "recommendations": [f"Verify {domain} data collection and sources"]
            }
        
        total_records = len(data)
        issues = []
        completeness_score = 1.0
        
        # Check for missing key fields based on domain
        key_fields = {
            "electricity": ["total_kwh", "total_cost", "billing_period_start", "billing_period_end"],
            "water": ["total_gallons", "total_cost", "billing_period_start", "billing_period_end"], 
            "waste": ["shipment_date", "waste_items"]
        }
        
        domain_fields = key_fields.get(domain, [])
        missing_field_count = 0
        
        for record in data:
            for field in domain_fields:
                if not record.get(field):
                    missing_field_count += 1
        
        if missing_field_count > 0:
            missing_percentage = (missing_field_count / (total_records * len(domain_fields))) * 100
            completeness_score -= missing_percentage / 100
            issues.append(f"Missing {missing_percentage:.1f}% of key field values")
        
        # Additional quality checks
        if total_records < 3:
            issues.append("Limited data points may affect analysis quality")
            completeness_score *= 0.8
        
        quality_score = max(0.0, min(1.0, completeness_score))
        
        return {
            "quality_score": quality_score,
            "completeness": completeness_score,
            "record_count": total_records,
            "issues": issues,
            "recommendations": [f"Improve {domain} data collection" if issues else f"{domain} data quality is adequate"]
        }
    
    def check_processing_mode(self, state: EnvironmentalAssessmentState) -> str:
        """Determine processing mode based on configuration and data availability."""
        processing_mode = state.get("processing_mode", "parallel")
        
        # Check if we have data for analysis
        raw_data = state.get("raw_data", {})
        if not any(raw_data.values()):
            logger.error("No environmental data available for analysis")
            return "error"
        
        # Check for any data collection errors
        if state.get("errors"):
            logger.warning("Errors detected during data collection")
            return "error"
        
        if processing_mode == "parallel" and self.parallel_processing:
            logger.info("Using parallel processing for domain analysis")
            return "parallel"
        else:
            logger.info("Using sequential processing for domain analysis")
            return "sequential"
    
    def analyze_electricity(self, state: EnvironmentalAssessmentState) -> EnvironmentalAssessmentState:
        """Analyze electricity consumption data."""
        logger.info("Analyzing electricity consumption data")
        state["current_step"] = "electricity_analysis"
        
        try:
            raw_data = state.get("raw_data", {})
            electricity_data = raw_data.get("electricity", [])
            
            if not electricity_data:
                logger.warning("No electricity data available for analysis")
                state["electricity_analysis"] = None
                return state
            
            # Prepare analysis input
            analysis_input = {
                "facility_id": state["facility_id"],
                "time_period": state.get("assessment_scope", {}).get("time_period", "assessment period"),
                "consumption_data": json.dumps(electricity_data, default=str, indent=2)
            }
            
            # Run electricity analysis chain
            start_time = datetime.utcnow()
            analysis_result = self.electricity_chain.invoke(analysis_input)
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create structured analysis result
            electricity_analysis = DomainAnalysis(
                domain="electricity",
                total_consumption=analysis_result.get("analysis_summary", {}).get("total_consumption"),
                consumption_unit=analysis_result.get("analysis_summary", {}).get("consumption_unit", "kWh"),
                consumption_trend=analysis_result.get("analysis_summary", {}).get("consumption_trend"),
                key_findings=analysis_result.get("analysis_summary", {}).get("key_findings", []),
                efficiency_metrics=analysis_result.get("efficiency_metrics", {}),
                cost_analysis=analysis_result.get("cost_analysis", {}),
                environmental_impact=analysis_result.get("environmental_impact", {}),
                data_quality_score=state.get("data_quality_metrics", {}).get("electricity", {}).get("quality_score", 1.0),
                processing_time=processing_time
            )
            
            state["electricity_analysis"] = electricity_analysis
            logger.info(f"Electricity analysis completed in {processing_time:.2f} seconds")
            
        except Exception as e:
            error_msg = f"Electricity analysis failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            state["electricity_analysis"] = None
        
        return state
    
    def analyze_water(self, state: EnvironmentalAssessmentState) -> EnvironmentalAssessmentState:
        """Analyze water consumption data."""
        logger.info("Analyzing water consumption data")
        state["current_step"] = "water_analysis"
        
        try:
            raw_data = state.get("raw_data", {})
            water_data = raw_data.get("water", [])
            
            if not water_data:
                logger.warning("No water data available for analysis")
                state["water_analysis"] = None
                return state
            
            # Prepare analysis input
            analysis_input = {
                "facility_id": state["facility_id"],
                "time_period": state.get("assessment_scope", {}).get("time_period", "assessment period"),
                "consumption_data": json.dumps(water_data, default=str, indent=2)
            }
            
            # Run water analysis chain
            start_time = datetime.utcnow()
            analysis_result = self.water_chain.invoke(analysis_input)
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create structured analysis result
            water_analysis = DomainAnalysis(
                domain="water",
                total_consumption=analysis_result.get("analysis_summary", {}).get("total_consumption"),
                consumption_unit=analysis_result.get("analysis_summary", {}).get("consumption_unit", "gallons"),
                consumption_trend=analysis_result.get("analysis_summary", {}).get("consumption_trend"),
                key_findings=analysis_result.get("analysis_summary", {}).get("key_findings", []),
                efficiency_metrics=analysis_result.get("efficiency_metrics", {}),
                cost_analysis=analysis_result.get("cost_analysis", {}),
                environmental_impact=analysis_result.get("environmental_impact", {}),
                data_quality_score=state.get("data_quality_metrics", {}).get("water", {}).get("quality_score", 1.0),
                processing_time=processing_time
            )
            
            state["water_analysis"] = water_analysis
            logger.info(f"Water analysis completed in {processing_time:.2f} seconds")
            
        except Exception as e:
            error_msg = f"Water analysis failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            state["water_analysis"] = None
        
        return state
    
    def analyze_waste(self, state: EnvironmentalAssessmentState) -> EnvironmentalAssessmentState:
        """Analyze waste generation data."""
        logger.info("Analyzing waste generation data")
        state["current_step"] = "waste_analysis"
        
        try:
            raw_data = state.get("raw_data", {})
            waste_data = raw_data.get("waste", [])
            
            if not waste_data:
                logger.warning("No waste data available for analysis")
                state["waste_analysis"] = None
                return state
            
            # Prepare analysis input
            analysis_input = {
                "facility_id": state["facility_id"],
                "time_period": state.get("assessment_scope", {}).get("time_period", "assessment period"),
                "consumption_data": json.dumps(waste_data, default=str, indent=2)
            }
            
            # Run waste analysis chain
            start_time = datetime.utcnow()
            analysis_result = self.waste_chain.invoke(analysis_input)
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create structured analysis result
            waste_analysis = DomainAnalysis(
                domain="waste",
                total_consumption=analysis_result.get("analysis_summary", {}).get("total_consumption"),
                consumption_unit=analysis_result.get("analysis_summary", {}).get("consumption_unit", "tons"),
                consumption_trend=analysis_result.get("analysis_summary", {}).get("consumption_trend"),
                key_findings=analysis_result.get("analysis_summary", {}).get("key_findings", []),
                efficiency_metrics=analysis_result.get("efficiency_metrics", {}),
                cost_analysis=analysis_result.get("cost_analysis", {}),
                environmental_impact=analysis_result.get("environmental_impact", {}),
                data_quality_score=state.get("data_quality_metrics", {}).get("waste", {}).get("quality_score", 1.0),
                processing_time=processing_time
            )
            
            state["waste_analysis"] = waste_analysis
            logger.info(f"Waste analysis completed in {processing_time:.2f} seconds")
            
        except Exception as e:
            error_msg = f"Waste analysis failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            state["waste_analysis"] = None
        
        return state
    
    def analyze_domains_parallel(self, state: EnvironmentalAssessmentState) -> EnvironmentalAssessmentState:
        """Analyze all environmental domains in parallel for improved performance."""
        logger.info("Starting parallel domain analysis")
        state["current_step"] = "parallel_domain_analysis"
        
        try:
            raw_data = state.get("raw_data", {})
            
            # Prepare analysis functions
            analysis_tasks = []
            
            if raw_data.get("electricity"):
                analysis_tasks.append(("electricity", self.electricity_chain, raw_data["electricity"]))
            
            if raw_data.get("water"):
                analysis_tasks.append(("water", self.water_chain, raw_data["water"]))
            
            if raw_data.get("waste"):
                analysis_tasks.append(("waste", self.waste_chain, raw_data["waste"]))
            
            if not analysis_tasks:
                logger.warning("No data available for parallel analysis")
                return state
            
            # Execute parallel analysis
            start_time = datetime.utcnow()
            results = {}
            
            with ThreadPoolExecutor(max_workers=min(3, len(analysis_tasks))) as executor:
                # Submit all analysis tasks
                future_to_domain = {}
                
                for domain, chain, data in analysis_tasks:
                    analysis_input = {
                        "facility_id": state["facility_id"],
                        "time_period": state.get("assessment_scope", {}).get("time_period", "assessment period"),
                        "consumption_data": json.dumps(data, default=str, indent=2)
                    }
                    
                    future = executor.submit(chain.invoke, analysis_input)
                    future_to_domain[future] = domain
                
                # Collect results
                for future in as_completed(future_to_domain):
                    domain = future_to_domain[future]
                    try:
                        result = future.result()
                        results[domain] = result
                        logger.info(f"Completed {domain} analysis")
                    except Exception as e:
                        error_msg = f"Parallel {domain} analysis failed: {str(e)}"
                        state["errors"].append(error_msg)
                        logger.error(error_msg)
                        results[domain] = None
            
            total_processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Convert results to structured format
            data_quality_metrics = state.get("data_quality_metrics", {})
            
            # Process electricity results
            if "electricity" in results and results["electricity"]:
                electricity_result = results["electricity"]
                state["electricity_analysis"] = DomainAnalysis(
                    domain="electricity",
                    total_consumption=electricity_result.get("analysis_summary", {}).get("total_consumption"),
                    consumption_unit=electricity_result.get("analysis_summary", {}).get("consumption_unit", "kWh"),
                    consumption_trend=electricity_result.get("analysis_summary", {}).get("consumption_trend"),
                    key_findings=electricity_result.get("analysis_summary", {}).get("key_findings", []),
                    efficiency_metrics=electricity_result.get("efficiency_metrics", {}),
                    cost_analysis=electricity_result.get("cost_analysis", {}),
                    environmental_impact=electricity_result.get("environmental_impact", {}),
                    data_quality_score=data_quality_metrics.get("electricity", {}).get("quality_score", 1.0),
                    processing_time=total_processing_time / len(analysis_tasks)  # Approximate per-domain time
                )
            
            # Process water results
            if "water" in results and results["water"]:
                water_result = results["water"]
                state["water_analysis"] = DomainAnalysis(
                    domain="water",
                    total_consumption=water_result.get("analysis_summary", {}).get("total_consumption"),
                    consumption_unit=water_result.get("analysis_summary", {}).get("consumption_unit", "gallons"),
                    consumption_trend=water_result.get("analysis_summary", {}).get("consumption_trend"),
                    key_findings=water_result.get("analysis_summary", {}).get("key_findings", []),
                    efficiency_metrics=water_result.get("efficiency_metrics", {}),
                    cost_analysis=water_result.get("cost_analysis", {}),
                    environmental_impact=water_result.get("environmental_impact", {}),
                    data_quality_score=data_quality_metrics.get("water", {}).get("quality_score", 1.0),
                    processing_time=total_processing_time / len(analysis_tasks)
                )
            
            # Process waste results
            if "waste" in results and results["waste"]:
                waste_result = results["waste"]
                state["waste_analysis"] = DomainAnalysis(
                    domain="waste",
                    total_consumption=waste_result.get("analysis_summary", {}).get("total_consumption"),
                    consumption_unit=waste_result.get("analysis_summary", {}).get("consumption_unit", "tons"),
                    consumption_trend=waste_result.get("analysis_summary", {}).get("consumption_trend"),
                    key_findings=waste_result.get("analysis_summary", {}).get("key_findings", []),
                    efficiency_metrics=waste_result.get("efficiency_metrics", {}),
                    cost_analysis=waste_result.get("cost_analysis", {}),
                    environmental_impact=waste_result.get("environmental_impact", {}),
                    data_quality_score=data_quality_metrics.get("waste", {}).get("quality_score", 1.0),
                    processing_time=total_processing_time / len(analysis_tasks)
                )
            
            logger.info(f"Parallel domain analysis completed in {total_processing_time:.2f} seconds")
            
        except Exception as e:
            error_msg = f"Parallel domain analysis failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
        
        return state
    
    def assess_environmental_risks(self, state: EnvironmentalAssessmentState) -> EnvironmentalAssessmentState:
        """Assess environmental risks based on domain analysis results."""
        logger.info("Assessing environmental risks")
        state["current_step"] = "risk_assessment"
        state["status"] = AssessmentStatus.ASSESSING_RISKS
        
        try:
            # Prepare risk assessment input
            risk_input = {
                "facility_id": state["facility_id"],
                "assessment_scope": state.get("assessment_scope", {}),
                "electricity_analysis": state.get("electricity_analysis").dict() if state.get("electricity_analysis") else {},
                "water_analysis": state.get("water_analysis").dict() if state.get("water_analysis") else {},
                "waste_analysis": state.get("waste_analysis").dict() if state.get("waste_analysis") else {}
            }
            
            # Run risk assessment chain
            start_time = datetime.utcnow()
            risk_result = self.risk_assessment_chain.invoke(risk_input)
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Extract risk factors and summary
            risk_factors = risk_result.get("risk_factors", [])
            risk_summary = risk_result.get("risk_summary", {})
            
            state["identified_risks"] = risk_factors
            state["risk_assessment_summary"] = risk_summary
            state["overall_risk_rating"] = risk_summary.get("overall_risk_rating", "unknown")
            
            logger.info(f"Risk assessment completed in {processing_time:.2f} seconds. "
                       f"Identified {len(risk_factors)} risks. Overall rating: {state['overall_risk_rating']}")
            
        except Exception as e:
            error_msg = f"Risk assessment failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            state["identified_risks"] = []
            state["risk_assessment_summary"] = {}
        
        return state
    
    def generate_recommendations(self, state: EnvironmentalAssessmentState) -> EnvironmentalAssessmentState:
        """Generate environmental sustainability recommendations."""
        logger.info("Generating environmental recommendations")
        state["current_step"] = "recommendation_generation"
        state["status"] = AssessmentStatus.GENERATING_RECOMMENDATIONS
        
        try:
            # Prepare recommendations input
            rec_input = {
                "facility_id": state["facility_id"],
                "assessment_scope": state.get("assessment_scope", {}),
                "electricity_analysis": state.get("electricity_analysis").dict() if state.get("electricity_analysis") else {},
                "water_analysis": state.get("water_analysis").dict() if state.get("water_analysis") else {},
                "waste_analysis": state.get("waste_analysis").dict() if state.get("waste_analysis") else {},
                "risk_factors": state.get("identified_risks", []),
                "risk_assessment_summary": state.get("risk_assessment_summary", {})
            }
            
            # Run recommendations chain
            start_time = datetime.utcnow()
            rec_result = self.recommendations_chain.invoke(rec_input)
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Extract recommendations and implementation plan
            recommendations = rec_result.get("recommendations", [])
            implementation_plan = rec_result.get("implementation_plan", {})
            
            state["recommendations"] = recommendations
            state["implementation_plan"] = implementation_plan
            
            logger.info(f"Recommendations generation completed in {processing_time:.2f} seconds. "
                       f"Generated {len(recommendations)} recommendations")
            
        except Exception as e:
            error_msg = f"Recommendations generation failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            state["recommendations"] = []
            state["implementation_plan"] = {}
        
        return state
    
    def correlate_domains(self, state: EnvironmentalAssessmentState) -> EnvironmentalAssessmentState:
        """Analyze cross-domain correlations and integrated insights."""
        logger.info("Analyzing cross-domain correlations")
        state["current_step"] = "cross_domain_correlation"
        state["status"] = AssessmentStatus.CORRELATING_DOMAINS
        
        try:
            # Prepare cross-domain analysis input
            analysis_results = {
                "electricity": state.get("electricity_analysis").dict() if state.get("electricity_analysis") else None,
                "water": state.get("water_analysis").dict() if state.get("water_analysis") else None,
                "waste": state.get("waste_analysis").dict() if state.get("waste_analysis") else None
            }
            
            # Filter out None values
            available_domains = {k: v for k, v in analysis_results.items() if v is not None}
            
            if len(available_domains) < 2:
                logger.warning("Insufficient domains for cross-correlation analysis")
                state["domain_correlations"] = []
                state["integrated_insights"] = []
                return state
            
            cross_domain_input = {
                "facility_id": state["facility_id"],
                "time_period": state.get("assessment_scope", {}).get("time_period", "assessment period"),
                "consumption_data": json.dumps(available_domains, default=str, indent=2)
            }
            
            # Run cross-domain analysis chain
            start_time = datetime.utcnow()
            correlation_result = self.cross_domain_chain.invoke(cross_domain_input)
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Extract correlations and insights
            correlations_data = correlation_result.get("correlation_analysis", [])
            insights_data = correlation_result.get("integrated_insights", [])
            
            # Convert to structured format
            domain_correlations = []
            for corr_data in correlations_data:
                try:
                    correlation = CrossDomainCorrelation(**corr_data)
                    domain_correlations.append(correlation)
                except Exception as e:
                    logger.warning(f"Failed to validate correlation data: {str(e)}")
                    continue
            
            state["domain_correlations"] = domain_correlations
            state["integrated_insights"] = insights_data
            
            logger.info(f"Cross-domain correlation analysis completed in {processing_time:.2f} seconds. "
                       f"Found {len(domain_correlations)} correlations and {len(insights_data)} insights")
            
        except Exception as e:
            error_msg = f"Cross-domain correlation analysis failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            state["domain_correlations"] = []
            state["integrated_insights"] = []
        
        return state
    
    def compile_final_report(self, state: EnvironmentalAssessmentState) -> EnvironmentalAssessmentState:
        """Compile comprehensive environmental assessment report."""
        logger.info("Compiling final environmental assessment report")
        state["current_step"] = "report_compilation"
        
        try:
            # Compile comprehensive report
            report = {
                "assessment_metadata": {
                    "assessment_id": state["assessment_id"],
                    "facility_id": state["facility_id"],
                    "assessment_date": datetime.utcnow().isoformat(),
                    "assessment_scope": state.get("assessment_scope", {}),
                    "processing_mode": state.get("processing_mode", "unknown"),
                    "llm_model": state.get("llm_model", "unknown"),
                    "total_processing_time": state.get("processing_time", 0)
                },
                
                "executive_summary": {
                    "overall_status": state.get("status", "unknown"),
                    "domains_analyzed": [
                        domain for domain in ["electricity", "water", "waste"]
                        if state.get(f"{domain}_analysis") is not None
                    ],
                    "total_risks_identified": len(state.get("identified_risks", [])),
                    "overall_risk_rating": state.get("overall_risk_rating", "unknown"),
                    "total_recommendations": len(state.get("recommendations", [])),
                    "key_findings": self._extract_key_findings(state),
                    "immediate_actions_required": self._identify_immediate_actions(state)
                },
                
                "domain_analysis": {
                    "electricity": state.get("electricity_analysis").dict() if state.get("electricity_analysis") else None,
                    "water": state.get("water_analysis").dict() if state.get("water_analysis") else None,
                    "waste": state.get("waste_analysis").dict() if state.get("waste_analysis") else None
                },
                
                "risk_assessment": {
                    "summary": state.get("risk_assessment_summary", {}),
                    "identified_risks": [
                        risk.dict() if hasattr(risk, 'dict') else risk 
                        for risk in state.get("identified_risks", [])
                    ],
                    "overall_risk_rating": state.get("overall_risk_rating")
                },
                
                "recommendations": {
                    "summary": state.get("implementation_plan", {}),
                    "recommendations": [
                        rec.dict() if hasattr(rec, 'dict') else rec 
                        for rec in state.get("recommendations", [])
                    ],
                    "quick_wins": self._identify_quick_wins(state),
                    "strategic_initiatives": self._identify_strategic_initiatives(state)
                },
                
                "cross_domain_analysis": {
                    "correlations": [
                        corr.dict() if hasattr(corr, 'dict') else corr 
                        for corr in state.get("domain_correlations", [])
                    ],
                    "integrated_insights": state.get("integrated_insights", []),
                    "optimization_opportunities": self._identify_optimization_opportunities(state)
                },
                
                "data_quality": state.get("data_quality_metrics", {}),
                
                "errors_and_warnings": {
                    "errors": state.get("errors", []),
                    "retry_count": state.get("retry_count", 0),
                    "processing_notes": [
                        f"Assessment completed with {len(state.get('errors', []))} errors",
                        f"Used {state.get('processing_mode', 'unknown')} processing mode",
                        f"LLM model: {state.get('llm_model', 'unknown')}"
                    ]
                }
            }
            
            state["final_report"] = report
            logger.info("Final report compilation completed successfully")
            
        except Exception as e:
            error_msg = f"Report compilation failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            state["final_report"] = {"error": "Report compilation failed", "details": str(e)}
        
        return state
    
    def _extract_key_findings(self, state: EnvironmentalAssessmentState) -> List[str]:
        """Extract key findings from all domain analyses."""
        key_findings = []
        
        for domain in ["electricity", "water", "waste"]:
            analysis = state.get(f"{domain}_analysis")
            if analysis and hasattr(analysis, 'key_findings'):
                for finding in analysis.key_findings:
                    key_findings.append(f"{domain.title()}: {finding}")
        
        return key_findings[:10]  # Limit to top 10 findings
    
    def _identify_immediate_actions(self, state: EnvironmentalAssessmentState) -> List[str]:
        """Identify recommendations that require immediate action."""
        immediate_actions = []
        
        recommendations = state.get("recommendations", [])
        for rec in recommendations:
            if hasattr(rec, 'priority') and rec.priority in ["critical", "high"]:
                if hasattr(rec, 'timeframe') and "immediate" in rec.timeframe.lower():
                    immediate_actions.append(rec.title if hasattr(rec, 'title') else str(rec))
        
        risks = state.get("identified_risks", [])
        for risk in risks:
            if hasattr(risk, 'overall_risk_score') and risk.overall_risk_score >= 15:  # High risk threshold
                immediate_actions.append(f"Address high-risk factor: {risk.risk_name if hasattr(risk, 'risk_name') else 'Unknown risk'}")
        
        return immediate_actions[:5]  # Limit to top 5 immediate actions
    
    def _identify_quick_wins(self, state: EnvironmentalAssessmentState) -> List[Dict[str, Any]]:
        """Identify quick win recommendations."""
        quick_wins = []
        
        recommendations = state.get("recommendations", [])
        for rec in recommendations:
            if hasattr(rec, 'implementation_effort') and rec.implementation_effort in ["minimal", "low"]:
                if hasattr(rec, 'priority') and rec.priority in ["high", "medium"]:
                    quick_wins.append({
                        "title": rec.title if hasattr(rec, 'title') else "Quick Win",
                        "description": rec.description if hasattr(rec, 'description') else "",
                        "effort": rec.implementation_effort if hasattr(rec, 'implementation_effort') else "low",
                        "timeframe": rec.timeframe if hasattr(rec, 'timeframe') else "short-term"
                    })
        
        return quick_wins[:5]  # Limit to top 5 quick wins
    
    def _identify_strategic_initiatives(self, state: EnvironmentalAssessmentState) -> List[Dict[str, Any]]:
        """Identify strategic long-term recommendations."""
        strategic_initiatives = []
        
        recommendations = state.get("recommendations", [])
        for rec in recommendations:
            if hasattr(rec, 'timeframe') and "long" in rec.timeframe.lower():
                if hasattr(rec, 'category') and rec.category in ["strategic_initiatives", "technology_integration"]:
                    strategic_initiatives.append({
                        "title": rec.title if hasattr(rec, 'title') else "Strategic Initiative",
                        "description": rec.description if hasattr(rec, 'description') else "",
                        "category": rec.category if hasattr(rec, 'category') else "strategic",
                        "estimated_impact": getattr(rec, 'estimated_savings', 'high impact')
                    })
        
        return strategic_initiatives[:5]  # Limit to top 5 strategic initiatives
    
    def _identify_optimization_opportunities(self, state: EnvironmentalAssessmentState) -> List[str]:
        """Identify cross-domain optimization opportunities."""
        opportunities = []
        
        correlations = state.get("domain_correlations", [])
        for corr in correlations:
            if hasattr(corr, 'optimization_opportunities'):
                opportunities.extend(corr.optimization_opportunities)
        
        insights = state.get("integrated_insights", [])
        for insight in insights:
            if isinstance(insight, dict) and insight.get("actionability") == "immediate":
                opportunities.append(insight.get("insight", ""))
        
        return opportunities[:5]  # Limit to top 5 opportunities
    
    def complete_assessment(self, state: EnvironmentalAssessmentState) -> EnvironmentalAssessmentState:
        """Complete the environmental assessment workflow."""
        logger.info("Completing environmental assessment")
        
        state["status"] = AssessmentStatus.COMPLETED
        state["current_step"] = "completed"
        state["processing_time"] = datetime.utcnow().timestamp() - state.get("start_time", datetime.utcnow().timestamp())
        
        logger.info(f"Environmental assessment completed in {state['processing_time']:.2f} seconds")
        logger.info(f"Final status: {state['status']}")
        
        return state
    
    def handle_error(self, state: EnvironmentalAssessmentState) -> EnvironmentalAssessmentState:
        """Handle errors during environmental assessment with retry logic."""
        state["retry_count"] += 1
        
        logger.error(f"Error in environmental assessment. Step: {state.get('current_step')}. "
                    f"Total retries: {state['retry_count']}")
        logger.error(f"Errors: {state['errors']}")
        
        if state["retry_count"] >= self.max_retries:
            state["status"] = AssessmentStatus.FAILED
            logger.error("Maximum retries exceeded. Environmental assessment failed.")
        else:
            state["status"] = AssessmentStatus.RETRY
            logger.info(f"Retrying assessment, attempt {state['retry_count']}/{self.max_retries}")
        
        return state
    
    # Conditional edge check methods
    def check_risks_complete(self, state: EnvironmentalAssessmentState) -> str:
        """Check if risk assessment completed successfully."""
        current_step = "assess_risks"
        step_retry_count = state.get("step_retry_count", 0)
        
        # Check for errors
        recent_errors = [error for error in state.get("errors", []) if "risk assessment" in error.lower()]
        if recent_errors:
            if step_retry_count < self.max_step_retries:
                state["step_retry_count"] = step_retry_count + 1
                logger.warning(f"Retrying {current_step}, attempt {step_retry_count + 1}/{self.max_step_retries}")
                return "retry"
            else:
                logger.error(f"Max step retries exceeded for {current_step}")
                return "error"
        
        # Check if we have risk assessment results
        risks = state.get("identified_risks", [])
        risk_summary = state.get("risk_assessment_summary", {})
        
        if not risks and not risk_summary:
            if step_retry_count < self.max_step_retries:
                state["step_retry_count"] = step_retry_count + 1
                logger.warning(f"No risk assessment results, retrying {current_step}")
                return "retry"
            else:
                logger.error(f"Max step retries exceeded for {current_step} - no results")
                return "error"
        
        # Reset step retry counter for next step
        state["step_retry_count"] = 0
        logger.info(f"{current_step} completed successfully with {len(risks)} risks identified")
        return "continue"
    
    def check_recommendations_complete(self, state: EnvironmentalAssessmentState) -> str:
        """Check if recommendations generation completed successfully."""
        current_step = "generate_recommendations"
        step_retry_count = state.get("step_retry_count", 0)
        
        # Check for errors
        recent_errors = [error for error in state.get("errors", []) if "recommendation" in error.lower()]
        if recent_errors:
            if step_retry_count < self.max_step_retries:
                state["step_retry_count"] = step_retry_count + 1
                logger.warning(f"Retrying {current_step}, attempt {step_retry_count + 1}/{self.max_step_retries}")
                return "retry"
            else:
                logger.error(f"Max step retries exceeded for {current_step}")
                return "error"
        
        # Check if we have recommendations
        recommendations = state.get("recommendations", [])
        
        if not recommendations:
            if step_retry_count < self.max_step_retries:
                state["step_retry_count"] = step_retry_count + 1
                logger.warning(f"No recommendations generated, retrying {current_step}")
                return "retry"
            else:
                logger.error(f"Max step retries exceeded for {current_step} - no recommendations")
                return "error"
        
        # Reset step retry counter for next step
        state["step_retry_count"] = 0
        logger.info(f"{current_step} completed successfully with {len(recommendations)} recommendations")
        return "continue"
    
    def check_retry_needed(self, state: EnvironmentalAssessmentState) -> str:
        """Check if retry is needed or assessment should fail."""
        if state["status"] == AssessmentStatus.RETRY and state["retry_count"] < self.max_retries:
            logger.info(f"Retrying assessment, attempt {state['retry_count']}/{self.max_retries}")
            # Clear step retry counter for fresh retry
            state["step_retry_count"] = 0
            return "retry"
        else:
            logger.error("Assessment failed - maximum retries exceeded")
            return "fail"
    
    def assess_facility_environment(
        self,
        facility_id: str,
        assessment_scope: Optional[Dict[str, Any]] = None,
        processing_mode: ProcessingMode = ProcessingMode.PARALLEL,
        output_format: str = "json"
    ) -> EnvironmentalAssessmentState:
        """
        Perform comprehensive environmental assessment for a facility.
        
        Args:
            facility_id: Unique facility identifier
            assessment_scope: Scope parameters (date ranges, domains, etc.)
            processing_mode: Processing mode (sequential, parallel, mixed)
            output_format: Output format (json, markdown, pdf)
            
        Returns:
            Final assessment state with comprehensive results
        """
        # Generate unique assessment ID
        assessment_id = f"env_assessment_{facility_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize state
        initial_state: EnvironmentalAssessmentState = {
            "facility_id": facility_id,
            "assessment_id": assessment_id,
            "assessment_scope": assessment_scope or {},
            "processing_mode": processing_mode.value,
            "output_format": output_format,
            
            "raw_data": {},
            "data_quality_metrics": {},
            
            "electricity_analysis": None,
            "water_analysis": None,
            "waste_analysis": None,
            
            "identified_risks": [],
            "overall_risk_rating": None,
            "risk_assessment_summary": {},
            
            "recommendations": [],
            "implementation_plan": {},
            
            "domain_correlations": [],
            "integrated_insights": [],
            
            "status": AssessmentStatus.PENDING,
            "current_step": "initialization",
            "errors": [],
            "retry_count": 0,
            "step_retry_count": 0,
            "processing_time": None,
            
            "llm_model": self.llm_model,
            "max_retries": self.max_retries,
            "parallel_processing": self.parallel_processing,
            
            "final_report": None,
            "start_time": datetime.utcnow().timestamp()
        }
        
        # Execute workflow
        try:
            logger.info(f"Starting environmental assessment for facility {facility_id}")
            final_state = self.workflow.invoke(initial_state)
            
            # Save report if requested
            if final_state.get("final_report") and output_format in ["json", "markdown"]:
                self._save_assessment_report(final_state, output_format)
            
            return final_state
            
        except Exception as e:
            logger.error(f"Environmental assessment workflow failed for {facility_id}: {str(e)}")
            error_state = initial_state.copy()
            error_state["status"] = AssessmentStatus.FAILED
            error_state["errors"] = [f"Workflow execution failed: {str(e)}"]
            error_state["processing_time"] = datetime.utcnow().timestamp() - error_state["start_time"]
            return error_state
    
    def _save_assessment_report(self, state: EnvironmentalAssessmentState, output_format: str):
        """Save the assessment report to file."""
        try:
            facility_id = state["facility_id"]
            assessment_id = state["assessment_id"]
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            if output_format == "json":
                file_name = f"environmental_assessment_{facility_id}_{timestamp}.json"
                file_path = self.output_dir / file_name
                
                with open(file_path, 'w') as f:
                    json.dump(state["final_report"], f, indent=2, default=str)
                
            elif output_format == "markdown":
                file_name = f"environmental_assessment_{facility_id}_{timestamp}.md"
                file_path = self.output_dir / file_name
                
                with open(file_path, 'w') as f:
                    self._write_markdown_report(f, state["final_report"])
            
            logger.info(f"Assessment report saved to: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save assessment report: {str(e)}")
    
    def _write_markdown_report(self, file, report: Dict[str, Any]):
        """Write assessment report in Markdown format."""
        file.write(f"# Environmental Assessment Report\n\n")
        
        metadata = report.get("assessment_metadata", {})
        file.write(f"**Facility:** {metadata.get('facility_id', 'Unknown')}\n")
        file.write(f"**Assessment ID:** {metadata.get('assessment_id', 'Unknown')}\n")
        file.write(f"**Date:** {metadata.get('assessment_date', 'Unknown')}\n")
        file.write(f"**Processing Time:** {metadata.get('total_processing_time', 0):.2f} seconds\n\n")
        
        # Executive Summary
        exec_summary = report.get("executive_summary", {})
        file.write("## Executive Summary\n\n")
        file.write(f"- **Overall Status:** {exec_summary.get('overall_status', 'Unknown')}\n")
        file.write(f"- **Domains Analyzed:** {', '.join(exec_summary.get('domains_analyzed', []))}\n")
        file.write(f"- **Risks Identified:** {exec_summary.get('total_risks_identified', 0)}\n")
        file.write(f"- **Overall Risk Rating:** {exec_summary.get('overall_risk_rating', 'Unknown')}\n")
        file.write(f"- **Recommendations:** {exec_summary.get('total_recommendations', 0)}\n\n")
        
        # Key Findings
        if exec_summary.get("key_findings"):
            file.write("### Key Findings\n\n")
            for finding in exec_summary["key_findings"]:
                file.write(f"- {finding}\n")
            file.write("\n")
        
        # Immediate Actions
        if exec_summary.get("immediate_actions_required"):
            file.write("### Immediate Actions Required\n\n")
            for action in exec_summary["immediate_actions_required"]:
                file.write(f"- {action}\n")
            file.write("\n")
        
        # Domain Analysis
        domain_analysis = report.get("domain_analysis", {})
        for domain in ["electricity", "water", "waste"]:
            analysis = domain_analysis.get(domain)
            if analysis:
                file.write(f"## {domain.title()} Analysis\n\n")
                file.write(f"- **Total Consumption:** {analysis.get('total_consumption', 'N/A')} {analysis.get('consumption_unit', '')}\n")
                file.write(f"- **Trend:** {analysis.get('consumption_trend', 'N/A')}\n")
                file.write(f"- **Data Quality Score:** {analysis.get('data_quality_score', 'N/A'):.2f}\n\n")
                
                key_findings = analysis.get("key_findings", [])
                if key_findings:
                    file.write("### Key Findings\n\n")
                    for finding in key_findings:
                        file.write(f"- {finding}\n")
                    file.write("\n")
        
        # Risk Assessment
        risk_assessment = report.get("risk_assessment", {})
        if risk_assessment.get("identified_risks"):
            file.write("## Risk Assessment\n\n")
            file.write(f"**Overall Risk Rating:** {risk_assessment.get('overall_risk_rating', 'Unknown')}\n\n")
            
            for risk in risk_assessment["identified_risks"][:5]:  # Top 5 risks
                file.write(f"### {risk.get('risk_name', 'Unknown Risk')}\n")
                file.write(f"- **Category:** {risk.get('risk_category', 'N/A')}\n")
                file.write(f"- **Domain:** {risk.get('domain', 'N/A')}\n")
                file.write(f"- **Risk Score:** {risk.get('overall_risk_score', 'N/A')}\n")
                file.write(f"- **Description:** {risk.get('description', 'N/A')}\n\n")
        
        # Recommendations
        recommendations = report.get("recommendations", {})
        if recommendations.get("recommendations"):
            file.write("## Recommendations\n\n")
            
            for rec in recommendations["recommendations"][:5]:  # Top 5 recommendations
                file.write(f"### {rec.get('title', 'Recommendation')}\n")
                file.write(f"- **Priority:** {rec.get('priority', 'N/A')}\n")
                file.write(f"- **Domain:** {rec.get('domain', 'N/A')}\n")
                file.write(f"- **Timeframe:** {rec.get('timeframe', 'N/A')}\n")
                file.write(f"- **Description:** {rec.get('description', 'N/A')}\n\n")
    
    def close(self):
        """Close connections and cleanup resources."""
        try:
            if hasattr(self, 'driver') and self.driver:
                self.driver.close()
                logger.info("Neo4j connection closed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


# Convenience functions for easy usage
def create_environmental_assessment_orchestrator(
    neo4j_uri: str = None,
    neo4j_username: str = None,
    neo4j_password: str = None,
    neo4j_database: str = "neo4j",
    llm_model: str = "gpt-4o",
    **kwargs
) -> EnvironmentalAssessmentOrchestrator:
    """
    Factory function to create an environmental assessment orchestrator.
    
    Args:
        neo4j_uri: Neo4j URI (falls back to NEO4J_URI env var)
        neo4j_username: Neo4j username (falls back to NEO4J_USERNAME env var)
        neo4j_password: Neo4j password (falls back to NEO4J_PASSWORD env var)
        neo4j_database: Neo4j database name
        llm_model: LLM model to use
        **kwargs: Additional arguments for the orchestrator
        
    Returns:
        Configured EnvironmentalAssessmentOrchestrator instance
    """
    # Use environment variables as fallback
    neo4j_uri = neo4j_uri or os.getenv('NEO4J_URI')
    neo4j_username = neo4j_username or os.getenv('NEO4J_USERNAME')
    neo4j_password = neo4j_password or os.getenv('NEO4J_PASSWORD')
    
    if not all([neo4j_uri, neo4j_username, neo4j_password]):
        raise ValueError("Neo4j connection parameters must be provided either as arguments or environment variables")
    
    return EnvironmentalAssessmentOrchestrator(
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
        # Create orchestrator
        orchestrator = create_environmental_assessment_orchestrator()
        
        # Example assessment
        if len(sys.argv) > 1:
            facility_id = sys.argv[1]
            
            assessment_scope = {
                "time_period": "last_30_days",
                "start_date": "2024-07-01",
                "end_date": "2024-08-31",
                "domains": ["electricity", "water", "waste"]
            }
            
            result = orchestrator.assess_facility_environment(
                facility_id=facility_id,
                assessment_scope=assessment_scope,
                processing_mode=ProcessingMode.PARALLEL
            )
            
            print(f"Assessment Status: {result['status']}")
            print(f"Overall Risk Rating: {result.get('overall_risk_rating', 'N/A')}")
            print(f"Recommendations: {len(result.get('recommendations', []))}")
            print(f"Processing Time: {result.get('processing_time', 0):.2f} seconds")
            
            if result.get('final_report'):
                exec_summary = result['final_report'].get('executive_summary', {})
                print(f"Domains Analyzed: {', '.join(exec_summary.get('domains_analyzed', []))}")
                print(f"Risks Identified: {exec_summary.get('total_risks_identified', 0)}")
                
        else:
            print("Environmental Assessment Orchestrator initialized successfully")
            print("Usage: python environmental_assessment_orchestrator.py <facility_id>")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        if 'orchestrator' in locals():
            orchestrator.close()