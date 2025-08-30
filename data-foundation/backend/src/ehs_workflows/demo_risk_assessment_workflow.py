"""
Demonstration workflow for EHS Risk Assessment Agent integration.
This workflow showcases how to use the risk assessment agent for various scenarios
including facility assessment, document-triggered assessment, and scheduled assessments.
"""

import os
import logging
import time
import json
from typing import Dict, List, Any, Optional, TypedDict
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Risk Assessment Agent imports
from agents.risk_assessment.agent import (
    RiskAssessmentAgent,
    RiskAssessmentState,
    RiskLevel,
    RiskCategory,
    AssessmentStatus
)

# Database imports
from shared.common_fn import create_graph_database_connection

logger = logging.getLogger(__name__)


@dataclass
class DemoConfiguration:
    """Configuration for demonstration scenarios."""
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str = "neo4j"
    llm_model: str = "gpt-4"
    enable_langsmith: bool = True
    output_dir: str = "./demo_outputs"
    verbose: bool = True


class DemoScenario(str, Enum):
    """Available demonstration scenarios."""
    FACILITY_RISK_ASSESSMENT = "facility_risk_assessment"
    DOCUMENT_TRIGGERED_ASSESSMENT = "document_triggered_assessment"
    SCHEDULED_ASSESSMENT = "scheduled_assessment"
    BATCH_FACILITY_ASSESSMENT = "batch_facility_assessment"
    COMPARATIVE_RISK_ANALYSIS = "comparative_risk_analysis"
    CUSTOM_SCENARIO = "custom_scenario"


class DemoState(TypedDict):
    """State for demonstration workflow."""
    scenario: str
    configuration: Dict[str, Any]
    scenario_params: Dict[str, Any]
    
    # Execution state
    current_step: str
    assessments_completed: List[Dict[str, Any]]
    risk_results: List[Dict[str, Any]]
    analysis_results: Dict[str, Any]
    
    # Output
    demo_report: Dict[str, Any]
    report_file_path: Optional[str]
    
    # Error handling
    errors: List[str]
    status: str
    processing_time: Optional[float]


class EHSRiskAssessmentDemo:
    """
    Demonstration workflow for EHS Risk Assessment Agent.
    
    This workflow provides various demonstration scenarios to showcase
    the capabilities of the risk assessment system including:
    - Single facility assessments
    - Document-triggered assessments  
    - Scheduled bulk assessments
    - Comparative risk analysis
    - Custom assessment scenarios
    """
    
    def __init__(self, config: DemoConfiguration):
        """
        Initialize the demonstration workflow.
        
        Args:
            config: Demonstration configuration
        """
        self.config = config
        
        # Initialize Risk Assessment Agent
        try:
            self.risk_agent = RiskAssessmentAgent(
                neo4j_uri=config.neo4j_uri,
                neo4j_username=config.neo4j_username,
                neo4j_password=config.neo4j_password,
                neo4j_database=config.neo4j_database,
                llm_model=config.llm_model,
                enable_langsmith=config.enable_langsmith
            )
            logger.info("Risk Assessment Agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Risk Assessment Agent: {str(e)}")
            raise
        
        # Initialize database connection
        try:
            self.graph = create_graph_database_connection(
                config.neo4j_uri,
                config.neo4j_username, 
                config.neo4j_password,
                config.neo4j_database
            )
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Build workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the demonstration workflow graph."""
        workflow = StateGraph(DemoState)
        
        # Add nodes
        workflow.add_node("setup_scenario", self.setup_scenario)
        workflow.add_node("execute_facility_assessment", self.execute_facility_assessment)
        workflow.add_node("execute_document_triggered", self.execute_document_triggered)
        workflow.add_node("execute_scheduled_assessment", self.execute_scheduled_assessment)
        workflow.add_node("execute_batch_assessment", self.execute_batch_assessment)
        workflow.add_node("execute_comparative_analysis", self.execute_comparative_analysis)
        workflow.add_node("execute_custom_scenario", self.execute_custom_scenario)
        workflow.add_node("analyze_results", self.analyze_results)
        workflow.add_node("generate_report", self.generate_report)
        workflow.add_node("complete", self.complete)
        workflow.add_node("handle_error", self.handle_error)
        
        # Add conditional routing directly from setup_scenario
        workflow.add_conditional_edges(
            "setup_scenario",
            self.route_scenario,
            {
                DemoScenario.FACILITY_RISK_ASSESSMENT: "execute_facility_assessment",
                DemoScenario.DOCUMENT_TRIGGERED_ASSESSMENT: "execute_document_triggered",
                DemoScenario.SCHEDULED_ASSESSMENT: "execute_scheduled_assessment",
                DemoScenario.BATCH_FACILITY_ASSESSMENT: "execute_batch_assessment",
                DemoScenario.COMPARATIVE_RISK_ANALYSIS: "execute_comparative_analysis",
                DemoScenario.CUSTOM_SCENARIO: "execute_custom_scenario",
                "error": "handle_error"
            }
        )
        
        # All scenario executions lead to analysis
        workflow.add_edge("execute_facility_assessment", "analyze_results")
        workflow.add_edge("execute_document_triggered", "analyze_results")
        workflow.add_edge("execute_scheduled_assessment", "analyze_results")
        workflow.add_edge("execute_batch_assessment", "analyze_results")
        workflow.add_edge("execute_comparative_analysis", "analyze_results")
        workflow.add_edge("execute_custom_scenario", "analyze_results")
        
        workflow.add_edge("analyze_results", "generate_report")
        workflow.add_edge("generate_report", "complete")
        workflow.add_edge("complete", END)
        
        # Error handling
        workflow.add_edge("handle_error", END)
        
        # Set entry point
        workflow.set_entry_point("setup_scenario")
        
        return workflow.compile()
    
    def setup_scenario(self, state: DemoState) -> DemoState:
        """
        Setup the demonstration scenario based on configuration.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        logger.info(f"Setting up demonstration scenario: {state['scenario']}")
        
        try:
            state["current_step"] = "setup"
            state["assessments_completed"] = []
            state["risk_results"] = []
            state["analysis_results"] = {}
            state["errors"] = []
            state["status"] = "running"
            
            # Validate scenario configuration
            scenario = state["scenario"]
            params = state.get("scenario_params", {})
            
            if scenario == DemoScenario.FACILITY_RISK_ASSESSMENT:
                # Validate facility ID is provided
                if not params.get("facility_id"):
                    raise ValueError("facility_id is required for facility risk assessment")
            
            elif scenario == DemoScenario.DOCUMENT_TRIGGERED_ASSESSMENT:
                # Validate document ID is provided
                if not params.get("document_id"):
                    raise ValueError("document_id is required for document-triggered assessment")
            
            elif scenario == DemoScenario.BATCH_FACILITY_ASSESSMENT:
                # Validate facility list or query is provided
                if not params.get("facility_ids") and not params.get("facility_query"):
                    raise ValueError("facility_ids list or facility_query is required for batch assessment")
            
            elif scenario == DemoScenario.COMPARATIVE_RISK_ANALYSIS:
                # Validate comparison parameters
                if not params.get("comparison_criteria"):
                    raise ValueError("comparison_criteria is required for comparative analysis")
            
            # Log scenario setup
            if self.config.verbose:
                logger.info(f"Scenario parameters: {params}")
                
            logger.info("Scenario setup completed successfully")
            
        except Exception as e:
            state["errors"].append(f"Setup error: {str(e)}")
            logger.error(f"Failed to setup scenario: {str(e)}")
        
        return state
    
    def route_scenario(self, state: DemoState) -> str:
        """Route to appropriate scenario execution."""
        if state.get("errors"):
            return "error"
        
        scenario = state["scenario"]
        
        # Map scenarios to execution nodes
        scenario_mapping = {
            DemoScenario.FACILITY_RISK_ASSESSMENT: DemoScenario.FACILITY_RISK_ASSESSMENT,
            DemoScenario.DOCUMENT_TRIGGERED_ASSESSMENT: DemoScenario.DOCUMENT_TRIGGERED_ASSESSMENT,
            DemoScenario.SCHEDULED_ASSESSMENT: DemoScenario.SCHEDULED_ASSESSMENT,
            DemoScenario.BATCH_FACILITY_ASSESSMENT: DemoScenario.BATCH_FACILITY_ASSESSMENT,
            DemoScenario.COMPARATIVE_RISK_ANALYSIS: DemoScenario.COMPARATIVE_RISK_ANALYSIS,
            DemoScenario.CUSTOM_SCENARIO: DemoScenario.CUSTOM_SCENARIO
        }
        
        return scenario_mapping.get(scenario, "error")
    
    def execute_facility_assessment(self, state: DemoState) -> DemoState:
        """
        Execute single facility risk assessment demonstration.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with assessment results
        """
        logger.info("Executing facility risk assessment demonstration")
        
        try:
            state["current_step"] = "facility_assessment"
            params = state["scenario_params"]
            facility_id = params["facility_id"]
            
            # Configure assessment scope
            assessment_scope = {
                "risk_categories": params.get("risk_categories", [
                    RiskCategory.ENVIRONMENTAL,
                    RiskCategory.HEALTH_SAFETY,
                    RiskCategory.REGULATORY_COMPLIANCE
                ]),
                "assessment_depth": params.get("assessment_depth", "comprehensive"),
                "include_historical_data": params.get("include_historical_data", True),
                "time_horizon": params.get("time_horizon", "1_year"),
                "assessment_trigger": "demonstration"
            }
            
            # Add facility context if provided
            if params.get("facility_context"):
                assessment_scope.update(params["facility_context"])
            
            # Execute assessment
            logger.info(f"Starting risk assessment for facility: {facility_id}")
            start_time = time.time()
            
            assessment_results = self.risk_agent.assess_facility_risk(
                facility_id=facility_id,
                assessment_scope=assessment_scope,
                metadata={
                    "demo_scenario": "facility_assessment",
                    "demo_timestamp": datetime.utcnow().isoformat()
                }
            )
            
            processing_time = time.time() - start_time
            
            # Record results
            assessment_record = {
                "facility_id": facility_id,
                "assessment_type": "single_facility",
                "processing_time": processing_time,
                "results": assessment_results,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            state["assessments_completed"].append(assessment_record)
            state["risk_results"].append(assessment_results)
            
            if self.config.verbose:
                logger.info(f"Assessment completed in {processing_time:.2f} seconds")
                logger.info(f"Risk level: {assessment_results.get('risk_assessment', {}).get('overall_risk_level', 'Unknown')}")
            
        except Exception as e:
            state["errors"].append(f"Facility assessment error: {str(e)}")
            logger.error(f"Failed to execute facility assessment: {str(e)}")
        
        return state
    
    def execute_document_triggered(self, state: DemoState) -> DemoState:
        """
        Execute document-triggered risk assessment demonstration.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with assessment results
        """
        logger.info("Executing document-triggered risk assessment demonstration")
        
        try:
            state["current_step"] = "document_triggered_assessment"
            params = state["scenario_params"]
            document_id = params["document_id"]
            
            # First, identify the facility associated with the document
            facility_query = """
            MATCH (d:Document {id: $document_id})
            OPTIONAL MATCH (d)-[*1..3]-(f:Facility)
            RETURN DISTINCT f.id as facility_id, f.name as facility_name
            LIMIT 1
            """
            
            result = self.graph.query(facility_query, {"document_id": document_id})
            
            if not result:
                raise ValueError(f"No facility found associated with document: {document_id}")
            
            facility_id = result[0]["facility_id"]
            facility_name = result[0]["facility_name"]
            
            logger.info(f"Document {document_id} associated with facility: {facility_name} ({facility_id})")
            
            # Configure assessment scope with document context
            assessment_scope = {
                "risk_categories": params.get("risk_categories", [
                    RiskCategory.ENVIRONMENTAL,
                    RiskCategory.REGULATORY_COMPLIANCE
                ]),
                "assessment_depth": "focused",
                "document_trigger": {
                    "document_id": document_id,
                    "trigger_reason": params.get("trigger_reason", "document_processed"),
                    "document_type": params.get("document_type", "unknown")
                },
                "focus_areas": params.get("focus_areas", ["document_compliance", "operational_changes"]),
                "assessment_trigger": "document_processing"
            }
            
            # Execute assessment
            logger.info(f"Starting document-triggered assessment for facility: {facility_id}")
            start_time = time.time()
            
            assessment_results = self.risk_agent.assess_facility_risk(
                facility_id=facility_id,
                assessment_scope=assessment_scope,
                metadata={
                    "demo_scenario": "document_triggered",
                    "trigger_document_id": document_id,
                    "demo_timestamp": datetime.utcnow().isoformat()
                }
            )
            
            processing_time = time.time() - start_time
            
            # Record results
            assessment_record = {
                "facility_id": facility_id,
                "facility_name": facility_name,
                "trigger_document_id": document_id,
                "assessment_type": "document_triggered",
                "processing_time": processing_time,
                "results": assessment_results,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            state["assessments_completed"].append(assessment_record)
            state["risk_results"].append(assessment_results)
            
            if self.config.verbose:
                logger.info(f"Document-triggered assessment completed in {processing_time:.2f} seconds")
            
        except Exception as e:
            state["errors"].append(f"Document-triggered assessment error: {str(e)}")
            logger.error(f"Failed to execute document-triggered assessment: {str(e)}")
        
        return state
    
    def execute_scheduled_assessment(self, state: DemoState) -> DemoState:
        """
        Execute scheduled risk assessment demonstration.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with assessment results
        """
        logger.info("Executing scheduled risk assessment demonstration")
        
        try:
            state["current_step"] = "scheduled_assessment"
            params = state["scenario_params"]
            
            # Get facilities for scheduled assessment
            schedule_config = params.get("schedule_config", {})
            assessment_frequency = schedule_config.get("frequency", "monthly")
            
            # Query facilities that need scheduled assessment
            facilities_query = """
            MATCH (f:Facility)
            OPTIONAL MATCH (f)-[:HAS_RISK_ASSESSMENT]->(ra:RiskAssessment)
            WHERE ra IS NULL OR ra.assessment_date < datetime() - duration('P30D')
            RETURN f.id as facility_id, f.name as facility_name, 
                   max(ra.assessment_date) as last_assessment
            ORDER BY last_assessment ASC NULLS FIRST
            LIMIT $limit
            """
            
            limit = params.get("max_facilities", 3)  # Limit for demo purposes
            facilities_result = self.graph.query(facilities_query, {"limit": limit})
            
            if not facilities_result:
                logger.warning("No facilities found requiring scheduled assessment")
                return state
            
            logger.info(f"Found {len(facilities_result)} facilities for scheduled assessment")
            
            # Execute assessments for each facility
            for facility in facilities_result:
                facility_id = facility["facility_id"]
                facility_name = facility["facility_name"]
                
                try:
                    # Configure assessment scope for scheduled assessment
                    assessment_scope = {
                        "risk_categories": [
                            RiskCategory.ENVIRONMENTAL,
                            RiskCategory.HEALTH_SAFETY,
                            RiskCategory.REGULATORY_COMPLIANCE,
                            RiskCategory.OPERATIONAL
                        ],
                        "assessment_depth": "standard",
                        "assessment_trigger": "scheduled",
                        "schedule_info": {
                            "frequency": assessment_frequency,
                            "last_assessment": facility.get("last_assessment"),
                            "assessment_type": "routine_scheduled"
                        }
                    }
                    
                    logger.info(f"Executing scheduled assessment for: {facility_name}")
                    start_time = time.time()
                    
                    assessment_results = self.risk_agent.assess_facility_risk(
                        facility_id=facility_id,
                        assessment_scope=assessment_scope,
                        metadata={
                            "demo_scenario": "scheduled_assessment",
                            "schedule_frequency": assessment_frequency,
                            "demo_timestamp": datetime.utcnow().isoformat()
                        }
                    )
                    
                    processing_time = time.time() - start_time
                    
                    # Record results
                    assessment_record = {
                        "facility_id": facility_id,
                        "facility_name": facility_name,
                        "assessment_type": "scheduled",
                        "schedule_frequency": assessment_frequency,
                        "processing_time": processing_time,
                        "results": assessment_results,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    state["assessments_completed"].append(assessment_record)
                    state["risk_results"].append(assessment_results)
                    
                    if self.config.verbose:
                        logger.info(f"Scheduled assessment for {facility_name} completed in {processing_time:.2f} seconds")
                
                except Exception as e:
                    error_msg = f"Scheduled assessment failed for {facility_name}: {str(e)}"
                    state["errors"].append(error_msg)
                    logger.error(error_msg)
            
            logger.info(f"Scheduled assessment demonstration completed for {len(state['assessments_completed'])} facilities")
            
        except Exception as e:
            state["errors"].append(f"Scheduled assessment error: {str(e)}")
            logger.error(f"Failed to execute scheduled assessment: {str(e)}")
        
        return state
    
    def execute_batch_assessment(self, state: DemoState) -> DemoState:
        """
        Execute batch facility assessment demonstration.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with assessment results
        """
        logger.info("Executing batch facility assessment demonstration")
        
        try:
            state["current_step"] = "batch_assessment"
            params = state["scenario_params"]
            
            # Get facilities for batch assessment
            if params.get("facility_ids"):
                # Use provided facility IDs
                facility_ids = params["facility_ids"]
                
                # Get facility names
                facilities_query = """
                MATCH (f:Facility)
                WHERE f.id IN $facility_ids
                RETURN f.id as facility_id, f.name as facility_name
                """
                
                facilities_result = self.graph.query(facilities_query, {"facility_ids": facility_ids})
                
            elif params.get("facility_query"):
                # Use custom query to select facilities
                facilities_result = self.graph.query(params["facility_query"])
                
            else:
                # Default: get first few facilities
                facilities_query = """
                MATCH (f:Facility)
                RETURN f.id as facility_id, f.name as facility_name
                LIMIT 5
                """
                
                facilities_result = self.graph.query(facilities_query)
            
            if not facilities_result:
                raise ValueError("No facilities found for batch assessment")
            
            logger.info(f"Starting batch assessment for {len(facilities_result)} facilities")
            
            # Configure batch assessment parameters
            batch_config = {
                "assessment_depth": params.get("assessment_depth", "standard"),
                "parallel_processing": params.get("parallel_processing", False),
                "risk_categories": params.get("risk_categories", [
                    RiskCategory.ENVIRONMENTAL,
                    RiskCategory.HEALTH_SAFETY,
                    RiskCategory.REGULATORY_COMPLIANCE
                ]),
                "comparative_analysis": params.get("comparative_analysis", True)
            }
            
            batch_start_time = time.time()
            
            # Execute assessments
            for i, facility in enumerate(facilities_result):
                facility_id = facility["facility_id"]
                facility_name = facility["facility_name"]
                
                try:
                    # Configure assessment scope
                    assessment_scope = {
                        "risk_categories": batch_config["risk_categories"],
                        "assessment_depth": batch_config["assessment_depth"],
                        "assessment_trigger": "batch_processing",
                        "batch_info": {
                            "batch_size": len(facilities_result),
                            "facility_index": i + 1,
                            "batch_id": f"demo_batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                        }
                    }
                    
                    logger.info(f"Assessing facility {i+1}/{len(facilities_result)}: {facility_name}")
                    start_time = time.time()
                    
                    assessment_results = self.risk_agent.assess_facility_risk(
                        facility_id=facility_id,
                        assessment_scope=assessment_scope,
                        metadata={
                            "demo_scenario": "batch_assessment",
                            "batch_index": i + 1,
                            "batch_size": len(facilities_result),
                            "demo_timestamp": datetime.utcnow().isoformat()
                        }
                    )
                    
                    processing_time = time.time() - start_time
                    
                    # Record results
                    assessment_record = {
                        "facility_id": facility_id,
                        "facility_name": facility_name,
                        "assessment_type": "batch",
                        "batch_index": i + 1,
                        "processing_time": processing_time,
                        "results": assessment_results,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    state["assessments_completed"].append(assessment_record)
                    state["risk_results"].append(assessment_results)
                    
                    if self.config.verbose:
                        risk_level = assessment_results.get("risk_assessment", {}).get("overall_risk_level", "Unknown")
                        logger.info(f"Facility {facility_name}: Risk level {risk_level} (processed in {processing_time:.2f}s)")
                
                except Exception as e:
                    error_msg = f"Batch assessment failed for {facility_name}: {str(e)}"
                    state["errors"].append(error_msg)
                    logger.error(error_msg)
            
            total_batch_time = time.time() - batch_start_time
            
            logger.info(f"Batch assessment completed: {len(state['assessments_completed'])} assessments in {total_batch_time:.2f} seconds")
            
            # Store batch summary
            state["analysis_results"]["batch_summary"] = {
                "total_facilities": len(facilities_result),
                "successful_assessments": len(state["assessments_completed"]),
                "failed_assessments": len([e for e in state["errors"] if "Batch assessment failed" in e]),
                "total_processing_time": total_batch_time,
                "average_processing_time": total_batch_time / max(len(state["assessments_completed"]), 1)
            }
            
        except Exception as e:
            state["errors"].append(f"Batch assessment error: {str(e)}")
            logger.error(f"Failed to execute batch assessment: {str(e)}")
        
        return state
    
    def execute_comparative_analysis(self, state: DemoState) -> DemoState:
        """
        Execute comparative risk analysis demonstration.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with comparative analysis results
        """
        logger.info("Executing comparative risk analysis demonstration")
        
        try:
            state["current_step"] = "comparative_analysis"
            params = state["scenario_params"]
            
            comparison_criteria = params["comparison_criteria"]
            
            # First, execute assessments for comparison facilities
            facilities_query = """
            MATCH (f:Facility)
            RETURN f.id as facility_id, f.name as facility_name
            LIMIT $limit
            """
            
            limit = params.get("max_facilities", 3)
            facilities_result = self.graph.query(facilities_query, {"limit": limit})
            
            if len(facilities_result) < 2:
                raise ValueError("At least 2 facilities required for comparative analysis")
            
            logger.info(f"Executing comparative analysis for {len(facilities_result)} facilities")
            
            # Execute assessments for each facility
            for facility in facilities_result:
                facility_id = facility["facility_id"]
                facility_name = facility["facility_name"]
                
                try:
                    # Configure assessment scope for comparison
                    assessment_scope = {
                        "risk_categories": params.get("risk_categories", [
                            RiskCategory.ENVIRONMENTAL,
                            RiskCategory.HEALTH_SAFETY,
                            RiskCategory.REGULATORY_COMPLIANCE
                        ]),
                        "assessment_depth": "comprehensive",
                        "assessment_trigger": "comparative_analysis",
                        "comparison_mode": True,
                        "comparison_criteria": comparison_criteria
                    }
                    
                    logger.info(f"Assessing facility for comparison: {facility_name}")
                    start_time = time.time()
                    
                    assessment_results = self.risk_agent.assess_facility_risk(
                        facility_id=facility_id,
                        assessment_scope=assessment_scope,
                        metadata={
                            "demo_scenario": "comparative_analysis",
                            "comparison_criteria": comparison_criteria,
                            "demo_timestamp": datetime.utcnow().isoformat()
                        }
                    )
                    
                    processing_time = time.time() - start_time
                    
                    # Record results
                    assessment_record = {
                        "facility_id": facility_id,
                        "facility_name": facility_name,
                        "assessment_type": "comparative",
                        "comparison_criteria": comparison_criteria,
                        "processing_time": processing_time,
                        "results": assessment_results,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    state["assessments_completed"].append(assessment_record)
                    state["risk_results"].append(assessment_results)
                    
                except Exception as e:
                    error_msg = f"Comparative assessment failed for {facility_name}: {str(e)}"
                    state["errors"].append(error_msg)
                    logger.error(error_msg)
            
            # Perform comparative analysis
            if len(state["assessments_completed"]) >= 2:
                comparative_results = self._perform_comparative_analysis(
                    state["assessments_completed"],
                    comparison_criteria
                )
                state["analysis_results"]["comparative_analysis"] = comparative_results
                
                if self.config.verbose:
                    logger.info("Comparative analysis completed successfully")
            
        except Exception as e:
            state["errors"].append(f"Comparative analysis error: {str(e)}")
            logger.error(f"Failed to execute comparative analysis: {str(e)}")
        
        return state
    
    def execute_custom_scenario(self, state: DemoState) -> DemoState:
        """
        Execute custom demonstration scenario.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with custom scenario results
        """
        logger.info("Executing custom demonstration scenario")
        
        try:
            state["current_step"] = "custom_scenario"
            params = state["scenario_params"]
            
            custom_config = params.get("custom_config", {})
            
            # Execute custom assessment logic based on configuration
            # This is a flexible node that can be configured for specific demo needs
            
            logger.info("Custom scenario demonstration completed")
            
        except Exception as e:
            state["errors"].append(f"Custom scenario error: {str(e)}")
            logger.error(f"Failed to execute custom scenario: {str(e)}")
        
        return state
    
    def _perform_comparative_analysis(
        self, 
        assessments: List[Dict[str, Any]], 
        criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform comparative analysis on assessment results.
        
        Args:
            assessments: List of assessment results
            criteria: Comparison criteria
            
        Returns:
            Comparative analysis results
        """
        try:
            # Extract risk scores and levels
            risk_data = []
            for assessment in assessments:
                facility_name = assessment["facility_name"]
                results = assessment["results"]
                
                if "risk_assessment" in results:
                    risk_assessment = results["risk_assessment"]
                    risk_data.append({
                        "facility_name": facility_name,
                        "facility_id": assessment["facility_id"],
                        "risk_level": risk_assessment.overall_risk_level.value,
                        "risk_score": risk_assessment.risk_score,
                        "risk_factors_count": len(risk_assessment.risk_factors)
                    })
            
            # Perform comparison analysis
            if risk_data:
                # Sort by risk score
                sorted_by_score = sorted(risk_data, key=lambda x: x["risk_score"], reverse=True)
                
                # Calculate statistics
                risk_scores = [item["risk_score"] for item in risk_data]
                
                comparative_results = {
                    "comparison_summary": {
                        "facilities_compared": len(risk_data),
                        "highest_risk": sorted_by_score[0] if sorted_by_score else None,
                        "lowest_risk": sorted_by_score[-1] if sorted_by_score else None,
                        "average_risk_score": sum(risk_scores) / len(risk_scores) if risk_scores else 0,
                        "risk_score_range": {
                            "min": min(risk_scores) if risk_scores else 0,
                            "max": max(risk_scores) if risk_scores else 0
                        }
                    },
                    "facility_rankings": sorted_by_score,
                    "risk_level_distribution": self._calculate_risk_distribution(risk_data),
                    "comparison_criteria": criteria,
                    "analysis_timestamp": datetime.utcnow().isoformat()
                }
                
                return comparative_results
            
            return {"error": "No valid risk data for comparison"}
            
        except Exception as e:
            logger.error(f"Comparative analysis failed: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _calculate_risk_distribution(self, risk_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of risk levels."""
        distribution = {}
        for item in risk_data:
            risk_level = item["risk_level"]
            distribution[risk_level] = distribution.get(risk_level, 0) + 1
        return distribution
    
    def analyze_results(self, state: DemoState) -> DemoState:
        """
        Analyze demonstration results and generate insights.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with analysis results
        """
        logger.info("Analyzing demonstration results")
        
        try:
            state["current_step"] = "analysis"
            
            # Compile analysis results
            analysis = {
                "scenario": state["scenario"],
                "total_assessments": len(state["assessments_completed"]),
                "successful_assessments": len([a for a in state["assessments_completed"] if a.get("results")]),
                "errors_encountered": len(state["errors"]),
                "processing_times": [a["processing_time"] for a in state["assessments_completed"] if "processing_time" in a]
            }
            
            # Calculate timing statistics
            if analysis["processing_times"]:
                analysis["timing_stats"] = {
                    "total_time": sum(analysis["processing_times"]),
                    "average_time": sum(analysis["processing_times"]) / len(analysis["processing_times"]),
                    "min_time": min(analysis["processing_times"]),
                    "max_time": max(analysis["processing_times"])
                }
            
            # Risk level analysis
            if state["risk_results"]:
                risk_levels = []
                risk_scores = []
                
                for result in state["risk_results"]:
                    if "risk_assessment" in result:
                        risk_assessment = result["risk_assessment"]
                        risk_levels.append(risk_assessment.overall_risk_level.value)
                        risk_scores.append(risk_assessment.risk_score)
                
                analysis["risk_analysis"] = {
                    "risk_level_distribution": self._calculate_risk_distribution([{"risk_level": rl} for rl in risk_levels]),
                    "risk_score_stats": {
                        "average": sum(risk_scores) / len(risk_scores) if risk_scores else 0,
                        "min": min(risk_scores) if risk_scores else 0,
                        "max": max(risk_scores) if risk_scores else 0
                    } if risk_scores else {}
                }
            
            # Add scenario-specific analysis
            if state["scenario"] == DemoScenario.BATCH_FACILITY_ASSESSMENT:
                analysis.update(state["analysis_results"].get("batch_summary", {}))
            elif state["scenario"] == DemoScenario.COMPARATIVE_RISK_ANALYSIS:
                analysis["comparative_results"] = state["analysis_results"].get("comparative_analysis", {})
            
            state["analysis_results"] = analysis
            
            if self.config.verbose:
                logger.info(f"Analysis completed: {analysis['total_assessments']} assessments processed")
            
        except Exception as e:
            state["errors"].append(f"Analysis error: {str(e)}")
            logger.error(f"Failed to analyze results: {str(e)}")
        
        return state
    
    def generate_report(self, state: DemoState) -> DemoState:
        """
        Generate demonstration report.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with report
        """
        logger.info("Generating demonstration report")
        
        try:
            state["current_step"] = "report_generation"
            
            # Compile demonstration report
            report = {
                "demo_metadata": {
                    "scenario": state["scenario"],
                    "timestamp": datetime.utcnow().isoformat(),
                    "configuration": state["configuration"],
                    "scenario_parameters": state["scenario_params"]
                },
                "execution_summary": {
                    "total_assessments": len(state["assessments_completed"]),
                    "successful_assessments": len([a for a in state["assessments_completed"] if a.get("results")]),
                    "failed_assessments": len(state["errors"]),
                    "total_processing_time": sum([a.get("processing_time", 0) for a in state["assessments_completed"]])
                },
                "assessments_completed": state["assessments_completed"],
                "analysis_results": state["analysis_results"],
                "errors": state["errors"]
            }
            
            # Add scenario-specific sections
            if state["scenario"] == DemoScenario.COMPARATIVE_RISK_ANALYSIS:
                report["comparative_analysis"] = state["analysis_results"].get("comparative_analysis", {})
            
            # Save report to file
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            report_filename = f"risk_assessment_demo_{state['scenario']}_{timestamp}.json"
            report_path = os.path.join(self.config.output_dir, report_filename)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            state["demo_report"] = report
            state["report_file_path"] = report_path
            
            logger.info(f"Demonstration report saved to: {report_path}")
            
        except Exception as e:
            state["errors"].append(f"Report generation error: {str(e)}")
            logger.error(f"Failed to generate report: {str(e)}")
        
        return state
    
    def complete(self, state: DemoState) -> DemoState:
        """Complete the demonstration workflow."""
        state["status"] = "completed"
        state["processing_time"] = time.time() - state.get("start_time", time.time())
        
        logger.info(f"Risk Assessment demonstration completed successfully")
        logger.info(f"Scenario: {state['scenario']}")
        logger.info(f"Assessments completed: {len(state['assessments_completed'])}")
        logger.info(f"Report saved to: {state.get('report_file_path', 'Not saved')}")
        
        return state
    
    def handle_error(self, state: DemoState) -> DemoState:
        """Handle demonstration errors."""
        state["status"] = "failed"
        logger.error(f"Demonstration failed with errors: {state['errors']}")
        return state
    
    def run_demonstration(
        self,
        scenario: DemoScenario,
        scenario_params: Optional[Dict[str, Any]] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> DemoState:
        """
        Run a complete risk assessment demonstration.
        
        Args:
            scenario: Demonstration scenario to execute
            scenario_params: Scenario-specific parameters
            config_overrides: Configuration overrides
            
        Returns:
            Final demonstration state
        """
        # Initialize state
        initial_state: DemoState = {
            "scenario": scenario,
            "configuration": {
                "neo4j_uri": self.config.neo4j_uri,
                "llm_model": self.config.llm_model,
                "enable_langsmith": self.config.enable_langsmith,
                "output_dir": self.config.output_dir,
                "verbose": self.config.verbose,
                **(config_overrides or {})
            },
            "scenario_params": scenario_params or {},
            "current_step": "initialization",
            "assessments_completed": [],
            "risk_results": [],
            "analysis_results": {},
            "demo_report": {},
            "report_file_path": None,
            "errors": [],
            "status": "starting",
            "processing_time": None,
            "start_time": time.time()
        }
        
        # Execute workflow
        final_state = self.workflow.invoke(initial_state)
        
        return final_state
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'risk_agent'):
            try:
                self.risk_agent.close()
                logger.info("Risk Assessment Agent closed")
            except Exception as e:
                logger.error(f"Error closing Risk Assessment Agent: {str(e)}")


# Convenience functions for common demonstration scenarios

def demo_facility_risk_assessment(
    config: DemoConfiguration,
    facility_id: str,
    risk_categories: Optional[List[RiskCategory]] = None
) -> DemoState:
    """
    Run a facility risk assessment demonstration.
    
    Args:
        config: Demonstration configuration
        facility_id: Facility to assess
        risk_categories: Risk categories to evaluate
        
    Returns:
        Demonstration results
    """
    demo = EHSRiskAssessmentDemo(config)
    
    scenario_params = {
        "facility_id": facility_id,
        "risk_categories": risk_categories or [
            RiskCategory.ENVIRONMENTAL,
            RiskCategory.HEALTH_SAFETY,
            RiskCategory.REGULATORY_COMPLIANCE
        ],
        "assessment_depth": "comprehensive"
    }
    
    try:
        result = demo.run_demonstration(
            DemoScenario.FACILITY_RISK_ASSESSMENT,
            scenario_params
        )
        return result
    finally:
        demo.close()


def demo_batch_facility_assessment(
    config: DemoConfiguration,
    facility_ids: Optional[List[str]] = None,
    max_facilities: int = 5
) -> DemoState:
    """
    Run a batch facility assessment demonstration.
    
    Args:
        config: Demonstration configuration
        facility_ids: Specific facilities to assess
        max_facilities: Maximum number of facilities to assess
        
    Returns:
        Demonstration results
    """
    demo = EHSRiskAssessmentDemo(config)
    
    scenario_params = {
        "facility_ids": facility_ids,
        "max_facilities": max_facilities,
        "assessment_depth": "standard",
        "comparative_analysis": True
    }
    
    try:
        result = demo.run_demonstration(
            DemoScenario.BATCH_FACILITY_ASSESSMENT,
            scenario_params
        )
        return result
    finally:
        demo.close()


def demo_comparative_risk_analysis(
    config: DemoConfiguration,
    comparison_criteria: Dict[str, Any],
    max_facilities: int = 3
) -> DemoState:
    """
    Run a comparative risk analysis demonstration.
    
    Args:
        config: Demonstration configuration
        comparison_criteria: Criteria for comparison
        max_facilities: Maximum facilities to compare
        
    Returns:
        Demonstration results
    """
    demo = EHSRiskAssessmentDemo(config)
    
    scenario_params = {
        "comparison_criteria": comparison_criteria,
        "max_facilities": max_facilities,
        "risk_categories": [
            RiskCategory.ENVIRONMENTAL,
            RiskCategory.HEALTH_SAFETY,
            RiskCategory.REGULATORY_COMPLIANCE
        ]
    }
    
    try:
        result = demo.run_demonstration(
            DemoScenario.COMPARATIVE_RISK_ANALYSIS,
            scenario_params
        )
        return result
    finally:
        demo.close()