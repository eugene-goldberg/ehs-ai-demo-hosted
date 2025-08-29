"""
LangGraph-based document processing workflow for EHS AI Platform with Risk Assessment Integration.
Extends the enhanced workflow to include comprehensive risk assessment after document processing.
This provides intelligent risk analysis of processed documents and facilities.
"""

import os
import logging
import contextlib
from typing import Dict, List, Any, Optional, TypedDict
from datetime import datetime
from enum import Enum
import json

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

# Enhanced workflow imports
from .ingestion_workflow_enhanced import (
    EnhancedIngestionWorkflow,
    DocumentState as BaseDocumentState,
    ProcessingStatus
)

# Risk Assessment Agent import
from ..agents.risk_assessment.agent import (
    RiskAssessmentAgent,
    RiskAssessmentState,
    RiskLevel,
    RiskCategory,
    AssessmentStatus
)

# Phase 1 Enhancement Imports
from ..phase1_enhancements.audit_trail_service import AuditTrailService
from ..phase1_enhancements.rejection_workflow_service import (
    RejectionWorkflowService, 
    RejectionReason, 
    ValidationResult
)
from ..phase1_enhancements.prorating_service import ProRatingService
from ..shared.common_fn import create_graph_database_connection
from ..langsmith_config import config as langsmith_config, tracing_context, tag_ingestion_trace
from ..setup_langsmith import setup_langsmith_tracing, set_langsmith_project

logger = logging.getLogger(__name__)


# Enhanced State with Risk Assessment Fields
class DocumentStateWithRisk(BaseDocumentState):
    """Extended document state including risk assessment fields."""
    
    # Risk Assessment Fields
    risk_assessment_enabled: bool
    risk_assessment_id: Optional[str]
    risk_assessment_status: Optional[str]
    risk_assessment_results: Optional[Dict[str, Any]]
    risk_level: Optional[str]
    risk_score: Optional[float]
    risk_factors: Optional[List[Dict[str, Any]]]
    risk_recommendations: Optional[List[Dict[str, Any]]]
    facility_risk_context: Optional[Dict[str, Any]]
    risk_processing_time: Optional[float]
    risk_errors: List[str]


class RiskAssessmentIntegratedWorkflow(EnhancedIngestionWorkflow):
    """
    Enhanced ingestion workflow with integrated risk assessment capabilities.
    
    This workflow extends the base enhanced workflow to include:
    1. Risk assessment after document processing completion
    2. Integration with the risk assessment agent
    3. Storage of risk assessment results in Neo4j
    4. LangSmith tracing for risk assessment operations
    5. Conditional logic to enable/disable risk assessment
    6. Error handling for risk assessment failures
    """
    
    def __init__(
        self,
        llama_parse_api_key: str,
        neo4j_uri: str,
        neo4j_username: str,
        neo4j_password: str,
        neo4j_database: str = "neo4j",
        llm_model: str = "gpt-4",
        max_retries: int = 3,
        enable_phase1_features: bool = True,
        enable_risk_assessment: bool = True,
        storage_path: str = "/app/storage/",
        risk_assessment_methodology: str = "comprehensive"
    ):
        """
        Initialize the risk assessment integrated workflow.
        
        Args:
            llama_parse_api_key: API key for LlamaParse
            neo4j_uri: Neo4j connection URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            neo4j_database: Neo4j database name
            llm_model: LLM model to use
            max_retries: Maximum retry attempts
            enable_phase1_features: Enable Phase 1 features
            enable_risk_assessment: Enable risk assessment integration
            storage_path: Base storage path for audit trail files
            risk_assessment_methodology: Risk assessment methodology to use
        """
        # Initialize base enhanced workflow
        super().__init__(
            llama_parse_api_key=llama_parse_api_key,
            neo4j_uri=neo4j_uri,
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password,
            neo4j_database=neo4j_database,
            llm_model=llm_model,
            max_retries=max_retries,
            enable_phase1_features=enable_phase1_features,
            storage_path=storage_path
        )
        
        self.enable_risk_assessment = enable_risk_assessment
        self.risk_methodology = risk_assessment_methodology
        
        # Initialize Risk Assessment Agent if enabled
        if self.enable_risk_assessment:
            try:
                self.risk_agent = RiskAssessmentAgent(
                    neo4j_uri=neo4j_uri,
                    neo4j_username=neo4j_username,
                    neo4j_password=neo4j_password,
                    neo4j_database=neo4j_database,
                    llm_model=llm_model,
                    max_retries=max_retries,
                    enable_langsmith=langsmith_config.is_available,
                    risk_assessment_methodology=risk_assessment_methodology
                )
                logger.info("Risk Assessment Agent initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Risk Assessment Agent: {str(e)}")
                self.enable_risk_assessment = False
                self.risk_agent = None
                logger.warning("Risk assessment disabled due to initialization failure")
        else:
            self.risk_agent = None
            logger.info("Risk assessment disabled by configuration")
        
        # Rebuild workflow to include risk assessment
        self.workflow = self._build_workflow_with_risk_assessment()
    
    def _build_workflow_with_risk_assessment(self) -> StateGraph:
        """
        Build the enhanced LangGraph workflow with risk assessment integration.
        
        Returns:
            Compiled workflow graph with risk assessment nodes
        """
        # Create workflow
        workflow = StateGraph(DocumentStateWithRisk)
        
        # Add all existing nodes from enhanced workflow
        workflow.add_node("store_source_file", self.store_source_file)
        workflow.add_node("validate", self.validate_document)
        workflow.add_node("validate_document_quality", self.validate_document_quality)
        workflow.add_node("parse", self.parse_document)
        workflow.add_node("check_for_duplicates", self.check_for_duplicates)
        workflow.add_node("extract", self.extract_data)
        workflow.add_node("transform", self.transform_data)
        workflow.add_node("validate_data", self.validate_extracted_data)
        workflow.add_node("process_prorating", self.process_prorating)
        workflow.add_node("load", self.load_to_neo4j)
        workflow.add_node("index", self.index_document)
        workflow.add_node("complete", self.complete_processing)
        workflow.add_node("handle_error", self.handle_error)
        workflow.add_node("handle_rejection", self.handle_rejection)
        
        # Add new risk assessment nodes
        workflow.add_node("initialize_risk_assessment", self.initialize_risk_assessment)
        workflow.add_node("perform_risk_assessment", self.perform_risk_assessment)
        workflow.add_node("store_risk_results", self.store_risk_results)
        workflow.add_node("finalize_risk_assessment", self.finalize_risk_assessment)
        workflow.add_node("handle_risk_error", self.handle_risk_error)
        
        # Add existing edges (same as enhanced workflow)
        workflow.add_edge("store_source_file", "validate")
        
        # Conditional validation flow
        workflow.add_conditional_edges(
            "validate",
            self.check_validation_status,
            {
                "continue": "validate_document_quality",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "validate_document_quality",
            self.check_quality_validation,
            {
                "passed": "parse",
                "rejected": "handle_rejection",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("parse", "check_for_duplicates")
        
        workflow.add_conditional_edges(
            "check_for_duplicates",
            self.check_duplicate_status,
            {
                "unique": "extract",
                "duplicate": "handle_rejection",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("extract", "transform")
        workflow.add_edge("transform", "validate_data")
        
        # Conditional edges for validation
        workflow.add_conditional_edges(
            "validate_data",
            self.check_validation,
            {
                "valid": "process_prorating",
                "invalid": "handle_error",
                "retry": "extract"
            }
        )
        
        # Pro-rating processing
        workflow.add_conditional_edges(
            "process_prorating",
            self.check_prorating_needed,
            {
                "prorating_needed": "load",
                "skip_prorating": "load"
            }
        )
        
        workflow.add_edge("load", "index")
        workflow.add_edge("index", "complete")
        
        # NEW: Risk assessment integration after complete processing
        workflow.add_conditional_edges(
            "complete",
            self.check_risk_assessment_needed,
            {
                "risk_assessment_needed": "initialize_risk_assessment",
                "skip_risk_assessment": END
            }
        )
        
        # Risk assessment flow
        workflow.add_edge("initialize_risk_assessment", "perform_risk_assessment")
        
        workflow.add_conditional_edges(
            "perform_risk_assessment",
            self.check_risk_assessment_status,
            {
                "success": "store_risk_results",
                "error": "handle_risk_error"
            }
        )
        
        workflow.add_edge("store_risk_results", "finalize_risk_assessment")
        workflow.add_edge("finalize_risk_assessment", END)
        
        # Risk assessment error handling
        workflow.add_conditional_edges(
            "handle_risk_error",
            self.check_risk_retry_needed,
            {
                "retry": "perform_risk_assessment",
                "skip": "finalize_risk_assessment"
            }
        )
        
        # Error and rejection handling (same as enhanced workflow)
        workflow.add_conditional_edges(
            "handle_error",
            self.check_retry,
            {
                "retry": "validate",
                "fail": END
            }
        )
        
        workflow.add_edge("handle_rejection", END)
        
        # Set entry point
        workflow.set_entry_point("store_source_file")
        
        # Compile
        return workflow.compile()
    
    def initialize_risk_assessment(self, state: DocumentStateWithRisk) -> DocumentStateWithRisk:
        """
        Initialize risk assessment for the processed document.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with risk assessment initialization
        """
        logger.info(f"Initializing risk assessment for document: {state['document_id']}")
        
        try:
            # Initialize risk assessment fields
            state["risk_assessment_enabled"] = self.enable_risk_assessment
            state["risk_assessment_id"] = f"risk_{state['document_id']}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            state["risk_assessment_status"] = AssessmentStatus.INITIATED.value
            state["risk_assessment_results"] = None
            state["risk_level"] = None
            state["risk_score"] = None
            state["risk_factors"] = []
            state["risk_recommendations"] = []
            state["facility_risk_context"] = {}
            state["risk_processing_time"] = None
            state["risk_errors"] = []
            
            # Extract facility context from processed document
            facility_context = self._extract_facility_context(state)
            state["facility_risk_context"] = facility_context
            
            # Phase 1: Log risk assessment initialization
            if self.enable_phase1:
                state["phase1_processing"]["risk_assessment_initialized"] = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "risk_assessment_id": state["risk_assessment_id"],
                    "facility_context": facility_context
                }
            
            logger.info(f"Risk assessment initialized: {state['risk_assessment_id']}")
            
        except Exception as e:
            state["risk_errors"].append(f"Risk assessment initialization error: {str(e)}")
            logger.error(f"Failed to initialize risk assessment: {str(e)}")
        
        return state
    
    def perform_risk_assessment(self, state: DocumentStateWithRisk) -> DocumentStateWithRisk:
        """
        Perform comprehensive risk assessment using the risk assessment agent.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with risk assessment results
        """
        logger.info(f"Performing risk assessment: {state['risk_assessment_id']}")
        
        risk_start_time = datetime.utcnow().timestamp()
        
        try:
            if not self.risk_agent:
                raise ValueError("Risk Assessment Agent not initialized")
            
            # Prepare assessment scope from document context
            assessment_scope = {
                "document_id": state["document_id"],
                "document_type": state["document_type"],
                "extracted_data": state.get("extracted_data", {}),
                "facility_context": state.get("facility_risk_context", {}),
                "assessment_trigger": "document_processing",
                "analysis_depth": "comprehensive"
            }
            
            # Determine facility ID from context or extracted data
            facility_id = self._determine_facility_id(state)
            
            if facility_id:
                # FIXED: Use the same LangSmith project as the main ingestion workflow
                # Instead of creating a session-specific project, use the main "ehs-ai-demo-ingestion" project
                context_manager = (
                    contextlib.nullcontext()  # Use the default project that's already set
                    if not langsmith_config.is_available 
                    else contextlib.nullcontext()  # Don't override the project
                )
                
                # Ensure we're using the same project as setup_langsmith.py
                if langsmith_config.is_available:
                    # Set the project to match the main ingestion workflow
                    set_langsmith_project("ehs-ai-demo-ingestion")
                
                with context_manager:
                    risk_results = self.risk_agent.assess_facility_risk(
                        facility_id=facility_id,
                        assessment_scope=assessment_scope,
                        metadata={
                            "source_document_id": state["document_id"],
                            "document_type": state["document_type"],
                            "workflow_integration": True
                        }
                    )
                
                # Extract and store risk assessment results
                state["risk_assessment_results"] = self._process_risk_results(risk_results)
                state["risk_assessment_status"] = risk_results.get("status", AssessmentStatus.COMPLETED.value)
                
                # Extract specific risk data
                if risk_results.get("risk_assessment"):
                    risk_assessment = risk_results["risk_assessment"]
                    state["risk_level"] = risk_assessment.overall_risk_level.value
                    state["risk_score"] = risk_assessment.risk_score
                    state["risk_factors"] = [
                        {
                            "id": factor.id,
                            "name": factor.name,
                            "category": factor.category.value,
                            "description": factor.description,
                            "severity": factor.severity,
                            "probability": factor.probability,
                            "confidence": factor.confidence
                        }
                        for factor in risk_assessment.risk_factors
                    ]
                
                # Extract recommendations
                if risk_results.get("recommendations"):
                    recommendations = risk_results["recommendations"]
                    state["risk_recommendations"] = [
                        {
                            "id": rec.id,
                            "title": rec.title,
                            "description": rec.description,
                            "priority": rec.priority,
                            "estimated_impact": rec.estimated_impact,
                            "implementation_timeline": rec.implementation_timeline
                        }
                        for rec in recommendations.recommendations
                    ]
                
                # Calculate processing time
                risk_end_time = datetime.utcnow().timestamp()
                state["risk_processing_time"] = risk_end_time - risk_start_time
                
                # Phase 1: Log risk assessment completion
                if self.enable_phase1:
                    state["phase1_processing"]["risk_assessment_completed"] = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "processing_time": state["risk_processing_time"],
                        "risk_level": state.get("risk_level"),
                        "risk_score": state.get("risk_score"),
                        "num_risk_factors": len(state.get("risk_factors", [])),
                        "num_recommendations": len(state.get("risk_recommendations", []))
                    }
                
                logger.info(f"Risk assessment completed with level: {state.get('risk_level', 'unknown')}")
                
            else:
                # No facility context available - skip risk assessment
                state["risk_assessment_status"] = "skipped"
                state["risk_errors"].append("No facility context available for risk assessment")
                logger.warning("Risk assessment skipped - no facility context available")
                
        except Exception as e:
            state["risk_errors"].append(f"Risk assessment error: {str(e)}")
            state["risk_assessment_status"] = AssessmentStatus.FAILED.value
            
            # Phase 1: Log risk assessment failure
            if self.enable_phase1:
                state["phase1_processing"]["risk_assessment_failed"] = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": str(e)
                }
            
            logger.error(f"Risk assessment failed: {str(e)}")
        
        return state
    
    def store_risk_results(self, state: DocumentStateWithRisk) -> DocumentStateWithRisk:
        """
        Store risk assessment results in Neo4j using direct Cypher queries.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state after storing risk results
        """
        logger.info("Storing risk assessment results in Neo4j")
        
        try:
            if not state.get("risk_assessment_results"):
                logger.warning("No risk assessment results to store")
                return state
            
            # Check if graph connection is available
            if not hasattr(self, 'graph') or not self.graph:
                logger.error("No Neo4j graph connection available")
                state["risk_errors"].append("No Neo4j graph connection available")
                return state
            
            nodes_created = 0
            relationships_created = 0
            
            # Create RiskAssessment node
            risk_assessment_query = """
            MERGE (ra:RiskAssessment {id: $risk_assessment_id})
            SET ra.document_id = $document_id,
                ra.assessment_date = $assessment_date,
                ra.risk_level = $risk_level,
                ra.risk_score = $risk_score,
                ra.assessment_status = $assessment_status,
                ra.methodology = $methodology,
                ra.processing_time = $processing_time,
                ra.created_by = $created_by,
                ra.updated_at = $updated_at
            RETURN ra
            """
            
            risk_assessment_params = {
                "risk_assessment_id": state["risk_assessment_id"],
                "document_id": state["document_id"],
                "assessment_date": datetime.utcnow().isoformat(),
                "risk_level": state.get("risk_level"),
                "risk_score": state.get("risk_score"),
                "assessment_status": state.get("risk_assessment_status"),
                "methodology": self.risk_methodology,
                "processing_time": state.get("risk_processing_time"),
                "created_by": "workflow_integration",
                "updated_at": datetime.utcnow().isoformat()
            }
            
            try:
                result = self.graph.query(risk_assessment_query, risk_assessment_params)
                if result:
                    nodes_created += 1
                    logger.info(f"Created RiskAssessment node: {state['risk_assessment_id']}")
            except Exception as e:
                logger.error(f"Failed to create RiskAssessment node: {str(e)}")
                state["risk_errors"].append(f"RiskAssessment node creation error: {str(e)}")
            
            # Create RiskFactor nodes and relationships
            for factor in state.get("risk_factors", []):
                try:
                    risk_factor_query = """
                    MERGE (rf:RiskFactor {id: $factor_id})
                    SET rf.name = $name,
                        rf.category = $category,
                        rf.description = $description,
                        rf.severity = $severity,
                        rf.probability = $probability,
                        rf.confidence = $confidence,
                        rf.created_at = $created_at,
                        rf.updated_at = $updated_at
                    RETURN rf
                    """
                    
                    risk_factor_params = {
                        "factor_id": factor["id"],
                        "name": factor["name"],
                        "category": factor["category"],
                        "description": factor["description"],
                        "severity": factor["severity"],
                        "probability": factor["probability"],
                        "confidence": factor["confidence"],
                        "created_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat()
                    }
                    
                    result = self.graph.query(risk_factor_query, risk_factor_params)
                    if result:
                        nodes_created += 1
                        
                        # Create relationship between RiskAssessment and RiskFactor
                        relationship_query = """
                        MATCH (ra:RiskAssessment {id: $risk_assessment_id})
                        MATCH (rf:RiskFactor {id: $factor_id})
                        MERGE (ra)-[r:IDENTIFIES]->(rf)
                        SET r.created_at = $created_at
                        RETURN r
                        """
                        
                        rel_params = {
                            "risk_assessment_id": state["risk_assessment_id"],
                            "factor_id": factor["id"],
                            "created_at": datetime.utcnow().isoformat()
                        }
                        
                        rel_result = self.graph.query(relationship_query, rel_params)
                        if rel_result:
                            relationships_created += 1
                            logger.info(f"Created RiskFactor and relationship: {factor['id']}")
                            
                except Exception as e:
                    logger.error(f"Failed to create RiskFactor {factor.get('id', 'unknown')}: {str(e)}")
                    state["risk_errors"].append(f"RiskFactor creation error: {str(e)}")
            
            # Create Recommendation nodes and relationships
            for rec in state.get("risk_recommendations", []):
                try:
                    recommendation_query = """
                    MERGE (rr:RiskRecommendation {id: $rec_id})
                    SET rr.title = $title,
                        rr.description = $description,
                        rr.priority = $priority,
                        rr.estimated_impact = $estimated_impact,
                        rr.implementation_timeline = $implementation_timeline,
                        rr.created_at = $created_at,
                        rr.updated_at = $updated_at
                    RETURN rr
                    """
                    
                    recommendation_params = {
                        "rec_id": rec["id"],
                        "title": rec["title"],
                        "description": rec["description"],
                        "priority": rec["priority"],
                        "estimated_impact": rec["estimated_impact"],
                        "implementation_timeline": rec["implementation_timeline"],
                        "created_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat()
                    }
                    
                    result = self.graph.query(recommendation_query, recommendation_params)
                    if result:
                        nodes_created += 1
                        
                        # Create relationship between RiskAssessment and RiskRecommendation
                        relationship_query = """
                        MATCH (ra:RiskAssessment {id: $risk_assessment_id})
                        MATCH (rr:RiskRecommendation {id: $rec_id})
                        MERGE (ra)-[r:RECOMMENDS]->(rr)
                        SET r.created_at = $created_at
                        RETURN r
                        """
                        
                        rel_params = {
                            "risk_assessment_id": state["risk_assessment_id"],
                            "rec_id": rec["id"],
                            "created_at": datetime.utcnow().isoformat()
                        }
                        
                        rel_result = self.graph.query(relationship_query, rel_params)
                        if rel_result:
                            relationships_created += 1
                            logger.info(f"Created RiskRecommendation and relationship: {rec['id']}")
                            
                except Exception as e:
                    logger.error(f"Failed to create RiskRecommendation {rec.get('id', 'unknown')}: {str(e)}")
                    state["risk_errors"].append(f"RiskRecommendation creation error: {str(e)}")
            
            # Create relationship between Document and RiskAssessment
            try:
                doc_relationship_query = """
                MATCH (d:Document {id: $document_id})
                MATCH (ra:RiskAssessment {id: $risk_assessment_id})
                MERGE (d)-[r:HAS_RISK_ASSESSMENT]->(ra)
                SET r.created_at = $created_at
                RETURN r
                """
                
                doc_rel_params = {
                    "document_id": state["document_id"],
                    "risk_assessment_id": state["risk_assessment_id"],
                    "created_at": datetime.utcnow().isoformat()
                }
                
                result = self.graph.query(doc_relationship_query, doc_rel_params)
                if result:
                    relationships_created += 1
                    logger.info(f"Created Document-RiskAssessment relationship")
                    
            except Exception as e:
                logger.error(f"Failed to create Document-RiskAssessment relationship: {str(e)}")
                state["risk_errors"].append(f"Document relationship creation error: {str(e)}")
            
            # Try to link to facility if available
            facility_id = self._determine_facility_id(state)
            if facility_id and facility_id != f"facility_from_doc_{state['document_id']}":
                try:
                    facility_relationship_query = """
                    MATCH (f:Facility {id: $facility_id})
                    MATCH (ra:RiskAssessment {id: $risk_assessment_id})
                    MERGE (f)-[r:HAS_RISK_ASSESSMENT]->(ra)
                    SET r.created_at = $created_at
                    RETURN r
                    """
                    
                    facility_rel_params = {
                        "facility_id": facility_id,
                        "risk_assessment_id": state["risk_assessment_id"],
                        "created_at": datetime.utcnow().isoformat()
                    }
                    
                    result = self.graph.query(facility_relationship_query, facility_rel_params)
                    if result:
                        relationships_created += 1
                        logger.info(f"Created Facility-RiskAssessment relationship")
                        
                except Exception as e:
                    logger.warning(f"Could not link to facility {facility_id}: {str(e)}")
                    # This is not a critical error, so we don't add it to risk_errors
            
            # Phase 1: Log risk results storage
            if self.enable_phase1:
                state["phase1_processing"]["risk_results_stored"] = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "nodes_created": nodes_created,
                    "relationships_created": relationships_created,
                    "has_errors": bool(state.get("risk_errors"))
                }
            
            logger.info(f"Successfully stored {nodes_created} risk nodes and {relationships_created} relationships")
            
        except Exception as e:
            error_msg = f"Risk storage error: {str(e)}"
            state["risk_errors"].append(error_msg)
            logger.error(f"Failed to store risk assessment results: {str(e)}")
        
        return state
    
    def finalize_risk_assessment(self, state: DocumentStateWithRisk) -> DocumentStateWithRisk:
        """
        Finalize risk assessment processing.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with finalized risk assessment
        """
        logger.info("Finalizing risk assessment")
        
        try:
            # Update final status
            if state.get("risk_errors"):
                state["risk_assessment_status"] = AssessmentStatus.FAILED.value
                logger.warning(f"Risk assessment completed with errors: {len(state['risk_errors'])}")
            else:
                state["risk_assessment_status"] = AssessmentStatus.COMPLETED.value
                logger.info("Risk assessment completed successfully")
            
            # Phase 1: Finalize risk assessment in audit trail
            if self.enable_phase1:
                state["phase1_processing"]["risk_assessment_finalized"] = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "final_status": state["risk_assessment_status"],
                    "has_errors": bool(state.get("risk_errors")),
                    "error_count": len(state.get("risk_errors", [])),
                    "total_processing_time": state.get("risk_processing_time", 0)
                }
            
        except Exception as e:
            state["risk_errors"].append(f"Risk finalization error: {str(e)}")
            logger.error(f"Failed to finalize risk assessment: {str(e)}")
        
        return state
    
    def handle_risk_error(self, state: DocumentStateWithRisk) -> DocumentStateWithRisk:
        """
        Handle errors in risk assessment processing.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with error handling
        """
        logger.error(f"Handling risk assessment error for: {state['risk_assessment_id']}")
        
        # Log all risk errors
        for error in state.get("risk_errors", []):
            logger.error(f"Risk assessment error: {error}")
        
        # Phase 1: Log error handling
        if self.enable_phase1:
            state["phase1_processing"]["risk_error_handled"] = {
                "timestamp": datetime.utcnow().isoformat(),
                "errors": state.get("risk_errors", []),
                "retry_count": state.get("retry_count", 0)
            }
        
        return state
    
    # Helper methods
    def _extract_facility_context(self, state: DocumentStateWithRisk) -> Dict[str, Any]:
        """Extract facility context from processed document."""
        context = {}
        
        try:
            # Extract from document metadata
            metadata = state.get("upload_metadata", {})
            context.update(metadata)
            
            # Extract from processed data
            extracted_data = state.get("extracted_data", {})
            if extracted_data:
                # Look for facility-related fields
                for field in ["facility_name", "facility_id", "site_id", "location", "address"]:
                    if field in extracted_data:
                        context[field] = extracted_data[field]
            
            # Extract from document type
            context["document_type"] = state.get("document_type")
            context["processing_date"] = datetime.utcnow().isoformat()
            
        except Exception as e:
            logger.warning(f"Failed to extract facility context: {str(e)}")
        
        return context
    
    def _determine_facility_id(self, state: DocumentStateWithRisk) -> Optional[str]:
        """Determine facility ID from various sources."""
        
        # Check extracted data
        extracted_data = state.get("extracted_data", {})
        if "facility_id" in extracted_data:
            return extracted_data["facility_id"]
        
        # Check metadata
        metadata = state.get("upload_metadata", {})
        if "facility_id" in metadata:
            return metadata["facility_id"]
        
        # Check facility context
        facility_context = state.get("facility_risk_context", {})
        if "facility_id" in facility_context:
            return facility_context["facility_id"]
        
        # Use document ID as fallback facility reference
        return f"facility_from_doc_{state['document_id']}"
    
    def _process_risk_results(self, risk_results: RiskAssessmentState) -> Dict[str, Any]:
        """Process and sanitize risk assessment results for storage."""
        
        processed_results = {
            "assessment_id": risk_results.get("assessment_id"),
            "facility_id": risk_results.get("facility_id"),
            "status": risk_results.get("status"),
            "processing_time": risk_results.get("processing_time"),
            "current_step": risk_results.get("current_step"),
            "errors": risk_results.get("errors", [])
        }
        
        # Add risk assessment if available
        if risk_results.get("risk_assessment"):
            risk_assessment = risk_results["risk_assessment"]
            processed_results["risk_assessment"] = {
                "overall_risk_level": risk_assessment.overall_risk_level.value,
                "risk_score": risk_assessment.risk_score,
                "assessment_date": risk_assessment.assessment_date.isoformat(),
                "methodology": risk_assessment.methodology,
                "confidence_level": risk_assessment.confidence_level
            }
        
        return processed_results
    
    # Conditional edge check methods for risk assessment
    def check_risk_assessment_needed(self, state: DocumentStateWithRisk) -> str:
        """Check if risk assessment should be performed."""
        
        # Perform risk assessment if:
        # 1. Risk assessment is enabled
        # 2. We have a document_id (so we know what document triggered the workflow)
        # 3. The workflow didn't completely fail at the start
        
        if not self.enable_risk_assessment:
            logger.info("Risk assessment disabled by configuration")
            return "skip_risk_assessment"
        
        if not state.get("document_id"):
            logger.info("No document_id available - skipping risk assessment")
            return "skip_risk_assessment"
        
        # Allow risk assessment to run independently even if document processing had issues
        logger.info(f"Risk assessment needed for document: {state['document_id']}")
        return "risk_assessment_needed"
    
    def check_risk_assessment_status(self, state: DocumentStateWithRisk) -> str:
        """Check the status of risk assessment."""
        
        if state.get("risk_errors"):
            return "error"
        
        if state.get("risk_assessment_status") == AssessmentStatus.COMPLETED.value:
            return "success"
        
        if state.get("risk_assessment_status") == AssessmentStatus.FAILED.value:
            return "error"
        
        # Default to success if no explicit errors
        return "success"
    
    def check_risk_retry_needed(self, state: DocumentStateWithRisk) -> str:
        """Check if risk assessment retry is needed."""
        
        retry_count = state.get("retry_count", 0)
        
        if retry_count < self.max_retries:
            state["retry_count"] = retry_count + 1
            return "retry"
        else:
            return "skip"
    
    def process_document(
        self,
        file_path: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentStateWithRisk:
        """
        Process a single document through the enhanced workflow with risk assessment.
        
        Args:
            file_path: Path to the document
            document_id: Unique document identifier
            metadata: Additional metadata
            
        Returns:
            Final workflow state with risk assessment results
        """
        # Initialize enhanced state with risk assessment fields
        initial_state: DocumentStateWithRisk = {
            # Base fields from enhanced workflow
            "file_path": file_path,
            "document_id": document_id,
            "upload_metadata": metadata or {},
            "document_type": None,
            "parsed_content": None,
            "extracted_data": None,
            "validation_results": None,
            "indexed": False,
            "errors": [],
            "retry_count": 0,
            "neo4j_nodes": None,
            "neo4j_relationships": None,
            "processing_time": None,
            "status": ProcessingStatus.PENDING,
            
            # Phase 1 fields
            "source_file_path": None,
            "original_filename": None,
            "audit_trail_id": None,
            "rejection_id": None,
            "rejection_reason": None,
            "rejection_status": None,
            "validation_score": None,
            "is_duplicate": False,
            "prorating_allocation_id": None,
            "phase1_processing": {},
            
            # Risk assessment fields
            "risk_assessment_enabled": self.enable_risk_assessment,
            "risk_assessment_id": None,
            "risk_assessment_status": None,
            "risk_assessment_results": None,
            "risk_level": None,
            "risk_score": None,
            "risk_factors": None,
            "risk_recommendations": None,
            "facility_risk_context": None,
            "risk_processing_time": None,
            "risk_errors": [],
            
            # Timing
            "start_time": datetime.utcnow().timestamp()
        }
        
        # Run enhanced workflow with risk assessment
        final_state = self.workflow.invoke(initial_state)
        
        return final_state
    
    def close(self):
        """Close the workflow and clean up resources."""
        super().close()
        
        if self.risk_agent:
            try:
                self.risk_agent.close()
                logger.info("Risk Assessment Agent closed successfully")
            except Exception as e:
                logger.error(f"Error closing Risk Assessment Agent: {str(e)}")


# Factory function for easy instantiation
def create_risk_integrated_workflow(
    llama_parse_api_key: str,
    neo4j_uri: str,
    neo4j_username: str,
    neo4j_password: str,
    neo4j_database: str = "neo4j",
    llm_model: str = "gpt-4",
    enable_risk_assessment: bool = True,
    **kwargs
) -> RiskAssessmentIntegratedWorkflow:
    """
    Factory function to create a risk assessment integrated workflow.
    
    Args:
        llama_parse_api_key: API key for LlamaParse
        neo4j_uri: Neo4j connection URI
        neo4j_username: Neo4j username
        neo4j_password: Neo4j password
        neo4j_database: Neo4j database name
        llm_model: LLM model to use
        enable_risk_assessment: Enable risk assessment integration
        **kwargs: Additional configuration options
        
    Returns:
        Configured RiskAssessmentIntegratedWorkflow instance
    """
    return RiskAssessmentIntegratedWorkflow(
        llama_parse_api_key=llama_parse_api_key,
        neo4j_uri=neo4j_uri,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password,
        neo4j_database=neo4j_database,
        llm_model=llm_model,
        enable_risk_assessment=enable_risk_assessment,
        **kwargs
    )


# Alias for backward compatibility
IngestionWorkflowWithRiskAssessment = RiskAssessmentIntegratedWorkflow