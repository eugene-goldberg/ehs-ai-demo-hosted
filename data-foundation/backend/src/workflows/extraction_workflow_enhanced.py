"""
Enhanced LangGraph-based data extraction workflow for EHS AI Platform.
Integrates Phase 1 enhancements: audit trail, pro-rating, and enhanced reporting.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, TypedDict
from datetime import datetime
from enum import Enum
from pathlib import Path

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from neo4j import GraphDatabase
from neo4j.graph import Node, Relationship
from pydantic import BaseModel, Field

# Phase 1 imports
from ..phase1_enhancements.audit_trail_service import AuditTrailService
from ..phase1_enhancements.rejection_workflow_service import RejectionWorkflowService
from ..phase1_enhancements.prorating_service import ProRatingService
from ..shared.common_fn import create_graph_database_connection

logger = logging.getLogger(__name__)

# Import transcript logging utilities
try:
    from src.utils.transcript_forwarder import forward_transcript_entry
except ImportError:
    # If import fails, create a no-op function
    def forward_transcript_entry(role, content, context=None):
        pass


# Enhanced State definitions with Phase 1 fields
class ExtractionState(TypedDict):
    """Enhanced state for data extraction workflow with Phase 1 features."""
    # Existing input fields
    query_config: Dict[str, Any]  # Contains query type, parameters, filters
    output_format: str  # json, csv, pdf, html
    report_template: Optional[str]
    
    # Existing processing state
    queries: List[Dict[str, Any]]  # Cypher queries to execute
    query_results: List[Dict[str, Any]]  # Results from each query
    graph_objects: List[Dict[str, Any]]  # Neo4j objects involved
    analysis_results: Optional[Dict[str, Any]]  # LLM analysis of results
    
    # Existing output
    report_data: Dict[str, Any]  # Structured report data
    report_file_path: Optional[str]
    processing_time: Optional[float]
    
    # Existing error handling
    errors: List[str]
    status: str  # pending, processing, completed, failed
    
    # Phase 1 Enhancement Fields
    prorating_allocations: List[Dict[str, Any]]  # Pro-rating allocation results
    audit_trail_entries: List[Dict[str, Any]]    # Extraction audit trail
    include_source_files: bool                   # Flag for including source files in report
    phase1_processing: Dict[str, Any]            # Phase 1 processing metadata


class QueryType(str, Enum):
    """Enhanced query types with Phase 1 additions."""
    # Existing query types
    FACILITY_EMISSIONS = "facility_emissions"
    UTILITY_CONSUMPTION = "utility_consumption"
    WATER_CONSUMPTION = "water_consumption"
    WASTE_GENERATION = "waste_generation"
    COMPLIANCE_STATUS = "compliance_status"
    TREND_ANALYSIS = "trend_analysis"
    CUSTOM = "custom"
    
    # Phase 1 query types
    UTILITY_ALLOCATIONS = "utility_allocations"
    MONTHLY_USAGE_REPORT = "monthly_usage_report"
    REJECTED_DOCUMENTS = "rejected_documents"


class DataExtractionWorkflowEnhanced:
    """
    Enhanced LangGraph workflow for extracting and reporting on EHS data from Neo4j.
    Includes Phase 1 enhancements for audit trail, pro-rating, and enhanced reporting.
    """
    
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_username: str,
        neo4j_password: str,
        llm_model: str = "gpt-4",
        output_dir: str = "./reports",
        enable_phase1_features: bool = True
    ):
        """
        Initialize the enhanced data extraction workflow.
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            llm_model: LLM model to use for analysis
            output_dir: Directory for output reports
            enable_phase1_features: Enable Phase 1 enhancements
        """
        # Neo4j connection
        self.driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_username, neo4j_password)
        )
        
        # Configure LLM
        if "claude" in llm_model.lower():
            self.llm = ChatAnthropic(model=llm_model)
        else:
            self.llm = ChatOpenAI(model=llm_model, temperature=0)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Phase 1 service initialization
        self.enable_phase1 = enable_phase1_features
        if self.enable_phase1:
            try:
                self.graph = create_graph_database_connection(
                    neo4j_uri, neo4j_username, neo4j_password, "neo4j"
                )
                
                # Initialize Phase 1 services
                self.audit_trail_service = AuditTrailService(self.graph)
                self.rejection_service = RejectionWorkflowService(self.graph)
                self.prorating_service = ProRatingService(self.graph)
                
                logger.info("Phase 1 services initialized for extraction workflow")
            except Exception as e:
                logger.warning(f"Phase 1 services initialization failed: {str(e)}")
                self.enable_phase1 = False
        
        # Pre-defined query templates with Phase 1 enhancements
        self.query_templates = self._initialize_query_templates()
        
        # Build enhanced workflow graph
        self.workflow = self._build_workflow()
    
    def _initialize_query_templates(self) -> Dict[str, List[str]]:
        """Initialize pre-defined Cypher query templates with Phase 1 additions."""
        templates = {
            QueryType.FACILITY_EMISSIONS: [
                """
                MATCH (f:Facility)
                OPTIONAL MATCH (f)<-[:BILLED_TO]-(b:UtilityBill)
                OPTIONAL MATCH (b)-[:RESULTED_IN]->(e:Emission)
                RETURN f, b, e
                ORDER BY b.billing_period_end DESC
                """,
                """
                MATCH (f:Facility)
                MATCH (f)<-[:BILLED_TO]-(b:UtilityBill)-[:RESULTED_IN]->(e:Emission)
                RETURN f.name as facility, 
                       SUM(e.amount) as total_emissions,
                       e.unit as unit,
                       COUNT(b) as bill_count
                """
            ],
            QueryType.UTILITY_CONSUMPTION: [
                """
                MATCH (d:Document)-[:EXTRACTED_TO]->(b:UtilityBill)
                WHERE 'Utilitybill' IN labels(d) OR 'UtilityBill' IN labels(d)
                RETURN d, b
                ORDER BY b.billing_period_end DESC
                """,
                """
                MATCH (b:UtilityBill)
                WHERE b.billing_period_start >= $start_date 
                  AND b.billing_period_end <= $end_date
                RETURN SUM(b.total_kwh) as total_consumption,
                       AVG(b.total_cost) as avg_cost,
                       COUNT(b) as bill_count
                """
            ],
            QueryType.WATER_CONSUMPTION: [
                """
                MATCH (d:Document)-[:EXTRACTED_TO]->(w:WaterBill)
                WHERE 'Waterbill' IN labels(d) OR 'WaterBill' IN labels(d)
                RETURN d, w
                ORDER BY w.billing_period_end DESC
                """,
                """
                MATCH (w:WaterBill)
                RETURN w
                ORDER BY w.billing_period_end DESC
                """,
                """
                MATCH (w:WaterBill)
                OPTIONAL MATCH (w)-[:BILLED_TO]->(f:Facility)
                OPTIONAL MATCH (w)-[:BILLED_FOR]->(c:Customer)
                OPTIONAL MATCH (w)-[:PROVIDED_BY]->(p:UtilityProvider)
                OPTIONAL MATCH (w)-[:RESULTED_IN]->(e:Emission)
                OPTIONAL MATCH (m:Meter)-[:RECORDED_IN]->(w)
                RETURN w, f, c, p, e, collect(m) as meters
                ORDER BY w.billing_period_end DESC
                """,
                """
                MATCH (w:WaterBill)
                WHERE w.billing_period_start >= $start_date 
                  AND w.billing_period_end <= $end_date
                RETURN SUM(w.total_gallons) as total_water_usage,
                       AVG(w.total_cost) as avg_cost,
                       COUNT(w) as bill_count
                """
            ],
            QueryType.WASTE_GENERATION: [
                """
                MATCH (d:Document)-[:TRACKS]->(wm:WasteManifest)
                WHERE 'Wastemanifest' IN labels(d) OR 'WasteManifest' IN labels(d)
                MATCH (wm)-[:DOCUMENTS]->(ws:WasteShipment)
                MATCH (ws)-[:GENERATED_BY]->(g:WasteGenerator)
                MATCH (ws)-[:TRANSPORTED_BY]->(t:Transporter)  
                MATCH (ws)-[:DISPOSED_AT]->(df:DisposalFacility)
                OPTIONAL MATCH (ws)-[:CONTAINS_WASTE]->(wi:WasteItem)
                OPTIONAL MATCH (ws)-[:RESULTED_IN]->(e:Emission)
                RETURN d, wm, ws, g, t, df, collect(wi) as waste_items, collect(e) as emissions
                ORDER BY ws.shipment_date DESC
                """,
                """
                MATCH (wm:WasteManifest)-[:DOCUMENTS]->(ws:WasteShipment)
                MATCH (ws)-[:GENERATED_BY]->(g:WasteGenerator)
                MATCH (ws)-[:CONTAINS_WASTE]->(wi:WasteItem)
                OPTIONAL MATCH (ws)-[:RESULTED_IN]->(e:Emission)
                WHERE ws.shipment_date >= $start_date 
                  AND ws.shipment_date <= $end_date
                RETURN g.name as generator,
                       g.epa_id as generator_epa_id,
                       SUM(wi.quantity) as total_waste_quantity,
                       wi.unit as quantity_unit,
                       COUNT(DISTINCT wm) as manifest_count,
                       COUNT(wi) as waste_item_count,
                       SUM(e.amount) as total_emissions
                ORDER BY total_waste_quantity DESC
                """,
                """
                MATCH (wm:WasteManifest)-[:DOCUMENTS]->(ws:WasteShipment)
                MATCH (ws)-[:DISPOSED_AT]->(df:DisposalFacility)
                MATCH (ws)-[:CONTAINS_WASTE]->(wi:WasteItem)
                OPTIONAL MATCH (ws)-[:RESULTED_IN]->(e:Emission)
                WHERE ws.shipment_date >= $start_date 
                  AND ws.shipment_date <= $end_date
                RETURN df.name as disposal_facility,
                       df.epa_id as facility_epa_id,
                       df.state as facility_state,
                       COUNT(DISTINCT wm) as manifests_received,
                       SUM(wi.quantity) as total_waste_received,
                       wi.unit as quantity_unit,
                       collect(DISTINCT wi.disposal_method) as disposal_methods,
                       SUM(e.amount) as total_emissions_from_disposal
                ORDER BY total_waste_received DESC
                """,
                """
                MATCH (wm:WasteManifest)-[:DOCUMENTS]->(ws:WasteShipment)
                MATCH (ws)-[:TRANSPORTED_BY]->(t:Transporter)
                MATCH (ws)-[:CONTAINS_WASTE]->(wi:WasteItem)
                WHERE ws.shipment_date >= $start_date 
                  AND ws.shipment_date <= $end_date
                RETURN t.name as transporter,
                       t.epa_id as transporter_epa_id,
                       COUNT(DISTINCT wm) as manifests_transported,
                       SUM(wi.quantity) as total_waste_transported,
                       wi.unit as quantity_unit,
                       COUNT(DISTINCT ws) as unique_shipments_transported,
                       collect(DISTINCT ws.transport_method) as transport_methods
                ORDER BY total_waste_transported DESC
                """,
                """
                MATCH (wm:WasteManifest)-[:DOCUMENTS]->(ws:WasteShipment)
                MATCH (ws)-[:CONTAINS_WASTE]->(wi:WasteItem)
                WHERE ws.shipment_date >= $start_date 
                  AND ws.shipment_date <= $end_date
                RETURN wi.waste_code as waste_code,
                       wi.description as waste_description,
                       COUNT(wi) as item_count,
                       SUM(wi.quantity) as total_quantity,
                       wi.unit as quantity_unit,
                       collect(DISTINCT wi.disposal_method) as disposal_methods
                ORDER BY total_quantity DESC
                """,
                """
                MATCH (wm:WasteManifest)-[:DOCUMENTS]->(ws:WasteShipment)
                WHERE ws.shipment_date >= $start_date 
                  AND ws.shipment_date <= $end_date
                WITH date(ws.shipment_date) as shipment_date, 
                     wm, 
                     ws,
                     [(ws)-[:CONTAINS_WASTE]->(wi:WasteItem) | wi] as waste_items
                RETURN shipment_date,
                       COUNT(DISTINCT wm) as manifests_generated,
                       SUM(reduce(total = 0, wi in waste_items | total + wi.quantity)) as daily_waste_total,
                       AVG(ws.total_weight) as avg_shipment_weight
                ORDER BY shipment_date DESC
                """,
                """
                MATCH (wm:WasteManifest)-[:DOCUMENTS]->(ws:WasteShipment)
                MATCH (ws)-[:RESULTED_IN]->(e:Emission)
                MATCH (ws)-[:CONTAINS_WASTE]->(wi:WasteItem)
                WHERE ws.shipment_date >= $start_date 
                  AND ws.shipment_date <= $end_date
                RETURN e.emission_type as emission_type,
                       e.unit as emission_unit,
                       COUNT(e) as emission_records,
                       SUM(e.amount) as total_emissions,
                       AVG(e.amount) as avg_emissions_per_record,
                       SUM(wi.quantity) as associated_waste_quantity,
                       AVG(e.amount / wi.quantity) as avg_emission_factor
                ORDER BY total_emissions DESC
                """,
                """
                MATCH (d:Document:Wastemanifest)-[:TRACKS]->(wm:WasteManifest)
                MATCH (wm)-[:DOCUMENTS]->(ws:WasteShipment)
                WHERE d.status IS NOT NULL
                OPTIONAL MATCH (ws)-[:CONTAINS_WASTE]->(wi:WasteItem)
                RETURN d.status as manifest_status,
                       COUNT(DISTINCT wm) as manifest_count,
                       SUM(wi.quantity) as total_waste_quantity,
                       MIN(ws.shipment_date) as earliest_shipment_date,
                       MAX(ws.shipment_date) as latest_shipment_date,
                       MIN(wm.issue_date) as earliest_issue_date,
                       MAX(wm.issue_date) as latest_issue_date,
                       COUNT(DISTINCT ws) as unique_shipments
                ORDER BY manifest_count DESC
                """
            ],
            QueryType.COMPLIANCE_STATUS: [
                """
                MATCH (p:Permit)
                OPTIONAL MATCH (p)<-[:PERMITS]-(f:Facility)
                WHERE p.expiry_date < datetime()
                RETURN p, f, 'expired' as status
                """,
                """
                MATCH (p:Permit)
                WHERE p.expiry_date > datetime() 
                  AND p.expiry_date < datetime() + duration('P90D')
                RETURN p, 'expiring_soon' as status
                """
            ],
            QueryType.TREND_ANALYSIS: [
                """
                MATCH (b:UtilityBill)
                RETURN b.billing_period_end as period,
                       b.total_kwh as consumption,
                       b.total_cost as cost
                ORDER BY b.billing_period_end
                """,
                """
                MATCH (b:UtilityBill)-[:RESULTED_IN]->(e:Emission)
                RETURN b.billing_period_end as period,
                       e.amount as emissions
                ORDER BY b.billing_period_end
                """
            ],
            
            # Phase 1 Enhanced Query Templates
            QueryType.UTILITY_ALLOCATIONS: [
                """
                MATCH (pa:ProRatingAllocation)
                OPTIONAL MATCH (pa)-[:ALLOCATED_TO]->(facility:Facility)
                OPTIONAL MATCH (pa)-[:BASED_ON]->(doc:Document)
                WHERE pa.period_start >= $start_date AND pa.period_end <= $end_date
                RETURN pa, facility, doc
                ORDER BY pa.created_at DESC
                """,
                """
                MATCH (pa:ProRatingAllocation)-[:HAS_DISTRIBUTION]->(pd:ProRatingDistribution)
                MATCH (pd)-[:ALLOCATED_TO]->(facility:Facility)
                WHERE pa.period_start >= $start_date AND pa.period_end <= $end_date
                RETURN facility.name as facility_name,
                       SUM(pd.allocated_amount) as total_allocated,
                       pa.utility_type as utility_type,
                       COUNT(pd) as distribution_count,
                       AVG(pd.allocated_percentage) as avg_percentage
                ORDER BY total_allocated DESC
                """
            ],
            
            QueryType.MONTHLY_USAGE_REPORT: [
                """
                MATCH (pa:ProRatingAllocation)
                WHERE pa.period_start >= $start_date AND pa.period_end <= $end_date
                MATCH (pa)-[:HAS_DISTRIBUTION]->(pd:ProRatingDistribution)
                MATCH (pd)-[:ALLOCATED_TO]->(facility:Facility)
                RETURN 
                    date(pa.period_start) as period_start,
                    date(pa.period_end) as period_end,
                    pa.utility_type as utility_type,
                    pa.total_amount as total_amount,
                    pa.allocation_method as method,
                    collect({
                        facility_name: facility.name,
                        allocated_amount: pd.allocated_amount,
                        allocated_percentage: pd.allocated_percentage,
                        allocation_basis: pd.allocation_basis
                    }) as distributions
                ORDER BY pa.period_start DESC
                """
            ],
            
            QueryType.REJECTED_DOCUMENTS: [
                """
                MATCH (rr:RejectionRecord)
                OPTIONAL MATCH (rr)-[:REJECTED_DOCUMENT]->(d:Document)
                WHERE rr.created_at >= $start_date AND rr.created_at <= $end_date
                RETURN rr, d
                ORDER BY rr.created_at DESC
                """,
                """
                MATCH (rr:RejectionRecord)
                WHERE rr.created_at >= $start_date AND rr.created_at <= $end_date
                RETURN 
                    rr.rejection_reason as reason,
                    rr.rejection_status as status,
                    COUNT(rr) as count,
                    AVG(rr.quality_score) as avg_quality_score
                ORDER BY count DESC
                """
            ]
        }
        
        return templates
    
    def _build_workflow(self) -> StateGraph:
        """Build the enhanced LangGraph workflow with Phase 1 nodes."""
        # Create workflow
        workflow = StateGraph(ExtractionState)
        
        # Add existing nodes
        workflow.add_node("prepare_queries", self.prepare_queries)
        workflow.add_node("identify_objects", self.identify_graph_objects)
        workflow.add_node("execute_queries", self.execute_queries)
        workflow.add_node("analyze_results", self.analyze_results)
        workflow.add_node("generate_report", self.generate_report)
        workflow.add_node("save_report", self.save_report)
        workflow.add_node("complete", self.complete_processing)
        workflow.add_node("handle_error", self.handle_error)
        
        # Add Phase 1 nodes
        if self.enable_phase1:
            workflow.add_node("calculate_prorations", self.calculate_prorations)
            workflow.add_node("generate_allocation_report", self.generate_allocation_report)
            workflow.add_node("include_audit_trail", self.include_audit_trail)
        
        # Add edges with conditional Phase 1 routing
        workflow.add_edge("prepare_queries", "identify_objects")
        workflow.add_edge("identify_objects", "execute_queries")
        
        if self.enable_phase1:
            # Enhanced workflow with Phase 1 features
            workflow.add_conditional_edges(
                "execute_queries",
                self.check_phase1_processing_needed,
                {
                    "prorating_needed": "calculate_prorations",
                    "standard_analysis": "analyze_results"
                }
            )
            workflow.add_edge("calculate_prorations", "analyze_results")
        else:
            # Standard workflow
            workflow.add_edge("execute_queries", "analyze_results")
        
        workflow.add_edge("analyze_results", "generate_report")
        
        if self.enable_phase1:
            workflow.add_conditional_edges(
                "generate_report",
                self.check_audit_trail_needed,
                {
                    "include_audit": "include_audit_trail",
                    "skip_audit": "save_report"
                }
            )
            workflow.add_edge("include_audit_trail", "save_report")
        else:
            workflow.add_edge("generate_report", "save_report")
        
        workflow.add_edge("save_report", "complete")
        workflow.add_edge("complete", END)
        
        # Error handling
        workflow.add_conditional_edges(
            "handle_error",
            lambda state: "fail" if len(state["errors"]) > 3 else "retry",
            {
                "fail": END,
                "retry": "prepare_queries"
            }
        )
        
        # Set entry point
        workflow.set_entry_point("prepare_queries")
        
        # Compile
        return workflow.compile()
    
    def prepare_queries(self, state: ExtractionState) -> ExtractionState:
        """
        Prepare Cypher queries based on query configuration with Phase 1 enhancements.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        logger.info(f"Preparing enhanced queries for type: {state['query_config'].get('type')}")
        
        try:
            query_type = state["query_config"].get("type", QueryType.CUSTOM)
            
            # Initialize Phase 1 processing
            state["phase1_processing"] = {}
            state["prorating_allocations"] = []
            state["audit_trail_entries"] = []
            state["include_source_files"] = state["query_config"].get("include_source_files", False)
            
            if query_type == QueryType.CUSTOM:
                # Use custom queries provided
                queries = state["query_config"].get("queries", [])
            else:
                # Use predefined templates
                template_queries = self.query_templates.get(query_type, [])
                queries = []
                
                for template in template_queries:
                    # Substitute parameters if provided
                    params = state["query_config"].get("parameters", {})
                    queries.append({
                        "query": template,
                        "parameters": params
                    })
            
            state["queries"] = queries
            state["status"] = "processing"
            
            # Create audit trail for extraction process
            if self.enable_phase1:
                try:
                    audit_entry = self.audit_trail_service.create_audit_trail(
                        document_name=f"extraction_request_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                        action="extraction_started",
                        user_id=state["query_config"].get("user_id", "system"),
                        metadata={
                            "query_type": query_type,
                            "query_count": len(queries),
                            "output_format": state["output_format"]
                        }
                    )
                    state["audit_trail_entries"].append({
                        "audit_id": audit_entry.audit_id,
                        "action": "extraction_started",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    state["phase1_processing"]["audit_trail"] = "created"
                except Exception as e:
                    logger.warning(f"Failed to create audit trail: {str(e)}")
            
            logger.info(f"Prepared {len(queries)} queries")
            
        except Exception as e:
            state["errors"].append(f"Query preparation error: {str(e)}")
            logger.error(f"Failed to prepare queries: {str(e)}")
        
        return state
    
    def identify_graph_objects(self, state: ExtractionState) -> ExtractionState:
        """
        Identify Neo4j objects that will be accessed by the queries.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        logger.info("Identifying graph objects involved in queries")
        
        try:
            graph_objects = []
            
            with self.driver.session() as session:
                # Get schema information
                schema_query = """
                CALL db.schema.visualization()
                """
                
                # Get node counts by label
                node_count_query = """
                MATCH (n)
                RETURN labels(n) as labels, count(n) as count
                """
                
                # Get relationship counts
                rel_count_query = """
                MATCH ()-[r]->()
                RETURN type(r) as type, count(r) as count
                """
                
                # Execute schema queries
                try:
                    node_counts = session.run(node_count_query).data()
                    rel_counts = session.run(rel_count_query).data()
                    
                    graph_objects = {
                        "nodes": node_counts,
                        "relationships": rel_counts,
                        "total_nodes": sum(item["count"] for item in node_counts),
                        "total_relationships": sum(item["count"] for item in rel_counts)
                    }
                    
                    # Phase 1: Add Phase 1 node counts if enabled
                    if self.enable_phase1:
                        phase1_nodes = session.run("""
                        MATCH (n)
                        WHERE any(label IN labels(n) WHERE label IN ['AuditTrailEntry', 'RejectionRecord', 'ProRatingAllocation', 'ProRatingDistribution'])
                        RETURN labels(n) as labels, count(n) as count
                        """).data()
                        
                        graph_objects["phase1_nodes"] = phase1_nodes
                        
                except Exception as e:
                    logger.warning(f"Could not fetch schema info: {str(e)}")
                    graph_objects = {"error": "Could not fetch schema information"}
            
            state["graph_objects"] = [graph_objects]
            
        except Exception as e:
            state["errors"].append(f"Object identification error: {str(e)}")
            logger.error(f"Failed to identify objects: {str(e)}")
        
        return state
    
    def _serialize_neo4j_value(self, value):
        """
        Convert Neo4j objects to serializable format.
        
        Args:
            value: Any value that might be a Neo4j object
            
        Returns:
            Serializable representation of the value
        """
        if isinstance(value, Node):
            # Convert Node to dict
            return {
                'id': value.id,
                'labels': list(value.labels),
                'properties': dict(value)
            }
        elif isinstance(value, Relationship):
            # Convert Relationship to dict
            return {
                'id': value.id,
                'type': value.type,
                'start_node': value.start_node.id,
                'end_node': value.end_node.id,
                'properties': dict(value)
            }
        elif isinstance(value, list):
            # Recursively process lists
            return [self._serialize_neo4j_value(item) for item in value]
        elif isinstance(value, dict):
            # Recursively process dictionaries
            return {k: self._serialize_neo4j_value(v) for k, v in value.items()}
        else:
            # Return primitive values as-is
            return value
    
    def execute_queries(self, state: ExtractionState) -> ExtractionState:
        """
        Execute Cypher queries against Neo4j with enhanced Phase 1 support.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        logger.info(f"Executing {len(state['queries'])} enhanced queries")
        
        query_results = []
        
        try:
            with self.driver.session() as session:
                for i, query_info in enumerate(state["queries"]):
                    query = query_info["query"]
                    parameters = query_info.get("parameters", {})
                    
                    logger.info(f"Executing query {i+1}")
                    
                    try:
                        # Execute query
                        result = session.run(query, parameters)
                        records = []
                        
                        # Process results with proper serialization
                        for record in result:
                            # Convert the entire record to a serializable format
                            record_dict = {}
                            for key, value in dict(record).items():
                                record_dict[key] = self._serialize_neo4j_value(value)
                            records.append(record_dict)
                        
                        query_results.append({
                            "query": query,
                            "parameters": parameters,
                            "results": records,
                            "record_count": len(records),
                            "status": "success"
                        })
                        
                        logger.info(f"Query {i+1} returned {len(records)} records")
                        
                        # Phase 1: Update audit trail
                        if self.enable_phase1 and state["audit_trail_entries"]:
                            try:
                                self.audit_trail_service.add_activity(
                                    state["audit_trail_entries"][0]["audit_id"],
                                    f"query_{i+1}_executed",
                                    details={
                                        "query_type": state["query_config"].get("type"),
                                        "record_count": len(records),
                                        "query_index": i+1
                                    }
                                )
                            except Exception as e:
                                logger.warning(f"Failed to update audit trail: {str(e)}")
                        
                    except Exception as e:
                        query_results.append({
                            "query": query,
                            "parameters": parameters,
                            "error": str(e),
                            "status": "failed"
                        })
                        logger.error(f"Query {i+1} failed: {str(e)}")
            
            state["query_results"] = query_results
            
        except Exception as e:
            state["errors"].append(f"Query execution error: {str(e)}")
            logger.error(f"Failed to execute queries: {str(e)}")
        
        return state
    
    def check_phase1_processing_needed(self, state: ExtractionState) -> str:
        """Check if Phase 1 specific processing is needed."""
        if not self.enable_phase1:
            return "standard_analysis"
        
        query_type = state["query_config"].get("type")
        
        # Check if pro-rating calculation is needed
        if query_type in [QueryType.UTILITY_ALLOCATIONS, QueryType.MONTHLY_USAGE_REPORT]:
            return "prorating_needed"
        
        return "standard_analysis"
    
    def calculate_prorations(self, state: ExtractionState) -> ExtractionState:
        """
        Calculate pro-rating allocations for utility bill reports.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        logger.info("Calculating pro-rating allocations")
        
        try:
            if not self.enable_phase1:
                logger.warning("Phase 1 features not enabled, skipping pro-rating calculation")
                return state
            
            query_type = state["query_config"].get("type")
            
            if query_type in [QueryType.UTILITY_ALLOCATIONS, QueryType.MONTHLY_USAGE_REPORT]:
                # Extract allocation data from query results
                allocation_data = []
                
                for result in state["query_results"]:
                    if result["status"] == "success":
                        for record in result["results"]:
                            # Process pro-rating allocation data
                            if "pa" in record:  # ProRatingAllocation node
                                allocation_info = self._serialize_neo4j_value(record["pa"])
                                allocation_data.append(allocation_info)
                
                # Calculate additional pro-rating metrics if needed
                if allocation_data:
                    # Calculate summary metrics
                    total_allocations = len(allocation_data)
                    total_amount = sum(
                        allocation.get("properties", {}).get("total_amount", 0) 
                        for allocation in allocation_data
                    )
                    
                    # Group by utility type
                    utility_summary = {}
                    for allocation in allocation_data:
                        props = allocation.get("properties", {})
                        utility_type = props.get("utility_type", "unknown")
                        if utility_type not in utility_summary:
                            utility_summary[utility_type] = {
                                "count": 0,
                                "total_amount": 0,
                                "allocations": []
                            }
                        utility_summary[utility_type]["count"] += 1
                        utility_summary[utility_type]["total_amount"] += props.get("total_amount", 0)
                        utility_summary[utility_type]["allocations"].append(allocation)
                    
                    state["prorating_allocations"] = allocation_data
                    state["phase1_processing"]["prorating"] = {
                        "total_allocations": total_allocations,
                        "total_amount": total_amount,
                        "utility_summary": utility_summary,
                        "calculated_at": datetime.utcnow().isoformat()
                    }
                    
                    logger.info(f"Calculated pro-rating data for {total_allocations} allocations")
                    
                    # Update audit trail
                    if state["audit_trail_entries"]:
                        try:
                            self.audit_trail_service.add_activity(
                                state["audit_trail_entries"][0]["audit_id"],
                                "prorating_calculated",
                                details={
                                    "total_allocations": total_allocations,
                                    "total_amount": total_amount,
                                    "utility_types": list(utility_summary.keys())
                                }
                            )
                        except Exception as e:
                            logger.warning(f"Failed to update audit trail: {str(e)}")
                else:
                    logger.info("No pro-rating allocation data found in query results")
            
        except Exception as e:
            state["errors"].append(f"Pro-rating calculation error: {str(e)}")
            logger.error(f"Failed to calculate pro-rating data: {str(e)}")
        
        return state
    
    def generate_allocation_report(self, state: ExtractionState) -> ExtractionState:
        """
        Generate allocation-specific report data.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        logger.info("Generating allocation report")
        
        try:
            if not self.enable_phase1 or not state["prorating_allocations"]:
                return state
            
            # Generate allocation-specific report sections
            allocation_report = {
                "allocation_summary": state["phase1_processing"].get("prorating", {}),
                "detailed_allocations": [],
                "facility_breakdown": {},
                "period_analysis": {}
            }
            
            # Process each allocation for detailed reporting
            for allocation in state["prorating_allocations"]:
                props = allocation.get("properties", {})
                
                detailed_info = {
                    "allocation_id": allocation.get("id"),
                    "period_start": props.get("period_start"),
                    "period_end": props.get("period_end"),
                    "utility_type": props.get("utility_type"),
                    "total_amount": props.get("total_amount"),
                    "allocation_method": props.get("allocation_method"),
                    "created_at": props.get("created_at")
                }
                
                allocation_report["detailed_allocations"].append(detailed_info)
            
            # Add allocation report to main report data
            if "report_data" not in state:
                state["report_data"] = {}
            
            state["report_data"]["allocation_report"] = allocation_report
            state["phase1_processing"]["allocation_report_generated"] = True
            
            logger.info("Allocation report generated successfully")
            
        except Exception as e:
            state["errors"].append(f"Allocation report generation error: {str(e)}")
            logger.error(f"Failed to generate allocation report: {str(e)}")
        
        return state
    
    def analyze_results(self, state: ExtractionState) -> ExtractionState:
        """
        Use LLM to analyze query results and generate insights with Phase 1 enhancements.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        logger.info("Analyzing query results with enhanced LLM analysis")
        
        try:
            # Prepare context for LLM
            context = {
                "query_type": state["query_config"].get("type"),
                "total_queries": len(state["queries"]),
                "successful_queries": sum(1 for r in state["query_results"] if r["status"] == "success"),
                "total_records": sum(r.get("record_count", 0) for r in state["query_results"])
            }
            
            # Add Phase 1 context if available
            if self.enable_phase1:
                context["phase1_features"] = {
                    "prorating_data_available": len(state["prorating_allocations"]) > 0,
                    "audit_trail_active": len(state["audit_trail_entries"]) > 0,
                    "prorating_summary": state["phase1_processing"].get("prorating", {})
                }
            
            # Create enhanced analysis prompt
            prompt = f"""
            Analyze the following enhanced EHS data extraction results and provide insights:
            
            Query Type: {context['query_type']}
            Total Queries Executed: {context['total_queries']}
            Successful Queries: {context['successful_queries']}
            Total Records Retrieved: {context['total_records']}
            
            Phase 1 Enhancement Status:
            - Pro-rating Data Available: {context.get('phase1_features', {}).get('prorating_data_available', False)}
            - Audit Trail Active: {context.get('phase1_features', {}).get('audit_trail_active', False)}
            
            Query Results:
            {json.dumps(state['query_results'], indent=2)[:4000]}  # Increased limit for Phase 1 data
            
            Pro-rating Data (if available):
            {json.dumps(state['prorating_allocations'][:5], indent=2) if state['prorating_allocations'] else 'No pro-rating data'}
            
            Please provide:
            1. Key findings from the data
            2. Any patterns or trends observed
            3. Potential data quality issues
            4. Pro-rating allocation insights (if applicable)
            5. Recommendations based on the analysis
            6. Phase 1 feature utilization assessment
            
            Format your response as structured JSON with enhanced Phase 1 insights.
            """
            
            # Log enhanced LLM analysis request
            try:
                forward_transcript_entry(
                    role="user",
                    content=f"Enhanced EHS Data Analysis Request (with Phase 1 features):\n{prompt[:1000]}... (truncated to 1000 chars)",
                    context={
                        "component": "extraction_workflow_enhanced",
                        "function": "analyze_results",
                        "query_type": state.get('query_type', 'unknown'),
                        "result_count": len(state.get('query_results', [])),
                        "has_prorating": bool(state.get('prorating_allocations')),
                        "timestamp": datetime.now().isoformat()
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to forward enhanced analysis request transcript: {e}")
            
            # Get LLM analysis
            response = self.llm.invoke([
                SystemMessage(content="You are an enhanced EHS data analyst with Phase 1 capabilities. Provide structured analysis of query results including pro-rating and audit insights."),
                HumanMessage(content=prompt)
            ])
            
            # Log enhanced LLM analysis response
            try:
                forward_transcript_entry(
                    role="assistant",
                    content=response.content,
                    context={
                        "component": "extraction_workflow_enhanced",
                        "function": "analyze_results",
                        "timestamp": datetime.now().isoformat()
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to forward enhanced analysis response transcript: {e}")
            
            # Parse response
            try:
                analysis = json.loads(response.content)
            except:
                analysis = {
                    "summary": response.content,
                    "findings": ["Analysis completed but could not parse structured response"],
                    "phase1_insights": {
                        "prorating_analysis": "Available but not parsed",
                        "audit_trail_summary": "Active"
                    }
                }
            
            # Enhanced analysis with Phase 1 data
            if self.enable_phase1:
                analysis["phase1_analysis"] = {
                    "prorating_summary": state["phase1_processing"].get("prorating", {}),
                    "audit_trail_count": len(state["audit_trail_entries"]),
                    "processing_metadata": state["phase1_processing"]
                }
            
            state["analysis_results"] = analysis
            
            # Update audit trail
            if self.enable_phase1 and state["audit_trail_entries"]:
                try:
                    self.audit_trail_service.add_activity(
                        state["audit_trail_entries"][0]["audit_id"],
                        "analysis_completed",
                        details={
                            "analysis_type": "enhanced_llm_analysis",
                            "findings_count": len(analysis.get("findings", [])),
                            "phase1_features_used": True
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to update audit trail: {str(e)}")
            
        except Exception as e:
            state["errors"].append(f"Analysis error: {str(e)}")
            logger.error(f"Failed to analyze results: {str(e)}")
            state["analysis_results"] = {"error": "Analysis failed"}
        
        return state
    
    def generate_report(self, state: ExtractionState) -> ExtractionState:
        """
        Generate enhanced structured report from query results and analysis.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        logger.info(f"Generating enhanced report in format: {state['output_format']}")
        
        try:
            # Compile enhanced report data
            report_data = {
                "metadata": {
                    "title": f"Enhanced EHS Data Extraction Report - {state['query_config'].get('type', 'Custom')}",
                    "generated_at": datetime.utcnow().isoformat(),
                    "query_type": state["query_config"].get("type"),
                    "parameters": state["query_config"].get("parameters", {}),
                    "output_format": state["output_format"],
                    "phase1_features_enabled": self.enable_phase1
                },
                "summary": {
                    "total_queries": len(state["queries"]),
                    "successful_queries": sum(1 for r in state["query_results"] if r["status"] == "success"),
                    "failed_queries": sum(1 for r in state["query_results"] if r["status"] == "failed"),
                    "total_records": sum(r.get("record_count", 0) for r in state["query_results"]),
                    "graph_objects": state["graph_objects"]
                },
                "queries_executed": state["queries"],
                "query_results": state["query_results"],
                "analysis": state.get("analysis_results", {}),
                "errors": state["errors"]
            }
            
            # Add Phase 1 enhancements to report
            if self.enable_phase1:
                report_data["phase1_enhancements"] = {
                    "prorating_allocations": {
                        "count": len(state["prorating_allocations"]),
                        "data": state["prorating_allocations"],
                        "summary": state["phase1_processing"].get("prorating", {})
                    },
                    "audit_trail": {
                        "entries": state["audit_trail_entries"],
                        "active": len(state["audit_trail_entries"]) > 0
                    },
                    "processing_metadata": state["phase1_processing"],
                    "source_files_included": state["include_source_files"]
                }
                
                # Add allocation report if generated
                if state["report_data"].get("allocation_report"):
                    report_data["allocation_report"] = state["report_data"]["allocation_report"]
            
            state["report_data"] = report_data
            
            logger.info("Enhanced report data generated successfully")
            
        except Exception as e:
            state["errors"].append(f"Report generation error: {str(e)}")
            logger.error(f"Failed to generate report: {str(e)}")
        
        return state
    
    def check_audit_trail_needed(self, state: ExtractionState) -> str:
        """Check if audit trail inclusion is needed."""
        if not self.enable_phase1:
            return "skip_audit"
        
        if state["query_config"].get("include_audit_trail", False) or state["audit_trail_entries"]:
            return "include_audit"
        
        return "skip_audit"
    
    def include_audit_trail(self, state: ExtractionState) -> ExtractionState:
        """
        Include audit trail information in the final report.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        logger.info("Including audit trail in report")
        
        try:
            if not self.enable_phase1 or not state["audit_trail_entries"]:
                return state
            
            # Finalize audit trail with completion
            if state["audit_trail_entries"]:
                try:
                    self.audit_trail_service.add_activity(
                        state["audit_trail_entries"][0]["audit_id"],
                        "report_generation_completed",
                        details={
                            "report_format": state["output_format"],
                            "total_records_processed": sum(r.get("record_count", 0) for r in state["query_results"]),
                            "phase1_features_used": list(state["phase1_processing"].keys())
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to finalize audit trail: {str(e)}")
            
            # Add audit trail summary to report
            audit_summary = {
                "audit_entries_count": len(state["audit_trail_entries"]),
                "extraction_start_time": state["audit_trail_entries"][0]["timestamp"] if state["audit_trail_entries"] else None,
                "extraction_completed": datetime.utcnow().isoformat(),
                "activities_tracked": True
            }
            
            if "report_data" in state:
                state["report_data"]["audit_summary"] = audit_summary
            
            logger.info("Audit trail successfully included in report")
            
        except Exception as e:
            state["errors"].append(f"Audit trail inclusion error: {str(e)}")
            logger.error(f"Failed to include audit trail: {str(e)}")
        
        return state
    
    def save_report(self, state: ExtractionState) -> ExtractionState:
        """
        Save enhanced report to file in requested format.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        logger.info("Saving enhanced report to file")
        
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            query_type = state["query_config"].get("type", "custom")
            
            if state["output_format"] == "json":
                # Save as JSON
                file_name = f"ehs_enhanced_report_{query_type}_{timestamp}.json"
                file_path = self.output_dir / file_name
                
                with open(file_path, 'w') as f:
                    json.dump(state["report_data"], f, indent=2)
                
            elif state["output_format"] in ["txt", "text"]:
                # Save as formatted text with Phase 1 enhancements
                file_name = f"ehs_enhanced_report_{query_type}_{timestamp}.txt"
                file_path = self.output_dir / file_name
                
                with open(file_path, 'w') as f:
                    # Write header
                    f.write("=" * 80 + "\n")
                    f.write(state["report_data"]["metadata"]["title"] + "\n")
                    f.write("=" * 80 + "\n\n")
                    
                    # Write metadata
                    f.write(f"Generated: {state['report_data']['metadata']['generated_at']}\n")
                    f.write(f"Query Type: {state['report_data']['metadata']['query_type']}\n")
                    f.write(f"Phase 1 Features: {'Enabled' if state['report_data']['metadata']['phase1_features_enabled'] else 'Disabled'}\n\n")
                    
                    # Write summary
                    f.write("SUMMARY\n")
                    f.write("-" * 40 + "\n")
                    summary = state["report_data"]["summary"]
                    f.write(f"Total Queries: {summary['total_queries']}\n")
                    f.write(f"Successful: {summary['successful_queries']}\n")
                    f.write(f"Failed: {summary['failed_queries']}\n")
                    f.write(f"Total Records: {summary['total_records']}\n\n")
                    
                    # Write Phase 1 enhancements summary
                    if self.enable_phase1 and "phase1_enhancements" in state["report_data"]:
                        f.write("PHASE 1 ENHANCEMENTS\n")
                        f.write("-" * 40 + "\n")
                        phase1 = state["report_data"]["phase1_enhancements"]
                        f.write(f"Pro-rating Allocations: {phase1['prorating_allocations']['count']}\n")
                        f.write(f"Audit Trail Entries: {len(phase1['audit_trail']['entries'])}\n")
                        f.write(f"Source Files Included: {phase1['source_files_included']}\n\n")
                    
                    # Write graph objects
                    f.write("GRAPH OBJECTS\n")
                    f.write("-" * 40 + "\n")
                    f.write(json.dumps(summary['graph_objects'], indent=2) + "\n\n")
                    
                    # Write query results
                    f.write("QUERY RESULTS\n")
                    f.write("=" * 80 + "\n\n")
                    
                    for i, result in enumerate(state["report_data"]["query_results"]):
                        f.write(f"Query {i+1}\n")
                        f.write("-" * 40 + "\n")
                        f.write(f"Query: {result['query']}\n")
                        if result.get("parameters"):
                            f.write(f"Parameters: {json.dumps(result['parameters'])}\n")
                        f.write(f"Status: {result['status']}\n")
                        
                        if result["status"] == "success":
                            f.write(f"Records: {result['record_count']}\n")
                            if result["results"]:
                                f.write("Results (first 5):\n")
                                
                                # Enhanced formatting for different query types
                                if state["query_config"].get("type") == QueryType.WASTE_GENERATION:
                                    self._format_waste_results(f, result["results"][:5])
                                elif state["query_config"].get("type") in [QueryType.UTILITY_ALLOCATIONS, QueryType.MONTHLY_USAGE_REPORT]:
                                    self._format_prorating_results(f, result["results"][:5])
                                else:
                                    for j, record in enumerate(result["results"][:5]):
                                        f.write(f"  {j+1}: {json.dumps(record, indent=2)}\n")
                                
                                if result["record_count"] > 5:
                                    f.write(f"  ... and {result['record_count'] - 5} more\n")
                        else:
                            f.write(f"Error: {result.get('error', 'Unknown error')}\n")
                        
                        f.write("\n")
                    
                    # Write Phase 1 specific sections
                    if self.enable_phase1 and "phase1_enhancements" in state["report_data"]:
                        # Pro-rating allocation details
                        if state["report_data"]["phase1_enhancements"]["prorating_allocations"]["count"] > 0:
                            f.write("PRO-RATING ALLOCATIONS\n")
                            f.write("=" * 80 + "\n")
                            prorating_data = state["report_data"]["phase1_enhancements"]["prorating_allocations"]
                            f.write(f"Total Allocations: {prorating_data['count']}\n")
                            if prorating_data['summary']:
                                f.write(f"Total Amount: {prorating_data['summary'].get('total_amount', 0)}\n")
                                f.write(f"Utility Types: {', '.join(prorating_data['summary'].get('utility_summary', {}).keys())}\n")
                            f.write("\n")
                        
                        # Audit trail summary
                        if "audit_summary" in state["report_data"]:
                            f.write("AUDIT TRAIL SUMMARY\n")
                            f.write("=" * 80 + "\n")
                            audit = state["report_data"]["audit_summary"]
                            f.write(f"Audit Entries: {audit['audit_entries_count']}\n")
                            f.write(f"Extraction Started: {audit['extraction_start_time']}\n")
                            f.write(f"Extraction Completed: {audit['extraction_completed']}\n")
                            f.write("\n")
                    
                    # Write analysis
                    if state["report_data"].get("analysis"):
                        f.write("ENHANCED ANALYSIS\n")
                        f.write("=" * 80 + "\n")
                        f.write(json.dumps(state["report_data"]["analysis"], indent=2) + "\n")
            
            else:
                # Default to JSON for unsupported formats
                file_name = f"ehs_enhanced_report_{query_type}_{timestamp}.json"
                file_path = self.output_dir / file_name
                
                with open(file_path, 'w') as f:
                    json.dump(state["report_data"], f, indent=2)
            
            state["report_file_path"] = str(file_path)
            logger.info(f"Enhanced report saved to: {file_path}")
            
        except Exception as e:
            state["errors"].append(f"Save error: {str(e)}")
            logger.error(f"Failed to save report: {str(e)}")
        
        return state
    
    def _format_waste_results(self, f, results):
        """Format waste manifest results for text output."""
        for j, record in enumerate(results):
            f.write(f"  Record {j+1}:\n")
            
            # Handle waste manifest data formatting
            for key, value in record.items():
                if key in ['generator', 'transporter', 'disposal_facility'] and isinstance(value, dict):
                    props = value.get('properties', {})
                    epa_id = props.get('epa_id', 'N/A')
                    name = props.get('name', 'N/A')
                    f.write(f"    {key.replace('_', ' ').title()}: {name} (EPA ID: {epa_id})\n")
                elif key == 'waste_items' and isinstance(value, list):
                    f.write(f"    Waste Items ({len(value)} items):\n")
                    for idx, item in enumerate(value[:3]):  # Show first 3 items
                        if isinstance(item, dict) and 'properties' in item:
                            props = item['properties']
                            qty = props.get('quantity', 'N/A')
                            unit = props.get('unit', '')
                            desc = props.get('description', 'N/A')
                            hazardous = props.get('hazardous', False)
                            f.write(f"      {idx+1}. {desc} - {qty} {unit} {'(Hazardous)' if hazardous else '(Non-Hazardous)'}\n")
                    if len(value) > 3:
                        f.write(f"      ... and {len(value) - 3} more items\n")
                elif key == 'emissions' and isinstance(value, list):
                    if value:
                        f.write(f"    Emissions ({len(value)} records):\n")
                        for idx, emission in enumerate(value[:2]):  # Show first 2 emissions
                            if isinstance(emission, dict) and 'properties' in emission:
                                props = emission['properties']
                                amount = props.get('amount', 'N/A')
                                unit = props.get('unit', '')
                                emission_type = props.get('emission_type', 'N/A')
                                f.write(f"      {idx+1}. {emission_type}: {amount} {unit}\n")
                        if len(value) > 2:
                            f.write(f"      ... and {len(value) - 2} more emissions\n")
                elif isinstance(value, (str, int, float)):
                    # Format quantities with proper units
                    if 'quantity' in key.lower() and isinstance(value, (int, float)):
                        f.write(f"    {key.replace('_', ' ').title()}: {value:,.2f}\n")
                    elif 'epa_id' in key.lower():
                        f.write(f"    {key.replace('_', ' ').title()}: {value}\n")
                    else:
                        f.write(f"    {key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")
    
    def _format_prorating_results(self, f, results):
        """Format pro-rating allocation results for text output."""
        for j, record in enumerate(results):
            f.write(f"  Pro-rating Record {j+1}:\n")
            
            for key, value in record.items():
                if key == 'pa' and isinstance(value, dict):  # ProRatingAllocation node
                    props = value.get('properties', {})
                    f.write(f"    Allocation ID: {value.get('id', 'N/A')}\n")
                    f.write(f"    Period: {props.get('period_start', 'N/A')} to {props.get('period_end', 'N/A')}\n")
                    f.write(f"    Utility Type: {props.get('utility_type', 'N/A')}\n")
                    f.write(f"    Total Amount: ${props.get('total_amount', 0):,.2f}\n")
                    f.write(f"    Allocation Method: {props.get('allocation_method', 'N/A')}\n")
                elif key == 'facility' and isinstance(value, dict):
                    props = value.get('properties', {})
                    f.write(f"    Facility: {props.get('name', 'N/A')}\n")
                elif key == 'distributions' and isinstance(value, list):
                    f.write(f"    Distributions ({len(value)} facilities):\n")
                    for idx, dist in enumerate(value[:5]):  # Show first 5 distributions
                        f.write(f"      {idx+1}. {dist.get('facility_name', 'N/A')}: ")
                        f.write(f"${dist.get('allocated_amount', 0):,.2f} ")
                        f.write(f"({dist.get('allocated_percentage', 0):.1f}%)\n")
                    if len(value) > 5:
                        f.write(f"      ... and {len(value) - 5} more distributions\n")
                elif isinstance(value, (str, int, float)):
                    if 'amount' in key.lower() and isinstance(value, (int, float)):
                        f.write(f"    {key.replace('_', ' ').title()}: ${value:,.2f}\n")
                    elif 'percentage' in key.lower() and isinstance(value, (int, float)):
                        f.write(f"    {key.replace('_', ' ').title()}: {value:.1f}%\n")
                    else:
                        f.write(f"    {key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")
    
    def complete_processing(self, state: ExtractionState) -> ExtractionState:
        """Complete the enhanced extraction workflow."""
        state["status"] = "completed"
        state["processing_time"] = datetime.utcnow().timestamp() - state.get("start_time", 0)
        
        # Finalize Phase 1 processing
        if self.enable_phase1:
            state["phase1_processing"]["completed_at"] = datetime.utcnow().isoformat()
            state["phase1_processing"]["total_processing_time"] = state["processing_time"]
        
        logger.info(f"Enhanced extraction completed in {state['processing_time']:.2f} seconds")
        logger.info(f"Report saved to: {state.get('report_file_path', 'Not saved')}")
        
        if self.enable_phase1:
            logger.info(f"Phase 1 features processed: {list(state['phase1_processing'].keys())}")
        
        return state
    
    def handle_error(self, state: ExtractionState) -> ExtractionState:
        """Handle workflow errors with Phase 1 audit trail."""
        logger.error(f"Errors in enhanced extraction workflow: {state['errors']}")
        
        # Log errors to audit trail if available
        if self.enable_phase1 and state.get("audit_trail_entries"):
            try:
                self.audit_trail_service.add_activity(
                    state["audit_trail_entries"][0]["audit_id"],
                    "extraction_failed",
                    details={
                        "error_count": len(state["errors"]),
                        "errors": state["errors"][:5]  # Limit error details
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to log errors to audit trail: {str(e)}")
        
        state["status"] = "failed"
        return state
    
    def extract_data(
        self,
        query_type: str = QueryType.CUSTOM,
        queries: Optional[List[Dict[str, Any]]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        output_format: str = "json",
        report_template: Optional[str] = None,
        include_audit_trail: bool = True,
        include_source_files: bool = False,
        user_id: Optional[str] = None
    ) -> ExtractionState:
        """
        Execute enhanced data extraction workflow with Phase 1 features.
        
        Args:
            query_type: Type of query to execute
            queries: Custom queries (if query_type is CUSTOM)
            parameters: Query parameters
            output_format: Output format (json, txt, csv, html)
            report_template: Optional custom report template
            include_audit_trail: Include audit trail in report
            include_source_files: Include source files in report
            user_id: User ID for audit trail
            
        Returns:
            Final workflow state
        """
        # Initialize enhanced state
        initial_state: ExtractionState = {
            "query_config": {
                "type": query_type,
                "queries": queries or [],
                "parameters": parameters or {},
                "include_audit_trail": include_audit_trail,
                "user_id": user_id
            },
            "output_format": output_format,
            "report_template": report_template,
            "queries": [],
            "query_results": [],
            "graph_objects": [],
            "analysis_results": None,
            "report_data": {},
            "report_file_path": None,
            "processing_time": None,
            "errors": [],
            "status": "pending",
            "start_time": datetime.utcnow().timestamp(),
            # Phase 1 fields
            "prorating_allocations": [],
            "audit_trail_entries": [],
            "include_source_files": include_source_files,
            "phase1_processing": {}
        }
        
        # Run enhanced workflow
        final_state = self.workflow.invoke(initial_state)
        
        return final_state
    
    def close(self):
        """Close connections."""
        if hasattr(self, 'driver'):
            self.driver.close()
        
        if self.enable_phase1 and hasattr(self, 'graph'):
            # Close graph connection if needed
            pass