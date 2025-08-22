"""
LangGraph-based data extraction workflow for EHS AI Platform.
Queries existing Neo4j data and generates comprehensive reports.
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

logger = logging.getLogger(__name__)


# State definitions
class ExtractionState(TypedDict):
    """State for data extraction workflow."""
    # Input
    query_config: Dict[str, Any]  # Contains query type, parameters, filters
    output_format: str  # json, csv, pdf, html
    report_template: Optional[str]
    
    # Processing state
    queries: List[Dict[str, Any]]  # Cypher queries to execute
    query_results: List[Dict[str, Any]]  # Results from each query
    graph_objects: List[Dict[str, Any]]  # Neo4j objects involved
    analysis_results: Optional[Dict[str, Any]]  # LLM analysis of results
    
    # Output
    report_data: Dict[str, Any]  # Structured report data
    report_file_path: Optional[str]
    processing_time: Optional[float]
    
    # Error handling
    errors: List[str]
    status: str  # pending, processing, completed, failed


class QueryType(str, Enum):
    """Types of queries supported."""
    FACILITY_EMISSIONS = "facility_emissions"
    UTILITY_CONSUMPTION = "utility_consumption"
    WATER_CONSUMPTION = "water_consumption"
    WASTE_GENERATION = "waste_generation"
    COMPLIANCE_STATUS = "compliance_status"
    TREND_ANALYSIS = "trend_analysis"
    CUSTOM = "custom"


class DataExtractionWorkflow:
    """
    LangGraph workflow for extracting and reporting on EHS data from Neo4j.
    """
    
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_username: str,
        neo4j_password: str,
        llm_model: str = "gpt-4",
        output_dir: str = "./reports"
    ):
        """
        Initialize the data extraction workflow.
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            llm_model: LLM model to use for analysis
            output_dir: Directory for output reports
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
        
        # Pre-defined query templates
        self.query_templates = self._initialize_query_templates()
        
        # Build workflow graph
        self.workflow = self._build_workflow()
    
    def _initialize_query_templates(self) -> Dict[str, List[str]]:
        """Initialize pre-defined Cypher query templates."""
        return {
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
            ]
        }
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create workflow
        workflow = StateGraph(ExtractionState)
        
        # Add nodes
        workflow.add_node("prepare_queries", self.prepare_queries)
        workflow.add_node("identify_objects", self.identify_graph_objects)
        workflow.add_node("execute_queries", self.execute_queries)
        workflow.add_node("analyze_results", self.analyze_results)
        workflow.add_node("generate_report", self.generate_report)
        workflow.add_node("save_report", self.save_report)
        workflow.add_node("complete", self.complete_processing)
        workflow.add_node("handle_error", self.handle_error)
        
        # Add edges
        workflow.add_edge("prepare_queries", "identify_objects")
        workflow.add_edge("identify_objects", "execute_queries")
        workflow.add_edge("execute_queries", "analyze_results")
        workflow.add_edge("analyze_results", "generate_report")
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
        Prepare Cypher queries based on query configuration.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        logger.info(f"Preparing queries for type: {state['query_config'].get('type')}")
        
        try:
            query_type = state["query_config"].get("type", QueryType.CUSTOM)
            
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
        Execute Cypher queries against Neo4j.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        logger.info(f"Executing {len(state['queries'])} queries")
        
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
    
    def analyze_results(self, state: ExtractionState) -> ExtractionState:
        """
        Use LLM to analyze query results and generate insights.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        logger.info("Analyzing query results with LLM")
        
        try:
            # Prepare context for LLM
            context = {
                "query_type": state["query_config"].get("type"),
                "total_queries": len(state["queries"]),
                "successful_queries": sum(1 for r in state["query_results"] if r["status"] == "success"),
                "total_records": sum(r.get("record_count", 0) for r in state["query_results"])
            }
            
            # Create analysis prompt
            prompt = f"""
            Analyze the following EHS data extraction results and provide insights:
            
            Query Type: {context['query_type']}
            Total Queries Executed: {context['total_queries']}
            Successful Queries: {context['successful_queries']}
            Total Records Retrieved: {context['total_records']}
            
            Query Results:
            {json.dumps(state['query_results'], indent=2)[:3000]}  # Truncate for token limits
            
            Please provide:
            1. Key findings from the data
            2. Any patterns or trends observed
            3. Potential data quality issues
            4. Recommendations based on the analysis
            
            Format your response as structured JSON.
            """
            
            # Get LLM analysis
            response = self.llm.invoke([
                SystemMessage(content="You are an EHS data analyst. Provide structured analysis of query results."),
                HumanMessage(content=prompt)
            ])
            
            # Parse response
            try:
                analysis = json.loads(response.content)
            except:
                analysis = {
                    "summary": response.content,
                    "findings": ["Analysis completed but could not parse structured response"]
                }
            
            state["analysis_results"] = analysis
            
        except Exception as e:
            state["errors"].append(f"Analysis error: {str(e)}")
            logger.error(f"Failed to analyze results: {str(e)}")
            state["analysis_results"] = {"error": "Analysis failed"}
        
        return state
    
    def generate_report(self, state: ExtractionState) -> ExtractionState:
        """
        Generate structured report from query results and analysis.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        logger.info(f"Generating report in format: {state['output_format']}")
        
        try:
            # Compile report data
            report_data = {
                "metadata": {
                    "title": f"EHS Data Extraction Report - {state['query_config'].get('type', 'Custom')}",
                    "generated_at": datetime.utcnow().isoformat(),
                    "query_type": state["query_config"].get("type"),
                    "parameters": state["query_config"].get("parameters", {}),
                    "output_format": state["output_format"]
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
            
            state["report_data"] = report_data
            
            logger.info("Report data generated successfully")
            
        except Exception as e:
            state["errors"].append(f"Report generation error: {str(e)}")
            logger.error(f"Failed to generate report: {str(e)}")
        
        return state
    
    def save_report(self, state: ExtractionState) -> ExtractionState:
        """
        Save report to file in requested format.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state
        """
        logger.info("Saving report to file")
        
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            query_type = state["query_config"].get("type", "custom")
            
            if state["output_format"] == "json":
                # Save as JSON
                file_name = f"ehs_report_{query_type}_{timestamp}.json"
                file_path = self.output_dir / file_name
                
                with open(file_path, 'w') as f:
                    json.dump(state["report_data"], f, indent=2)
                
            elif state["output_format"] in ["txt", "text"]:
                # Save as formatted text
                file_name = f"ehs_report_{query_type}_{timestamp}.txt"
                file_path = self.output_dir / file_name
                
                with open(file_path, 'w') as f:
                    # Write header
                    f.write("=" * 80 + "\n")
                    f.write(state["report_data"]["metadata"]["title"] + "\n")
                    f.write("=" * 80 + "\n\n")
                    
                    # Write metadata
                    f.write(f"Generated: {state['report_data']['metadata']['generated_at']}\n")
                    f.write(f"Query Type: {state['report_data']['metadata']['query_type']}\n\n")
                    
                    # Write summary
                    f.write("SUMMARY\n")
                    f.write("-" * 40 + "\n")
                    summary = state["report_data"]["summary"]
                    f.write(f"Total Queries: {summary['total_queries']}\n")
                    f.write(f"Successful: {summary['successful_queries']}\n")
                    f.write(f"Failed: {summary['failed_queries']}\n")
                    f.write(f"Total Records: {summary['total_records']}\n\n")
                    
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
                                
                                # Special formatting for waste manifest data
                                if state["query_config"].get("type") == QueryType.WASTE_GENERATION:
                                    self._format_waste_results(f, result["results"][:5])
                                else:
                                    for j, record in enumerate(result["results"][:5]):
                                        f.write(f"  {j+1}: {json.dumps(record, indent=2)}\n")
                                
                                if result["record_count"] > 5:
                                    f.write(f"  ... and {result['record_count'] - 5} more\n")
                        else:
                            f.write(f"Error: {result.get('error', 'Unknown error')}\n")
                        
                        f.write("\n")
                    
                    # Write analysis
                    if state["report_data"].get("analysis"):
                        f.write("ANALYSIS\n")
                        f.write("=" * 80 + "\n")
                        f.write(json.dumps(state["report_data"]["analysis"], indent=2) + "\n")
            
            else:
                # Default to JSON for unsupported formats
                file_name = f"ehs_report_{query_type}_{timestamp}.json"
                file_path = self.output_dir / file_name
                
                with open(file_path, 'w') as f:
                    json.dump(state["report_data"], f, indent=2)
            
            state["report_file_path"] = str(file_path)
            logger.info(f"Report saved to: {file_path}")
            
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
    
    def complete_processing(self, state: ExtractionState) -> ExtractionState:
        """Complete the extraction workflow."""
        state["status"] = "completed"
        state["processing_time"] = datetime.utcnow().timestamp() - state.get("start_time", 0)
        
        logger.info(f"Extraction completed in {state['processing_time']:.2f} seconds")
        logger.info(f"Report saved to: {state.get('report_file_path', 'Not saved')}")
        
        return state
    
    def handle_error(self, state: ExtractionState) -> ExtractionState:
        """Handle workflow errors."""
        logger.error(f"Errors in extraction workflow: {state['errors']}")
        state["status"] = "failed"
        return state
    
    def extract_data(
        self,
        query_type: str = QueryType.CUSTOM,
        queries: Optional[List[Dict[str, Any]]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        output_format: str = "json",
        report_template: Optional[str] = None
    ) -> ExtractionState:
        """
        Execute data extraction workflow.
        
        Args:
            query_type: Type of query to execute
            queries: Custom queries (if query_type is CUSTOM)
            parameters: Query parameters
            output_format: Output format (json, txt, csv, html)
            report_template: Optional custom report template
            
        Returns:
            Final workflow state
        """
        # Initialize state
        initial_state: ExtractionState = {
            "query_config": {
                "type": query_type,
                "queries": queries or [],
                "parameters": parameters or {}
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
            "start_time": datetime.utcnow().timestamp()
        }
        
        # Run workflow
        final_state = self.workflow.invoke(initial_state)
        
        return final_state
    
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()