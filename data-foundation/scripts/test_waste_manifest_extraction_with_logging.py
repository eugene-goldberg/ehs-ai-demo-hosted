#!/usr/bin/env python3
"""
Enhanced test script for waste manifest data extraction from Neo4j with comprehensive logging.
Queries the graph, generates comprehensive waste generation reports, and logs every action.
"""

import os
import sys
import logging
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables from backend/.env
env_path = Path(__file__).parent.parent / "backend" / ".env"
load_dotenv(env_path, override=True)

from backend.src.workflows.extraction_workflow import DataExtractionWorkflow, QueryType

# Set up comprehensive logging
def setup_logging():
    """Set up comprehensive logging to both console and file."""
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Create log file path
    log_file = log_dir / "waste_manifest_extraction.log"
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler - detailed logging
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler - important messages only
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger, log_file

def log_neo4j_results(logger, results, query_name, output_file):
    """Log Neo4j query results in detail to both log file and separate output file."""
    logger.info(f"Processing results for query: {query_name}")
    logger.info(f"Number of results returned: {len(results) if results else 0}")
    
    if not results:
        logger.warning(f"No results returned for query: {query_name}")
        return
    
    # Log summary to main log
    logger.info(f"First few results from {query_name}:")
    for i, record in enumerate(results[:3], 1):
        logger.info(f"  Record {i}: {dict(record)}")
    
    if len(results) > 3:
        logger.info(f"  ... and {len(results) - 3} more records")
    
    # Write detailed results to output file
    try:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"QUERY: {query_name}\n")
            f.write(f"TIMESTAMP: {datetime.now().isoformat()}\n")
            f.write(f"RESULT COUNT: {len(results)}\n")
            f.write(f"{'='*80}\n\n")
            
            for i, record in enumerate(results, 1):
                f.write(f"Record {i}:\n")
                f.write("-" * 40 + "\n")
                record_dict = dict(record)
                f.write(json.dumps(record_dict, indent=2, default=str))
                f.write("\n\n")
        
        logger.info(f"Detailed results written to: {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to write results to output file: {e}")

def execute_comprehensive_neo4j_dump(workflow, logger, output_file):
    """Execute comprehensive Neo4j data dump queries."""
    logger.info("Starting comprehensive Neo4j data dump")
    
    # Comprehensive queries to capture all waste-related data
    dump_queries = [
        {
            "name": "All_WasteManifest_Nodes",
            "query": """
                MATCH (m:WasteManifest)
                RETURN 
                    m.manifest_id as manifest_id,
                    m.date_shipped as date_shipped,
                    m.quantity as quantity,
                    m.unit as unit,
                    m.hazardous as hazardous,
                    m.waste_code as waste_code,
                    m.description as description,
                    m.proper_shipping_name as proper_shipping_name,
                    m.un_number as un_number,
                    m.packing_group as packing_group,
                    m.hazard_class as hazard_class,
                    properties(m) as all_properties
                ORDER BY m.date_shipped DESC, m.manifest_id
            """,
            "parameters": {}
        },
        {
            "name": "All_Generator_Nodes",
            "query": """
                MATCH (g:Generator)
                RETURN 
                    g.name as name,
                    g.epa_id as epa_id,
                    g.address as address,
                    g.city as city,
                    g.state as state,
                    g.zip_code as zip_code,
                    g.phone as phone,
                    g.contact_person as contact_person,
                    properties(g) as all_properties
                ORDER BY g.name
            """,
            "parameters": {}
        },
        {
            "name": "All_DisposalFacility_Nodes",
            "query": """
                MATCH (df:DisposalFacility)
                RETURN 
                    df.name as name,
                    df.epa_id as epa_id,
                    df.address as address,
                    df.city as city,
                    df.state as state,
                    df.zip_code as zip_code,
                    df.disposal_method as disposal_method,
                    df.permit_number as permit_number,
                    properties(df) as all_properties
                ORDER BY df.name
            """,
            "parameters": {}
        },
        {
            "name": "All_Transporter_Nodes",
            "query": """
                MATCH (t:Transporter)
                RETURN 
                    t.name as name,
                    t.epa_id as epa_id,
                    t.address as address,
                    t.city as city,
                    t.state as state,
                    t.license_number as license_number,
                    properties(t) as all_properties
                ORDER BY t.name
            """,
            "parameters": {}
        },
        {
            "name": "All_Emission_Nodes",
            "query": """
                MATCH (e:Emission)
                RETURN 
                    e.emission_type as emission_type,
                    e.amount as amount,
                    e.unit as unit,
                    e.emission_factor as emission_factor,
                    e.calculation_method as calculation_method,
                    e.date_calculated as date_calculated,
                    properties(e) as all_properties
                ORDER BY e.amount DESC
            """,
            "parameters": {}
        },
        {
            "name": "All_WasteManifest_Relationships",
            "query": """
                MATCH (m:WasteManifest)-[r]-(n)
                RETURN 
                    m.manifest_id as manifest_id,
                    type(r) as relationship_type,
                    labels(n) as connected_node_labels,
                    CASE 
                        WHEN 'Generator' IN labels(n) THEN n.name
                        WHEN 'Transporter' IN labels(n) THEN n.name
                        WHEN 'DisposalFacility' IN labels(n) THEN n.name
                        WHEN 'Emission' IN labels(n) THEN n.emission_type
                        ELSE toString(id(n))
                    END as connected_node_name,
                    properties(r) as relationship_properties,
                    id(startNode(r)) as start_node_id,
                    id(endNode(r)) as end_node_id
                ORDER BY m.manifest_id, relationship_type
            """,
            "parameters": {}
        },
        {
            "name": "Manifest_Generator_Relationships",
            "query": """
                MATCH (m:WasteManifest)-[:GENERATED_BY]->(g:Generator)
                RETURN 
                    m.manifest_id as manifest_id,
                    m.date_shipped as date_shipped,
                    m.quantity as quantity,
                    m.unit as unit,
                    m.hazardous as hazardous,
                    g.name as generator_name,
                    g.epa_id as generator_epa_id,
                    g.address as generator_address
                ORDER BY m.date_shipped DESC
            """,
            "parameters": {}
        },
        {
            "name": "Manifest_Transporter_Relationships",
            "query": """
                MATCH (m:WasteManifest)-[:TRANSPORTED_BY]->(t:Transporter)
                RETURN 
                    m.manifest_id as manifest_id,
                    m.date_shipped as date_shipped,
                    t.name as transporter_name,
                    t.epa_id as transporter_epa_id,
                    t.license_number as license_number
                ORDER BY m.date_shipped DESC
            """,
            "parameters": {}
        },
        {
            "name": "Manifest_DisposalFacility_Relationships",
            "query": """
                MATCH (m:WasteManifest)-[:DISPOSED_AT]->(df:DisposalFacility)
                RETURN 
                    m.manifest_id as manifest_id,
                    m.date_shipped as date_shipped,
                    m.quantity as quantity,
                    m.unit as unit,
                    df.name as disposal_facility_name,
                    df.epa_id as disposal_facility_epa_id,
                    df.disposal_method as disposal_method
                ORDER BY m.date_shipped DESC
            """,
            "parameters": {}
        },
        {
            "name": "Manifest_Emission_Relationships",
            "query": """
                MATCH (m:WasteManifest)-[:RESULTED_IN]->(e:Emission)
                RETURN 
                    m.manifest_id as manifest_id,
                    m.date_shipped as date_shipped,
                    m.quantity as waste_quantity,
                    m.unit as waste_unit,
                    e.emission_type as emission_type,
                    e.amount as emission_amount,
                    e.unit as emission_unit,
                    e.emission_factor as emission_factor
                ORDER BY m.date_shipped DESC
            """,
            "parameters": {}
        },
        {
            "name": "Database_Statistics",
            "query": """
                CALL db.stats.retrieve('GRAPH COUNTS') YIELD data
                RETURN data
                
                UNION
                
                MATCH (n)
                RETURN 'Total Nodes' as metric, count(n) as value
                
                UNION
                
                MATCH ()-[r]->()
                RETURN 'Total Relationships' as metric, count(r) as value
                
                UNION
                
                MATCH (m:WasteManifest)
                RETURN 'WasteManifest Nodes' as metric, count(m) as value
                
                UNION
                
                MATCH (g:Generator)
                RETURN 'Generator Nodes' as metric, count(g) as value
                
                UNION
                
                MATCH (t:Transporter)
                RETURN 'Transporter Nodes' as metric, count(t) as value
                
                UNION
                
                MATCH (df:DisposalFacility)
                RETURN 'DisposalFacility Nodes' as metric, count(df) as value
                
                UNION
                
                MATCH (e:Emission)
                RETURN 'Emission Nodes' as metric, count(e) as value
            """,
            "parameters": {}
        }
    ]
    
    logger.info(f"Executing {len(dump_queries)} comprehensive data dump queries")
    
    # Execute each dump query
    for i, query_config in enumerate(dump_queries, 1):
        query_name = query_config["name"]
        logger.info(f"Executing dump query {i}/{len(dump_queries)}: {query_name}")
        
        try:
            # Execute the query through the workflow
            state = workflow.extract_data(
                query_type=QueryType.CUSTOM,
                queries=[query_config],
                output_format="json"
            )
            
            if state.get("status") == "completed":
                logger.info(f"✓ Query {query_name} completed successfully")
                
                # Extract and log results
                if state.get("query_results"):
                    results = state["query_results"][0].get("results", [])
                    log_neo4j_results(logger, results, query_name, output_file)
                else:
                    logger.warning(f"No query results found for {query_name}")
            else:
                logger.error(f"✗ Query {query_name} failed")
                if state.get("errors"):
                    for error in state["errors"]:
                        logger.error(f"  Error: {error}")
                        
        except Exception as e:
            logger.error(f"Exception during query {query_name}: {e}", exc_info=True)
    
    logger.info("Comprehensive Neo4j data dump completed")

def test_waste_manifest_extraction_with_logging():
    """Test waste manifest data extraction and report generation with comprehensive logging."""
    # Set up logging
    logger, log_file = setup_logging()
    
    logger.info("=" * 100)
    logger.info("STARTING ENHANCED WASTE MANIFEST DATA EXTRACTION WITH COMPREHENSIVE LOGGING")
    logger.info("=" * 100)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Script started at: {datetime.now().isoformat()}")
    
    # Create output file for Neo4j data
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "waste_manifest_neo4j_data.txt"
    
    logger.info(f"Neo4j data output file: {output_file}")
    
    # Initialize output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"WASTE MANIFEST NEO4J DATA EXTRACTION\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"{'='*80}\n\n")
        logger.info("Output file initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize output file: {e}")
        return
    
    workflow = None
    
    try:
        logger.info("Initializing DataExtractionWorkflow...")
        logger.info("Neo4j Connection Parameters:")
        logger.info(f"  URI: bolt://localhost:7687")
        logger.info(f"  Username: neo4j")
        logger.info(f"  Password: [REDACTED]")
        logger.info(f"  LLM Model: gpt-4")
        logger.info(f"  Output Directory: ./reports")
        
        # Initialize extraction workflow
        workflow = DataExtractionWorkflow(
            neo4j_uri="bolt://localhost:7687",
            neo4j_username="neo4j",
            neo4j_password="EhsAI2024!",
            llm_model="gpt-4",
            output_dir="./reports"
        )
        
        logger.info("✓ DataExtractionWorkflow initialized successfully")
        
        # Execute comprehensive Neo4j data dump first
        logger.info("PHASE 1: Executing comprehensive Neo4j data dump")
        execute_comprehensive_neo4j_dump(workflow, logger, output_file)
        
        logger.info("PHASE 2: Starting standard extraction workflow tests")
        
        # Test 1: Basic waste manifest query
        logger.info("TEST 1: Basic waste manifest query")
        logger.info("Starting basic waste generation data extraction...")
        
        try:
            state1 = workflow.extract_data(
                query_type=QueryType.WASTE_GENERATION,
                output_format="txt"
            )
            
            if state1.get("status") == "completed":
                logger.info(f"✓ TEST 1 PASSED: Waste generation report generated: {state1.get('report_file_path')}")
                
                # Log detailed results if available
                if state1.get("query_results"):
                    for i, query_result in enumerate(state1["query_results"]):
                        results = query_result.get("results", [])
                        log_neo4j_results(logger, results, f"BasicWasteGeneration_Query{i+1}", output_file)
            else:
                logger.error("✗ TEST 1 FAILED: Waste generation report generation failed")
                if state1.get("errors"):
                    for error in state1["errors"]:
                        logger.error(f"  Error: {error}")
                        
        except Exception as e:
            logger.error(f"TEST 1 EXCEPTION: {e}", exc_info=True)
        
        # Test 2: Waste generation with date range
        logger.info("TEST 2: Waste generation with date range")
        logger.info("Starting filtered waste generation extraction (2025-06-01 to 2025-06-30)...")
        
        try:
            state2 = workflow.extract_data(
                query_type=QueryType.WASTE_GENERATION,
                parameters={
                    "start_date": "2025-06-01",
                    "end_date": "2025-06-30"
                },
                output_format="json"
            )
            
            if state2.get("status") == "completed":
                logger.info(f"✓ TEST 2 PASSED: Filtered waste report generated: {state2.get('report_file_path')}")
                
                # Log detailed results if available
                if state2.get("query_results"):
                    for i, query_result in enumerate(state2["query_results"]):
                        results = query_result.get("results", [])
                        log_neo4j_results(logger, results, f"FilteredWasteGeneration_Query{i+1}", output_file)
            else:
                logger.error("✗ TEST 2 FAILED: Filtered waste report generation failed")
                if state2.get("errors"):
                    for error in state2["errors"]:
                        logger.error(f"  Error: {error}")
                        
        except Exception as e:
            logger.error(f"TEST 2 EXCEPTION: {e}", exc_info=True)
        
        # Test 3: Waste generation by generator
        logger.info("TEST 3: Waste generation by generator")
        logger.info("Starting waste generation by generator analysis...")
        
        generator_queries = [
            {
                "query": """
                    MATCH (m:WasteManifest)
                    MATCH (m)-[:GENERATED_BY]->(g:Generator)
                    OPTIONAL MATCH (m)-[:TRANSPORTED_BY]->(t:Transporter)
                    OPTIONAL MATCH (m)-[:DISPOSED_AT]->(df:DisposalFacility)
                    OPTIONAL MATCH (m)-[:RESULTED_IN]->(e:Emission)
                    RETURN 
                        g.name as generator_name,
                        g.epa_id as generator_epa_id,
                        g.address as generator_address,
                        COUNT(DISTINCT m) as manifest_count,
                        SUM(m.quantity) as total_quantity,
                        m.unit as unit,
                        COUNT(DISTINCT CASE WHEN m.hazardous = true THEN m END) as hazardous_manifests,
                        COUNT(DISTINCT CASE WHEN m.hazardous = false THEN m END) as nonhazardous_manifests,
                        SUM(CASE WHEN m.hazardous = true THEN m.quantity ELSE 0 END) as hazardous_quantity,
                        SUM(CASE WHEN m.hazardous = false THEN m.quantity ELSE 0 END) as nonhazardous_quantity,
                        SUM(e.amount) as total_emissions,
                        collect(DISTINCT t.name) as transporters,
                        collect(DISTINCT df.name) as disposal_facilities
                    ORDER BY total_quantity DESC
                """,
                "parameters": {}
            }
        ]
        
        try:
            logger.info("Executing generator analysis query...")
            state3 = workflow.extract_data(
                query_type=QueryType.CUSTOM,
                queries=generator_queries,
                output_format="txt"
            )
            
            if state3.get("status") == "completed":
                logger.info(f"✓ TEST 3 PASSED: Waste by generator report generated: {state3.get('report_file_path')}")
                
                # Log and display results
                if state3.get("query_results"):
                    results = state3["query_results"][0].get("results", [])
                    log_neo4j_results(logger, results, "WasteByGenerator", output_file)
                    
                    if results:
                        logger.info("Top Waste Generators:")
                        for i, record in enumerate(results[:3], 1):
                            logger.info(f"  {i}. {record.get('generator_name')}")
                            logger.info(f"     EPA ID: {record.get('generator_epa_id')}")
                            logger.info(f"     Total Quantity: {record.get('total_quantity')} {record.get('unit')}")
                            logger.info(f"     Hazardous: {record.get('hazardous_quantity')} {record.get('unit')}")
                            logger.info(f"     Non-Hazardous: {record.get('nonhazardous_quantity')} {record.get('unit')}")
                            logger.info(f"     Total Emissions: {record.get('total_emissions')} kg CO2")
            else:
                logger.error("✗ TEST 3 FAILED: Waste by generator report generation failed")
                if state3.get("errors"):
                    for error in state3["errors"]:
                        logger.error(f"  Error: {error}")
                        
        except Exception as e:
            logger.error(f"TEST 3 EXCEPTION: {e}", exc_info=True)
        
        # Test 4: Disposal facility analysis
        logger.info("TEST 4: Disposal facility analysis")
        logger.info("Starting disposal facility analysis...")
        
        disposal_queries = [
            {
                "query": """
                    MATCH (m:WasteManifest)
                    MATCH (m)-[:DISPOSED_AT]->(df:DisposalFacility)
                    OPTIONAL MATCH (m)-[:GENERATED_BY]->(g:Generator)
                    OPTIONAL MATCH (m)-[:RESULTED_IN]->(e:Emission)
                    RETURN 
                        df.name as facility_name,
                        df.epa_id as facility_epa_id,
                        df.address as facility_address,
                        df.disposal_method as disposal_method,
                        COUNT(DISTINCT m) as manifest_count,
                        SUM(m.quantity) as total_quantity,
                        m.unit as unit,
                        COUNT(DISTINCT CASE WHEN m.hazardous = true THEN m END) as hazardous_manifests,
                        COUNT(DISTINCT CASE WHEN m.hazardous = false THEN m END) as nonhazardous_manifests,
                        SUM(CASE WHEN m.hazardous = true THEN m.quantity ELSE 0 END) as hazardous_quantity,
                        SUM(CASE WHEN m.hazardous = false THEN m.quantity ELSE 0 END) as nonhazardous_quantity,
                        SUM(e.amount) as total_emissions,
                        COUNT(DISTINCT g) as unique_generators,
                        collect(DISTINCT g.name)[0..5] as top_generators
                    ORDER BY total_quantity DESC
                """,
                "parameters": {}
            }
        ]
        
        try:
            logger.info("Executing disposal facility analysis query...")
            state4 = workflow.extract_data(
                query_type=QueryType.CUSTOM,
                queries=disposal_queries,
                output_format="txt"
            )
            
            if state4.get("status") == "completed":
                logger.info(f"✓ TEST 4 PASSED: Disposal facility analysis generated: {state4.get('report_file_path')}")
                
                # Log and display results
                if state4.get("query_results"):
                    results = state4["query_results"][0].get("results", [])
                    log_neo4j_results(logger, results, "DisposalFacilityAnalysis", output_file)
                    
                    if results:
                        logger.info("Top Disposal Facilities:")
                        for i, record in enumerate(results[:3], 1):
                            logger.info(f"  {i}. {record.get('facility_name')}")
                            logger.info(f"     EPA ID: {record.get('facility_epa_id')}")
                            logger.info(f"     Disposal Method: {record.get('disposal_method')}")
                            logger.info(f"     Total Quantity: {record.get('total_quantity')} {record.get('unit')}")
                            logger.info(f"     Unique Generators: {record.get('unique_generators')}")
                            logger.info(f"     Total Emissions: {record.get('total_emissions')} kg CO2")
            else:
                logger.error("✗ TEST 4 FAILED: Disposal facility analysis generation failed")
                if state4.get("errors"):
                    for error in state4["errors"]:
                        logger.error(f"  Error: {error}")
                        
        except Exception as e:
            logger.error(f"TEST 4 EXCEPTION: {e}", exc_info=True)
        
        # Test 5: Emission analysis from waste
        logger.info("TEST 5: Emission analysis from waste")
        logger.info("Starting emission analysis from waste...")
        
        emission_queries = [
            {
                "query": """
                    MATCH (m:WasteManifest)-[:RESULTED_IN]->(e:Emission)
                    MATCH (m)-[:GENERATED_BY]->(g:Generator)
                    MATCH (m)-[:DISPOSED_AT]->(df:DisposalFacility)
                    RETURN 
                        e.emission_type as emission_type,
                        COUNT(DISTINCT m) as manifest_count,
                        SUM(e.amount) as total_emissions,
                        AVG(e.amount) as avg_emissions_per_manifest,
                        e.unit as emission_unit,
                        SUM(m.quantity) as waste_quantity,
                        m.unit as waste_unit,
                        AVG(e.emission_factor) as avg_emission_factor,
                        COUNT(DISTINCT g) as unique_generators,
                        COUNT(DISTINCT df) as unique_disposal_facilities,
                        collect(DISTINCT CASE WHEN m.hazardous = true THEN 'Hazardous' ELSE 'Non-Hazardous' END) as waste_types
                    ORDER BY total_emissions DESC
                """,
                "parameters": {}
            }
        ]
        
        try:
            logger.info("Executing emission analysis query...")
            state5 = workflow.extract_data(
                query_type=QueryType.CUSTOM,
                queries=emission_queries,
                output_format="json"
            )
            
            if state5.get("status") == "completed":
                logger.info(f"✓ TEST 5 PASSED: Emission analysis report generated: {state5.get('report_file_path')}")
                
                # Log results
                if state5.get("query_results"):
                    results = state5["query_results"][0].get("results", [])
                    log_neo4j_results(logger, results, "EmissionAnalysis", output_file)
            else:
                logger.error("✗ TEST 5 FAILED: Emission analysis report generation failed")
                if state5.get("errors"):
                    for error in state5["errors"]:
                        logger.error(f"  Error: {error}")
                        
        except Exception as e:
            logger.error(f"TEST 5 EXCEPTION: {e}", exc_info=True)
        
        # Test 6: Combined summary report
        logger.info("TEST 6: Combined summary report")
        logger.info("Generating combined waste summary report...")
        
        summary_queries = [
            {
                "query": """
                    MATCH (m:WasteManifest)
                    OPTIONAL MATCH (m)-[:GENERATED_BY]->(g:Generator)
                    OPTIONAL MATCH (m)-[:DISPOSED_AT]->(df:DisposalFacility)
                    OPTIONAL MATCH (m)-[:RESULTED_IN]->(e:Emission)
                    WITH 
                        COUNT(DISTINCT m) as total_manifests,
                        SUM(m.quantity) as total_waste_quantity,
                        SUM(CASE WHEN m.hazardous = true THEN m.quantity ELSE 0 END) as total_hazardous,
                        SUM(CASE WHEN m.hazardous = false THEN m.quantity ELSE 0 END) as total_nonhazardous,
                        COUNT(DISTINCT g) as unique_generators,
                        COUNT(DISTINCT df) as unique_disposal_facilities,
                        SUM(e.amount) as total_emissions,
                        MIN(m.date_shipped) as earliest_date,
                        MAX(m.date_shipped) as latest_date
                    RETURN 
                        total_manifests,
                        total_waste_quantity,
                        total_hazardous,
                        total_nonhazardous,
                        unique_generators,
                        unique_disposal_facilities,
                        total_emissions,
                        earliest_date,
                        latest_date,
                        ROUND((total_hazardous * 100.0 / total_waste_quantity), 2) as hazardous_percentage,
                        ROUND((total_nonhazardous * 100.0 / total_waste_quantity), 2) as nonhazardous_percentage,
                        ROUND((total_emissions / total_waste_quantity), 4) as emissions_per_unit_waste
                """,
                "parameters": {}
            }
        ]
        
        try:
            logger.info("Executing combined summary query...")
            state6 = workflow.extract_data(
                query_type=QueryType.CUSTOM,
                queries=summary_queries,
                output_format="txt"
            )
            
            if state6.get("status") == "completed":
                logger.info(f"✓ TEST 6 PASSED: Combined summary report generated: {state6.get('report_file_path')}")
                
                # Log and display summary results
                if state6.get("query_results"):
                    results = state6["query_results"][0].get("results", [])
                    log_neo4j_results(logger, results, "CombinedSummary", output_file)
                    
                    if results:
                        summary = results[0]
                        logger.info("Waste Management Summary:")
                        logger.info("=" * 50)
                        logger.info(f"Total Manifests: {summary.get('total_manifests')}")
                        logger.info(f"Total Waste Generated: {summary.get('total_waste_quantity')} tons")
                        logger.info(f"  - Hazardous: {summary.get('total_hazardous')} tons ({summary.get('hazardous_percentage')}%)")
                        logger.info(f"  - Non-Hazardous: {summary.get('total_nonhazardous')} tons ({summary.get('nonhazardous_percentage')}%)")
                        logger.info(f"Total Emissions: {summary.get('total_emissions')} kg CO2")
                        logger.info(f"Emissions per Unit Waste: {summary.get('emissions_per_unit_waste')} kg CO2/ton")
                        logger.info(f"Unique Generators: {summary.get('unique_generators')}")
                        logger.info(f"Unique Disposal Facilities: {summary.get('unique_disposal_facilities')}")
                        logger.info(f"Date Range: {summary.get('earliest_date')} to {summary.get('latest_date')}")
            else:
                logger.error("✗ TEST 6 FAILED: Combined summary report generation failed")
                if state6.get("errors"):
                    for error in state6["errors"]:
                        logger.error(f"  Error: {error}")
                        
        except Exception as e:
            logger.error(f"TEST 6 EXCEPTION: {e}", exc_info=True)
        
        # Test 7: Top generators and facilities list
        logger.info("TEST 7: Top generators and facilities list")
        logger.info("Generating top generators and facilities list...")
        
        top_entities_queries = [
            {
                "query": """
                    // Top 5 Generators by total waste
                    MATCH (m:WasteManifest)-[:GENERATED_BY]->(g:Generator)
                    WITH g, SUM(m.quantity) as total_quantity
                    ORDER BY total_quantity DESC
                    LIMIT 5
                    RETURN 'Generator' as entity_type, g.name as name, g.epa_id as epa_id, 
                           total_quantity, 'tons' as unit, g.address as address
                    
                    UNION
                    
                    // Top 5 Disposal Facilities by total waste received
                    MATCH (m:WasteManifest)-[:DISPOSED_AT]->(df:DisposalFacility)
                    WITH df, SUM(m.quantity) as total_quantity
                    ORDER BY total_quantity DESC
                    LIMIT 5
                    RETURN 'Disposal Facility' as entity_type, df.name as name, df.epa_id as epa_id,
                           total_quantity, 'tons' as unit, df.address as address
                    
                    ORDER BY entity_type, total_quantity DESC
                """,
                "parameters": {}
            }
        ]
        
        try:
            logger.info("Executing top entities query...")
            state7 = workflow.extract_data(
                query_type=QueryType.CUSTOM,
                queries=top_entities_queries,
                output_format="txt"
            )
            
            if state7.get("status") == "completed":
                logger.info(f"✓ TEST 7 PASSED: Top entities report generated: {state7.get('report_file_path')}")
                
                # Log and display top entities
                if state7.get("query_results"):
                    results = state7["query_results"][0].get("results", [])
                    log_neo4j_results(logger, results, "TopEntities", output_file)
                    
                    if results:
                        generators = [r for r in results if r.get('entity_type') == 'Generator']
                        facilities = [r for r in results if r.get('entity_type') == 'Disposal Facility']
                        
                        if generators:
                            logger.info("Top Waste Generators:")
                            for i, gen in enumerate(generators, 1):
                                logger.info(f"  {i}. {gen.get('name')} ({gen.get('epa_id')})")
                                logger.info(f"     Total: {gen.get('total_quantity')} {gen.get('unit')}")
                        
                        if facilities:
                            logger.info("Top Disposal Facilities:")
                            for i, fac in enumerate(facilities, 1):
                                logger.info(f"  {i}. {fac.get('name')} ({fac.get('epa_id')})")
                                logger.info(f"     Total: {fac.get('total_quantity')} {fac.get('unit')}")
            else:
                logger.error("✗ TEST 7 FAILED: Top entities report generation failed")
                if state7.get("errors"):
                    for error in state7["errors"]:
                        logger.error(f"  Error: {error}")
                        
        except Exception as e:
            logger.error(f"TEST 7 EXCEPTION: {e}", exc_info=True)
        
        logger.info("=" * 100)
        logger.info("ALL TESTS COMPLETED - Enhanced Waste Manifest Extraction Tests Finished")
        logger.info("=" * 100)
        logger.info(f"Complete log file available at: {log_file}")
        logger.info(f"Complete Neo4j data dump available at: {output_file}")
        logger.info(f"Script completed at: {datetime.now().isoformat()}")
        
    except Exception as e:
        logger.error(f"CRITICAL ERROR - Test failed with exception: {e}", exc_info=True)
        
    finally:
        logger.info("Performing cleanup...")
        if workflow:
            try:
                workflow.driver.close()
                logger.info("✓ Neo4j driver closed successfully")
            except Exception as e:
                logger.error(f"Error closing Neo4j driver: {e}")
        logger.info("Cleanup completed")

if __name__ == "__main__":
    test_waste_manifest_extraction_with_logging()