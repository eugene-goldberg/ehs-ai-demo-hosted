#!/usr/bin/env python3
"""
Enhanced test script for waste manifest ingestion with comprehensive logging.
Tests the full pipeline with the waste manifest PDF and writes detailed logs.
"""

import os
import sys
import logging
import traceback
from pathlib import Path
from datetime import datetime
import json
from dotenv import load_dotenv
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables from backend/.env
env_path = Path(__file__).parent.parent / "backend" / ".env"
load_dotenv(env_path, override=True)

from backend.src.workflows.ingestion_workflow import IngestionWorkflow
from neo4j import GraphDatabase

# Create logs directory if it doesn't exist
logs_dir = Path(__file__).parent / "logs"
logs_dir.mkdir(exist_ok=True)

# Configure comprehensive logging
log_filename = logs_dir / "waste_manifest_ingestion.log"

# Set up logger with multiple handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create formatters
detailed_formatter = logging.Formatter(
    '%(asctime)s.%(msecs)03d [%(levelname)8s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

# File handler for detailed logs
file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(detailed_formatter)

# Console handler for real-time monitoring
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(console_formatter)

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Also set up root logger to capture all module logs
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.handlers.clear()
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Suppress some noisy loggers
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def log_system_info():
    """Log system and environment information."""
    logger.info("=" * 80)
    logger.info("WASTE MANIFEST INGESTION TEST - ENHANCED LOGGING")
    logger.info("=" * 80)
    
    logger.info(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file location: {log_filename}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Environment variables check
    logger.info("\nEnvironment Variables Check:")
    env_vars = ["LLAMA_PARSE_API_KEY", "OPENAI_API_KEY"]
    for var in env_vars:
        value = os.getenv(var)
        if value:
            logger.info(f"  ✓ {var}: Set (length: {len(value)})")
            logger.debug(f"  {var}: {value[:10]}...{value[-4:] if len(value) > 14 else value}")
        else:
            logger.error(f"  ✗ {var}: Not set!")
    
    logger.info("=" * 80)


def clear_neo4j_data():
    """Clear all data from Neo4j database with detailed logging."""
    logger.info("\n" + "=" * 60)
    logger.info("CLEARING NEO4J DATA")
    logger.info("=" * 60)
    
    start_time = time.time()
    logger.info("Connecting to Neo4j database...")
    
    driver = GraphDatabase.driver(
        'bolt://localhost:7687', 
        auth=('neo4j', 'EhsAI2024!')
    )
    
    try:
        with driver.session() as session:
            # First, count existing data
            logger.info("Counting existing data before deletion...")
            
            node_count_result = session.run('MATCH (n) WHERE NOT n:__Node__ RETURN count(n) as count')
            existing_nodes = node_count_result.single()["count"]
            logger.info(f"  Existing nodes: {existing_nodes}")
            
            rel_count_result = session.run('MATCH (a)-[r]->(b) WHERE NOT a:__Node__ AND NOT b:__Node__ RETURN count(r) as count')
            existing_rels = rel_count_result.single()["count"]
            logger.info(f"  Existing relationships: {existing_rels}")
            
            if existing_nodes > 0 or existing_rels > 0:
                logger.info(f"Deleting all existing data ({existing_nodes} nodes, {existing_rels} relationships)...")
                session.run('MATCH (n) DETACH DELETE n')
                
                # Verify deletion
                verify_result = session.run('MATCH (n) RETURN count(n) as count')
                remaining_nodes = verify_result.single()["count"]
                
                if remaining_nodes == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"✓ Neo4j data cleared successfully in {elapsed:.2f}s")
                else:
                    logger.warning(f"⚠ Some nodes remain after deletion: {remaining_nodes}")
            else:
                logger.info("✓ Neo4j database was already empty")
                
    except Exception as e:
        logger.error(f"✗ Failed to clear Neo4j data: {e}")
        logger.debug(f"Stack trace: {traceback.format_exc()}")
        raise
    finally:
        driver.close()
        logger.debug("Neo4j connection closed")


def log_workflow_initialization():
    """Log workflow initialization details."""
    logger.info("\n" + "=" * 60)
    logger.info("INITIALIZING WORKFLOW")
    logger.info("=" * 60)
    
    # Check file existence
    file_path = Path("data/waste_manifest.pdf")
    logger.info(f"Checking input file: {file_path}")
    
    if file_path.exists():
        file_size = file_path.stat().st_size
        logger.info(f"✓ File exists - Size: {file_size} bytes ({file_size/1024:.1f} KB)")
        logger.debug(f"File path: {file_path.absolute()}")
    else:
        logger.error(f"✗ File not found: {file_path}")
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    logger.info("Initializing IngestionWorkflow...")
    start_time = time.time()
    
    try:
        workflow = IngestionWorkflow(
            llama_parse_api_key=os.getenv("LLAMA_PARSE_API_KEY"),
            neo4j_uri="bolt://localhost:7687",
            neo4j_username="neo4j",
            neo4j_password="EhsAI2024!",
            llm_model="gpt-4"
        )
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Workflow initialized successfully in {elapsed:.2f}s")
        logger.debug(f"Workflow class: {type(workflow).__name__}")
        
        return workflow
        
    except Exception as e:
        logger.error(f"✗ Failed to initialize workflow: {e}")
        logger.debug(f"Stack trace: {traceback.format_exc()}")
        raise


def process_document_with_logging(workflow, file_path, document_id, metadata):
    """Process document with detailed step-by-step logging."""
    logger.info("\n" + "=" * 60)
    logger.info("DOCUMENT PROCESSING WORKFLOW")
    logger.info("=" * 60)
    
    logger.info(f"Starting document processing...")
    logger.info(f"  Document ID: {document_id}")
    logger.info(f"  File path: {file_path}")
    logger.info(f"  Metadata: {json.dumps(metadata, indent=2)}")
    
    start_time = time.time()
    
    try:
        # Log the start of each major workflow step
        logger.info("\n🔄 STEP 1: Document Parsing")
        logger.info("-" * 40)
        step_start = time.time()
        
        result = workflow.process_document(
            file_path=file_path,
            document_id=document_id,
            metadata=metadata
        )
        
        total_elapsed = time.time() - start_time
        logger.info(f"\n✓ Document processing completed in {total_elapsed:.2f}s")
        
        # Log detailed results
        logger.info("\n" + "=" * 50)
        logger.info("PROCESSING RESULTS SUMMARY")
        logger.info("=" * 50)
        
        if result.get("success"):
            logger.info("✓ WORKFLOW STATUS: SUCCESS")
            
            # Log basic metrics
            logger.info(f"  Document ID: {result.get('document_id', 'N/A')}")
            logger.info(f"  Processing time: {total_elapsed:.2f} seconds")
            
            if 'neo4j_node_count' in result:
                logger.info(f"  Neo4j nodes created: {result['neo4j_node_count']}")
            if 'neo4j_relationship_count' in result:
                logger.info(f"  Neo4j relationships created: {result['neo4j_relationship_count']}")
            
            # Log extracted data in detail
            if result.get("extracted_data"):
                logger.info("\n📊 EXTRACTED DATA DETAILS:")
                logger.info("-" * 30)
                data = result["extracted_data"]
                
                # Convert to pretty JSON for logging
                json_data = json.dumps(data, indent=2, default=str)
                logger.debug(f"Full extracted data:\n{json_data}")
                
                # Log key fields
                key_fields = [
                    'manifest_number', 'ship_date', 'generator_name', 'generator_epa_id',
                    'transporter_name', 'transporter_epa_id', 'disposal_facility_name',
                    'disposal_facility_epa_id'
                ]
                
                for field in key_fields:
                    if field in data:
                        logger.info(f"  {field}: {data[field]}")
                
                # Log waste items details
                if data.get('waste_items'):
                    logger.info(f"\n  waste_items count: {len(data['waste_items'])}")
                    for i, item in enumerate(data['waste_items'], 1):
                        logger.info(f"    Item {i}:")
                        for key, value in item.items():
                            logger.info(f"      {key}: {value}")
                        
                # Log calculated emissions
                if 'total_emissions_kg_co2' in data:
                    logger.info(f"\n  🌱 Total emissions: {data['total_emissions_kg_co2']} kg CO2")
                    
            else:
                logger.warning("⚠ No extracted data found in results")
                
        else:
            logger.error("✗ WORKFLOW STATUS: FAILED")
            if result.get("errors"):
                logger.error("  Errors encountered:")
                for i, error in enumerate(result["errors"], 1):
                    logger.error(f"    {i}. {error}")
                    
        return result
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"✗ Document processing failed after {elapsed:.2f}s: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.debug(f"Full stack trace:\n{traceback.format_exc()}")
        
        # Return error result
        return {
            "success": False,
            "document_id": document_id,
            "errors": [str(e)],
            "processing_time": elapsed
        }


def verify_neo4j_data_detailed():
    """Verify data in Neo4j after ingestion with comprehensive logging."""
    logger.info("\n" + "=" * 60)
    logger.info("DETAILED NEO4J DATA VERIFICATION")
    logger.info("=" * 60)
    
    start_time = time.time()
    logger.info("Connecting to Neo4j for verification...")
    
    driver = GraphDatabase.driver(
        'bolt://localhost:7687', 
        auth=('neo4j', 'EhsAI2024!')
    )
    
    verification_results = {
        'node_counts': {},
        'relationship_counts': {},
        'data_samples': {},
        'relationship_verification': {}
    }
    
    try:
        with driver.session() as session:
            # 1. Count nodes by type
            logger.info("\n🏢 NODE TYPE ANALYSIS")
            logger.info("-" * 30)
            
            labels = [
                'Document', 'WasteManifest', 'WasteShipment', 'WasteGenerator', 
                'Transporter', 'DisposalFacility', 'WasteItem', 'Emission'
            ]
            
            total_nodes = 0
            for label in labels:
                query = f'MATCH (n:{label}) RETURN count(n) as count'
                logger.debug(f"Executing query: {query}")
                
                result = session.run(query)
                count = result.single()["count"]
                verification_results['node_counts'][label] = count
                total_nodes += count
                
                status = "✓" if count > 0 else "✗"
                logger.info(f"  {status} {label}: {count}")
                
                # Get sample data for non-zero counts
                if count > 0:
                    sample_query = f'MATCH (n:{label}) RETURN n LIMIT 1'
                    sample_result = session.run(sample_query)
                    sample_record = sample_result.single()
                    if sample_record:
                        sample_node = dict(sample_record['n'])
                        verification_results['data_samples'][label] = sample_node
                        logger.debug(f"  Sample {label} properties: {json.dumps(sample_node, indent=2, default=str)}")
            
            logger.info(f"\nTotal nodes created: {total_nodes}")
            
            # 2. Count relationships by type
            logger.info("\n🔗 RELATIONSHIP TYPE ANALYSIS")
            logger.info("-" * 35)
            
            rel_query = """
                MATCH (a)-[r]->(b)
                WHERE NOT a:__Node__ AND NOT b:__Node__
                RETURN type(r) as relationship, count(r) as count
                ORDER BY relationship
            """
            logger.debug(f"Executing relationship query: {rel_query}")
            
            result = session.run(rel_query)
            total_relationships = 0
            
            for record in result:
                rel_type = record['relationship']
                count = record['count']
                verification_results['relationship_counts'][rel_type] = count
                total_relationships += count
                
                status = "✓" if count > 0 else "✗"
                logger.info(f"  {status} {rel_type}: {count}")
                
                # Get sample relationship data
                sample_rel_query = f"""
                    MATCH (a)-[r:{rel_type}]->(b)
                    RETURN a, r, b LIMIT 1
                """
                sample_result = session.run(sample_rel_query)
                sample_record = sample_result.single()
                if sample_record:
                    logger.debug(f"  Sample {rel_type}: {labels(sample_record['a'])} -> {labels(sample_record['b'])}")
            
            logger.info(f"\nTotal relationships created: {total_relationships}")
            
            # 3. Detailed data inspection
            logger.info("\n📋 DETAILED DATA INSPECTION")
            logger.info("-" * 35)
            
            # Waste Manifest details
            logger.info("\n  WasteManifest Details:")
            wm_query = """
                MATCH (wm:WasteManifest)
                RETURN wm.manifest_number as manifest_number,
                       wm.ship_date as ship_date,
                       wm.waste_category as category,
                       wm.created_at as created_at
            """
            result = session.run(wm_query)
            for record in result:
                logger.info(f"    📄 Manifest: {record['manifest_number']}")
                logger.info(f"        Ship Date: {record['ship_date']}")
                logger.info(f"        Category: {record['category']}")
                logger.info(f"        Created: {record['created_at']}")
            
            # Waste Shipment details
            logger.info("\n  WasteShipment Details:")
            ws_query = """
                MATCH (ws:WasteShipment)
                RETURN ws.total_quantity as quantity,
                       ws.unit as unit,
                       ws.status as status,
                       ws.shipment_id as shipment_id
            """
            result = session.run(ws_query)
            for record in result:
                logger.info(f"    📦 Shipment: {record['shipment_id']}")
                logger.info(f"        Quantity: {record['quantity']} {record['unit']}")
                logger.info(f"        Status: {record['status']}")
            
            # Participants details
            logger.info("\n  Participants:")
            
            # Generator
            gen_query = "MATCH (g:WasteGenerator) RETURN g.name as name, g.epa_id as epa_id, g.address as address"
            result = session.run(gen_query)
            for record in result:
                logger.info(f"    🏭 Generator: {record['name']}")
                logger.info(f"        EPA ID: {record['epa_id']}")
                if record['address']:
                    logger.info(f"        Address: {record['address']}")
            
            # Transporter
            trans_query = "MATCH (t:Transporter) RETURN t.name as name, t.epa_id as epa_id"
            result = session.run(trans_query)
            for record in result:
                logger.info(f"    🚛 Transporter: {record['name']}")
                logger.info(f"        EPA ID: {record['epa_id']}")
            
            # Disposal Facility
            disp_query = "MATCH (d:DisposalFacility) RETURN d.name as name, d.epa_id as epa_id"
            result = session.run(disp_query)
            for record in result:
                logger.info(f"    🏢 Disposal Facility: {record['name']}")
                logger.info(f"        EPA ID: {record['epa_id']}")
            
            # Waste Items
            logger.info("\n  Waste Items:")
            wi_query = """
                MATCH (wi:WasteItem)
                RETURN wi.description as description,
                       wi.quantity as quantity,
                       wi.unit as unit,
                       wi.waste_code as waste_code,
                       wi.hazard_class as hazard_class
                ORDER BY wi.description
            """
            result = session.run(wi_query)
            for i, record in enumerate(result, 1):
                logger.info(f"    {i}. 🗑️ {record['description']}")
                logger.info(f"        Quantity: {record['quantity']} {record['unit']}")
                logger.info(f"        Waste Code: {record['waste_code']}")
                logger.info(f"        Hazard Class: {record['hazard_class']}")
            
            # Emissions
            logger.info("\n  Emissions Calculation:")
            em_query = """
                MATCH (wm:WasteManifest)-[:RESULTED_IN]->(e:Emission)
                RETURN e.amount as amount,
                       e.unit as unit,
                       e.emission_factor as factor,
                       e.calculation_method as method,
                       e.emission_category as category,
                       e.calculated_at as calculated_at
            """
            result = session.run(em_query)
            for record in result:
                logger.info(f"    🌱 Emissions: {record['amount']} {record['unit']}")
                logger.info(f"        Factor: {record['factor']}")
                logger.info(f"        Method: {record['method']}")
                logger.info(f"        Category: {record['category']}")
                logger.info(f"        Calculated: {record['calculated_at']}")
            
            # 4. Relationship verification
            logger.info("\n🔍 RELATIONSHIP VERIFICATION")
            logger.info("-" * 35)
            
            relationship_checks = [
                ('WasteManifest -> WasteShipment', 'MATCH (wm:WasteManifest)-[:CONTAINS]->(ws:WasteShipment) RETURN count(*) as count'),
                ('WasteShipment -> WasteGenerator', 'MATCH (ws:WasteShipment)-[:GENERATED_BY]->(wg:WasteGenerator) RETURN count(*) as count'),
                ('WasteShipment -> Transporter', 'MATCH (ws:WasteShipment)-[:TRANSPORTED_BY]->(t:Transporter) RETURN count(*) as count'),
                ('WasteShipment -> DisposalFacility', 'MATCH (ws:WasteShipment)-[:DISPOSED_AT]->(df:DisposalFacility) RETURN count(*) as count'),
                ('WasteShipment -> WasteItem', 'MATCH (ws:WasteShipment)-[:CONTAINS]->(wi:WasteItem) RETURN count(*) as count'),
                ('WasteManifest -> Emission', 'MATCH (wm:WasteManifest)-[:RESULTED_IN]->(e:Emission) RETURN count(*) as count'),
                ('Document -> WasteManifest', 'MATCH (d:Document)-[:EXTRACTED_AS]->(wm:WasteManifest) RETURN count(*) as count')
            ]
            
            passed_rel_checks = 0
            for check_name, query in relationship_checks:
                logger.debug(f"Executing relationship check: {query}")
                result = session.run(query)
                count = result.single()["count"]
                verification_results['relationship_verification'][check_name] = count
                
                status = "✓" if count > 0 else "✗"
                logger.info(f"  {status} {check_name}: {count}")
                if count > 0:
                    passed_rel_checks += 1
            
            total_rel_checks = len(relationship_checks)
            rel_success_rate = (passed_rel_checks / total_rel_checks) * 100
            
            # 5. Data quality checks
            logger.info("\n🎯 DATA QUALITY CHECKS")
            logger.info("-" * 30)
            
            quality_checks = []
            
            # Check for required fields in WasteManifest
            wm_required_fields = """
                MATCH (wm:WasteManifest)
                WHERE wm.manifest_number IS NOT NULL 
                  AND wm.ship_date IS NOT NULL
                RETURN count(wm) as valid_count
            """
            result = session.run(wm_required_fields)
            valid_manifests = result.single()["valid_count"]
            wm_total_result = session.run("MATCH (wm:WasteManifest) RETURN count(wm) as count")
            total_manifests = wm_total_result.single()["count"]
            
            if total_manifests > 0:
                wm_quality = (valid_manifests / total_manifests) * 100
                logger.info(f"  WasteManifest data completeness: {valid_manifests}/{total_manifests} ({wm_quality:.1f}%)")
                quality_checks.append(wm_quality == 100.0)
            
            # Check for waste items with quantities
            wi_quantity_check = """
                MATCH (wi:WasteItem)
                WHERE wi.quantity IS NOT NULL AND wi.quantity > 0
                RETURN count(wi) as valid_count
            """
            result = session.run(wi_quantity_check)
            valid_items = result.single()["valid_count"]
            wi_total_result = session.run("MATCH (wi:WasteItem) RETURN count(wi) as count")
            total_items = wi_total_result.single()["count"]
            
            if total_items > 0:
                wi_quality = (valid_items / total_items) * 100
                logger.info(f"  WasteItem quantity completeness: {valid_items}/{total_items} ({wi_quality:.1f}%)")
                quality_checks.append(wi_quality >= 80.0)  # Allow some flexibility
            
            elapsed = time.time() - start_time
            logger.info(f"\n✓ Verification completed in {elapsed:.2f}s")
            
            # Summary
            verification_results.update({
                'total_nodes': total_nodes,
                'total_relationships': total_relationships,
                'relationship_success_rate': rel_success_rate,
                'passed_relationship_checks': passed_rel_checks,
                'total_relationship_checks': total_rel_checks,
                'data_quality_passed': all(quality_checks) if quality_checks else False
            })
            
            return verification_results
            
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"✗ Verification failed after {elapsed:.2f}s: {e}")
        logger.debug(f"Stack trace: {traceback.format_exc()}")
        return None
    finally:
        driver.close()
        logger.debug("Neo4j verification connection closed")


def labels(node):
    """Helper function to extract node labels for logging."""
    return list(node.labels) if hasattr(node, 'labels') else ['Unknown']


def generate_comprehensive_report(workflow_result, verification_results):
    """Generate a comprehensive test report with detailed analysis."""
    logger.info("\n" + "=" * 80)
    logger.info("COMPREHENSIVE TEST REPORT")
    logger.info("=" * 80)
    
    report_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Report generated: {report_timestamp}")
    logger.info(f"Log file: {log_filename}")
    
    # Test overview
    logger.info("\n📋 TEST OVERVIEW")
    logger.info("-" * 20)
    logger.info(f"Test script: test_waste_manifest_ingestion_with_logging.py")
    logger.info(f"Input file: data/waste_manifest.pdf")
    logger.info(f"Workflow model: gpt-4")
    
    # Workflow results
    logger.info("\n🔄 WORKFLOW EXECUTION")
    logger.info("-" * 25)
    
    success = workflow_result.get("success", False)
    status_icon = "✅" if success else "❌"
    logger.info(f"{status_icon} Overall Status: {'SUCCESS' if success else 'FAILED'}")
    
    if success:
        logger.info(f"  Document ID: {workflow_result.get('document_id', 'N/A')}")
        logger.info(f"  Processing time: {workflow_result.get('processing_time', 0):.2f}s")
        
        if 'neo4j_node_count' in workflow_result:
            logger.info(f"  Nodes created: {workflow_result['neo4j_node_count']}")
        if 'neo4j_relationship_count' in workflow_result:
            logger.info(f"  Relationships created: {workflow_result['neo4j_relationship_count']}")
    else:
        logger.info("  ❌ Workflow failed!")
        if workflow_result.get("errors"):
            for i, error in enumerate(workflow_result["errors"], 1):
                logger.info(f"    {i}. {error}")
    
    # Verification results
    if verification_results:
        logger.info("\n🔍 DATA VERIFICATION")
        logger.info("-" * 22)
        
        # Node analysis
        logger.info("  Node Creation:")
        total_nodes = verification_results.get('total_nodes', 0)
        node_counts = verification_results.get('node_counts', {})
        
        for node_type, count in node_counts.items():
            status = "✅" if count > 0 else "❌"
            logger.info(f"    {status} {node_type}: {count}")
        
        logger.info(f"    📊 Total nodes: {total_nodes}")
        
        # Relationship analysis
        logger.info("\n  Relationship Creation:")
        total_rels = verification_results.get('total_relationships', 0)
        rel_counts = verification_results.get('relationship_counts', {})
        
        for rel_type, count in rel_counts.items():
            status = "✅" if count > 0 else "❌"
            logger.info(f"    {status} {rel_type}: {count}")
        
        logger.info(f"    📊 Total relationships: {total_rels}")
        
        # Relationship verification
        logger.info("\n  Relationship Integrity:")
        rel_verification = verification_results.get('relationship_verification', {})
        passed_checks = verification_results.get('passed_relationship_checks', 0)
        total_checks = verification_results.get('total_relationship_checks', 0)
        success_rate = verification_results.get('relationship_success_rate', 0)
        
        for check_name, count in rel_verification.items():
            status = "✅" if count > 0 else "❌"
            logger.info(f"    {status} {check_name}: {count}")
        
        logger.info(f"    📈 Success rate: {passed_checks}/{total_checks} ({success_rate:.1f}%)")
        
        # Data quality
        quality_passed = verification_results.get('data_quality_passed', False)
        quality_status = "✅" if quality_passed else "⚠️"
        logger.info(f"\n  {quality_status} Data Quality: {'PASSED' if quality_passed else 'NEEDS REVIEW'}")
        
    else:
        logger.info("\n❌ DATA VERIFICATION: FAILED")
    
    # Final assessment
    logger.info("\n🎯 FINAL ASSESSMENT")
    logger.info("-" * 22)
    
    overall_success = (
        workflow_result.get("success", False) and
        verification_results is not None and
        verification_results.get('total_nodes', 0) > 0 and
        verification_results.get('relationship_success_rate', 0) >= 80.0
    )
    
    if overall_success:
        logger.info("🎉 TEST PASSED - Waste manifest ingestion completed successfully!")
        logger.info("   ✅ Document processed")
        logger.info("   ✅ Data extracted")
        logger.info("   ✅ Neo4j nodes created")
        logger.info("   ✅ Relationships established")
        logger.info("   ✅ Data verification passed")
    else:
        logger.info("⚠️  TEST INCOMPLETE - Some issues detected:")
        if not workflow_result.get("success", False):
            logger.info("   ❌ Workflow execution failed")
        if verification_results is None:
            logger.info("   ❌ Data verification failed")
        elif verification_results.get('total_nodes', 0) == 0:
            logger.info("   ❌ No data created in Neo4j")
        elif verification_results.get('relationship_success_rate', 0) < 80.0:
            logger.info("   ⚠️  Relationship integrity issues")
    
    logger.info("\n📁 ARTIFACTS CREATED")
    logger.info("-" * 22)
    logger.info(f"  📄 Detailed log file: {log_filename}")
    if verification_results:
        logger.info(f"  📊 Neo4j nodes: {verification_results.get('total_nodes', 0)}")
        logger.info(f"  🔗 Neo4j relationships: {verification_results.get('total_relationships', 0)}")
    
    logger.info("\n💡 MONITORING")
    logger.info("-" * 15)
    logger.info(f"Monitor live logs with: tail -f {log_filename}")
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETED")
    logger.info("=" * 80)


def main():
    """Main test execution function."""
    overall_start_time = time.time()
    
    try:
        # System information
        log_system_info()
        
        # Clear existing data
        clear_neo4j_data()
        
        # Initialize workflow
        workflow = log_workflow_initialization()
        
        # Process document
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        document_id = f"waste_manifest_{timestamp}"
        metadata = {
            "source": "test_enhanced", 
            "document_type": "waste_manifest",
            "test_run": True,
            "enhanced_logging": True
        }
        
        workflow_result = process_document_with_logging(
            workflow=workflow,
            file_path="data/waste_manifest.pdf",
            document_id=document_id,
            metadata=metadata
        )
        
        # Verify Neo4j data
        verification_results = verify_neo4j_data_detailed()
        
        # Generate comprehensive report
        generate_comprehensive_report(workflow_result, verification_results)
        
        # Final timing
        total_elapsed = time.time() - overall_start_time
        logger.info(f"\n⏱️  Total test execution time: {total_elapsed:.2f} seconds")
        
        return workflow_result.get("success", False)
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Test interrupted by user")
        return False
    except Exception as e:
        total_elapsed = time.time() - overall_start_time
        logger.error(f"\n💥 Test failed with unexpected error after {total_elapsed:.2f}s: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.debug(f"Full traceback:\n{traceback.format_exc()}")
        return False
    finally:
        # Ensure logs are flushed
        for handler in logger.handlers:
            handler.flush()


if __name__ == "__main__":
    success = main()
    exit_code = 0 if success else 1
    logger.info(f"Exiting with code: {exit_code}")
    sys.exit(exit_code)