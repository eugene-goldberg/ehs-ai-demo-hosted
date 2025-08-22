#!/usr/bin/env python3
"""
Comprehensive script to ingest all EHS documents into the system.
Processes electric bill, water bill, and waste manifest documents.
Includes comprehensive logging, error handling, and Neo4j verification.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables from backend/.env
env_path = Path(__file__).parent.parent / "backend" / ".env"
load_dotenv(env_path, override=True)

from backend.src.workflows.ingestion_workflow import IngestionWorkflow

# Create logs directory if it doesn't exist
script_dir = Path(__file__).parent
logs_dir = script_dir / "logs"
logs_dir.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(logs_dir / f'ingestion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def get_neo4j_summary():
    """Get comprehensive database summary with detailed statistics."""
    driver = GraphDatabase.driver(
        'bolt://localhost:7687', 
        auth=('neo4j', 'EhsAI2024!')
    )
    
    summary = {
        'total_nodes': 0,
        'total_relationships': 0,
        'node_counts': {},
        'relationship_counts': {},
        'detailed_stats': {}
    }
    
    try:
        with driver.session() as session:
            # Get total node count
            result = session.run('MATCH (n) RETURN count(n) as count')
            summary['total_nodes'] = result.single()['count']
            
            # Get total relationship count
            result = session.run('MATCH ()-[r]->() RETURN count(r) as count')
            summary['total_relationships'] = result.single()['count']
            
            # Get node counts by label
            result = session.run('CALL db.labels() YIELD label RETURN label')
            labels = [record['label'] for record in result]
            
            for label in labels:
                if not label.startswith('_'):  # Skip system labels
                    result = session.run(f'MATCH (n:{label}) RETURN count(n) as count')
                    summary['node_counts'][label] = result.single()['count']
            
            # Get relationship counts by type
            result = session.run("""
                MATCH ()-[r]->() 
                RETURN type(r) as rel_type, count(r) as count
                ORDER BY rel_type
            """)
            for record in result:
                summary['relationship_counts'][record['rel_type']] = record['count']
            
            # Get detailed statistics for key entity types
            entity_queries = {
                'Documents': 'MATCH (d:Document) RETURN count(d) as count',
                'Facilities': 'MATCH (f:Facility) RETURN count(f) as count',
                'Customers': 'MATCH (c:Customer) RETURN count(c) as count',
                'UtilityProviders': 'MATCH (u:UtilityProvider) RETURN count(u) as count',
                'UtilityBills': 'MATCH (u:UtilityBill) RETURN count(u) as count',
                'WaterBills': 'MATCH (w:WaterBill) RETURN count(w) as count',
                'WasteManifests': 'MATCH (w:WasteManifest) RETURN count(w) as count',
                'WasteItems': 'MATCH (w:WasteItem) RETURN count(w) as count',
                'Meters': 'MATCH (m:Meter) RETURN count(m) as count',
                'Emissions': 'MATCH (e:Emission) RETURN count(e) as count'
            }
            
            for entity, query in entity_queries.items():
                try:
                    result = session.run(query)
                    summary['detailed_stats'][entity] = result.single()['count']
                except Exception as e:
                    summary['detailed_stats'][entity] = 0
                    
    except Exception as e:
        logger.error(f"Failed to get Neo4j summary: {e}")
    finally:
        driver.close()
    
    return summary


def clear_neo4j_database():
    """Clear all data from Neo4j database."""
    driver = GraphDatabase.driver(
        'bolt://localhost:7687', 
        auth=('neo4j', 'EhsAI2024!')
    )
    
    try:
        with driver.session() as session:
            # Delete all nodes and relationships
            logger.info("Clearing all Neo4j data...")
            result = session.run("MATCH (n) DETACH DELETE n")
            logger.info("✓ Neo4j database cleared successfully")
            
            # Verify database is empty
            count_result = session.run("MATCH (n) RETURN COUNT(n) as count").single()
            node_count = count_result['count']
            
            if node_count == 0:
                logger.info(f"✓ Database verified empty: {node_count} nodes")
                return True
            else:
                logger.warning(f"⚠️ Database may not be fully cleared: {node_count} nodes remaining")
                return False
                
    except Exception as e:
        logger.error(f"✗ Failed to clear Neo4j database: {e}")
        return False
    finally:
        driver.close()


def print_summary(title, summary):
    """Print database summary in a formatted way."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    print(f"Total Nodes: {summary['total_nodes']}")
    print(f"Total Relationships: {summary['total_relationships']}")
    
    if summary['node_counts']:
        print(f"\nNode Counts by Label:")
        for label, count in sorted(summary['node_counts'].items()):
            print(f"  {label}: {count}")
    
    if summary['relationship_counts']:
        print(f"\nRelationship Counts by Type:")
        for rel_type, count in sorted(summary['relationship_counts'].items()):
            print(f"  {rel_type}: {count}")
    
    if summary['detailed_stats']:
        print(f"\nDetailed Entity Statistics:")
        for entity, count in summary['detailed_stats'].items():
            if count > 0:
                print(f"  {entity}: {count}")


def generate_document_id(doc_type):
    """Generate a unique document ID with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
    return f"{doc_type}_{timestamp}"


def process_document(workflow, file_path, doc_type):
    """Process a single document and return detailed result summary."""
    document_id = generate_document_id(doc_type)
    
    logger.info(f"\n{'*'*60}")
    logger.info(f"Processing {doc_type.upper()}: {file_path}")
    logger.info(f"Document ID: {document_id}")
    logger.info(f"{'*'*60}")
    
    start_time = datetime.now()
    
    try:
        result = workflow.process_document(
            file_path=file_path,
            document_id=document_id,
            metadata={
                "source": "comprehensive_batch_ingestion", 
                "document_type": doc_type,
                "processed_at": start_time.isoformat()
            }
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Extract detailed information from result
        status = result.get("status", "unknown")
        errors = result.get("errors", [])
        nodes_created = len(result.get("neo4j_nodes", []))
        relationships_created = len(result.get("neo4j_relationships", []))
        validation_results = result.get("validation_results", {})
        
        # Log comprehensive status
        if status == "completed":
            logger.info(f"✓ {doc_type.upper()} PROCESSED SUCCESSFULLY")
            logger.info(f"  • Document ID: {document_id}")
            logger.info(f"  • Processing Time: {processing_time:.2f} seconds")
            logger.info(f"  • Nodes Created: {nodes_created}")
            logger.info(f"  • Relationships Created: {relationships_created}")
            
            # Show validation results
            if validation_results:
                logger.info(f"  • Validation: {'PASSED' if validation_results.get('valid', False) else 'FAILED'}")
                if validation_results.get('warnings'):
                    logger.warning(f"  • Validation Warnings: {len(validation_results['warnings'])}")
                    for warning in validation_results['warnings']:
                        logger.warning(f"    - {warning}")
            
            # Show key extracted data
            extracted = result.get("extracted_data", {})
            if doc_type == "electric_bill":
                logger.info(f"  • Account Number: {extracted.get('account_number', 'N/A')}")
                logger.info(f"  • Total kWh: {extracted.get('total_kwh', 'N/A')}")
                logger.info(f"  • Total Cost: ${extracted.get('total_cost', 'N/A')}")
                logger.info(f"  • Billing Period: {extracted.get('billing_period_start', 'N/A')} to {extracted.get('billing_period_end', 'N/A')}")
                
            elif doc_type == "water_bill":
                logger.info(f"  • Total Gallons: {extracted.get('total_gallons', 'N/A')}")
                logger.info(f"  • Total Cost: ${extracted.get('total_cost', 'N/A')}")
                logger.info(f"  • Billing Period: {extracted.get('billing_period_start', 'N/A')} to {extracted.get('billing_period_end', 'N/A')}")
                
            elif doc_type == "waste_manifest":
                logger.info(f"  • Manifest Number: {extracted.get('manifest_tracking_number', 'N/A')}")
                logger.info(f"  • Total Waste Quantity: {extracted.get('total_waste_quantity', 'N/A')} {extracted.get('total_waste_unit', '')}")
                logger.info(f"  • Disposal Method: {extracted.get('disposal_method', 'N/A')}")
                logger.info(f"  • Generator: {extracted.get('generator_name', 'N/A')}")
                logger.info(f"  • Disposal Facility: {extracted.get('facility_name', 'N/A')}")
                waste_items = extracted.get('waste_items', [])
                logger.info(f"  • Waste Items: {len(waste_items)} types")
                
            return {
                "success": True, 
                "errors": [], 
                "nodes": nodes_created, 
                "relationships": relationships_created,
                "processing_time": processing_time,
                "document_id": document_id,
                "validation": validation_results
            }
        else:
            logger.error(f"✗ {doc_type.upper()} PROCESSING FAILED")
            logger.error(f"  • Status: {status}")
            logger.error(f"  • Processing Time: {processing_time:.2f} seconds")
            for error in errors:
                logger.error(f"  • {error}")
            
            return {
                "success": False, 
                "errors": errors, 
                "nodes": 0, 
                "relationships": 0,
                "processing_time": processing_time,
                "document_id": document_id,
                "validation": validation_results
            }
            
    except Exception as e:
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        logger.error(f"✗ EXCEPTION PROCESSING {doc_type.upper()}: {e}")
        logger.error(f"  • Processing Time: {processing_time:.2f} seconds")
        return {
            "success": False, 
            "errors": [str(e)], 
            "nodes": 0, 
            "relationships": 0,
            "processing_time": processing_time,
            "document_id": document_id,
            "validation": {}
        }


def query_neo4j_verification():
    """Query Neo4j for comprehensive verification of ingested data."""
    driver = GraphDatabase.driver(
        'bolt://localhost:7687', 
        auth=('neo4j', 'EhsAI2024!')
    )
    
    verification_data = {}
    
    try:
        with driver.session() as session:
            # Get sample data from each document type
            queries = {
                "Electric Bills": """
                    MATCH (d:Document)-[:EXTRACTED_TO]->(ub:UtilityBill)
                    OPTIONAL MATCH (ub)-[:BILLED_TO]->(f:Facility)
                    OPTIONAL MATCH (ub)-[:RESULTED_IN]->(e:Emission)
                    RETURN d.id as doc_id, d.account_number, ub.total_kwh, ub.total_cost, 
                           f.name as facility, e.amount as emission_amount
                    LIMIT 5
                """,
                "Water Bills": """
                    MATCH (d:Document)-[:EXTRACTED_TO]->(wb:WaterBill)
                    OPTIONAL MATCH (wb)-[:BILLED_TO]->(f:Facility)
                    OPTIONAL MATCH (wb)-[:RESULTED_IN]->(e:Emission)
                    RETURN d.id as doc_id, wb.total_gallons, wb.total_cost, 
                           f.name as facility, e.amount as emission_amount
                    LIMIT 5
                """,
                "Waste Manifests": """
                    MATCH (d:Document)-[:TRACKS]->(wm:WasteManifest)-[:DOCUMENTS]->(ws:WasteShipment)
                    OPTIONAL MATCH (ws)-[:GENERATED_BY]->(wg:WasteGenerator)
                    OPTIONAL MATCH (ws)-[:DISPOSED_AT]->(df:DisposalFacility)
                    OPTIONAL MATCH (ws)-[:RESULTED_IN]->(e:Emission)
                    RETURN d.id as doc_id, wm.manifest_tracking_number, wm.total_quantity,
                           wg.name as generator, df.name as disposal_facility, e.amount as emission_amount
                    LIMIT 5
                """,
                "All Emissions": """
                    MATCH (e:Emission)
                    RETURN e.id, e.amount, e.unit, e.source_type, e.calculation_method
                    ORDER BY e.amount DESC
                    LIMIT 10
                """,
                "All Facilities": """
                    MATCH (f:Facility)
                    OPTIONAL MATCH (f)<-[:BILLED_TO]-()
                    OPTIONAL MATCH (f)<-[:MONITORS]-(m:Meter)
                    RETURN f.id, f.name, f.address, count(m) as meter_count
                    ORDER BY f.name
                    LIMIT 10
                """
            }
            
            for query_name, query in queries.items():
                try:
                    result = session.run(query)
                    verification_data[query_name] = [dict(record) for record in result]
                except Exception as e:
                    logger.error(f"Error running verification query '{query_name}': {e}")
                    verification_data[query_name] = []
                    
    except Exception as e:
        logger.error(f"Failed to run verification queries: {e}")
    finally:
        driver.close()
    
    return verification_data


def print_verification_results(verification_data):
    """Print verification results in a formatted way."""
    print(f"\n{'='*80}")
    print("NEO4J DATA VERIFICATION")
    print(f"{'='*80}")
    
    for query_name, data in verification_data.items():
        print(f"\n{query_name}:")
        print("-" * len(query_name))
        
        if not data:
            print("  No data found")
            continue
            
        for i, record in enumerate(data, 1):
            print(f"  {i}. {dict(record)}")
            if i >= 3:  # Limit display to first 3 records
                if len(data) > 3:
                    print(f"     ... and {len(data) - 3} more records")
                break


def main():
    """Main comprehensive ingestion script."""
    print("=" * 100)
    print("EHS AI DEMO - COMPREHENSIVE DOCUMENT INGESTION")
    print("Processing: Electric Bill, Water Bill, and Waste Manifest")
    print("=" * 100)
    
    # Check environment variables
    required_env_vars = {
        "LLAMA_PARSE_API_KEY": os.getenv("LLAMA_PARSE_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "NEO4J_URI": os.getenv("NEO4J_URI"),
        "NEO4J_USERNAME": os.getenv("NEO4J_USERNAME"),
        "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD")
    }
    
    missing_vars = [var for var, value in required_env_vars.items() if not value]
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return
    
    logger.info("✓ All required environment variables found")
    
    # Get initial database state
    logger.info("Getting initial database state...")
    initial_summary = get_neo4j_summary()
    print_summary("DATABASE STATE - BEFORE INGESTION", initial_summary)
    
    # Clear the database before ingestion
    logger.info("\nClearing Neo4j database before ingestion...")
    if clear_neo4j_database():
        logger.info("✓ Database cleared successfully")
        # Get database state after clearing
        post_clear_summary = get_neo4j_summary()
        print_summary("DATABASE STATE - AFTER CLEARING", post_clear_summary)
    else:
        logger.error("✗ Failed to clear database, continuing anyway...")
    
    # Initialize workflow
    logger.info(f"\nInitializing IngestionWorkflow...")
    try:
        workflow = IngestionWorkflow(
            llama_parse_api_key=required_env_vars["LLAMA_PARSE_API_KEY"],
            neo4j_uri=required_env_vars["NEO4J_URI"],
            neo4j_username=required_env_vars["NEO4J_USERNAME"], 
            neo4j_password=required_env_vars["NEO4J_PASSWORD"],
            llm_model="gpt-4o-2024-11-20"  # Use the latest model from env config
        )
        logger.info("✓ IngestionWorkflow initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize workflow: {e}")
        return
    
    # Process all documents
    # Use absolute paths based on script location
    data_dir = Path(__file__).parent.parent / "data"
    documents_to_process = [
        (str(data_dir / "electric_bill.pdf"), "electric_bill"),
        (str(data_dir / "water_bill.pdf"), "water_bill"),
        (str(data_dir / "waste_manifest.pdf"), "waste_manifest")
    ]
    
    results = []
    total_nodes_created = 0
    total_relationships_created = 0
    total_processing_time = 0
    successful_ingestions = 0
    
    script_start_time = datetime.now()
    
    for file_path, doc_type in documents_to_process:
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"✗ File not found: {file_path}")
            results.append({
                "success": False, 
                "errors": [f"File not found: {file_path}"], 
                "nodes": 0, 
                "relationships": 0,
                "processing_time": 0,
                "document_id": f"{doc_type}_missing",
                "validation": {}
            })
            continue
        
        logger.info(f"File exists: {file_path} ({os.path.getsize(file_path)} bytes)")
        
        result = process_document(workflow, file_path, doc_type)
        results.append(result)
        
        if result["success"]:
            successful_ingestions += 1
            total_nodes_created += result["nodes"]
            total_relationships_created += result["relationships"]
        
        total_processing_time += result["processing_time"]
    
    script_end_time = datetime.now()
    script_total_time = (script_end_time - script_start_time).total_seconds()
    
    # Get final database state
    logger.info("Getting final database state...")
    final_summary = get_neo4j_summary()
    print_summary("DATABASE STATE - AFTER INGESTION", final_summary)
    
    # Get verification data
    logger.info("Running Neo4j verification queries...")
    verification_data = query_neo4j_verification()
    print_verification_results(verification_data)
    
    # Show comprehensive ingestion summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE INGESTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total Script Runtime: {script_total_time:.2f} seconds")
    print(f"Total Processing Time: {total_processing_time:.2f} seconds")
    print(f"Documents Attempted: {len(documents_to_process)}")
    print(f"Successful Ingestions: {successful_ingestions}")
    print(f"Failed Ingestions: {len(documents_to_process) - successful_ingestions}")
    print(f"\nData Created This Session:")
    print(f"  • Nodes Created: {total_nodes_created}")
    print(f"  • Relationships Created: {total_relationships_created}")
    
    # Calculate database growth
    nodes_added = final_summary['total_nodes'] - initial_summary['total_nodes']
    relationships_added = final_summary['total_relationships'] - initial_summary['total_relationships']
    
    print(f"\nDatabase Growth:")
    print(f"  • Total Nodes Added: {nodes_added}")
    print(f"  • Total Relationships Added: {relationships_added}")
    
    # Show detailed results for each document
    print(f"\nDetailed Results by Document:")
    for i, result in enumerate(results):
        doc_name, doc_type = documents_to_process[i] if i < len(documents_to_process) else ("Unknown", "unknown")
        status_icon = "✓" if result["success"] else "✗"
        print(f"  {status_icon} {doc_type.upper()}: {result['processing_time']:.2f}s, "
              f"{result['nodes']} nodes, {result['relationships']} relationships")
        if result.get("document_id"):
            print(f"    Document ID: {result['document_id']}")
        if not result["success"] and result["errors"]:
            for error in result["errors"][:2]:  # Show first 2 errors
                print(f"    Error: {error}")
    
    # Show validation summary
    print(f"\nValidation Summary:")
    for i, result in enumerate(results):
        if result["success"]:
            doc_type = documents_to_process[i][1]
            validation = result.get("validation", {})
            if validation:
                status = "PASSED" if validation.get("valid", False) else "FAILED"
                warnings = len(validation.get("warnings", []))
                issues = len(validation.get("issues", []))
                print(f"  • {doc_type.upper()}: {status} ({warnings} warnings, {issues} issues)")
    
    # Show any errors encountered
    all_errors = []
    for i, result in enumerate(results):
        if not result["success"]:
            doc_name = documents_to_process[i][1] if i < len(documents_to_process) else "unknown"
            all_errors.extend([f"{doc_name}: {error}" for error in result["errors"]])
    
    if all_errors:
        print(f"\nErrors Encountered:")
        for error in all_errors:
            print(f"  • {error}")
    
    # Final status
    print(f"\n{'='*80}")
    if successful_ingestions == len(documents_to_process):
        print("🎉 ALL DOCUMENTS INGESTED SUCCESSFULLY!")
        print("   The EHS knowledge graph has been populated with comprehensive data.")
    elif successful_ingestions > 0:
        print(f"⚠️  PARTIAL SUCCESS: {successful_ingestions}/{len(documents_to_process)} documents ingested")
        print("   Some documents were processed, but issues occurred with others.")
    else:
        print("❌ ALL INGESTIONS FAILED!")
        print("   No documents were successfully processed. Check logs for details.")
    print(f"{'='*80}")
    
    # Log file location
    log_filename = f'ingestion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    print(f"\nDetailed logs saved to: {logs_dir / log_filename}")


if __name__ == "__main__":
    main()