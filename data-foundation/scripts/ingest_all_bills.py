#!/usr/bin/env python3
"""
Simple script to ingest all utility bills into the EHS system.
Processes electric bill and water bill without clearing the database.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_neo4j_summary():
    """Get current database summary."""
    driver = GraphDatabase.driver(
        'bolt://localhost:7687', 
        auth=('neo4j', 'EhsAI2024!')
    )
    
    summary = {
        'total_nodes': 0,
        'total_relationships': 0,
        'node_counts': {},
        'relationship_counts': {}
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
                
    except Exception as e:
        logger.error(f"Failed to get Neo4j summary: {e}")
    finally:
        driver.close()
    
    return summary


def print_summary(title, summary):
    """Print database summary in a formatted way."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Total Nodes: {summary['total_nodes']}")
    print(f"Total Relationships: {summary['total_relationships']}")
    
    if summary['node_counts']:
        print(f"\nNode Counts by Type:")
        for label, count in sorted(summary['node_counts'].items()):
            print(f"  {label}: {count}")
    
    if summary['relationship_counts']:
        print(f"\nRelationship Counts by Type:")
        for rel_type, count in sorted(summary['relationship_counts'].items()):
            print(f"  {rel_type}: {count}")


def process_document(workflow, file_path, doc_type):
    """Process a single document and return result summary."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    document_id = f"{doc_type}_{timestamp}"
    
    logger.info(f"\nProcessing {doc_type}: {file_path}")
    logger.info("-" * 50)
    
    try:
        result = workflow.process_document(
            file_path=file_path,
            document_id=document_id,
            metadata={"source": "batch_ingestion", "document_type": doc_type}
        )
        
        # Extract key information from result
        status = result.get("status", "unknown")
        errors = result.get("errors", [])
        nodes_created = len(result.get("neo4j_nodes", []))
        relationships_created = len(result.get("neo4j_relationships", []))
        
        # Log status
        if status == "completed":
            logger.info(f"✓ {doc_type} processed successfully")
            logger.info(f"  • Document ID: {document_id}")
            logger.info(f"  • Nodes created: {nodes_created}")
            logger.info(f"  • Relationships created: {relationships_created}")
            
            # Show key extracted data
            extracted = result.get("extracted_data", {})
            if doc_type == "electric_bill":
                logger.info(f"  • Account: {extracted.get('account_number', 'N/A')}")
                logger.info(f"  • Total kWh: {extracted.get('total_kwh', 'N/A')}")
                logger.info(f"  • Total Cost: ${extracted.get('total_cost', 'N/A')}")
            elif doc_type == "water_bill":
                logger.info(f"  • Total Gallons: {extracted.get('total_gallons', 'N/A')}")
                logger.info(f"  • Total Cost: ${extracted.get('total_cost', 'N/A')}")
                
            return {"success": True, "errors": [], "nodes": nodes_created, "relationships": relationships_created}
        else:
            logger.error(f"✗ {doc_type} processing failed")
            for error in errors:
                logger.error(f"  • {error}")
            return {"success": False, "errors": errors, "nodes": 0, "relationships": 0}
            
    except Exception as e:
        logger.error(f"✗ Exception processing {doc_type}: {e}")
        return {"success": False, "errors": [str(e)], "nodes": 0, "relationships": 0}


def main():
    """Main ingestion script."""
    print("=" * 80)
    print("EHS AI DEMO - UTILITY BILLS INGESTION")
    print("=" * 80)
    
    # Check environment variables
    llama_parse_key = os.getenv("LLAMA_PARSE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not llama_parse_key:
        logger.error("LLAMA_PARSE_API_KEY not found in environment!")
        return
    
    if not openai_key:
        logger.error("OPENAI_API_KEY not found in environment!")
        return
    
    logger.info("Environment variables loaded successfully")
    
    # Get initial database state
    initial_summary = get_neo4j_summary()
    print_summary("DATABASE STATE - BEFORE INGESTION", initial_summary)
    
    # Initialize workflow
    logger.info(f"\nInitializing IngestionWorkflow...")
    try:
        workflow = IngestionWorkflow(
            llama_parse_api_key=llama_parse_key,
            neo4j_uri="bolt://localhost:7687",
            neo4j_username="neo4j", 
            neo4j_password="EhsAI2024!",
            llm_model="gpt-4"
        )
        logger.info("✓ Workflow initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize workflow: {e}")
        return
    
    # Process documents
    documents_to_process = [
        ("data/electric_bill.pdf", "electric_bill"),
        ("data/water_bill.pdf", "water_bill")
    ]
    
    results = []
    total_nodes_created = 0
    total_relationships_created = 0
    successful_ingestions = 0
    
    for file_path, doc_type in documents_to_process:
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"✗ File not found: {file_path}")
            results.append({"success": False, "errors": [f"File not found: {file_path}"], "nodes": 0, "relationships": 0})
            continue
        
        result = process_document(workflow, file_path, doc_type)
        results.append(result)
        
        if result["success"]:
            successful_ingestions += 1
            total_nodes_created += result["nodes"]
            total_relationships_created += result["relationships"]
    
    # Get final database state
    final_summary = get_neo4j_summary()
    print_summary("DATABASE STATE - AFTER INGESTION", final_summary)
    
    # Show ingestion summary
    print(f"\n{'='*60}")
    print("INGESTION SUMMARY")
    print(f"{'='*60}")
    print(f"Documents Processed: {len(documents_to_process)}")
    print(f"Successful Ingestions: {successful_ingestions}")
    print(f"Failed Ingestions: {len(documents_to_process) - successful_ingestions}")
    print(f"\nNodes Created This Session: {total_nodes_created}")
    print(f"Relationships Created This Session: {total_relationships_created}")
    
    # Calculate differences
    nodes_added = final_summary['total_nodes'] - initial_summary['total_nodes']
    relationships_added = final_summary['total_relationships'] - initial_summary['total_relationships']
    
    print(f"\nDatabase Growth:")
    print(f"  Total Nodes Added: {nodes_added}")
    print(f"  Total Relationships Added: {relationships_added}")
    
    # Show any errors
    all_errors = []
    for i, result in enumerate(results):
        if not result["success"]:
            doc_name = documents_to_process[i][1]
            all_errors.extend([f"{doc_name}: {error}" for error in result["errors"]])
    
    if all_errors:
        print(f"\nErrors Encountered:")
        for error in all_errors:
            print(f"  • {error}")
    
    print(f"\n{'='*60}")
    if successful_ingestions == len(documents_to_process):
        print("✓ ALL DOCUMENTS INGESTED SUCCESSFULLY!")
    elif successful_ingestions > 0:
        print(f"⚠ PARTIAL SUCCESS: {successful_ingestions}/{len(documents_to_process)} documents ingested")
    else:
        print("✗ ALL INGESTIONS FAILED!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()