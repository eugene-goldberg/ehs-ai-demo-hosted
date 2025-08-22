#!/usr/bin/env python3
"""
Test script for water bill ingestion in the EHS document processing pipeline.
Tests the full pipeline with the water bill PDF.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import json
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables from backend/.env
env_path = Path(__file__).parent.parent / "backend" / ".env"
load_dotenv(env_path, override=True)

from backend.src.workflows.ingestion_workflow import IngestionWorkflow
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clear_neo4j_data():
    """Clear all data from Neo4j database."""
    logger.info("Clearing existing Neo4j data...")
    
    driver = GraphDatabase.driver(
        'bolt://localhost:7687', 
        auth=('neo4j', 'EhsAI2024!')
    )
    
    try:
        with driver.session() as session:
            session.run('MATCH (n) DETACH DELETE n')
        logger.info("Neo4j data cleared successfully")
    except Exception as e:
        logger.error(f"Failed to clear Neo4j data: {e}")
    finally:
        driver.close()


def test_water_bill_ingestion():
    """Test water bill ingestion workflow."""
    logger.info("=" * 80)
    logger.info("Testing Water Bill Ingestion Workflow")
    logger.info("=" * 80)
    
    # Check environment variables
    llama_parse_key = os.getenv("LLAMA_PARSE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not llama_parse_key:
        logger.error("LLAMA_PARSE_API_KEY not set!")
        return
    
    if not openai_key:
        logger.error("OPENAI_API_KEY not set!")
        return
    
    # Clear existing data
    clear_neo4j_data()
    
    try:
        # Initialize workflow
        workflow = IngestionWorkflow(
            llama_parse_api_key=llama_parse_key,
            neo4j_uri="bolt://localhost:7687",
            neo4j_username="neo4j",
            neo4j_password="EhsAI2024!",
            llm_model="gpt-4"
        )
        
        # Process water bill
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = "data/water_bill.pdf"
        
        logger.info(f"Processing water bill: {file_path}")
        
        result = workflow.process_document(
            file_path=file_path,
            document_id=f"water_bill_{timestamp}",
            metadata={"source": "test", "document_type": "water_bill"}
        )
        
        # Log results
        logger.info("\n" + "=" * 40)
        logger.info("WORKFLOW RESULTS")
        logger.info("=" * 40)
        
        if result.get("success"):
            logger.info("✓ Workflow completed successfully")
            logger.info(f"✓ Document ID: {result.get('document_id')}")
            logger.info(f"✓ Neo4j Nodes created: {result.get('neo4j_node_count', 0)}")
            logger.info(f"✓ Neo4j Relationships created: {result.get('neo4j_relationship_count', 0)}")
            
            # Show extracted data summary
            if result.get("extracted_data"):
                logger.info("\nExtracted Data Summary:")
                data = result["extracted_data"]
                logger.info(f"  - Account Number: {data.get('account_number')}")
                logger.info(f"  - Billing Period: {data.get('billing_period_start')} to {data.get('billing_period_end')}")
                logger.info(f"  - Total Gallons: {data.get('total_gallons')}")
                logger.info(f"  - Total Cost: ${data.get('total_cost')}")
                logger.info(f"  - Customer: {data.get('customer_name')}")
                logger.info(f"  - Facility: {data.get('facility_name')}")
                logger.info(f"  - Provider: {data.get('provider_name')}")
                
                # Show meter readings
                if data.get("meter_readings"):
                    logger.info(f"  - Meter Readings: {len(data['meter_readings'])} meters")
                    for meter in data["meter_readings"]:
                        logger.info(f"    • {meter.get('meter_id')}: {meter.get('usage')} {meter.get('unit', 'gallons')}")
        else:
            logger.error("✗ Workflow failed!")
            if result.get("errors"):
                for error in result["errors"]:
                    logger.error(f"  - {error}")
        
        # Verify Neo4j data
        verify_neo4j_data()
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)


def verify_neo4j_data():
    """Verify data in Neo4j after ingestion."""
    logger.info("\n" + "=" * 40)
    logger.info("VERIFYING NEO4J DATA")
    logger.info("=" * 40)
    
    driver = GraphDatabase.driver(
        'bolt://localhost:7687', 
        auth=('neo4j', 'EhsAI2024!')
    )
    
    try:
        with driver.session() as session:
            # Count nodes by type
            logger.info("\nNode Counts:")
            labels = ['Document', 'WaterBill', 'Facility', 'Customer', 'UtilityProvider', 'Meter', 'Emission']
            for label in labels:
                result = session.run(f'MATCH (n:{label}) RETURN count(n) as count')
                count = result.single()["count"]
                logger.info(f"  {label}: {count}")
            
            # Count relationships
            logger.info("\nRelationship Counts:")
            result = session.run("""
                MATCH (a)-[r]->(b)
                WHERE NOT a:__Node__ AND NOT b:__Node__
                RETURN type(r) as relationship, count(r) as count
                ORDER BY relationship
            """)
            total_relationships = 0
            for record in result:
                logger.info(f"  {record['relationship']}: {record['count']}")
                total_relationships += record['count']
            logger.info(f"\nTotal Relationships: {total_relationships}")
            
            # Show water bill details
            logger.info("\nWater Bill Details:")
            result = session.run("""
                MATCH (w:WaterBill)
                RETURN w.billing_period_start as start_date,
                       w.billing_period_end as end_date,
                       w.total_gallons as gallons,
                       w.total_cost as cost
            """)
            for record in result:
                logger.info(f"  Period: {record['start_date']} to {record['end_date']}")
                logger.info(f"  Usage: {record['gallons']} gallons")
                logger.info(f"  Cost: ${record['cost']}")
            
            # Show emission calculation
            logger.info("\nWater Emission Calculation:")
            result = session.run("""
                MATCH (w:WaterBill)-[:RESULTED_IN]->(e:Emission)
                RETURN e.amount as amount,
                       e.unit as unit,
                       e.emission_factor as factor,
                       e.calculation_method as method
            """)
            for record in result:
                logger.info(f"  Emissions: {record['amount']} {record['unit']}")
                logger.info(f"  Factor: {record['factor']} kg CO2/gallon")
                logger.info(f"  Method: {record['method']}")
            
    except Exception as e:
        logger.error(f"Failed to verify Neo4j data: {e}")
    finally:
        driver.close()


if __name__ == "__main__":
    test_water_bill_ingestion()