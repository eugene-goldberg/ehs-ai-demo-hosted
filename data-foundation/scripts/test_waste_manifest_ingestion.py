#!/usr/bin/env python3
"""
Test script for waste manifest ingestion in the EHS document processing pipeline.
Tests the full pipeline with the waste manifest PDF.
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


def test_waste_manifest_ingestion():
    """Test waste manifest ingestion workflow."""
    logger.info("=" * 80)
    logger.info("Testing Waste Manifest Ingestion Workflow")
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
        
        # Process waste manifest
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = "data/waste_manifest.pdf"
        
        logger.info(f"Processing waste manifest: {file_path}")
        
        result = workflow.process_document(
            file_path=file_path,
            document_id=f"waste_manifest_{timestamp}",
            metadata={"source": "test", "document_type": "waste_manifest"}
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
                logger.info(f"  - Manifest Number: {data.get('manifest_number')}")
                logger.info(f"  - Ship Date: {data.get('ship_date')}")
                logger.info(f"  - Generator: {data.get('generator_name')} ({data.get('generator_epa_id')})")
                logger.info(f"  - Transporter: {data.get('transporter_name')} ({data.get('transporter_epa_id')})")
                logger.info(f"  - Disposal Facility: {data.get('disposal_facility_name')} ({data.get('disposal_facility_epa_id')})")
                logger.info(f"  - Total Waste Items: {len(data.get('waste_items', []))}")
                
                # Show waste items summary
                if data.get("waste_items"):
                    logger.info(f"  - Waste Items Summary:")
                    total_quantity = 0
                    for item in data["waste_items"]:
                        logger.info(f"    • {item.get('description', 'Unknown')}: {item.get('quantity', 0)} {item.get('unit', 'units')} ({item.get('waste_code', 'No code')})")
                        if item.get('quantity'):
                            try:
                                total_quantity += float(item['quantity'])
                            except (ValueError, TypeError):
                                pass
                    logger.info(f"  - Total Quantity: {total_quantity} units")
        else:
            logger.error("✗ Workflow failed!")
            if result.get("errors"):
                for error in result["errors"]:
                    logger.error(f"  - {error}")
        
        # Verify Neo4j data
        verify_neo4j_data()
        
        # Generate test report
        generate_test_report(result)
        
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
            labels = [
                'Document', 'WasteManifest', 'WasteShipment', 'WasteGenerator', 
                'Transporter', 'DisposalFacility', 'WasteItem', 'Emission'
            ]
            node_counts = {}
            for label in labels:
                result = session.run(f'MATCH (n:{label}) RETURN count(n) as count')
                count = result.single()["count"]
                node_counts[label] = count
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
            relationship_counts = {}
            for record in result:
                rel_type = record['relationship']
                count = record['count']
                relationship_counts[rel_type] = count
                logger.info(f"  {rel_type}: {count}")
                total_relationships += count
            logger.info(f"\nTotal Relationships: {total_relationships}")
            
            # Show waste manifest details
            logger.info("\nWaste Manifest Details:")
            result = session.run("""
                MATCH (wm:WasteManifest)
                RETURN wm.manifest_number as manifest_number,
                       wm.ship_date as ship_date,
                       wm.waste_category as category
            """)
            for record in result:
                logger.info(f"  Manifest Number: {record['manifest_number']}")
                logger.info(f"  Ship Date: {record['ship_date']}")
                logger.info(f"  Category: {record['category']}")
            
            # Show waste shipment details
            logger.info("\nWaste Shipment Details:")
            result = session.run("""
                MATCH (ws:WasteShipment)
                RETURN ws.total_quantity as quantity,
                       ws.unit as unit,
                       ws.status as status
            """)
            for record in result:
                logger.info(f"  Total Quantity: {record['quantity']} {record['unit']}")
                logger.info(f"  Status: {record['status']}")
            
            # Show generator, transporter, and disposal facility
            logger.info("\nParticipants:")
            
            # Generator
            result = session.run("""
                MATCH (g:WasteGenerator)
                RETURN g.name as name, g.epa_id as epa_id
            """)
            for record in result:
                logger.info(f"  Generator: {record['name']} (EPA ID: {record['epa_id']})")
            
            # Transporter
            result = session.run("""
                MATCH (t:Transporter)
                RETURN t.name as name, t.epa_id as epa_id
            """)
            for record in result:
                logger.info(f"  Transporter: {record['name']} (EPA ID: {record['epa_id']})")
            
            # Disposal Facility
            result = session.run("""
                MATCH (d:DisposalFacility)
                RETURN d.name as name, d.epa_id as epa_id
            """)
            for record in result:
                logger.info(f"  Disposal Facility: {record['name']} (EPA ID: {record['epa_id']})")
            
            # Show waste items
            logger.info("\nWaste Items:")
            result = session.run("""
                MATCH (wi:WasteItem)
                RETURN wi.description as description,
                       wi.quantity as quantity,
                       wi.unit as unit,
                       wi.waste_code as waste_code,
                       wi.hazard_class as hazard_class
                ORDER BY wi.description
            """)
            for record in result:
                logger.info(f"  • {record['description']}: {record['quantity']} {record['unit']} (Code: {record['waste_code']}, Hazard: {record['hazard_class']})")
            
            # Show emission calculation
            logger.info("\nWaste Emission Calculation:")
            result = session.run("""
                MATCH (wm:WasteManifest)-[:RESULTED_IN]->(e:Emission)
                RETURN e.amount as amount,
                       e.unit as unit,
                       e.emission_factor as factor,
                       e.calculation_method as method,
                       e.emission_category as category
            """)
            for record in result:
                logger.info(f"  Emissions: {record['amount']} {record['unit']}")
                logger.info(f"  Factor: {record['factor']} kg CO2/unit")
                logger.info(f"  Method: {record['method']}")
                logger.info(f"  Category: {record['category']}")
            
            # Verify key relationships exist
            logger.info("\nRelationship Verification:")
            
            # WasteManifest -> WasteShipment
            result = session.run("MATCH (wm:WasteManifest)-[:CONTAINS]->(ws:WasteShipment) RETURN count(*) as count")
            count = result.single()["count"]
            status = "✓" if count > 0 else "✗"
            logger.info(f"  {status} WasteManifest -> WasteShipment: {count}")
            
            # WasteShipment -> WasteGenerator
            result = session.run("MATCH (ws:WasteShipment)-[:GENERATED_BY]->(wg:WasteGenerator) RETURN count(*) as count")
            count = result.single()["count"]
            status = "✓" if count > 0 else "✗"
            logger.info(f"  {status} WasteShipment -> WasteGenerator: {count}")
            
            # WasteShipment -> Transporter
            result = session.run("MATCH (ws:WasteShipment)-[:TRANSPORTED_BY]->(t:Transporter) RETURN count(*) as count")
            count = result.single()["count"]
            status = "✓" if count > 0 else "✗"
            logger.info(f"  {status} WasteShipment -> Transporter: {count}")
            
            # WasteShipment -> DisposalFacility
            result = session.run("MATCH (ws:WasteShipment)-[:DISPOSED_AT]->(df:DisposalFacility) RETURN count(*) as count")
            count = result.single()["count"]
            status = "✓" if count > 0 else "✗"
            logger.info(f"  {status} WasteShipment -> DisposalFacility: {count}")
            
            # WasteShipment -> WasteItem
            result = session.run("MATCH (ws:WasteShipment)-[:CONTAINS]->(wi:WasteItem) RETURN count(*) as count")
            count = result.single()["count"]
            status = "✓" if count > 0 else "✗"
            logger.info(f"  {status} WasteShipment -> WasteItem: {count}")
            
            # WasteManifest -> Emission
            result = session.run("MATCH (wm:WasteManifest)-[:RESULTED_IN]->(e:Emission) RETURN count(*) as count")
            count = result.single()["count"]
            status = "✓" if count > 0 else "✗"
            logger.info(f"  {status} WasteManifest -> Emission: {count}")
            
    except Exception as e:
        logger.error(f"Failed to verify Neo4j data: {e}")
    finally:
        driver.close()


def generate_test_report(workflow_result):
    """Generate a comprehensive test report."""
    logger.info("\n" + "=" * 60)
    logger.info("WASTE MANIFEST INGESTION TEST REPORT")
    logger.info("=" * 60)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Test Date: {timestamp}")
    logger.info(f"Test File: waste_manifest.pdf")
    
    # Overall status
    success = workflow_result.get("success", False)
    logger.info(f"\nOverall Status: {'✓ PASSED' if success else '✗ FAILED'}")
    
    if success:
        logger.info(f"Document ID: {workflow_result.get('document_id')}")
        logger.info(f"Neo4j Nodes Created: {workflow_result.get('neo4j_node_count', 0)}")
        logger.info(f"Neo4j Relationships Created: {workflow_result.get('neo4j_relationship_count', 0)}")
    
    # Connect to Neo4j for detailed verification
    driver = GraphDatabase.driver(
        'bolt://localhost:7687', 
        auth=('neo4j', 'EhsAI2024!')
    )
    
    try:
        with driver.session() as session:
            logger.info("\nData Verification Checklist:")
            
            # Check each required node type
            checklist = [
                ('WasteManifest node created', 'WasteManifest'),
                ('WasteShipment node created', 'WasteShipment'),
                ('WasteGenerator node created', 'WasteGenerator'),
                ('Transporter node created', 'Transporter'),
                ('DisposalFacility node created', 'DisposalFacility'),
                ('WasteItem nodes created', 'WasteItem'),
                ('Emission node created', 'Emission')
            ]
            
            passed_checks = 0
            for check_name, label in checklist:
                result = session.run(f'MATCH (n:{label}) RETURN count(n) as count')
                count = result.single()["count"]
                status = "✓" if count > 0 else "✗"
                logger.info(f"  {status} {check_name}: {count} found")
                if count > 0:
                    passed_checks += 1
            
            # Check key relationships
            relationship_checks = [
                ('WasteManifest contains WasteShipment', 'MATCH (wm:WasteManifest)-[:CONTAINS]->(ws:WasteShipment) RETURN count(*) as count'),
                ('WasteShipment generated by WasteGenerator', 'MATCH (ws:WasteShipment)-[:GENERATED_BY]->(wg:WasteGenerator) RETURN count(*) as count'),
                ('WasteShipment transported by Transporter', 'MATCH (ws:WasteShipment)-[:TRANSPORTED_BY]->(t:Transporter) RETURN count(*) as count'),
                ('WasteShipment disposed at DisposalFacility', 'MATCH (ws:WasteShipment)-[:DISPOSED_AT]->(df:DisposalFacility) RETURN count(*) as count'),
                ('WasteShipment contains WasteItems', 'MATCH (ws:WasteShipment)-[:CONTAINS]->(wi:WasteItem) RETURN count(*) as count'),
                ('WasteManifest resulted in Emission', 'MATCH (wm:WasteManifest)-[:RESULTED_IN]->(e:Emission) RETURN count(*) as count')
            ]
            
            logger.info("\nRelationship Verification:")
            for check_name, query in relationship_checks:
                result = session.run(query)
                count = result.single()["count"]
                status = "✓" if count > 0 else "✗"
                logger.info(f"  {status} {check_name}: {count} found")
                if count > 0:
                    passed_checks += 1
            
            total_checks = len(checklist) + len(relationship_checks)
            logger.info(f"\nTest Summary: {passed_checks}/{total_checks} checks passed ({(passed_checks/total_checks)*100:.1f}%)")
            
            if passed_checks == total_checks:
                logger.info("🎉 All verification checks PASSED!")
            else:
                logger.info("⚠️  Some verification checks FAILED!")
    
    except Exception as e:
        logger.error(f"Failed to generate complete test report: {e}")
    finally:
        driver.close()
    
    logger.info("\n" + "=" * 60)


if __name__ == "__main__":
    test_waste_manifest_ingestion()