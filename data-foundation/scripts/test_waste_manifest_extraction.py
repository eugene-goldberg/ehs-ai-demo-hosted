#!/usr/bin/env python3
"""
Test script for waste manifest data extraction from Neo4j.
Queries the graph and generates comprehensive waste generation reports.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables from backend/.env
env_path = Path(__file__).parent.parent / "backend" / ".env"
load_dotenv(env_path, override=True)

from backend.src.workflows.extraction_workflow import DataExtractionWorkflow, QueryType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_waste_manifest_extraction():
    """Test waste manifest data extraction and report generation."""
    logger.info("=" * 80)
    logger.info("Testing Waste Manifest Data Extraction")
    logger.info("=" * 80)
    
    try:
        # Initialize extraction workflow
        workflow = DataExtractionWorkflow(
            neo4j_uri="bolt://localhost:7687",
            neo4j_username="neo4j",
            neo4j_password="EhsAI2024!",
            llm_model="gpt-4",
            output_dir="./reports"
        )
        
        # Extract waste generation data
        logger.info("\nExtracting waste generation data...")
        
        # Test 1: Basic waste manifest query
        state1 = workflow.extract_data(
            query_type=QueryType.WASTE_GENERATION,
            output_format="txt"
        )
        
        if state1.get("status") == "completed":
            logger.info(f"✓ Waste generation report generated: {state1.get('report_file_path')}")
        else:
            logger.error("✗ Waste generation report generation failed")
            if state1.get("errors"):
                for error in state1["errors"]:
                    logger.error(f"  - {error}")
        
        # Test 2: Waste generation with date range
        logger.info("\nExtracting waste generation for specific period...")
        
        state2 = workflow.extract_data(
            query_type=QueryType.WASTE_GENERATION,
            parameters={
                "start_date": "2025-06-01",
                "end_date": "2025-06-30"
            },
            output_format="json"
        )
        
        if state2.get("status") == "completed":
            logger.info(f"✓ Filtered waste report generated: {state2.get('report_file_path')}")
        else:
            logger.error("✗ Filtered waste report generation failed")
        
        # Test 3: Waste generation by generator
        logger.info("\nExtracting waste generation by generator...")
        
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
        
        state3 = workflow.extract_data(
            query_type=QueryType.CUSTOM,
            queries=generator_queries,
            output_format="txt"
        )
        
        if state3.get("status") == "completed":
            logger.info(f"✓ Waste by generator report generated: {state3.get('report_file_path')}")
            
            # Display some results
            if state3.get("query_results"):
                results = state3["query_results"][0].get("results", [])
                if results:
                    logger.info("\nTop Waste Generators:")
                    for i, record in enumerate(results[:3], 1):
                        logger.info(f"  {i}. {record.get('generator_name')}")
                        logger.info(f"     EPA ID: {record.get('generator_epa_id')}")
                        logger.info(f"     Total Quantity: {record.get('total_quantity')} {record.get('unit')}")
                        logger.info(f"     Hazardous: {record.get('hazardous_quantity')} {record.get('unit')}")
                        logger.info(f"     Non-Hazardous: {record.get('nonhazardous_quantity')} {record.get('unit')}")
                        logger.info(f"     Total Emissions: {record.get('total_emissions')} kg CO2")
        else:
            logger.error("✗ Waste by generator report generation failed")
        
        # Test 4: Disposal facility analysis
        logger.info("\nExtracting disposal facility analysis...")
        
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
        
        state4 = workflow.extract_data(
            query_type=QueryType.CUSTOM,
            queries=disposal_queries,
            output_format="txt"
        )
        
        if state4.get("status") == "completed":
            logger.info(f"✓ Disposal facility analysis generated: {state4.get('report_file_path')}")
            
            # Display some results
            if state4.get("query_results"):
                results = state4["query_results"][0].get("results", [])
                if results:
                    logger.info("\nTop Disposal Facilities:")
                    for i, record in enumerate(results[:3], 1):
                        logger.info(f"  {i}. {record.get('facility_name')}")
                        logger.info(f"     EPA ID: {record.get('facility_epa_id')}")
                        logger.info(f"     Disposal Method: {record.get('disposal_method')}")
                        logger.info(f"     Total Quantity: {record.get('total_quantity')} {record.get('unit')}")
                        logger.info(f"     Unique Generators: {record.get('unique_generators')}")
                        logger.info(f"     Total Emissions: {record.get('total_emissions')} kg CO2")
        else:
            logger.error("✗ Disposal facility analysis generation failed")
        
        # Test 5: Emission analysis from waste
        logger.info("\nExtracting emission analysis from waste...")
        
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
        
        state5 = workflow.extract_data(
            query_type=QueryType.CUSTOM,
            queries=emission_queries,
            output_format="json"
        )
        
        if state5.get("status") == "completed":
            logger.info(f"✓ Emission analysis report generated: {state5.get('report_file_path')}")
        else:
            logger.error("✗ Emission analysis report generation failed")
        
        # Test 6: Combined summary report
        logger.info("\nGenerating combined waste summary report...")
        
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
        
        state6 = workflow.extract_data(
            query_type=QueryType.CUSTOM,
            queries=summary_queries,
            output_format="txt"
        )
        
        if state6.get("status") == "completed":
            logger.info(f"✓ Combined summary report generated: {state6.get('report_file_path')}")
            
            # Display summary results
            if state6.get("query_results"):
                results = state6["query_results"][0].get("results", [])
                if results:
                    summary = results[0]
                    logger.info("\nWaste Management Summary:")
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
            logger.error("✗ Combined summary report generation failed")
        
        # Test 7: Top generators and facilities list
        logger.info("\nGenerating top generators and facilities list...")
        
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
        
        state7 = workflow.extract_data(
            query_type=QueryType.CUSTOM,
            queries=top_entities_queries,
            output_format="txt"
        )
        
        if state7.get("status") == "completed":
            logger.info(f"✓ Top entities report generated: {state7.get('report_file_path')}")
            
            # Display top entities
            if state7.get("query_results"):
                results = state7["query_results"][0].get("results", [])
                if results:
                    generators = [r for r in results if r.get('entity_type') == 'Generator']
                    facilities = [r for r in results if r.get('entity_type') == 'Disposal Facility']
                    
                    if generators:
                        logger.info("\nTop Waste Generators:")
                        for i, gen in enumerate(generators, 1):
                            logger.info(f"  {i}. {gen.get('name')} ({gen.get('epa_id')})")
                            logger.info(f"     Total: {gen.get('total_quantity')} {gen.get('unit')}")
                    
                    if facilities:
                        logger.info("\nTop Disposal Facilities:")
                        for i, fac in enumerate(facilities, 1):
                            logger.info(f"  {i}. {fac.get('name')} ({fac.get('epa_id')})")
                            logger.info(f"     Total: {fac.get('total_quantity')} {fac.get('unit')}")
        else:
            logger.error("✗ Top entities report generation failed")
        
        logger.info("\n" + "=" * 80)
        logger.info("Waste Manifest Extraction Tests Completed")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
    finally:
        # Clean up
        if 'workflow' in locals():
            workflow.driver.close()


if __name__ == "__main__":
    test_waste_manifest_extraction()