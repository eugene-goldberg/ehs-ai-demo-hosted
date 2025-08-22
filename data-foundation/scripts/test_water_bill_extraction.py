#!/usr/bin/env python3
"""
Test script for water bill data extraction from Neo4j.
Queries the graph and generates a comprehensive water bill report.
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


def test_water_bill_extraction():
    """Test water bill data extraction and report generation."""
    logger.info("=" * 80)
    logger.info("Testing Water Bill Data Extraction")
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
        
        # Extract water consumption data
        logger.info("\nExtracting water consumption data...")
        
        # Test 1: Basic water bill query
        state1 = workflow.extract_data(
            query_type=QueryType.WATER_CONSUMPTION,
            output_format="txt"
        )
        
        if state1.get("status") == "completed":
            logger.info(f"✓ Water consumption report generated: {state1.get('report_file_path')}")
        else:
            logger.error("✗ Water consumption report generation failed")
            if state1.get("errors"):
                for error in state1["errors"]:
                    logger.error(f"  - {error}")
        
        # Test 2: Water consumption with date range
        logger.info("\nExtracting water consumption for specific period...")
        
        state2 = workflow.extract_data(
            query_type=QueryType.WATER_CONSUMPTION,
            parameters={
                "start_date": "2025-06-01",
                "end_date": "2025-06-30"
            },
            output_format="json"
        )
        
        if state2.get("status") == "completed":
            logger.info(f"✓ Filtered water report generated: {state2.get('report_file_path')}")
        else:
            logger.error("✗ Filtered water report generation failed")
        
        # Test 3: Custom query for detailed water bill information
        logger.info("\nRunning custom query for detailed water bill data...")
        
        custom_queries = [
            {
                "query": """
                    MATCH (w:WaterBill)
                    MATCH (w)-[:BILLED_TO]->(f:Facility)
                    MATCH (w)-[:BILLED_FOR]->(c:Customer)
                    MATCH (w)-[:PROVIDED_BY]->(p:UtilityProvider)
                    OPTIONAL MATCH (w)-[:RESULTED_IN]->(e:Emission)
                    OPTIONAL MATCH (m:Meter)-[:RECORDED_IN]->(w)
                    RETURN 
                        w.billing_period_start as period_start,
                        w.billing_period_end as period_end,
                        w.total_gallons as gallons_used,
                        w.total_cost as total_cost,
                        w.water_consumption_cost as water_cost,
                        w.sewer_service_charge as sewer_charge,
                        w.stormwater_fee as stormwater_fee,
                        w.conservation_tax as conservation_tax,
                        w.infrastructure_surcharge as infrastructure_surcharge,
                        f.name as facility_name,
                        f.address as facility_address,
                        c.name as customer_name,
                        c.billing_address as billing_address,
                        p.name as provider_name,
                        p.phone as provider_phone,
                        e.amount as co2_emissions,
                        e.emission_factor as emission_factor,
                        collect({
                            meter_id: m.id,
                            usage: m.usage,
                            unit: m.unit,
                            service_type: m.service_type
                        }) as meter_readings
                    ORDER BY w.billing_period_end DESC
                """,
                "parameters": {}
            }
        ]
        
        state3 = workflow.extract_data(
            query_type=QueryType.CUSTOM,
            queries=custom_queries,
            output_format="txt"
        )
        
        if state3.get("status") == "completed":
            logger.info(f"✓ Detailed water bill report generated: {state3.get('report_file_path')}")
            
            # Display some results
            if state3.get("query_results"):
                results = state3["query_results"][0].get("results", [])
                if results:
                    logger.info("\nSample Water Bill Data:")
                    record = results[0]
                    logger.info(f"  Period: {record.get('period_start')} to {record.get('period_end')}")
                    logger.info(f"  Facility: {record.get('facility_name')}")
                    logger.info(f"  Customer: {record.get('customer_name')}")
                    logger.info(f"  Provider: {record.get('provider_name')}")
                    logger.info(f"  Water Usage: {record.get('gallons_used')} gallons")
                    logger.info(f"  Total Cost: ${record.get('total_cost')}")
                    logger.info(f"  CO2 Emissions: {record.get('co2_emissions')} kg")
        else:
            logger.error("✗ Detailed water bill report generation failed")
        
        # Test 4: Combined facility report (water + electric)
        logger.info("\nGenerating combined facility emissions report...")
        
        combined_query = [
            {
                "query": """
                    MATCH (f:Facility)
                    OPTIONAL MATCH (f)<-[:BILLED_TO]-(w:WaterBill)-[:RESULTED_IN]->(we:Emission)
                    OPTIONAL MATCH (f)<-[:BILLED_TO]-(u:UtilityBill)-[:RESULTED_IN]->(ue:Emission)
                    WITH f,
                         SUM(we.amount) as water_emissions,
                         SUM(ue.amount) as electric_emissions
                    RETURN 
                        f.name as facility_name,
                        f.address as facility_address,
                        water_emissions,
                        electric_emissions,
                        water_emissions + electric_emissions as total_emissions,
                        'kg_CO2' as unit
                """,
                "parameters": {}
            }
        ]
        
        state4 = workflow.extract_data(
            query_type=QueryType.CUSTOM,
            queries=combined_query,
            output_format="txt"
        )
        
        if state4.get("status") == "completed":
            logger.info(f"✓ Combined facility report generated: {state4.get('report_file_path')}")
        else:
            logger.error("✗ Combined facility report generation failed")
        
        logger.info("\n" + "=" * 80)
        logger.info("Water Bill Extraction Tests Completed")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
    finally:
        # Clean up
        if 'workflow' in locals():
            workflow.driver.close()


if __name__ == "__main__":
    test_water_bill_extraction()