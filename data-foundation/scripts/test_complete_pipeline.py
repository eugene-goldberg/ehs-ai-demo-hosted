#!/usr/bin/env python3
"""
Test the complete document processing pipeline with the electric bill.
This demonstrates:
1. PDF parsing (using PyMuPDF)
2. Data extraction (structured extraction)
3. Neo4j schema transformation
4. Knowledge graph creation (simulated)
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import fitz  # PyMuPDF

# Import our extraction function
sys.path.append(str(Path(__file__).parent))
from test_extract_utility_data import extract_utility_bill_data


def transform_to_neo4j_schema(extracted_data, document_id):
    """Transform extracted data to Neo4j nodes and relationships."""
    
    nodes = []
    relationships = []
    
    # Create Document node
    doc_node = {
        "labels": ["Document", "UtilityBill"],
        "properties": {
            "id": document_id,
            "type": "utility_bill",
            "account_number": extracted_data["account_number"],
            "statement_date": extracted_data["statement_date"],
            "uploaded_at": datetime.now().isoformat()
        }
    }
    nodes.append(doc_node)
    
    # Create UtilityBill node with consumption data
    bill_node = {
        "labels": ["UtilityBill"],
        "properties": {
            "id": f"bill_{document_id}",
            "billing_period_start": extracted_data["billing_period_start"],
            "billing_period_end": extracted_data["billing_period_end"],
            "total_kwh": extracted_data["total_kwh"],
            "peak_kwh": extracted_data["peak_kwh"],
            "off_peak_kwh": extracted_data["off_peak_kwh"],
            "total_cost": extracted_data["total_amount_due"],
            "due_date": extracted_data["due_date"]
        }
    }
    nodes.append(bill_node)
    
    # Create Facility node
    facility_node = {
        "labels": ["Facility"],
        "properties": {
            "id": "facility_apex_plant_a",
            "name": "Apex Manufacturing - Plant A",
            "address": extracted_data["service_address"]
        }
    }
    nodes.append(facility_node)
    
    # Create Emission node (calculated)
    emission_factor = 0.4  # kg CO2 per kWh
    emission_node = {
        "labels": ["Emission"],
        "properties": {
            "id": f"emission_{document_id}",
            "amount": extracted_data["total_kwh"] * emission_factor,
            "unit": "kg_CO2",
            "calculation_method": "grid_average_factor",
            "emission_factor": emission_factor
        }
    }
    nodes.append(emission_node)
    
    # Create relationships
    relationships.extend([
        {
            "source": doc_node["properties"]["id"],
            "target": bill_node["properties"]["id"],
            "type": "EXTRACTED_TO",
            "properties": {"extraction_date": datetime.now().isoformat()}
        },
        {
            "source": bill_node["properties"]["id"],
            "target": facility_node["properties"]["id"],
            "type": "BILLED_TO",
            "properties": {}
        },
        {
            "source": bill_node["properties"]["id"],
            "target": emission_node["properties"]["id"],
            "type": "RESULTED_IN",
            "properties": {"calculation_method": "grid_average"}
        }
    ])
    
    # Add meter relationships
    for meter in extracted_data["meter_readings"]:
        meter_node = {
            "labels": ["Meter"],
            "properties": {
                "id": meter["meter_id"],
                "type": "electricity",
                "previous_reading": meter["previous_reading"],
                "current_reading": meter["current_reading"],
                "usage": meter["current_reading"] - meter["previous_reading"]
            }
        }
        nodes.append(meter_node)
        
        relationships.append({
            "source": meter_node["properties"]["id"],
            "target": facility_node["properties"]["id"],
            "type": "MONITORS",
            "properties": {}
        })
    
    return nodes, relationships


def generate_cypher_queries(nodes, relationships):
    """Generate Cypher queries to create the knowledge graph."""
    
    queries = []
    
    # Create nodes
    for node in nodes:
        labels = ":".join(node["labels"])
        props = ", ".join([f"{k}: ${k}" for k in node["properties"].keys()])
        query = f"CREATE (n:{labels} {{{props}}}) RETURN n"
        queries.append({
            "query": query,
            "parameters": node["properties"]
        })
    
    # Create relationships
    for rel in relationships:
        query = f"""
        MATCH (a {{id: $source_id}})
        MATCH (b {{id: $target_id}})
        CREATE (a)-[r:{rel['type']} $properties]->(b)
        RETURN r
        """
        queries.append({
            "query": query,
            "parameters": {
                "source_id": rel["source"],
                "target_id": rel["target"],
                "properties": rel["properties"]
            }
        })
    
    return queries


def main():
    file_path = "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/data/electric_bill.pdf"
    document_id = f"electric_bill_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("EHS AI Platform - Document Processing Pipeline Test")
    print("=" * 60)
    
    # Step 1: Parse and Extract
    print("\n1. DOCUMENT PARSING & EXTRACTION")
    print("-" * 40)
    
    extracted_data = extract_utility_bill_data(file_path)
    print(f"✓ Successfully extracted data from PDF")
    print(f"  - Account: {extracted_data['account_number']}")
    print(f"  - Period: {extracted_data['billing_period_start']} to {extracted_data['billing_period_end']}")
    print(f"  - Total consumption: {extracted_data['total_kwh']:,} kWh")
    print(f"  - Amount due: ${extracted_data['total_amount_due']:,.2f}")
    
    # Step 2: Transform to Graph Schema
    print("\n2. TRANSFORM TO NEO4J SCHEMA")
    print("-" * 40)
    
    nodes, relationships = transform_to_neo4j_schema(extracted_data, document_id)
    
    print(f"✓ Created {len(nodes)} nodes:")
    for node in nodes:
        print(f"  - {'/'.join(node['labels'])}: {node['properties']['id']}")
    
    print(f"\n✓ Created {len(relationships)} relationships:")
    for rel in relationships:
        print(f"  - ({rel['source']})-[:{rel['type']}]->({rel['target']})")
    
    # Step 3: Generate Cypher Queries
    print("\n3. GENERATE CYPHER QUERIES")
    print("-" * 40)
    
    queries = generate_cypher_queries(nodes, relationships)
    print(f"✓ Generated {len(queries)} Cypher queries")
    
    # Save queries for inspection
    with open("cypher_queries.json", "w") as f:
        json.dump(queries, f, indent=2)
    print("  Queries saved to: cypher_queries.json")
    
    # Step 4: Calculate Metrics
    print("\n4. CALCULATED METRICS")
    print("-" * 40)
    
    emissions = extracted_data['total_kwh'] * 0.4
    daily_avg = extracted_data['total_kwh'] / 30  # 30 days in June
    cost_per_kwh = extracted_data['total_amount_due'] / extracted_data['total_kwh']
    
    print(f"✓ Environmental Impact:")
    print(f"  - CO2 Emissions: {emissions:,.2f} kg")
    print(f"  - Daily Average: {daily_avg:,.2f} kWh")
    print(f"  - Cost per kWh: ${cost_per_kwh:.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nThis demonstrates the complete flow:")
    print("1. PDF → Structured Data Extraction")
    print("2. Data → Neo4j Knowledge Graph Schema")
    print("3. Schema → Cypher Query Generation")
    print("4. Metrics → Environmental Impact Calculation")
    
    # Save complete results
    results = {
        "document_id": document_id,
        "extracted_data": extracted_data,
        "graph_schema": {
            "nodes": nodes,
            "relationships": relationships
        },
        "metrics": {
            "co2_emissions_kg": emissions,
            "daily_average_kwh": daily_avg,
            "cost_per_kwh": cost_per_kwh
        }
    }
    
    with open("pipeline_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nComplete results saved to: pipeline_test_results.json")


if __name__ == "__main__":
    main()