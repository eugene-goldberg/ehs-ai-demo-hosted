#!/usr/bin/env python3
"""
Debug script for Water Consumption endpoint issues.
This script investigates why water endpoints are returning empty data.
"""

import os
import sys
from datetime import datetime, timedelta
from neo4j import GraphDatabase

# Add the src directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from database.neo4j_client import create_neo4j_client

def connect_to_neo4j():
    """Establish connection to Neo4j database."""
    try:
        client = create_neo4j_client()
        if not client.connect():
            print("✗ Failed to connect to Neo4j")
            return None
        print("✓ Successfully connected to Neo4j")
        return client
    except Exception as e:
        print(f"✗ Failed to connect to Neo4j: {e}")
        return None

def inspect_water_consumption_nodes(client):
    """Inspect WaterConsumption nodes to understand their structure."""
    print("\n" + "="*60)
    print("INSPECTING WATERCONSUMPTION NODES")
    print("="*60)
    
    # Count total WaterConsumption nodes
    result = client.execute_read_query("MATCH (w:WaterConsumption) RETURN count(w) as total")
    total_count = result[0]["total"] if result else 0
    print(f"Total WaterConsumption nodes: {total_count}")
    
    if total_count == 0:
        print("✗ No WaterConsumption nodes found in database!")
        return None, None
    
    # Sample a few nodes to see their structure
    print("\nSample WaterConsumption nodes (first 5):")
    result = client.execute_read_query("""
        MATCH (w:WaterConsumption) 
        RETURN w 
        LIMIT 5
    """)
    
    sample_nodes = []
    for record in result:
        node_data = dict(record["w"])
        sample_nodes.append(node_data)
        print(f"Node data: {node_data}")
        print("-" * 40)
    
    # Get all unique property keys
    print("\nAll unique property keys in WaterConsumption nodes:")
    result = client.execute_read_query("""
        MATCH (w:WaterConsumption)
        UNWIND keys(w) as key
        RETURN DISTINCT key
        ORDER BY key
    """)
    
    property_keys = [record["key"] for record in result]
    print(f"Property keys: {property_keys}")
    
    return sample_nodes, property_keys

def test_current_water_query(client):
    """Test the current query used in environmental assessment service."""
    print("\n" + "="*60)
    print("TESTING CURRENT WATER QUERY")
    print("="*60)
    
    # This is likely the current query from the service
    current_query = """
        MATCH (w:WaterConsumption)
        WHERE w.location = $location
          AND w.timestamp >= $start_date
          AND w.timestamp <= $end_date
        RETURN w.timestamp as timestamp,
               w.consumption as consumption,
               w.unit as unit
        ORDER BY w.timestamp
    """
    
    print("Current query:")
    print(current_query)
    
    # Test with a sample location and date range
    test_params = {
        "location": "Building A",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31"
    }
    
    print(f"\nTesting with parameters: {test_params}")
    
    result = client.execute_read_query(current_query, test_params)
    print(f"Results found: {len(result)}")
    
    if len(result) > 0:
        print("Sample results:")
        for i, record in enumerate(result[:3]):
            print(f"  {i+1}. {dict(record)}")
    else:
        print("✗ No results found with current query!")
    
    return len(result) > 0

def find_correct_location_property(client, sample_nodes):
    """Find the correct property name for location/facility identification."""
    print("\n" + "="*60)
    print("FINDING CORRECT LOCATION PROPERTY")
    print("="*60)
    
    if not sample_nodes:
        print("No sample nodes available for analysis")
        return None
    
    # Look for common location-related property names
    location_candidates = ["location", "facility_id", "facility", "site", "building", "site_id"]
    
    found_properties = {}
    for candidate in location_candidates:
        if candidate in sample_nodes[0]:
            values = [node.get(candidate) for node in sample_nodes if candidate in node]
            found_properties[candidate] = values
            print(f"Found property '{candidate}': {values}")
    
    if found_properties:
        print(f"\nLocation-related properties found: {list(found_properties.keys())}")
        return found_properties
    else:
        print("No obvious location properties found. Full sample node:")
        print(sample_nodes[0])
        return None

def test_corrected_queries(client, location_properties):
    """Test queries with corrected property names."""
    print("\n" + "="*60)
    print("TESTING CORRECTED QUERIES")
    print("="*60)
    
    if not location_properties:
        print("No location properties to test")
        return None, None
    
    # Get all unique values for each location property
    for prop_name, sample_values in location_properties.items():
        print(f"\nTesting with property: {prop_name}")
        
        # Get all unique values for this property
        result = client.execute_read_query(f"""
            MATCH (w:WaterConsumption)
            WHERE w.{prop_name} IS NOT NULL
            RETURN DISTINCT w.{prop_name} as value
            ORDER BY value
        """)
        
        all_values = [record["value"] for record in result]
        print(f"All values for {prop_name}: {all_values}")
        
        if all_values:
            # Test query with first available value
            test_value = all_values[0]
            corrected_query = f"""
                MATCH (w:WaterConsumption)
                WHERE w.{prop_name} = $location
                  AND w.timestamp >= $start_date
                  AND w.timestamp <= $end_date
                RETURN w.timestamp as timestamp,
                       w.consumption as consumption,
                       w.unit as unit,
                       w.{prop_name} as location_identifier
                ORDER BY w.timestamp
            """
            
            test_params = {
                "location": test_value,
                "start_date": "2024-01-01",
                "end_date": "2024-12-31"
            }
            
            print(f"Testing corrected query with {prop_name} = '{test_value}':")
            result = client.execute_read_query(corrected_query, test_params)
            print(f"Results found: {len(result)}")
            
            if len(result) > 0:
                print("Sample results:")
                for i, record in enumerate(result[:3]):
                    print(f"  {i+1}. {dict(record)}")
                print(f"✓ SUCCESS: Found data using property '{prop_name}'")
                return prop_name, corrected_query
    
    return None, None

def test_date_formats(client):
    """Test different date formats in WaterConsumption nodes."""
    print("\n" + "="*60)
    print("TESTING DATE FORMATS")
    print("="*60)
    
    # Check timestamp formats
    result = client.execute_read_query("""
        MATCH (w:WaterConsumption)
        WHERE w.timestamp IS NOT NULL
        RETURN w.timestamp as timestamp, toString(w.timestamp) as timestamp_string
        ORDER BY w.timestamp
        LIMIT 10
    """)
    
    print("Sample timestamps:")
    for record in result:
        print(f"  Timestamp: {record['timestamp']} (as string: {record['timestamp_string']})")

def generate_fix_recommendation(correct_property, corrected_query):
    """Generate code fix recommendation."""
    print("\n" + "="*60)
    print("FIX RECOMMENDATION")
    print("="*60)
    
    print(f"The issue is that the query uses 'location' but the correct property is '{correct_property}'")
    print("\nUpdate the environmental assessment service to use:")
    print(f"Property name: {correct_property}")
    print("\nCorrected query:")
    print(corrected_query)

def main():
    """Main debugging function."""
    print("Starting Water Consumption Endpoint Debug")
    print("="*60)
    
    # Connect to Neo4j
    client = connect_to_neo4j()
    if not client:
        return
    
    try:
        # Step 1: Inspect WaterConsumption nodes
        sample_nodes, property_keys = inspect_water_consumption_nodes(client)
        
        if not sample_nodes:
            print("Cannot continue without WaterConsumption data")
            return
        
        # Step 2: Test current query
        current_works = test_current_water_query(client)
        
        if current_works:
            print("✓ Current query works! The issue might be elsewhere.")
            return
        
        # Step 3: Find correct location property
        location_properties = find_correct_location_property(client, sample_nodes)
        
        # Step 4: Test date formats
        test_date_formats(client)
        
        # Step 5: Test corrected queries
        if location_properties:
            correct_property, corrected_query = test_corrected_queries(client, location_properties)
            
            # Step 6: Generate fix recommendation
            if correct_property:
                generate_fix_recommendation(correct_property, corrected_query)
            else:
                print("\n✗ Could not determine the correct fix")
                print("Manual investigation required")
        else:
            print("\n✗ No location-related properties found")
            print("Manual investigation required")
    
    finally:
        client.close()

if __name__ == "__main__":
    main()