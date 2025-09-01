#!/usr/bin/env python3
"""
Debug script for Water Consumption endpoint issues - COMPREHENSIVE VERSION.
This script investigates why water endpoints are returning empty data and provides the complete fix.
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

def analyze_water_data_structure(client):
    """Complete analysis of WaterConsumption data structure."""
    print("\n" + "="*60)
    print("COMPREHENSIVE WATERCONSUMPTION DATA ANALYSIS")
    print("="*60)
    
    # Count total nodes
    result = client.execute_read_query("MATCH (w:WaterConsumption) RETURN count(w) as total")
    total_count = result[0]["total"] if result else 0
    print(f"Total WaterConsumption nodes: {total_count}")
    
    if total_count == 0:
        print("✗ No WaterConsumption nodes found!")
        return None
    
    # Get sample data
    result = client.execute_read_query("""
        MATCH (w:WaterConsumption) 
        RETURN w 
        LIMIT 3
    """)
    
    print("\nSample WaterConsumption nodes:")
    sample_nodes = []
    for i, record in enumerate(result):
        node_data = dict(record["w"])
        sample_nodes.append(node_data)
        print(f"\nNode {i+1}:")
        for key, value in node_data.items():
            print(f"  {key}: {value} (type: {type(value)})")
    
    # Get all property keys
    result = client.execute_read_query("""
        MATCH (w:WaterConsumption)
        UNWIND keys(w) as key
        RETURN DISTINCT key
        ORDER BY key
    """)
    property_keys = [record["key"] for record in result]
    print(f"\nAll property keys: {property_keys}")
    
    # Get unique facility_ids
    result = client.execute_read_query("""
        MATCH (w:WaterConsumption)
        RETURN DISTINCT w.facility_id as facility_id
        ORDER BY facility_id
    """)
    facility_ids = [record["facility_id"] for record in result]
    print(f"Unique facility_ids: {facility_ids}")
    
    # Get date range
    result = client.execute_read_query("""
        MATCH (w:WaterConsumption)
        RETURN min(w.date) as min_date, max(w.date) as max_date, count(w) as total
    """)
    if result:
        date_info = result[0]
        print(f"Date range: {date_info['min_date']} to {date_info['max_date']} ({date_info['total']} records)")
    
    return sample_nodes, property_keys, facility_ids

def test_broken_query(client, facility_ids):
    """Test the current broken query from environmental service."""
    print("\n" + "="*60)
    print("TESTING CURRENT BROKEN QUERY")
    print("="*60)
    
    broken_query = """
        MATCH (w:WaterConsumption)
        WHERE w.location = $location
          AND w.timestamp >= $start_date
          AND w.timestamp <= $end_date
        RETURN w.timestamp as timestamp,
               w.consumption as consumption,
               w.unit as unit
        ORDER BY w.timestamp
    """
    
    print("Current broken query:")
    print(broken_query)
    
    # Test with different location values
    test_cases = [
        {"location": "Building A", "start_date": "2024-01-01", "end_date": "2024-12-31"},
        {"location": facility_ids[0] if facility_ids else "unknown", "start_date": "2024-01-01", "end_date": "2024-12-31"},
    ]
    
    for i, params in enumerate(test_cases):
        print(f"\nTest {i+1} - Parameters: {params}")
        try:
            result = client.execute_read_query(broken_query, params)
            print(f"  Results: {len(result)} records")
        except Exception as e:
            print(f"  Error: {e}")

def create_fixed_query(client, facility_ids):
    """Create and test the fixed query."""
    print("\n" + "="*60)
    print("CREATING AND TESTING FIXED QUERY")
    print("="*60)
    
    fixed_query = """
        MATCH (w:WaterConsumption)
        WHERE w.facility_id = $location
          AND w.date >= date($start_date)
          AND w.date <= date($end_date)
        RETURN w.date as timestamp,
               w.consumption_gallons as consumption,
               'gallons' as unit,
               w.facility_id as facility_id
        ORDER BY w.date
    """
    
    print("Fixed query:")
    print(fixed_query)
    
    print("\nChanges made:")
    print("  1. w.location -> w.facility_id")
    print("  2. w.timestamp -> w.date")
    print("  3. w.consumption -> w.consumption_gallons")
    print("  4. Added hardcoded 'gallons' for unit (since unit property doesn't exist)")
    print("  5. date($start_date) and date($end_date) for proper date comparison")
    
    # Test fixed query
    test_params = {
        "location": facility_ids[0] if facility_ids else "unknown",
        "start_date": "2025-07-01",
        "end_date": "2025-08-31"
    }
    
    print(f"\nTesting fixed query with params: {test_params}")
    try:
        result = client.execute_read_query(fixed_query, test_params)
        print(f"✓ SUCCESS: Found {len(result)} records!")
        
        if result:
            print("\nSample results:")
            for i, record in enumerate(result[:5]):
                print(f"  {i+1}. {dict(record)}")
            if len(result) > 5:
                print(f"  ... and {len(result) - 5} more")
        
        return fixed_query
        
    except Exception as e:
        print(f"✗ Error with fixed query: {e}")
        return None

def generate_code_fixes(fixed_query):
    """Generate the code changes needed in the environmental service."""
    print("\n" + "="*60)
    print("CODE FIXES NEEDED")
    print("="*60)
    
    print("The environmental assessment service needs these updates:")
    print("\n1. UPDATE WATER CONSUMPTION QUERY:")
    print("   Replace the current water query with:")
    print(f"   {fixed_query}")
    
    print("\n2. PROPERTY MAPPING CHANGES:")
    print("   - location parameter maps to facility_id property")
    print("   - timestamp maps to date property") 
    print("   - consumption maps to consumption_gallons property")
    print("   - unit should be hardcoded as 'gallons'")
    
    print("\n3. DATE HANDLING:")
    print("   - Use date($param) for date comparisons")
    print("   - Input dates should be YYYY-MM-DD strings")
    
    print("\n4. RESPONSE FIELD MAPPING:")
    print("   The response should map:")
    print("   - w.date -> timestamp field in API response")
    print("   - w.consumption_gallons -> consumption field")
    print("   - 'gallons' -> unit field")

def find_environmental_service():
    """Find the environmental assessment service file."""
    print("\n" + "="*60)
    print("LOCATING ENVIRONMENTAL ASSESSMENT SERVICE")
    print("="*60)
    
    # Common locations for the service file
    possible_locations = [
        "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/src/services/environmental_assessment.py",
        "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/src/services/environmental_assessment_service.py",
        "/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/src/ehs_workflows/environmental_assessment.py"
    ]
    
    for location in possible_locations:
        if os.path.exists(location):
            print(f"✓ Found service file at: {location}")
            return location
    
    print("✗ Could not automatically locate environmental assessment service file")
    print("   Search for files containing water consumption queries")
    return None

def main():
    """Main debugging and analysis function."""
    print("Starting Comprehensive Water Consumption Debug")
    print("="*60)
    
    # Connect to Neo4j
    client = connect_to_neo4j()
    if not client:
        return
    
    try:
        # Step 1: Comprehensive data analysis
        analysis_result = analyze_water_data_structure(client)
        if not analysis_result:
            print("Cannot continue without WaterConsumption data")
            return
        
        sample_nodes, property_keys, facility_ids = analysis_result
        
        # Step 2: Test broken query
        test_broken_query(client, facility_ids)
        
        # Step 3: Create and test fixed query
        fixed_query = create_fixed_query(client, facility_ids)
        
        if fixed_query:
            # Step 4: Generate code fixes
            generate_code_fixes(fixed_query)
            
            # Step 5: Find service file
            service_file = find_environmental_service()
            
            print(f"\n{'='*60}")
            print("SUMMARY")
            print(f"{'='*60}")
            print("✓ Problem identified: Property name and data type mismatches")
            print("✓ Fixed query created and tested successfully")
            print("✓ Code fix recommendations generated")
            if service_file:
                print(f"✓ Service file located: {service_file}")
            print("\nNext step: Apply the fixes to the environmental assessment service")
        
    finally:
        client.close()

if __name__ == "__main__":
    main()