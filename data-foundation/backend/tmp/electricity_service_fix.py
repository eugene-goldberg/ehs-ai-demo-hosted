#!/usr/bin/env python3
"""
Electricity Service Fix

This script demonstrates the fix needed for the electricity endpoints to return data.
The issue is a property name mismatch between the service query and Neo4j schema.

Created: 2025-08-31
"""

import os
import sys
from datetime import datetime

# Add the project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.database.neo4j_client import Neo4jClient, ConnectionConfig
from src.services.environmental_assessment_service import EnvironmentalAssessmentService


def test_current_electricity_service():
    """Test the current electricity service to see the issue"""
    print("=== TESTING CURRENT ELECTRICITY SERVICE ===")
    
    try:
        # Create Neo4j client
        config = ConnectionConfig.from_env()
        client = Neo4jClient(config=config, enable_logging=False)
        
        if not client.connect():
            print("Failed to connect to Neo4j")
            return
        
        # Create service
        service = EnvironmentalAssessmentService(client)
        
        # Test electricity assessment
        result = service.assess_electricity_consumption()
        
        print(f"Service result keys: {list(result.keys())}")
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Raw data count: {len(result.get('raw_data', []))}")
            if result.get('raw_data'):
                print(f"First record: {result['raw_data'][0]}")
            print(f"Facts: {result.get('facts', {})}")
        
    except Exception as e:
        print(f"Error testing current service: {e}")


def test_fixed_electricity_query():
    """Test the electricity service with a fixed query"""
    print("\n=== TESTING FIXED ELECTRICITY QUERY ===")
    
    try:
        # Create Neo4j client  
        config = ConnectionConfig.from_env()
        client = Neo4jClient(config=config, enable_logging=False)
        
        if not client.connect():
            print("Failed to connect to Neo4j")
            return
        
        # Test the corrected query directly
        fixed_query = """
        MATCH (e:ElectricityConsumption)
        WHERE ($location IS NULL OR e.facility_id CONTAINS $location)
        AND ($start_date IS NULL OR e.date >= $start_date)
        AND ($end_date IS NULL OR e.date <= $end_date)
        RETURN e.facility_id as location, e.date as date, e.consumption_kwh as consumption,
               e.cost_usd as cost, 
               COALESCE(e.source_type, 'Unknown') as source_type, 
               COALESCE(e.efficiency_rating, 0.0) as efficiency
        ORDER BY e.date DESC
        LIMIT 10
        """
        
        records = client.execute_query(
            fixed_query,
            parameters={
                "location": None,
                "start_date": None,
                "end_date": None
            }
        )
        
        print(f"Fixed query returned {len(records)} records")
        if records:
            data = [dict(record) for record in records]
            print(f"Sample records:")
            for i, record in enumerate(data[:3]):
                print(f"  Record {i+1}: {record}")
            
            # Test the facts calculation with this data
            print("\n=== TESTING FACTS CALCULATION ===")
            service = EnvironmentalAssessmentService(client)
            facts = service._calculate_electricity_facts(data)
            print(f"Facts calculated from fixed data: {facts}")
        
    except Exception as e:
        print(f"Error testing fixed query: {e}")


def show_fix_instructions():
    """Show the exact fix needed"""
    print("\n" + "="*80)
    print("EXACT FIX REQUIRED")
    print("="*80)
    
    print("""
To fix the electricity endpoints, update the query in:
src/services/environmental_assessment_service.py 

Around line 117-124, replace the current query:

OLD (BROKEN) QUERY:
```
query = \"\"\"
MATCH (e:ElectricityConsumption)
WHERE ($location IS NULL OR e.location CONTAINS $location)
AND ($start_date IS NULL OR e.date >= $start_date)
AND ($end_date IS NULL OR e.date <= $end_date)
RETURN e.location as location, e.date as date, e.consumption_kwh as consumption,
       e.cost_usd as cost, e.source_type as source_type, e.efficiency_rating as efficiency
ORDER BY e.date DESC
\"\"\"
```

NEW (FIXED) QUERY:
```
query = \"\"\"
MATCH (e:ElectricityConsumption)
WHERE ($location IS NULL OR e.facility_id CONTAINS $location)
AND ($start_date IS NULL OR e.date >= $start_date)  
AND ($end_date IS NULL OR e.date <= $end_date)
RETURN e.facility_id as location, e.date as date, e.consumption_kwh as consumption,
       e.cost_usd as cost, 
       COALESCE(e.source_type, 'Unknown') as source_type, 
       COALESCE(e.efficiency_rating, 0.0) as efficiency
ORDER BY e.date DESC
\"\"\"
```

KEY CHANGES:
1. Changed 'e.location' to 'e.facility_id' in WHERE clause and RETURN statement
2. Added COALESCE() for optional properties to provide default values
3. This ensures the query returns actual data instead of NULL values

REASON FOR THE BUG:
- ElectricityConsumption nodes have 'facility_id' property, not 'location'
- ElectricityConsumption nodes don't have 'source_type' or 'efficiency_rating' properties
- The original query returned records with NULL values which were being filtered out or causing issues downstream
""")


def main():
    """Main function"""
    print("ELECTRICITY SERVICE DEBUG AND FIX")
    print("="*50)
    
    # Test current service
    test_current_electricity_service()
    
    # Test fixed query 
    test_fixed_electricity_query()
    
    # Show fix instructions
    show_fix_instructions()


if __name__ == '__main__':
    main()