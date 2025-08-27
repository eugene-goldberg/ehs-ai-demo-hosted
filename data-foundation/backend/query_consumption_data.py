#!/usr/bin/env python3

from neo4j import GraphDatabase
import json

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "EhsAI2024!"))

with driver.session() as session:
    # Query 1: Examine MonthlyUsageAllocation structure
    print("=== MonthlyUsageAllocation Node Structure ===")
    result = session.run("""
        MATCH (m:MonthlyUsageAllocation)
        WITH m LIMIT 1
        RETURN keys(m) as properties
    """)
    for record in result:
        print("Properties:", record["properties"])
    
    # Query 2: Get sample MonthlyUsageAllocation data
    print("\n=== Sample MonthlyUsageAllocation Data ===")
    result = session.run("""
        MATCH (m:MonthlyUsageAllocation)
        RETURN m
        LIMIT 3
    """)
    for record in result:
        node = dict(record["m"])
        print(json.dumps(node, indent=2, default=str))
    
    # Query 3: Count allocations by facility
    print("\n=== Allocations by Facility ===")
    result = session.run("""
        MATCH (m:MonthlyUsageAllocation)
        RETURN m.facility_id as facility, count(m) as count
        ORDER BY count DESC
    """)
    for record in result:
        print(f'Facility {record["facility"]}: {record["count"]} allocations')
    
    # Query 4: Check Document-Allocation relationships
    print("\n=== Document to Allocation Relationships ===")
    result = session.run("""
        MATCH (d:Document)-[:HAS_MONTHLY_ALLOCATION]->(m:MonthlyUsageAllocation)
        RETURN d.fileName as document, count(m) as allocations, 
               sum(m.allocated_cost) as total_cost,
               sum(m.allocated_usage) as total_usage
        ORDER BY allocations DESC
    """)
    for record in result:
        print(f'Document: {record["document"]}')
        print(f'  Allocations: {record["allocations"]}')
        print(f'  Total Cost: ${record["total_cost"]}')
        print(f'  Total Usage: {record["total_usage"]}')
    
    # Query 5: Look for electricity bills
    print("\n=== Electricity Bills ===")
    result = session.run("""
        MATCH (d:Electricitybill)
        RETURN d.fileName as file, d.total_cost as cost, d.total_usage as usage,
               d.start_date as start, d.end_date as end
        LIMIT 5
    """)
    for record in result:
        print(json.dumps(dict(record), indent=2, default=str))
    
    # Query 6: Look for water bills
    print("\n=== Water Bills ===")
    result = session.run("""
        MATCH (d)
        WHERE 'Waterbill' IN labels(d) OR 'WaterBill' IN labels(d)
        RETURN labels(d) as labels, d.fileName as file, 
               d.total_cost as cost, d.total_usage as usage
        LIMIT 5
    """)
    for record in result:
        print(json.dumps(dict(record), indent=2, default=str))
    
    # Query 7: Summary statistics
    print("\n=== Summary Statistics ===")
    result = session.run("""
        MATCH (m:MonthlyUsageAllocation)
        RETURN count(m) as total_allocations,
               count(DISTINCT m.facility_id) as facilities,
               min(m.usage_year) as earliest_year,
               max(m.usage_year) as latest_year,
               sum(m.allocated_cost) as total_cost,
               sum(m.allocated_usage) as total_usage
    """)
    for record in result:
        print(json.dumps(dict(record), indent=2, default=str))

driver.close()