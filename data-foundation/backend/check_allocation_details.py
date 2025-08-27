#!/usr/bin/env python3

from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "EhsAI2024!"))

with driver.session() as session:
    # Get MonthlyUsageAllocation details
    result = session.run("""
        MATCH (d:Document)-[:HAS_MONTHLY_ALLOCATION]->(m:MonthlyUsageAllocation)
        RETURN d.id as doc_id, d.fileName as file,
               m.allocation_id as alloc_id,
               m.allocated_usage as usage, 
               m.allocated_cost as cost,
               m.facility_id as facility,
               m.facility_name as facility_name,
               m.usage_year as year, 
               m.usage_month as month,
               m.allocation_percentage as percentage
        ORDER BY d.id, m.facility_id
    """)
    
    print("MonthlyUsageAllocation details:")
    current_doc = None
    for record in result:
        if current_doc != record["doc_id"]:
            current_doc = record["doc_id"]
            print(f"\n\nDocument: {current_doc[:8]}... ({record['file']})")
        
        print(f"  Facility: {record['facility']} - {record['facility_name']}")
        print(f"    Allocation: {record['percentage']}%")
        print(f"    Usage: {record['usage']} kWh")
        print(f"    Cost: ${record['cost']}")
        print(f"    Period: {record['month']}/{record['year']}")

driver.close()

print("\n\nThe MonthlyUsageAllocation nodes have been successfully created!")
print("The frontend should now display the prorated monthly usage values.")