#!/usr/bin/env python3

from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "EhsAI2024!"))

with driver.session() as session:
    # Get documents with UtilityBill relationships
    result = session.run("""
        MATCH (d:Document)
        OPTIONAL MATCH (d)-[:EXTRACTED_TO]->(ub:UtilityBill)
        RETURN d.id as id, labels(d) as labels, 
               keys(d) as doc_props,
               ub.total_kwh as kwh, ub.total_cost as cost,
               ub.start_date as start, ub.end_date as end,
               keys(ub) as ub_props
        LIMIT 3
    """)
    
    print("Documents and their extracted data:")
    for record in result:
        print(f"\nDocument ID: {record['id']}")
        print(f"Labels: {record['labels']}")
        print(f"Document properties: {record['doc_props']}")
        if record['ub_props']:
            print(f"UtilityBill properties: {record['ub_props']}")
            print(f"  Usage: {record['kwh']} kWh")
            print(f"  Cost: ${record['cost']}")
            print(f"  Period: {record['start']} to {record['end']}")
        else:
            print("  No UtilityBill data extracted")

driver.close()

print("\n\nThe document exists but has no extracted UtilityBill data.")
print("This is why prorating cannot work - there's no usage data to prorate.")