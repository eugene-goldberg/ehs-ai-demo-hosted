#!/usr/bin/env python3

from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "EhsAI2024!"))

with driver.session() as session:
    # Count MonthlyUsageAllocation nodes
    result = session.run("MATCH (m:MonthlyUsageAllocation) RETURN count(m) as count")
    count = result.single()['count']
    print(f'Total MonthlyUsageAllocation nodes: {count}')
    
    # Get electricity bills
    result = session.run("""
        MATCH (d:Document)
        WHERE 'Electricitybill' IN labels(d) OR 'ElectricityBill' IN labels(d)
        RETURN d.id as id, d.fileName as file, d.total_kwh as kwh, 
               d.total_cost as cost, d.start_date as start, d.end_date as end
        LIMIT 5
    """)
    print('\nElectricity bills in database:')
    for record in result:
        print(f'ID: {record["id"]}')
        print(f'  File: {record["file"]}')
        print(f'  Usage: {record["kwh"]} kWh')
        print(f'  Cost: ${record["cost"]}')
        print(f'  Period: {record["start"]} to {record["end"]}')
        print()

driver.close()

print("To create MonthlyUsageAllocation nodes, you need to run the prorating API")
print("for each document. The prorating feature will create these nodes.")