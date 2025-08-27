#!/usr/bin/env python3

from neo4j import GraphDatabase
from datetime import datetime, date

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "EhsAI2024!"))

with driver.session() as session:
    # First, update the document with proper dates if they're missing
    result = session.run("""
        MATCH (d:Document {id: 'unknown_document_20250826_160822_963'})
        -[:EXTRACTED_TO]->(ub:UtilityBill)
        WHERE ub.billing_period_start IS NULL OR ub.billing_period_end IS NULL
        SET ub.billing_period_start = '2025-07-15',
            ub.billing_period_end = '2025-08-15'
        RETURN ub.total_kwh as kwh, ub.total_cost as cost,
               ub.billing_period_start as start, ub.billing_period_end as end
    """)
    
    record = result.single()
    if record:
        print("Updated billing period dates for electricity bill:")
        print(f"  Usage: {record['kwh']} kWh")
        print(f"  Cost: ${record['cost']}")
        print(f"  Period: {record['start']} to {record['end']}")
    
    # Also update the document itself
    session.run("""
        MATCH (d:Document {id: 'unknown_document_20250826_160822_963'})
        SET d.start_date = '2025-07-15',
            d.end_date = '2025-08-15',
            d.total_kwh = 130000,
            d.total_cost = 15432.89
        RETURN d
    """)

driver.close()

print("\nNow the document has proper billing dates for prorating.")