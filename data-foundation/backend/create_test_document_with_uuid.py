#!/usr/bin/env python3

import uuid
from neo4j import GraphDatabase
from datetime import datetime

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "EhsAI2024!"))

# Generate a proper UUID
doc_uuid = str(uuid.uuid4())

with driver.session() as session:
    # Create a new test document with proper UUID and data
    result = session.run("""
        CREATE (d:Document:ElectricityBill {
            id: $id,
            fileName: 'test_electricity_bill.pdf',
            document_type: 'Electric Bill',
            recognition_confidence: 0.95,
            source: 'test',
            file_path: '/tmp/test_electricity_bill.pdf',
            uploaded_at: $timestamp,
            processed_at: $timestamp,
            statement_date: '2025-08-15',
            start_date: '2025-07-15',
            end_date: '2025-08-15',
            total_kwh: 130000,
            total_cost: 15432.89
        })
        -[:EXTRACTED_TO]->
        (ub:UtilityBill {
            id: $ub_id,
            billing_period_start: '2025-07-15',
            billing_period_end: '2025-08-15',
            due_date: '2025-08-30',
            peak_kwh: 80000,
            off_peak_kwh: 50000,
            total_kwh: 130000,
            total_cost: 15432.89,
            state_environmental_surcharge: 45.00,
            grid_infrastructure_fee: 120.00,
            base_service_charge: 75.00
        })
        RETURN d.id as doc_id, d.total_kwh as kwh, d.total_cost as cost
    """, {
        "id": doc_uuid,
        "ub_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat()
    })
    
    record = result.single()
    print(f"Created test electricity bill with proper UUID:")
    print(f"  Document ID: {record['doc_id']}")
    print(f"  Usage: {record['kwh']} kWh")
    print(f"  Cost: ${record['cost']}")

driver.close()

print(f"\nUse this document ID for prorating: {doc_uuid}")