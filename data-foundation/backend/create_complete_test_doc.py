#!/usr/bin/env python3

import uuid
from neo4j import GraphDatabase
from datetime import datetime

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "EhsAI2024!"))

# Create a test document with MonthlyUsageAllocation already linked
doc_id = str(uuid.uuid4())

with driver.session() as session:
    # Create document with electricity bill and allocations in a single transaction
    result = session.run("""
        // Create the document
        CREATE (d:Document:ElectricityBill {
            id: $doc_id,
            fileName: 'test_with_allocations.pdf',
            document_type: 'Electric Bill',
            recognition_confidence: 0.99,
            source: 'test_with_allocations',
            file_path: '/tmp/test_with_allocations.pdf',
            uploaded_at: $timestamp,
            processed_at: $timestamp,
            statement_date: '2025-08-15',
            start_date: '2025-07-15',
            end_date: '2025-08-15',
            total_kwh: 100000,
            total_cost: 12500.00,
            date_received: $timestamp
        })
        
        // Create utility bill
        CREATE (ub:UtilityBill {
            id: $ub_id,
            billing_period_start: '2025-07-15',
            billing_period_end: '2025-08-15',
            due_date: '2025-08-30',
            peak_kwh: 60000,
            off_peak_kwh: 40000,
            total_kwh: 100000,
            total_cost: 12500.00
        })
        
        // Create relationship
        CREATE (d)-[:EXTRACTED_TO]->(ub)
        
        // Create MonthlyUsageAllocation for July 2025
        CREATE (ma1:MonthlyUsageAllocation {
            allocation_id: $alloc1_id,
            usage_year: 2025,
            usage_month: 7,
            allocation_method: 'custom',
            allocation_percentage: 51.61,
            allocated_usage: 51610.0,
            allocated_cost: 6451.25,
            facility_id: 'facility_001',
            facility_name: 'Main Facility',
            days_in_month: 31,
            billing_days_in_month: 16,
            created_at: $timestamp
        })
        
        // Create MonthlyUsageAllocation for August 2025
        CREATE (ma2:MonthlyUsageAllocation {
            allocation_id: $alloc2_id,
            usage_year: 2025,
            usage_month: 8,
            allocation_method: 'custom',
            allocation_percentage: 48.39,
            allocated_usage: 48390.0,
            allocated_cost: 6048.75,
            facility_id: 'facility_001',
            facility_name: 'Main Facility',
            days_in_month: 31,
            billing_days_in_month: 15,
            created_at: $timestamp
        })
        
        // Create relationships
        CREATE (d)-[:HAS_MONTHLY_ALLOCATION]->(ma1)
        CREATE (d)-[:HAS_MONTHLY_ALLOCATION]->(ma2)
        
        RETURN d.id as doc_id, d.total_kwh as kwh, 
               ma2.allocated_usage as august_usage
    """, {
        "doc_id": doc_id,
        "ub_id": str(uuid.uuid4()),
        "alloc1_id": str(uuid.uuid4()),
        "alloc2_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat()
    })
    
    record = result.single()
    print(f"Created test document with MonthlyUsageAllocation nodes:")
    print(f"  Document ID: {record['doc_id']}")
    print(f"  Total Usage: {record['kwh']} kWh")
    print(f"  August 2025 Allocation: {record['august_usage']} kWh")
    print(f"\nThis document should now appear in the web app with prorated values.")

driver.close()