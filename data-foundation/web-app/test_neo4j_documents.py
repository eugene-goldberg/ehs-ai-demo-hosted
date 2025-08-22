#!/usr/bin/env python3
from neo4j import GraphDatabase
import json

# Neo4j connection
uri = 'bolt://localhost:7687'
username = 'neo4j'
password = 'EhsAI2024!'

try:
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    with driver.session() as session:
        # Query to get all documents
        print("=== Querying Neo4j for Documents ===")
        result = session.run("""
            MATCH (d:Document)
            RETURN d.id as id, 
                   d.type as document_type,
                   d.created_at as date_received,
                   labels(d) as labels,
                   d.account_number as account_number,
                   d.service_address as site,
                   d
            ORDER BY d.created_at DESC
        """)
        
        documents = []
        for record in result:
            print(f"\nDocument ID: {record['id']}")
            print(f"Labels: {record['labels']}")
            print(f"Type: {record['document_type']}")
            print(f"Site: {record['site']}")
            print(f"All properties: {dict(record['d'])}")
            
            # Extract document type from labels
            labels = record['labels']
            doc_type = 'Unknown'
            
            if 'Utilitybill' in labels or 'UtilityBill' in labels:
                doc_type = 'Electric Bill'
            elif 'Waterbill' in labels or 'WaterBill' in labels:
                doc_type = 'Water Bill'
            elif 'Wastemanifest' in labels or 'WasteManifest' in labels:
                doc_type = 'Waste Manifest'
            
            doc = {
                'id': record['id'],
                'document_type': doc_type,
                'date_received': record['date_received'] or record.get('created_at', '2025-08-18T10:00:00'),
                'site': record['site'] or record['account_number'] or 'Main Facility'
            }
            documents.append(doc)
        
        print(f"\n=== Total Documents Found: {len(documents)} ===")
        print("\nFormatted for frontend:")
        print(json.dumps(documents, indent=2))
        
    driver.close()
    
except Exception as e:
    print(f"Error: {e}")
