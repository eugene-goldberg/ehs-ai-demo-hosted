#!/usr/bin/env python3
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
username = os.getenv('NEO4J_USERNAME', 'neo4j')
password = os.getenv('NEO4J_PASSWORD', 'EhsAI2024!')

driver = GraphDatabase.driver(uri, auth=(username, password))

with driver.session() as session:
    # Check all nodes with those exact IDs regardless of label
    ids_to_check = ['HVAC-001', 'HVAC-002', 'COMP-001', 'COMP-002']
    
    for node_id in ids_to_check:
        query = "MATCH (n) WHERE n.id = $node_id RETURN labels(n) as labels, n.id as id"
        result = session.run(query, node_id=node_id)
        record = result.single()
        if record:
            print(f'Found node with id {node_id}: labels = {record["labels"]}')
        else:
            print(f'No node found with id {node_id}')
    
driver.close()