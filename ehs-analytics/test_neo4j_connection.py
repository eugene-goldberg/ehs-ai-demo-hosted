#!/usr/bin/env python3
"""Quick Neo4j connectivity test"""

import os
import sys
from datetime import datetime
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_neo4j_connection():
    """Test Neo4j database connection"""
    print(f"=== NEO4J CONNECTIVITY TEST ===")
    print(f"Time: {datetime.now()}")
    
    # Get connection details
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    
    print(f"URI: {uri}")
    print(f"Username: {username}")
    print(f"Password: {'*' * len(password) if password else 'NOT SET'}")
    
    if not password:
        print("ERROR: NEO4J_PASSWORD not set in environment")
        return False
    
    try:
        # Create driver
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # Test connection
        with driver.session() as session:
            result = session.run("RETURN 1 as test, timestamp() as ts")
            record = result.single()
            print(f"\n✅ CONNECTION SUCCESSFUL")
            print(f"Server timestamp: {record['ts']}")
            
            # Check for EHS nodes
            result = session.run("MATCH (n) RETURN labels(n) as labels, count(n) as count LIMIT 10")
            print(f"\nNode counts by label:")
            for record in result:
                if record['labels']:
                    print(f"  {record['labels'][0]}: {record['count']}")
        
        driver.close()
        return True
        
    except Exception as e:
        print(f"\n❌ CONNECTION FAILED")
        print(f"Error type: {type(e).__name__}")
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_neo4j_connection()
    sys.exit(0 if success else 1)