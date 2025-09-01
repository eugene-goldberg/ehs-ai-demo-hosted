#!/usr/bin/env python3
import os
import sys
from neo4j import GraphDatabase

def load_config():
    """Load Neo4j configuration from environment or defaults"""
    config = {
        'uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        'username': os.getenv('NEO4J_USERNAME', 'neo4j'),
        'password': os.getenv('NEO4J_PASSWORD', 'password')
    }
    
    # Try to load from .env file if it exists
    env_path = '.env'
    if os.path.exists(env_path):
        print(f"Loading configuration from {env_path}")
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    
                    # Map .env keys to config keys
                    if key == 'NEO4J_URI':
                        config['uri'] = value
                        print(f"Set URI to: {value}")
                    elif key == 'NEO4J_USERNAME':
                        config['username'] = value
                        print(f"Set username to: {value}")
                    elif key == 'NEO4J_PASSWORD':
                        config['password'] = value
                        print(f"Set password to: {'*' * len(value)}")
    
    return config

# Test the connection
config = load_config()
print(f"Attempting connection to {config['uri']} with user {config['username']}")

try:
    driver = GraphDatabase.driver(
        config['uri'], 
        auth=(config['username'], config['password'])
    )
    # Test connection
    with driver.session() as session:
        session.run("RETURN 1")
    print("✅ Connection successful!")
    driver.close()
except Exception as e:
    print(f"❌ Connection failed: {e}")
