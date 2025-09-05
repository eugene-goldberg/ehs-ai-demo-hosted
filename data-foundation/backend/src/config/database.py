"""Database configuration module"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection configuration
DATABASE_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
DATABASE_USER = os.getenv("NEO4J_USERNAME", "neo4j")
DATABASE_PASSWORD = os.getenv("NEO4J_PASSWORD", "EhsAI2024!")
DATABASE_NAME = os.getenv("NEO4J_DATABASE", "neo4j")

def get_db():
    """Get database connection parameters"""
    return {
        "uri": DATABASE_URI,
        "username": DATABASE_USER,
        "password": DATABASE_PASSWORD,
        "database": DATABASE_NAME
    }
