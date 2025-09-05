#!/usr/bin/env python3
"""
Risk Assessment Agent Runner Script

This script runs the RiskAssessmentAgent to populate the dashboard with
risks and recommendations for the specified site and category.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Add the backend/src path to sys.path for relative imports
current_dir = Path(__file__).parent.absolute()
backend_src_path = current_dir.parent
sys.path.insert(0, str(backend_src_path))

def setup_environment():
    """Set up environment variables and connections."""
    # Load environment variables from .env file
    env_path = current_dir.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        print(f"Warning: .env file not found at {env_path}")
    
    # Check for required environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    return openai_api_key

def create_neo4j_client():
    """Create and test Neo4j client connection."""
    try:
        # Import the actual Neo4jClient from the project
        from database.neo4j_client import Neo4jClient
        
        # Create client instance
        client = Neo4jClient()
        
        # Connect to Neo4j
        if not client.connect():
            raise ConnectionError("Failed to connect to Neo4j")
            
        print(f"Neo4j connection: Connection successful")
        
        return client
        
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        raise

def run_risk_assessment():
    """Run the risk assessment agent."""
    try:
        # Import the RiskAssessmentAgent
        from agents.risk_assessment_agent import RiskAssessmentAgent
        
        print("Setting up environment...")
        openai_api_key = setup_environment()
        
        print("Connecting to Neo4j...")
        neo4j_client = create_neo4j_client()
        
        print("Creating RiskAssessmentAgent instance...")
        agent = RiskAssessmentAgent(
            neo4j_client=neo4j_client,
            openai_api_key=openai_api_key
        )
        
        # Risk assessment parameters
        site_id = "algonquin_il"
        category = "electricity"
        
        print(f"\nRunning risk assessment for:")
        print(f"  Site ID: {site_id}")
        print(f"  Category: {category}")
        print("-" * 50)
        
        # Run the risk assessment
        results = agent.analyze_site_performance(site_id=site_id, category=category)
        
        print("\n" + "=" * 60)
        print("RISK ASSESSMENT RESULTS")
        print("=" * 60)
        
        # Print trend analysis
        if "trend_analysis" in results:
            print("\nTREND ANALYSIS:")
            print("-" * 30)
            trend = results["trend_analysis"]
            for key, value in trend.items():
                print(f"{key.replace('_', ' ').title()}: {value}")
        
        # Print risk assessment
        if "risk_assessment" in results:
            print("\nRISK ASSESSMENT:")
            print("-" * 30)
            risk = results["risk_assessment"]
            for key, value in risk.items():
                print(f"{key.replace('_', ' ').title()}: {value}")
        
        # Print recommendations
        if "recommendations" in results:
            print("\nRECOMMENDATIONS:")
            print("-" * 30)
            recommendations = results["recommendations"]
            if isinstance(recommendations, list):
                for i, rec in enumerate(recommendations, 1):
                    print(f"{i}. {rec}")
            else:
                print(recommendations)
        
        # Print full results for debugging
        print("\n" + "=" * 60)
        print("FULL RESULTS (for debugging):")
        print("=" * 60)
        print(results)
        
        # Verify data was stored in Neo4j
        print("\n" + "=" * 60)
        print("VERIFYING DATA STORAGE IN NEO4J")
        print("=" * 60)
        
        with neo4j_client.session() as session:
            # Check for risk assessment records
            query = """
            MATCH (r:RiskAssessment {site_id: $site_id, category: $category})
            RETURN count(r) as risk_count
            """
            result = session.run(query, site_id=site_id, category=category)
            risk_count = result.single()["risk_count"]
            print(f"Risk assessment records in Neo4j: {risk_count}")
            
            # Check for recommendation records
            query = """
            MATCH (rec:Recommendation)
            WHERE rec.site_id = $site_id AND rec.category = $category
            RETURN count(rec) as rec_count
            """
            result = session.run(query, site_id=site_id, category=category)
            rec_count = result.single()["rec_count"]
            print(f"Recommendation records in Neo4j: {rec_count}")
            
            # Show sample records
            query = """
            MATCH (r:RiskAssessment {site_id: $site_id, category: $category})
            RETURN r.risk_level as risk_level, r.assessment_date as date
            LIMIT 1
            """
            result = session.run(query, site_id=site_id, category=category)
            record = result.single()
            if record:
                print(f"Sample risk record - Level: {record['risk_level']}, Date: {record['date']}")
            else:
                print("No risk assessment records found")
        
        print("\n" + "=" * 60)
        print("RISK ASSESSMENT COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("Dashboard should now show updated risks and recommendations!")
        
        # Close Neo4j connection
        neo4j_client.close()
        
        return results
        
    except ImportError as e:
        print(f"Failed to import RiskAssessmentAgent: {e}")
        print("Please ensure the agent file exists at the specified path")
        raise
    except Exception as e:
        print(f"Error running risk assessment: {e}")
        raise

if __name__ == "__main__":
    try:
        results = run_risk_assessment()
    except Exception as e:
        print(f"Script failed with error: {e}")
        sys.exit(1)