#!/usr/bin/env python3
"""
Test script to verify dashboard queries work correctly.

This script connects to Neo4j and runs the exact queries used by the dashboard API
to verify they return expected results.
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
import json

# Load environment variables
load_dotenv()

def get_neo4j_driver():
    """Create and return Neo4j driver."""
    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    username = os.getenv('NEO4J_USERNAME', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD')
    
    if not password:
        raise ValueError("NEO4J_PASSWORD not found in environment variables")
    
    return GraphDatabase.driver(uri, auth=(username, password))

def test_risk_query(driver, site_id, category):
    """Test the risk assessment query."""
    print(f"\n=== Testing Risk Query ===")
    print(f"Site ID: {site_id}")
    print(f"Category: {category}")
    
    risk_query = """
    MATCH (s:Site {id: $site_id})-[:HAS_RISK]->(r:RiskAssessment {category: $category})
    RETURN r.risk_level as level, r.description as description, 
           r.confidence_score as confidence, r.factors as factors
    ORDER BY r.assessment_date DESC
    LIMIT 5
    """
    
    with driver.session() as session:
        try:
            result = session.run(risk_query, site_id=site_id, category=category)
            records = list(result)
            
            print(f"Found {len(records)} risk records:")
            for i, record in enumerate(records, 1):
                print(f"\nRisk {i}:")
                print(f"  Level: {record['level']}")
                print(f"  Description: {record['description']}")
                print(f"  Confidence: {record['confidence']}")
                print(f"  Factors: {record['factors']}")
                
            return records
            
        except Exception as e:
            print(f"Error executing risk query: {e}")
            return []

def test_recommendation_query(driver, site_id, category):
    """Test the recommendation query."""
    print(f"\n=== Testing Recommendation Query ===")
    print(f"Site ID: {site_id}")
    print(f"Category: {category}")
    
    rec_query = """
    MATCH (s:Site {id: $site_id})-[:HAS_RECOMMENDATION]->(r:Recommendation {category: $category})
    RETURN r.title as title, r.description as description, r.priority as priority
    ORDER BY r.priority DESC, r.created_date DESC
    LIMIT 5
    """
    
    with driver.session() as session:
        try:
            result = session.run(rec_query, site_id=site_id, category=category)
            records = list(result)
            
            print(f"Found {len(records)} recommendation records:")
            for i, record in enumerate(records, 1):
                print(f"\nRecommendation {i}:")
                print(f"  Title: {record['title']}")
                print(f"  Description: {record['description']}")
                print(f"  Priority: {record['priority']}")
                
            return records
            
        except Exception as e:
            print(f"Error executing recommendation query: {e}")
            return []

def test_site_exists(driver, site_id):
    """Test if the site exists in the database."""
    print(f"\n=== Testing Site Existence ===")
    print(f"Site ID: {site_id}")
    
    site_query = "MATCH (s:Site {id: $site_id}) RETURN s"
    
    with driver.session() as session:
        try:
            result = session.run(site_query, site_id=site_id)
            records = list(result)
            
            if records:
                print(f"Site '{site_id}' exists in database")
                site_data = dict(records[0]['s'])
                print(f"Site properties: {json.dumps(site_data, indent=2, default=str)}")
                return True
            else:
                print(f"Site '{site_id}' NOT found in database")
                return False
                
        except Exception as e:
            print(f"Error checking site existence: {e}")
            return False

def test_available_categories(driver, site_id):
    """Test what categories are available for the site."""
    print(f"\n=== Testing Available Categories ===")
    print(f"Site ID: {site_id}")
    
    # Check risk categories
    risk_cat_query = """
    MATCH (s:Site {id: $site_id})-[:HAS_RISK]->(r:RiskAssessment)
    RETURN DISTINCT r.category as category
    ORDER BY category
    """
    
    # Check recommendation categories
    rec_cat_query = """
    MATCH (s:Site {id: $site_id})-[:HAS_RECOMMENDATION]->(r:Recommendation)
    RETURN DISTINCT r.category as category
    ORDER BY category
    """
    
    with driver.session() as session:
        try:
            # Get risk categories
            risk_result = session.run(risk_cat_query, site_id=site_id)
            risk_categories = [record['category'] for record in risk_result]
            
            # Get recommendation categories
            rec_result = session.run(rec_cat_query, site_id=site_id)
            rec_categories = [record['category'] for record in rec_result]
            
            print(f"Available Risk Categories: {risk_categories}")
            print(f"Available Recommendation Categories: {rec_categories}")
            
            return risk_categories, rec_categories
            
        except Exception as e:
            print(f"Error checking available categories: {e}")
            return [], []

def main():
    """Main test function."""
    print("=== Dashboard Query Test Script ===")
    
    # Test parameters
    site_id = 'algonquin_il'
    category = 'electricity'
    
    try:
        # Initialize Neo4j connection
        driver = get_neo4j_driver()
        print("Successfully connected to Neo4j")
        
        # Test site existence
        site_exists = test_site_exists(driver, site_id)
        
        if site_exists:
            # Test available categories
            risk_cats, rec_cats = test_available_categories(driver, site_id)
            
            # Test the specific queries
            risk_results = test_risk_query(driver, site_id, category)
            rec_results = test_recommendation_query(driver, site_id, category)
            
            # Summary
            print(f"\n=== Test Summary ===")
            print(f"Site '{site_id}' exists: {site_exists}")
            print(f"Risk records found for '{category}': {len(risk_results)}")
            print(f"Recommendation records found for '{category}': {len(rec_results)}")
            
            if not risk_results and not rec_results:
                print(f"\nNo data found for category '{category}'")
                print(f"Available risk categories: {risk_cats}")
                print(f"Available recommendation categories: {rec_cats}")
        else:
            print(f"\nCannot proceed with tests - site '{site_id}' not found")
        
        driver.close()
        print("\nDatabase connection closed")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())