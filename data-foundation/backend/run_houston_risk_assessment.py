#!/usr/bin/env python3
"""
Updated script to run risk assessment for Houston TX with proper Neo4j connection initialization
 
This script properly initializes the Neo4j connection before running the assessment,
includes connection debugging, and has better error handling for connection problems.
 
Key improvements:
1. Properly initializes Neo4j connection before creating the agent
2. Uses Neo4j credentials from .env file
3. Includes connection debugging and health checks
4. Better error handling for connection problems
5. Comprehensive logging and status reporting
 
Author: AI Assistant
Date: 2025-09-11
"""
 
import sys
import os
import logging
import traceback
from pathlib import Path
 
# Add the src directory to Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))
 
# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/houston_risk_assessment.log')
    ]
)
logger = logging.getLogger(__name__)
 
try:
    # Load environment variables first
    from dotenv import load_dotenv
    load_dotenv()
    
    # Import required modules
    from database.neo4j_client import Neo4jClient, ConnectionConfig
    from agents.risk_assessment_agent import RiskAssessmentAgent
    
    print("Successfully imported required modules")
    logger.info("Required modules imported successfully")
    
except ImportError as e:
    error_msg = f"Error importing required modules: {e}"
    print(error_msg)
    logger.error(error_msg)
    print("Make sure you're running this from the backend directory")
    print("and that all dependencies are installed.")
    sys.exit(1)
 
def debug_neo4j_connection():
    """
    Debug Neo4j connection parameters and test connectivity
    
    Returns:
        tuple: (success: bool, config: ConnectionConfig, error_msg: str)
    """
    print("\n" + "="*60)
    print("NEO4J CONNECTION DEBUGGING")
    print("="*60)
    
    try:
        # Create configuration from environment variables
        config = ConnectionConfig.from_env()
        
        print(f"Neo4j URI: {config.uri}")
        print(f"Neo4j Username: {config.username}")
        print(f"Neo4j Database: {config.database}")
        print(f"Password configured: {'Yes' if config.password else 'No'}")
        
        # Test connection
        print("\nTesting Neo4j connection...")
        client = Neo4jClient(config=config, enable_logging=True)
        
        if client.connect():
            print("✓ Neo4j connection established successfully!")
            
            # Perform health check
            print("\nPerforming health check...")
            health_check = client.health_check()
            
            if health_check.is_healthy:
                print("✓ Neo4j health check passed!")
                print(f"  Connection time: {health_check.connection_time:.3f}s")
                
                if health_check.server_info:
                    version = health_check.server_info.get('neo4j_version', 'Unknown')
                    edition = health_check.server_info.get('edition', 'Unknown')
                    print(f"  Neo4j version: {version}")
                    print(f"  Neo4j edition: {edition}")
                
                # Get database statistics
                print("\nDatabase statistics:")
                db_info = client.get_database_info()
                print(f"  Nodes: {db_info.get('node_count', 0)}")
                print(f"  Relationships: {db_info.get('relationship_count', 0)}")
                print(f"  Labels: {db_info.get('labels', [])}")
                
                return True, config, None
                
            else:
                error_msg = f"Health check failed: {health_check.error_message}"
                print(f"✗ {error_msg}")
                logger.error(error_msg)
                return False, config, error_msg
        else:
            error_msg = "Failed to establish Neo4j connection"
            print(f"✗ {error_msg}")
            logger.error(error_msg)
            return False, config, error_msg
            
    except Exception as e:
        error_msg = f"Error during connection debugging: {e}"
        print(f"✗ {error_msg}")
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return False, None, error_msg
        
    finally:
        if 'client' in locals():
            client.close()
 
def initialize_neo4j_client():
    """
    Initialize and return a properly configured Neo4j client
    
    Returns:
        Neo4jClient: Configured and connected Neo4j client, or None if failed
    """
    try:
        # Create configuration from environment
        config = ConnectionConfig.from_env()
        
        # Create client with detailed logging enabled
        client = Neo4jClient(
            config=config,
            max_retries=3,
            retry_delay=2.0,
            enable_logging=True
        )
        
        print("Connecting to Neo4j database...")
        logger.info("Initializing Neo4j connection")
        
        if client.connect():
            print("✓ Neo4j client initialized and connected successfully")
            logger.info("Neo4j client connected successfully")
            return client
        else:
            print("✗ Failed to connect Neo4j client")
            logger.error("Failed to connect Neo4j client")
            return None
            
    except Exception as e:
        error_msg = f"Error initializing Neo4j client: {e}"
        print(f"✗ {error_msg}")
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return None
 
def verify_environment_variables():
    """
    Verify that all required environment variables are set
    
    Returns:
        tuple: (success: bool, missing_vars: list)
    """
    print("\n" + "="*60)
    print("ENVIRONMENT VARIABLES CHECK")
    print("="*60)
    
    required_vars = [
        'NEO4J_URI',
        'NEO4J_USERNAME',
        'NEO4J_PASSWORD',
        'NEO4J_DATABASE',
        'OPENAI_API_KEY'
    ]
    
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✓ {var}: {'*' * len(value) if 'password' in var.lower() or 'key' in var.lower() else value}")
        else:
            print(f"✗ {var}: NOT SET")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\nMissing required environment variables: {', '.join(missing_vars)}")
        return False, missing_vars
    else:
        print("\n✓ All required environment variables are set")
        return True, []
 
def run_houston_risk_assessment():
    """
    Main function to run the Houston TX risk assessment with proper initialization
    """
    print("="*80)
    print("HOUSTON TX RISK ASSESSMENT - VERSION 2")
    print("Enhanced with proper Neo4j connection initialization")
    print("="*80)
    
    try:
        # Step 1: Verify environment variables
        env_success, missing_vars = verify_environment_variables()
        if not env_success:
            print(f"Cannot continue: Missing environment variables: {missing_vars}")
            return False
        
        # Step 2: Debug Neo4j connection
        conn_success, config, error_msg = debug_neo4j_connection()
        if not conn_success:
            print(f"Cannot continue: Neo4j connection failed: {error_msg}")
            return False
        
        # Step 3: Initialize Neo4j client
        print("\n" + "="*60)
        print("INITIALIZING NEO4J CLIENT")
        print("="*60)
        
        neo4j_client = initialize_neo4j_client()
        if not neo4j_client:
            print("Cannot continue: Failed to initialize Neo4j client")
            return False
        
        # Step 4: Initialize Risk Assessment Agent
        print("\n" + "="*60)
        print("INITIALIZING RISK ASSESSMENT AGENT")
        print("="*60)
        
        print("Creating Risk Assessment Agent with Neo4j client...")
        logger.info("Initializing Risk Assessment Agent")
        
        agent = RiskAssessmentAgent(neo4j_client=neo4j_client)
        print("✓ Risk Assessment Agent initialized successfully")
        logger.info("Risk Assessment Agent initialized successfully")
        
        # Step 5: Run the assessment
        print("\n" + "="*60)
        print("RUNNING RISK ASSESSMENT")
        print("="*60)
        
        site_id = 'houston_tx'
        category = 'electricity'  # Can be changed to 'water' or 'waste' if needed
        
        print(f"Site ID: {site_id}")
        print(f"Category: {category}")
        print(f"Starting assessment...")
        
        logger.info(f"Starting risk assessment for {site_id}/{category}")
        
        results = agent.analyze_site_performance(site_id, category)
        
        # Step 6: Display results
        print("\n" + "="*60)
        print("RISK ASSESSMENT RESULTS")
        print("="*60)
        
        if 'error' in results:
            print(f"✗ Assessment Error: {results['error']}")
            logger.error(f"Assessment error: {results['error']}")
            return False
        
        print("\nTREND ANALYSIS:")
        print("-" * 40)
        trend_analysis = results.get('trend_analysis', {})
        if isinstance(trend_analysis, dict):
            for key, value in trend_analysis.items():
                print(f"{key}: {value}")
        else:
            print(trend_analysis)
        
        print("\nRISK ASSESSMENT:")
        print("-" * 40)
        risk_assessment = results.get('risk_assessment', {})
        if isinstance(risk_assessment, dict):
            risk_level = risk_assessment.get('risk_level', 'UNKNOWN')
            risk_probability = risk_assessment.get('risk_probability', 0)
            gap_percentage = risk_assessment.get('gap_percentage', 0)
            
            print(f"Risk Level: {risk_level}")
            print(f"Risk Probability: {risk_probability:.1%}")
            print(f"Gap Percentage: {gap_percentage:.1f}%")
            print(f"Goal Achievable: {risk_assessment.get('goal_achievable', 'Unknown')}")
            
            if 'analysis_text' in risk_assessment:
                print(f"Analysis: {risk_assessment['analysis_text'][:200]}...")
        else:
            print(risk_assessment)
        
        print("\nRECOMMENDATIONS:")
        print("-" * 40)
        recommendations = results.get('recommendations', {})
        if isinstance(recommendations, dict) and 'recommendations' in recommendations:
            rec_list = recommendations['recommendations']
            for i, rec in enumerate(rec_list[:5], 1):  # Show first 5 recommendations
                if isinstance(rec, str):
                    print(f"{i}. {rec[:150]}...")
                else:
                    print(f"{i}. {str(rec)[:150]}...")
            
            if len(rec_list) > 5:
                print(f"... and {len(rec_list) - 5} more recommendations")
        else:
            print(recommendations)
        
        print("\n" + "="*60)
        print("✓ ASSESSMENT COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        logger.info("Houston TX risk assessment completed successfully")
        return True
        
    except Exception as e:
        error_msg = f"Error during risk assessment: {e}"
        print(f"✗ {error_msg}")
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return False
        
    finally:
        # Clean up Neo4j connection
        if 'neo4j_client' in locals() and neo4j_client:
            print("\nClosing Neo4j connection...")
            neo4j_client.close()
            logger.info("Neo4j connection closed")
 
def main():
    """Main entry point"""
    try:
        success = run_houston_risk_assessment()
        if success:
            print(f"\nLog file available at: /tmp/houston_risk_assessment.log")
            sys.exit(0)
        else:
            print(f"\nAssessment failed. Check log file at: /tmp/houston_risk_assessment.log")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nAssessment interrupted by user")
        logger.info("Assessment interrupted by user")
        sys.exit(130)
    except Exception as e:
        error_msg = f"Unexpected error in main: {e}"
        print(f"✗ {error_msg}")
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        sys.exit(1)
 
if __name__ == "__main__":
    main()
