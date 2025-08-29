#!/usr/bin/env python3
"""
Standalone Risk Assessment Agent Test Script

This script provides a simple way to test the risk assessment agent directly without 
the full workflow. It:
1. Sets up the Python path properly to avoid import issues
2. Initializes the risk assessment agent directly
3. Runs a risk assessment on a test facility (e.g., "DEMO_FACILITY_001")
4. Prints out the risk assessment results
5. Uses the existing Neo4j connection to analyze any historical data

Usage:
    python3 test_risk_assessment_standalone.py [--facility-id FACILITY_ID] [--verbose]
    
Examples:
    python3 test_risk_assessment_standalone.py
    python3 test_risk_assessment_standalone.py --facility-id DEMO_FACILITY_002 --verbose
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

# Set up the Python path to include the backend src directory
backend_dir = Path(__file__).parent.absolute()
src_dir = backend_dir / 'src'
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(src_dir))

# Now import after path is set up
try:
    from src.agents.risk_assessment.agent import (
        RiskAssessmentAgent,
        create_risk_assessment_agent,
        RiskLevel,
    )
    from src.shared.common_fn import create_graph_database_connection
    from src.langsmith_config import config as langsmith_config
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running this from the backend directory and all dependencies are installed.")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Get environment variables
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'EhsAI2024!')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE', 'neo4j')

# Set up logging
def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('/tmp/risk_assessment_standalone.log')
        ]
    )
    
    # Reduce noise from external libraries
    logging.getLogger('neo4j').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

def check_environment() -> bool:
    """Check if required environment variables are set."""
    required_vars = ['OPENAI_API_KEY', 'NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("Please check your .env file and ensure all required variables are set.")
        return False
    
    return True

def test_neo4j_connection() -> Optional[Any]:
    """Test Neo4j connection and return the graph instance if successful."""
    try:
        uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        username = os.getenv('NEO4J_USERNAME', 'neo4j')
        password = os.getenv('NEO4J_PASSWORD', 'EhsAI2024!')
        database = os.getenv('NEO4J_DATABASE', 'neo4j')
        
        graph = create_graph_database_connection(uri, username, password, database)
        
        # Test the connection with a simple query
        result = graph.query("RETURN 'Connection successful' AS status")
        logger.info("✅ Neo4j connection established successfully")
        
        # Query for facility data
        facility_query = """
        MATCH (f:Facility)
        RETURN f.facility_id AS facility_id, f.name AS name, COUNT(f) AS count
        LIMIT 5
        """
        facilities = graph.query(facility_query)
        if facilities:
            logger.info(f"Found {len(facilities)} facilities in database")
            for facility in facilities:
                logger.info(f"  - Facility: {facility.get('facility_id', 'Unknown')} - {facility.get('name', 'No name')}")
        else:
            logger.info("No facilities found in database - will create test data")
        
        return graph
        
    except Exception as e:
        logger.error(f"❌ Neo4j connection failed: {e}")
        logger.error("Please ensure Neo4j is running and credentials are correct.")
        return None

def create_sample_data(graph, facility_id: str) -> bool:
    """Create sample data for testing if it doesn't exist."""
    try:
        # Check if facility exists
        check_query = """
        MATCH (f:Facility {facility_id: $facility_id})
        RETURN f
        """
        existing_facility = graph.query(check_query, {"facility_id": facility_id})
        
        if existing_facility:
            logger.info(f"✅ Facility {facility_id} already exists")
            return True
        
        logger.info(f"Creating sample data for facility: {facility_id}")
        
        # Create facility and sample data
        create_query = """
        // Create facility
        MERGE (f:Facility {facility_id: $facility_id})
        SET f.name = $facility_name,
            f.type = 'Manufacturing',
            f.location = 'Demo Location',
            f.created_date = datetime()
        
        // Create sample documents
        MERGE (d1:Document {document_id: $facility_id + '_safety_report_001'})
        SET d1.title = 'Safety Inspection Report',
            d1.type = 'safety_report',
            d1.facility_id = $facility_id,
            d1.upload_date = datetime() - duration('P7D'),
            d1.status = 'processed'
        
        MERGE (d2:Document {document_id: $facility_id + '_env_monitoring_001'})
        SET d2.title = 'Environmental Monitoring Data',
            d2.type = 'environmental_report',
            d2.facility_id = $facility_id,
            d2.upload_date = datetime() - duration('P14D'),
            d2.status = 'processed'
        
        // Create sample incidents
        MERGE (i1:Incident {incident_id: $facility_id + '_incident_001'})
        SET i1.type = 'safety',
            i1.severity = 'medium',
            i1.description = 'Minor equipment malfunction',
            i1.facility_id = $facility_id,
            i1.reported_date = datetime() - duration('P30D'),
            i1.status = 'resolved'
        
        MERGE (i2:Incident {incident_id: $facility_id + '_incident_002'})
        SET i2.type = 'environmental',
            i2.severity = 'low',
            i2.description = 'Temporary air quality reading spike',
            i2.facility_id = $facility_id,
            i2.reported_date = datetime() - duration('P60D'),
            i2.status = 'resolved'
        
        // Create relationships
        MERGE (f)-[:HAS_DOCUMENT]->(d1)
        MERGE (f)-[:HAS_DOCUMENT]->(d2)
        MERGE (f)-[:HAS_INCIDENT]->(i1)
        MERGE (f)-[:HAS_INCIDENT]->(i2)
        
        RETURN f, d1, d2, i1, i2
        """
        
        result = graph.query(create_query, {
            "facility_id": facility_id,
            "facility_name": f"Demo Facility {facility_id}"
        })
        
        logger.info("✅ Sample data created successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create sample data: {e}")
        return False

def run_risk_assessment_test(facility_id: str, verbose: bool = False) -> Dict[str, Any]:
    """Run the risk assessment test."""
    logger.info("=" * 80)
    logger.info("STARTING STANDALONE RISK ASSESSMENT TEST")
    logger.info("=" * 80)
    
    results = {
        "test_started": datetime.utcnow().isoformat(),
        "facility_id": facility_id,
        "status": "starting",
        "errors": [],
        "results": {}
    }
    
    try:
        # Step 1: Check environment
        logger.info("🔧 Step 1: Checking environment...")
        if not check_environment():
            results["status"] = "failed"
            results["errors"].append("Environment check failed")
            return results
        
        # Step 2: Test Neo4j connection
        logger.info("🔗 Step 2: Testing Neo4j connection...")
        graph = test_neo4j_connection()
        if not graph:
            results["status"] = "failed"
            results["errors"].append("Neo4j connection failed")
            return results
        
        # Step 3: Create sample data if needed
        logger.info("📊 Step 3: Setting up test data...")
        if not create_sample_data(graph, facility_id):
            results["status"] = "failed"
            results["errors"].append("Failed to create sample data")
            return results
        
        # Step 4: Initialize risk assessment agent
        logger.info("🤖 Step 4: Initializing risk assessment agent...")
        try:
            risk_agent = create_risk_assessment_agent(
                neo4j_uri=NEO4J_URI,
                neo4j_username=NEO4J_USERNAME,
                neo4j_password=NEO4J_PASSWORD,
                neo4j_database=NEO4J_DATABASE,
                llm_model="gpt-4"
            )
            logger.info("✅ Risk assessment agent initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize risk assessment agent: {e}")
            results["status"] = "failed"
            results["errors"].append(f"Agent initialization failed: {str(e)}")
            return results
        
        
        # Step 6: Run risk assessment
        logger.info("🏃 Step 6: Running risk assessment...")
        
        # Prepare assessment parameters
        assessment_scope = {
            "date_range": {
                "start": (datetime.utcnow() - timedelta(days=90)).isoformat(),
                "end": datetime.utcnow().isoformat()
            },
            "categories": ["environmental", "health", "safety", "compliance"]
        }
        
        metadata = {
            "source": "standalone_test",
            "priority": "normal",
            "requester": "test_script"
        }
        start_time = datetime.utcnow()
        
        try:
            # Invoke the risk assessment workflow
            final_state = risk_agent.assess_facility_risk(
                facility_id=facility_id,
                assessment_scope=assessment_scope,
                metadata=metadata
            )
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(f"✅ Risk assessment completed in {processing_time:.2f} seconds")
            
            # Extract results
            results["status"] = "completed"
            results["processing_time_seconds"] = processing_time
            results["final_status"] = final_state.get("status", "unknown")
            results["current_step"] = final_state.get("current_step", "unknown")
            
            if final_state.get("errors"):
                results["errors"].extend(final_state["errors"])
            
            # Extract risk assessment results
            if final_state.get("risk_assessment"):
                risk_assessment = final_state["risk_assessment"]
                results["results"]["risk_assessment"] = {
                    "overall_risk_level": risk_assessment.overall_risk_level if hasattr(risk_assessment, 'overall_risk_level') else "unknown",
                    "risk_score": risk_assessment.risk_score if hasattr(risk_assessment, 'risk_score') else 0.0,
                    "confidence_level": risk_assessment.confidence_level if hasattr(risk_assessment, 'confidence_level') else 0.0,
                    "methodology": risk_assessment.methodology if hasattr(risk_assessment, 'methodology') else "unknown",
                    "risk_factors_count": len(risk_assessment.risk_factors) if hasattr(risk_assessment, 'risk_factors') and risk_assessment.risk_factors else 0
                }
                
                # Log risk factors
                if hasattr(risk_assessment, 'risk_factors') and risk_assessment.risk_factors:
                    logger.info(f"🎯 Identified {len(risk_assessment.risk_factors)} risk factors:")
                    for i, factor in enumerate(risk_assessment.risk_factors[:5], 1):  # Show first 5
                        logger.info(f"  {i}. {factor.name} ({factor.category}) - Severity: {factor.severity}/10")
            
            # Extract recommendations
            if final_state.get("recommendations"):
                recommendations = final_state["recommendations"]
                results["results"]["recommendations"] = {
                    "count": len(recommendations.recommendations) if hasattr(recommendations, 'recommendations') and recommendations.recommendations else 0,
                    "estimated_risk_reduction": recommendations.estimated_risk_reduction if hasattr(recommendations, 'estimated_risk_reduction') else 0.0
                }
                
                # Log top recommendations
                if hasattr(recommendations, 'recommendations') and recommendations.recommendations:
                    logger.info(f"💡 Generated {len(recommendations.recommendations)} recommendations:")
                    for i, rec in enumerate(recommendations.recommendations[:3], 1):  # Show top 3
                        logger.info(f"  {i}. {rec.title} (Priority: {rec.priority})")
            
        except Exception as e:
            logger.error(f"❌ Risk assessment execution failed: {e}")
            results["status"] = "failed"
            results["errors"].append(f"Assessment execution failed: {str(e)}")
            return results
        
        # Step 7: Query historical data
        logger.info("📊 Step 7: Querying historical data...")
        try:
            historical_query = """
            MATCH (f:Facility {facility_id: $facility_id})-[:HAS_INCIDENT]->(i:Incident)
            RETURN i.type AS incident_type, i.severity AS severity, i.reported_date AS date
            ORDER BY i.reported_date DESC
            LIMIT 10
            """
            incidents = graph.query(historical_query, {"facility_id": facility_id})
            
            results["results"]["historical_incidents"] = len(incidents)
            if incidents:
                logger.info(f"📈 Found {len(incidents)} historical incidents")
                for incident in incidents[:3]:  # Show first 3
                    logger.info(f"  - {incident['incident_type']} incident (severity: {incident['severity']})")
            else:
                logger.info("No historical incidents found")
                
        except Exception as e:
            logger.warning(f"⚠️  Could not query historical data: {e}")
            results["errors"].append(f"Historical data query failed: {str(e)}")
        
        results["test_completed"] = datetime.utcnow().isoformat()
        
        logger.info("=" * 80)
        logger.info("✅ RISK ASSESSMENT TEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Unexpected error during risk assessment test: {e}")
        results["status"] = "failed"
        results["errors"].append(f"Unexpected error: {str(e)}")
        results["test_completed"] = datetime.utcnow().isoformat()
        return results

def print_results_summary(results: Dict[str, Any]):
    """Print a formatted summary of the test results."""
    print("\n" + "=" * 80)
    print("RISK ASSESSMENT TEST RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"📊 Test Status: {results['status'].upper()}")
    print(f"🏢 Facility ID: {results['facility_id']}")
    print(f"⏱️  Processing Time: {results.get('processing_time_seconds', 0):.2f} seconds")
    
    if results.get('errors'):
        print(f"⚠️  Errors ({len(results['errors'])}):")
        for error in results['errors']:
            print(f"   • {error}")
    
    if results.get('results'):
        r = results['results']
        print("\n📋 ASSESSMENT RESULTS:")
        
        if 'risk_assessment' in r:
            ra = r['risk_assessment']
            print(f"  🎯 Overall Risk Level: {ra.get('overall_risk_level', 'unknown').upper()}")
            print(f"  📊 Risk Score: {ra.get('risk_score', 0)}/100")
            print(f"  🎯 Confidence: {ra.get('confidence_level', 0)*100:.1f}%")
            print(f"  🔍 Risk Factors: {ra.get('risk_factors_count', 0)} identified")
        
        if 'recommendations' in r:
            rec = r['recommendations']
            print(f"  💡 Recommendations: {rec.get('count', 0)} generated")
            print(f"  📉 Expected Risk Reduction: {rec.get('estimated_risk_reduction', 0):.1f}%")
        
        if 'historical_incidents' in r:
            print(f"  📈 Historical Incidents: {r['historical_incidents']} found")
    
    print("\n" + "=" * 80)
    
    # Save detailed results to file
    results_file = f"/tmp/risk_assessment_test_{results['facility_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Detailed results saved to: {results_file}")
    except Exception as e:
        print(f"⚠️  Could not save results to file: {e}")

def main():
    """Main function to run the standalone risk assessment test."""
    parser = argparse.ArgumentParser(
        description='Standalone Risk Assessment Agent Test',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--facility-id', type=str, default='DEMO_FACILITY_001',
                       help='Facility ID to test (default: DEMO_FACILITY_001)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    global logger
    logger = setup_logging(args.verbose)
    
    # Run the test
    results = run_risk_assessment_test(args.facility_id, args.verbose)
    
    # Print summary
    print_results_summary(results)
    
    # Exit with appropriate code
    if results['status'] == 'completed':
        return 0
    else:
        return 1

if __name__ == '__main__':
    sys.exit(main())