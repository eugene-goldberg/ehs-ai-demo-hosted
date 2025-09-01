#!/usr/bin/env python3
"""
Debug Electricity Queries Script

This script connects directly to Neo4j to debug why electricity endpoints are returning empty data.
It will:
1. Check what ElectricityConsumption data exists in Neo4j
2. Test the exact query used in the environmental assessment service
3. Identify property name mismatches between service and Neo4j schema
4. Compare with working waste query
5. Provide fixes for electricity query issues

Created: 2025-08-31
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

# Add the project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, ClientError
except ImportError:
    print("Error: neo4j package not installed. Please install with: pip install neo4j")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'debug_electricity_queries.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ElectricityQueryDebugger:
    """Debug electricity data queries in Neo4j"""
    
    def __init__(self, uri: str, username: str, password: str):
        """Initialize Neo4j connection"""
        self.driver = None
        self.uri = uri
        self.username = username
        self.password = password
        self._connect()
    
    def _connect(self) -> None:
        """Establish connection to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"Successfully connected to Neo4j at {self.uri}")
        except ServiceUnavailable as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error connecting to Neo4j: {e}")
            raise
    
    def close(self) -> None:
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def check_electricity_data_exists(self) -> Dict[str, Any]:
        """Check what ElectricityConsumption data exists in Neo4j"""
        logger.info("=== CHECKING ELECTRICITY DATA EXISTENCE ===")
        
        with self.driver.session() as session:
            # Count total electricity records
            count_query = "MATCH (e:ElectricityConsumption) RETURN count(e) as count"
            count_result = session.run(count_query)
            total_count = count_result.single()['count']
            
            logger.info(f"Total ElectricityConsumption records: {total_count}")
            
            if total_count == 0:
                logger.warning("No ElectricityConsumption records found!")
                return {
                    'total_count': 0,
                    'sample_records': [],
                    'properties': [],
                    'issue': 'No electricity data exists in database'
                }
            
            # Get sample records to examine structure
            sample_query = """
            MATCH (e:ElectricityConsumption) 
            RETURN e 
            ORDER BY e.date DESC 
            LIMIT 5
            """
            sample_result = session.run(sample_query)
            sample_records = []
            
            for record in sample_result:
                node = record['e']
                sample_records.append(dict(node))
            
            # Get all property names that exist on ElectricityConsumption nodes
            properties_query = """
            MATCH (e:ElectricityConsumption)
            UNWIND keys(e) AS key
            RETURN DISTINCT key
            ORDER BY key
            """
            properties_result = session.run(properties_query)
            properties = [record['key'] for record in properties_result]
            
            logger.info(f"Properties found on ElectricityConsumption nodes: {properties}")
            logger.info("Sample records:")
            for i, record in enumerate(sample_records):
                logger.info(f"  Record {i+1}: {record}")
            
            return {
                'total_count': total_count,
                'sample_records': sample_records,
                'properties': properties,
                'issue': None
            }
    
    def test_service_electricity_query(self) -> Dict[str, Any]:
        """Test the exact query used in the environmental assessment service"""
        logger.info("=== TESTING SERVICE ELECTRICITY QUERY ===")
        
        # This is the exact query from environmental_assessment_service.py
        service_query = """
        MATCH (e:ElectricityConsumption)
        WHERE ($location IS NULL OR e.location CONTAINS $location)
        AND ($start_date IS NULL OR e.date >= $start_date)
        AND ($end_date IS NULL OR e.date <= $end_date)
        RETURN e.location as location, e.date as date, e.consumption_kwh as consumption,
               e.cost_usd as cost, e.source_type as source_type, e.efficiency_rating as efficiency
        ORDER BY e.date DESC
        """
        
        parameters = {
            "location": None,
            "start_date": None,
            "end_date": None
        }
        
        logger.info(f"Running service query with parameters: {parameters}")
        
        with self.driver.session() as session:
            try:
                result = session.run(service_query, parameters)
                records = [dict(record) for record in result]
                
                logger.info(f"Service query returned {len(records)} records")
                
                if records:
                    logger.info("Sample service query results:")
                    for i, record in enumerate(records[:3]):  # Show first 3
                        logger.info(f"  Record {i+1}: {record}")
                else:
                    logger.warning("Service query returned no records!")
                
                return {
                    'query': service_query,
                    'parameters': parameters,
                    'record_count': len(records),
                    'records': records[:5],  # First 5 records
                    'issue': 'No records returned' if not records else None
                }
                
            except Exception as e:
                logger.error(f"Error running service query: {e}")
                return {
                    'query': service_query,
                    'parameters': parameters,
                    'record_count': 0,
                    'records': [],
                    'error': str(e),
                    'issue': f'Query execution failed: {e}'
                }
    
    def analyze_property_mismatches(self, electricity_data: Dict, service_query_data: Dict) -> Dict[str, Any]:
        """Analyze property name mismatches between service query and actual data"""
        logger.info("=== ANALYZING PROPERTY MISMATCHES ===")
        
        # Properties expected by the service query
        expected_properties = [
            'location', 'date', 'consumption_kwh', 'cost_usd', 
            'source_type', 'efficiency_rating'
        ]
        
        # Properties that actually exist in the data
        actual_properties = electricity_data.get('properties', [])
        
        # Find missing properties
        missing_properties = [prop for prop in expected_properties if prop not in actual_properties]
        
        # Find extra properties (not used by service)
        extra_properties = [prop for prop in actual_properties if prop not in expected_properties]
        
        logger.info(f"Expected properties: {expected_properties}")
        logger.info(f"Actual properties: {actual_properties}")
        logger.info(f"Missing properties: {missing_properties}")
        logger.info(f"Extra properties (not used): {extra_properties}")
        
        # Analyze the impact
        issues = []
        if missing_properties:
            issues.append(f"Missing properties: {missing_properties}")
        
        # Check if any records would be filtered out due to property constraints
        if electricity_data['total_count'] > 0 and service_query_data['record_count'] == 0:
            issues.append("All records filtered out by service query constraints")
        
        return {
            'expected_properties': expected_properties,
            'actual_properties': actual_properties,
            'missing_properties': missing_properties,
            'extra_properties': extra_properties,
            'issues': issues,
            'total_records_available': electricity_data['total_count'],
            'records_returned_by_service_query': service_query_data['record_count']
        }
    
    def compare_with_working_waste_query(self) -> Dict[str, Any]:
        """Compare electricity query with the working waste query to understand differences"""
        logger.info("=== COMPARING WITH WORKING WASTE QUERY ===")
        
        # Test the waste query that works (from the service)
        waste_query = """
        MATCH (w:WasteGeneration)
        WHERE ($location IS NULL OR w.location CONTAINS $location)
        AND ($start_date IS NULL OR w.date >= $start_date)
        AND ($end_date IS NULL OR w.date <= $end_date)
        RETURN w.location as location, w.date as date, w.amount_pounds as amount,
               w.disposal_cost_usd as cost, w.waste_type as waste_type, w.disposal_method as disposal_method,
               0 as recycled
        ORDER BY w.date DESC
        """
        
        parameters = {
            "location": None,
            "start_date": None,
            "end_date": None
        }
        
        with self.driver.session() as session:
            try:
                # Test waste query
                waste_result = session.run(waste_query, parameters)
                waste_records = [dict(record) for record in waste_result]
                
                logger.info(f"Waste query returned {len(waste_records)} records")
                
                # Get waste data properties for comparison
                waste_properties_query = """
                MATCH (w:WasteGeneration)
                UNWIND keys(w) AS key
                RETURN DISTINCT key
                ORDER BY key
                """
                waste_props_result = session.run(waste_properties_query)
                waste_properties = [record['key'] for record in waste_props_result]
                
                # Get electricity properties for comparison
                elec_properties_query = """
                MATCH (e:ElectricityConsumption)
                UNWIND keys(e) AS key
                RETURN DISTINCT key
                ORDER BY key
                """
                elec_props_result = session.run(elec_properties_query)
                elec_properties = [record['key'] for record in elec_props_result]
                
                logger.info(f"Waste properties: {waste_properties}")
                logger.info(f"Electricity properties: {elec_properties}")
                
                return {
                    'waste_query_success': True,
                    'waste_records_count': len(waste_records),
                    'waste_properties': waste_properties,
                    'electricity_properties': elec_properties,
                    'waste_sample_records': waste_records[:3],
                    'analysis': self._analyze_query_differences()
                }
                
            except Exception as e:
                logger.error(f"Error testing waste query: {e}")
                return {
                    'waste_query_success': False,
                    'error': str(e),
                    'waste_properties': [],
                    'electricity_properties': [],
                    'analysis': {}
                }
    
    def _analyze_query_differences(self) -> Dict[str, Any]:
        """Analyze key differences between working waste query and electricity query"""
        
        analysis = {
            'key_observations': [
                "Waste query works because properties match the data model",
                "Electricity query may be looking for properties that don't exist",
                "Need to verify if electricity data has location property for filtering"
            ],
            'potential_issues': [
                "ElectricityConsumption nodes might not have 'location' property",
                "ElectricityConsumption nodes might not have 'source_type' property", 
                "ElectricityConsumption nodes might not have 'efficiency_rating' property",
                "Date format or filtering might be incompatible"
            ]
        }
        
        return analysis
    
    def generate_fixed_electricity_query(self, analysis_data: Dict) -> Dict[str, Any]:
        """Generate a corrected electricity query based on analysis"""
        logger.info("=== GENERATING FIXED ELECTRICITY QUERY ===")
        
        # Get actual properties available
        actual_properties = analysis_data.get('actual_properties', [])
        
        # Map service expectations to actual properties
        property_mapping = {
            'location': 'facility_id',  # Use facility_id instead of location
            'consumption_kwh': 'consumption_kwh',
            'cost_usd': 'cost_usd',
            'date': 'date'
        }
        
        # Build the corrected query
        corrected_query = """
        MATCH (e:ElectricityConsumption)
        WHERE ($location IS NULL OR e.facility_id CONTAINS $location)
        AND ($start_date IS NULL OR e.date >= $start_date)
        AND ($end_date IS NULL OR e.date <= $end_date)
        RETURN e.facility_id as location, e.date as date, e.consumption_kwh as consumption,
               e.cost_usd as cost, 
               COALESCE(e.source_type, 'Unknown') as source_type, 
               COALESCE(e.efficiency_rating, 0.0) as efficiency
        ORDER BY e.date DESC
        """
        
        # Alternative simplified query focusing only on existing properties
        simplified_query = """
        MATCH (e:ElectricityConsumption)
        RETURN e.facility_id as location, e.date as date, e.consumption_kwh as consumption,
               e.cost_usd as cost, 'Unknown' as source_type, 0.0 as efficiency
        ORDER BY e.date DESC
        LIMIT 10
        """
        
        logger.info("Generated corrected query:")
        logger.info(corrected_query)
        
        # Test the corrected query
        with self.driver.session() as session:
            try:
                result = session.run(simplified_query)
                test_records = [dict(record) for record in result]
                
                logger.info(f"Corrected query test returned {len(test_records)} records")
                if test_records:
                    logger.info("Sample corrected query results:")
                    for i, record in enumerate(test_records[:3]):
                        logger.info(f"  Record {i+1}: {record}")
                
                return {
                    'corrected_query': corrected_query,
                    'simplified_query': simplified_query,
                    'test_successful': True,
                    'test_records_count': len(test_records),
                    'test_sample_records': test_records[:3],
                    'property_mapping': property_mapping,
                    'fixes_needed': self._generate_fix_recommendations(analysis_data)
                }
                
            except Exception as e:
                logger.error(f"Error testing corrected query: {e}")
                return {
                    'corrected_query': corrected_query,
                    'simplified_query': simplified_query,
                    'test_successful': False,
                    'error': str(e),
                    'property_mapping': property_mapping,
                    'fixes_needed': self._generate_fix_recommendations(analysis_data)
                }
    
    def _generate_fix_recommendations(self, analysis_data: Dict) -> List[str]:
        """Generate specific recommendations to fix the electricity query issues"""
        
        fixes = []
        missing_properties = analysis_data.get('missing_properties', [])
        
        if 'location' in missing_properties:
            fixes.append(
                "CRITICAL: Replace 'e.location' with 'e.facility_id' in the WHERE clause and RETURN statement"
            )
        
        if 'source_type' in missing_properties:
            fixes.append(
                "OPTIONAL: Use COALESCE(e.source_type, 'Unknown') or default value for source_type"
            )
        
        if 'efficiency_rating' in missing_properties:
            fixes.append(
                "OPTIONAL: Use COALESCE(e.efficiency_rating, 0.0) or default value for efficiency_rating"
            )
        
        fixes.extend([
            "Update environmental_assessment_service.py line 122-123 to use correct property names",
            "Update service query to handle optional properties gracefully",
            "Consider adding missing properties to the data model if they're required for business logic"
        ])
        
        return fixes
    
    def run_comprehensive_debug(self) -> Dict[str, Any]:
        """Run comprehensive debugging analysis"""
        logger.info("Starting comprehensive electricity query debugging...")
        
        debug_results = {
            'timestamp': datetime.now().isoformat(),
            'debug_steps': []
        }
        
        try:
            # Step 1: Check if electricity data exists
            logger.info("Step 1: Checking electricity data existence...")
            electricity_data = self.check_electricity_data_exists()
            debug_results['electricity_data_check'] = electricity_data
            debug_results['debug_steps'].append("✓ Checked electricity data existence")
            
            # Step 2: Test service query
            logger.info("Step 2: Testing service electricity query...")
            service_query_data = self.test_service_electricity_query()
            debug_results['service_query_test'] = service_query_data
            debug_results['debug_steps'].append("✓ Tested service electricity query")
            
            # Step 3: Analyze property mismatches
            logger.info("Step 3: Analyzing property mismatches...")
            mismatch_analysis = self.analyze_property_mismatches(electricity_data, service_query_data)
            debug_results['property_analysis'] = mismatch_analysis
            debug_results['debug_steps'].append("✓ Analyzed property mismatches")
            
            # Step 4: Compare with working waste query
            logger.info("Step 4: Comparing with working waste query...")
            waste_comparison = self.compare_with_working_waste_query()
            debug_results['waste_comparison'] = waste_comparison
            debug_results['debug_steps'].append("✓ Compared with working waste query")
            
            # Step 5: Generate fixes
            logger.info("Step 5: Generating fix recommendations...")
            fix_recommendations = self.generate_fixed_electricity_query(mismatch_analysis)
            debug_results['fix_recommendations'] = fix_recommendations
            debug_results['debug_steps'].append("✓ Generated fix recommendations")
            
            # Summary
            debug_results['summary'] = self._generate_debug_summary(debug_results)
            
            logger.info("Comprehensive debugging completed successfully!")
            return debug_results
            
        except Exception as e:
            logger.error(f"Error during comprehensive debugging: {e}")
            debug_results['error'] = str(e)
            debug_results['summary'] = f"Debug failed with error: {e}"
            return debug_results
    
    def _generate_debug_summary(self, debug_results: Dict) -> Dict[str, Any]:
        """Generate a summary of debug findings"""
        
        summary = {
            'root_cause_identified': False,
            'main_issues': [],
            'confidence_level': 'high',
            'immediate_actions': [],
            'data_availability': 'unknown'
        }
        
        # Analyze electricity data availability
        elec_data = debug_results.get('electricity_data_check', {})
        if elec_data.get('total_count', 0) == 0:
            summary['root_cause_identified'] = True
            summary['main_issues'].append("No ElectricityConsumption data exists in database")
            summary['confidence_level'] = 'very_high'
            summary['immediate_actions'].append("Run create_environmental_models.py to populate sample data")
            summary['data_availability'] = 'none'
        else:
            summary['data_availability'] = 'exists'
            
            # Check service query results
            service_data = debug_results.get('service_query_test', {})
            if service_data.get('record_count', 0) == 0:
                summary['root_cause_identified'] = True
                summary['main_issues'].append("Service query returns no records despite data existing")
                
                # Check property analysis
                prop_analysis = debug_results.get('property_analysis', {})
                missing_props = prop_analysis.get('missing_properties', [])
                
                if 'location' in missing_props:
                    summary['main_issues'].append("Query uses 'location' property but data has 'facility_id'")
                    summary['immediate_actions'].append("Update query to use 'facility_id' instead of 'location'")
                
                if missing_props:
                    summary['main_issues'].append(f"Missing properties: {missing_props}")
                    summary['immediate_actions'].append("Update query to handle missing optional properties")
        
        return summary


def load_config() -> Dict[str, str]:
    """Load Neo4j configuration from environment or defaults"""
    config = {
        'uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        'username': os.getenv('NEO4J_USERNAME', 'neo4j'),
        'password': os.getenv('NEO4J_PASSWORD', 'password')
    }
    
    # Try to load from .env file if it exists
    env_path = os.path.join(project_root, '.env')
    if os.path.exists(env_path):
        logger.info(f"Loading configuration from {env_path}")
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key in config:
                        config[key] = value
    
    return config


def main():
    """Main function"""
    print("="*80)
    print("ELECTRICITY QUERIES DEBUG SCRIPT")
    print("="*80)
    
    try:
        # Load configuration
        config = load_config()
        logger.info(f"Connecting to Neo4j at {config['uri']}")
        
        # Create debugger
        debugger = ElectricityQueryDebugger(
            uri=config['uri'],
            username=config['username'],
            password=config['password']
        )
        
        # Run comprehensive debugging
        results = debugger.run_comprehensive_debug()
        
        # Print summary
        print("\n" + "="*60)
        print("DEBUG RESULTS SUMMARY")
        print("="*60)
        
        summary = results.get('summary', {})
        print(f"Root cause identified: {summary.get('root_cause_identified', False)}")
        print(f"Data availability: {summary.get('data_availability', 'unknown')}")
        print(f"Confidence level: {summary.get('confidence_level', 'unknown')}")
        
        print("\nMain issues identified:")
        for issue in summary.get('main_issues', []):
            print(f"  • {issue}")
        
        print("\nImmediate actions required:")
        for action in summary.get('immediate_actions', []):
            print(f"  → {action}")
        
        # Save detailed results to file
        results_file = os.path.join(os.path.dirname(__file__), 'electricity_debug_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {results_file}")
        print(f"Debug log saved to: {os.path.join(os.path.dirname(__file__), 'debug_electricity_queries.log')}")
        
        # Generate fix code if needed
        if results.get('fix_recommendations', {}).get('corrected_query'):
            print("\n" + "="*60)
            print("RECOMMENDED FIXES")
            print("="*60)
            
            fix_data = results['fix_recommendations']
            print("Replace the electricity query in environmental_assessment_service.py (line ~117-124) with:")
            print("\n" + fix_data['corrected_query'])
            
            print("\nSpecific fixes needed:")
            for fix in fix_data.get('fixes_needed', []):
                print(f"  • {fix}")
        
        debugger.close()
        
    except KeyboardInterrupt:
        logger.info("Debug script interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Debug script failed: {e}")
        print(f"\nERROR: Debug script failed with: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()