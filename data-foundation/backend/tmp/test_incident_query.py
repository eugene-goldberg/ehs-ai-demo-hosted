#!/usr/bin/env python3
"""
Test script to query Neo4j database for incident data structure and relationships.

This script:
1. Queries for all Incident nodes and their properties
2. Shows the structure and relationships of incidents
3. Checks how incidents relate to facilities/locations
4. Analyzes the incident data model in detail

Author: Test Runner Sub-agent
Date: 2025-08-30
"""

import sys
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add src directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
src_dir = os.path.join(backend_dir, 'src')
sys.path.insert(0, src_dir)

try:
    from langchain_neo4j import Neo4jGraph
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running this script in the virtual environment with required dependencies.")
    sys.exit(1)

class IncidentQueryTest:
    """Test class for querying incident data"""
    
    def __init__(self):
        self.graph = None
        self.test_id = f"incident_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_filename = f"incident_query_test_{self.test_id}.log"
        log_path = os.path.join(current_dir, log_filename)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        logger = logging.getLogger(f"IncidentQueryTest_{self.test_id}")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def connect_to_neo4j(self) -> bool:
        """Establish connection to Neo4j database"""
        try:
            # Load environment variables
            env_path = os.path.join(backend_dir, '.env')
            load_dotenv(env_path)
            
            neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
            neo4j_username = os.getenv('NEO4J_USERNAME', 'neo4j')
            neo4j_password = os.getenv('NEO4J_PASSWORD', 'EhsAI2024!')
            neo4j_database = os.getenv('NEO4J_DATABASE', 'neo4j')
            
            self.logger.info(f"Connecting to Neo4j at {neo4j_uri}")
            
            self.graph = Neo4jGraph(
                url=neo4j_uri,
                username=neo4j_username,
                password=neo4j_password,
                database=neo4j_database
            )
            
            # Test connection
            test_result = self.graph.query("RETURN 'connection_test' as test")
            if test_result and test_result[0]['test'] == 'connection_test':
                self.logger.info("Neo4j connection successful")
                return True
            else:
                raise Exception("Connection test failed")
                
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            return False
    
    def query_all_incidents(self) -> List[Dict]:
        """Query all Incident nodes and their properties"""
        try:
            self.logger.info("Querying all Incident nodes...")
            
            query = """
            MATCH (i:Incident)
            RETURN i,
                   labels(i) as node_labels,
                   keys(i) as properties,
                   elementId(i) as element_id
            ORDER BY i.incident_date DESC
            """
            
            results = self.graph.query(query)
            
            self.logger.info(f"Found {len(results)} Incident nodes")
            
            incidents = []
            for result in results:
                incident_data = dict(result['i'])
                incident_data['_labels'] = result['node_labels']
                incident_data['_properties'] = result['properties']
                incident_data['_element_id'] = result['element_id']
                incidents.append(incident_data)
                
                self.logger.info(f"Incident ID: {incident_data.get('incident_id', 'N/A')}")
                self.logger.info(f"  Type: {incident_data.get('incident_type', 'N/A')}")
                self.logger.info(f"  Date: {incident_data.get('incident_date', 'N/A')}")
                self.logger.info(f"  Severity: {incident_data.get('severity', 'N/A')}")
                self.logger.info(f"  Properties: {incident_data['_properties']}")
            
            return incidents
            
        except Exception as e:
            self.logger.error(f"Failed to query incidents: {e}")
            return []
    
    def analyze_incident_relationships(self) -> Dict:
        """Analyze relationships of Incident nodes"""
        try:
            self.logger.info("Analyzing Incident relationships...")
            
            # Query outgoing relationships
            outgoing_query = """
            MATCH (i:Incident)-[r]->(target)
            RETURN i.incident_id as incident_id,
                   type(r) as relationship_type,
                   labels(target) as target_labels,
                   target,
                   elementId(i) as incident_element_id,
                   elementId(target) as target_element_id
            """
            
            # Query incoming relationships
            incoming_query = """
            MATCH (source)-[r]->(i:Incident)
            RETURN i.incident_id as incident_id,
                   type(r) as relationship_type,
                   labels(source) as source_labels,
                   source,
                   elementId(i) as incident_element_id,
                   elementId(source) as source_element_id
            """
            
            outgoing_results = self.graph.query(outgoing_query)
            incoming_results = self.graph.query(incoming_query)
            
            relationships = {
                'outgoing': outgoing_results,
                'incoming': incoming_results
            }
            
            self.logger.info(f"Found {len(outgoing_results)} outgoing relationships")
            self.logger.info(f"Found {len(incoming_results)} incoming relationships")
            
            # Log outgoing relationships
            for rel in outgoing_results:
                self.logger.info(f"Outgoing: Incident {rel['incident_id']} -[{rel['relationship_type']}]-> {rel['target_labels']}")
                target_data = dict(rel['target'])
                self.logger.info(f"  Target: {json.dumps(target_data, indent=2, default=str)}")
            
            # Log incoming relationships
            for rel in incoming_results:
                self.logger.info(f"Incoming: {rel['source_labels']} -[{rel['relationship_type']}]-> Incident {rel['incident_id']}")
                source_data = dict(rel['source'])
                self.logger.info(f"  Source: {json.dumps(source_data, indent=2, default=str)}")
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"Failed to analyze relationships: {e}")
            return {}
    
    def query_facility_incident_connections(self) -> List[Dict]:
        """Query how incidents are connected to facilities/locations"""
        try:
            self.logger.info("Querying facility/location connections to incidents...")
            
            # Query various path patterns that might connect incidents to locations
            queries = [
                # Direct facility connections
                """
                MATCH (f:Facility)-[r1]-(i:Incident)
                RETURN 'Facility-Incident' as connection_type,
                       f.name as facility_name,
                       f.type as facility_type,
                       type(r1) as relationship1,
                       i.incident_id as incident_id,
                       i.incident_type as incident_type,
                       null as intermediate_node
                """,
                
                # Facility -> Area -> Incident patterns
                """
                MATCH (f:Facility)-[r1]-(a)-[r2]-(i:Incident)
                WHERE NOT labels(a) = ['Incident'] AND NOT labels(a) = ['Facility']
                RETURN 'Facility-Intermediate-Incident' as connection_type,
                       f.name as facility_name,
                       f.type as facility_type,
                       type(r1) as relationship1,
                       labels(a) as intermediate_labels,
                       type(r2) as relationship2,
                       i.incident_id as incident_id,
                       i.incident_type as incident_type,
                       a as intermediate_node
                """,
                
                # Any location-related connections
                """
                MATCH (loc)-[r]-(i:Incident)
                WHERE any(label in labels(loc) WHERE label IN ['Site', 'Building', 'Floor', 'Area', 'Facility', 'Location'])
                RETURN 'Location-Incident' as connection_type,
                       loc.name as location_name,
                       labels(loc) as location_labels,
                       type(r) as relationship,
                       i.incident_id as incident_id,
                       i.incident_type as incident_type,
                       loc as location_node
                """,
                
                # Check for site/building/area properties within incidents
                """
                MATCH (i:Incident)
                WHERE i.site_code IS NOT NULL OR i.building_name IS NOT NULL OR i.area_name IS NOT NULL
                RETURN 'Incident-Location-Properties' as connection_type,
                       i.site_code as site_code,
                       i.building_name as building_name,
                       i.area_name as area_name,
                       i.incident_id as incident_id,
                       i.incident_type as incident_type
                """
            ]
            
            all_connections = []
            
            for i, query in enumerate(queries, 1):
                self.logger.info(f"Running connection query {i}...")
                results = self.graph.query(query)
                
                if results:
                    self.logger.info(f"  Query {i} found {len(results)} connections")
                    all_connections.extend(results)
                    
                    for result in results:
                        self.logger.info(f"  Connection: {result['connection_type']}")
                        for key, value in result.items():
                            if value is not None and key != 'connection_type':
                                self.logger.info(f"    {key}: {value}")
                else:
                    self.logger.info(f"  Query {i} found no connections")
            
            return all_connections
            
        except Exception as e:
            self.logger.error(f"Failed to query facility connections: {e}")
            return []
    
    def analyze_incident_data_model(self) -> Dict:
        """Analyze the incident data model structure"""
        try:
            self.logger.info("Analyzing incident data model...")
            
            # Get all property names and types from incidents
            property_analysis_query = """
            MATCH (i:Incident)
            WITH i, keys(i) as props
            UNWIND props as prop
            RETURN prop as property_name,
                   collect(DISTINCT typename(i[prop])) as property_types,
                   count(*) as usage_count,
                   collect(DISTINCT i[prop])[0..5] as sample_values
            ORDER BY usage_count DESC
            """
            
            property_results = self.graph.query(property_analysis_query)
            
            # Get relationship type distribution
            relationship_analysis_query = """
            MATCH (i:Incident)-[r]-(other)
            RETURN type(r) as relationship_type,
               labels(other) as connected_node_types,
               count(*) as usage_count
            ORDER BY usage_count DESC
            """
            
            relationship_results = self.graph.query(relationship_analysis_query)
            
            analysis = {
                'properties': property_results,
                'relationships': relationship_results
            }
            
            self.logger.info("Property Analysis:")
            for prop in property_results:
                self.logger.info(f"  {prop['property_name']}: {prop['property_types']} (used {prop['usage_count']} times)")
                self.logger.info(f"    Sample values: {prop['sample_values']}")
            
            self.logger.info("Relationship Analysis:")
            for rel in relationship_results:
                self.logger.info(f"  {rel['relationship_type']} -> {rel['connected_node_types']} (used {rel['usage_count']} times)")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze data model: {e}")
            return {}
    
    def check_location_hierarchy(self) -> Dict:
        """Check if there's a location hierarchy that incidents connect to"""
        try:
            self.logger.info("Checking location hierarchy structure...")
            
            hierarchy_queries = [
                # Check for Site nodes and their structure
                """
                MATCH (s:Site)
                OPTIONAL MATCH (s)-[r1]->(b:Building)
                OPTIONAL MATCH (b)-[r2]->(f:Floor)
                OPTIONAL MATCH (f)-[r3]->(a:Area)
                RETURN s.name as site_name, s.code as site_code,
                       collect(DISTINCT b.name) as buildings,
                       collect(DISTINCT f.name) as floors,
                       collect(DISTINCT a.name) as areas,
                       count(DISTINCT b) as building_count,
                       count(DISTINCT f) as floor_count,
                       count(DISTINCT a) as area_count
                """,
                
                # Check for any location-type nodes
                """
                MATCH (n)
                WHERE any(label in labels(n) WHERE label IN ['Site', 'Building', 'Floor', 'Area', 'Location'])
                RETURN DISTINCT labels(n) as node_labels, count(n) as count,
                       collect(n.name)[0..5] as sample_names
                ORDER BY count DESC
                """,
                
                # Check test data locations
                """
                MATCH (n)
                WHERE n.source = 'test_data'
                RETURN DISTINCT labels(n) as node_labels, count(n) as count,
                       collect(n.name)[0..5] as sample_names
                ORDER BY count DESC
                """
            ]
            
            results = {}
            
            for i, query in enumerate(hierarchy_queries, 1):
                self.logger.info(f"Running hierarchy query {i}...")
                query_results = self.graph.query(query)
                results[f'query_{i}'] = query_results
                
                if query_results:
                    self.logger.info(f"  Found {len(query_results)} results")
                    for result in query_results:
                        self.logger.info(f"    {json.dumps(dict(result), indent=2, default=str)}")
                else:
                    self.logger.info("  No results found")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to check location hierarchy: {e}")
            return {}
    
    def generate_summary_report(self, incidents: List[Dict], relationships: Dict, 
                              connections: List[Dict], data_model: Dict, hierarchy: Dict) -> Dict:
        """Generate comprehensive summary report"""
        try:
            self.logger.info("Generating summary report...")
            
            report = {
                'test_id': self.test_id,
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_incidents': len(incidents),
                    'total_outgoing_relationships': len(relationships.get('outgoing', [])),
                    'total_incoming_relationships': len(relationships.get('incoming', [])),
                    'total_connections': len(connections),
                    'property_count': len(data_model.get('properties', [])),
                    'relationship_types': len(data_model.get('relationships', []))
                },
                'incident_details': {
                    'incidents': incidents,
                    'incident_types': list(set([i.get('incident_type') for i in incidents if i.get('incident_type')])),
                    'severities': list(set([i.get('severity') for i in incidents if i.get('severity')])),
                    'locations': list(set([i.get('area_name') for i in incidents if i.get('area_name')]))
                },
                'relationships': relationships,
                'location_connections': connections,
                'data_model': data_model,
                'location_hierarchy': hierarchy
            }
            
            # Write detailed report to file
            report_filename = f"incident_analysis_report_{self.test_id}.json"
            report_path = os.path.join(current_dir, report_filename)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Detailed report written to: {report_path}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary: {e}")
            return {}
    
    def run_incident_analysis(self) -> bool:
        """Run the complete incident analysis"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("STARTING INCIDENT DATA ANALYSIS")
            self.logger.info("=" * 80)
            self.logger.info(f"Test ID: {self.test_id}")
            
            # Connect to Neo4j
            if not self.connect_to_neo4j():
                return False
            
            analysis_steps = [
                ("Query All Incidents", lambda: self.query_all_incidents()),
                ("Analyze Relationships", lambda: self.analyze_incident_relationships()),
                ("Check Facility Connections", lambda: self.query_facility_incident_connections()),
                ("Analyze Data Model", lambda: self.analyze_incident_data_model()),
                ("Check Location Hierarchy", lambda: self.check_location_hierarchy())
            ]
            
            results = {}
            
            for step_name, step_function in analysis_steps:
                self.logger.info(f"\n--- {step_name} ---")
                
                try:
                    result = step_function()
                    results[step_name.lower().replace(' ', '_')] = result
                    self.logger.info(f"✓ {step_name} completed")
                        
                except Exception as e:
                    self.logger.error(f"✗ {step_name} failed with exception: {e}")
                    return False
            
            # Generate comprehensive report
            incidents = results['query_all_incidents']
            relationships = results['analyze_relationships']
            connections = results['check_facility_connections']
            data_model = results['analyze_data_model']
            hierarchy = results['check_location_hierarchy']
            
            summary = self.generate_summary_report(incidents, relationships, connections, data_model, hierarchy)
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("INCIDENT ANALYSIS SUMMARY")
            self.logger.info("=" * 80)
            self.logger.info(f"Total Incidents Found: {summary['summary']['total_incidents']}")
            self.logger.info(f"Incident Types: {summary['incident_details']['incident_types']}")
            self.logger.info(f"Severities: {summary['incident_details']['severities']}")
            self.logger.info(f"Associated Locations: {summary['incident_details']['locations']}")
            self.logger.info(f"Total Relationships: {summary['summary']['total_outgoing_relationships'] + summary['summary']['total_incoming_relationships']}")
            self.logger.info(f"Location Connections: {summary['summary']['total_connections']}")
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("INCIDENT DATA ANALYSIS COMPLETED")
            self.logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Analysis process failed: {e}")
            return False

def main():
    """Main execution function"""
    try:
        print("Starting Incident Data Query Test...")
        
        # Create test instance
        test = IncidentQueryTest()
        
        # Run analysis
        success = test.run_incident_analysis()
        
        if success:
            print("\n✓ Incident data analysis completed successfully!")
            print("Check the log file and JSON report for detailed results.")
        else:
            print("\n✗ Incident data analysis failed. Check the log file for details.")
            return False
            
        return True
            
    except KeyboardInterrupt:
        print("\n⚠ Analysis interrupted by user")
        return False
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
