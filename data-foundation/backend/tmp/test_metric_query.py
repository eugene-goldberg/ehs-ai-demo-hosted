#!/usr/bin/env python3
"""
MetricData Query Test Script

This script connects to Neo4j and examines the MetricData (EHSMetric) nodes
created by our test data seeding script to understand why the dashboard
shows 0 values.

Features:
- Connects to Neo4j using the same configuration as the backend
- Queries for EHSMetric nodes and their relationships
- Shows sample metric data and analyzes the structure
- Checks if metrics are properly linked to facilities/areas
- Provides debugging information for dashboard integration

Author: Test Agent
Date: 2025-08-30
"""

import sys
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

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

class MetricQueryTester:
    """Test class for querying and analyzing EHS metrics"""
    
    def __init__(self):
        self.graph = None
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        logger = logging.getLogger("MetricQueryTester")
        logger.setLevel(logging.INFO)
        logger.addHandler(console_handler)
        
        return logger
    
    def connect_to_neo4j(self) -> bool:
        """Establish connection to Neo4j database"""
        try:
            # Load environment variables
            env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
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
    
    def analyze_database_structure(self):
        """Analyze the overall database structure"""
        self.logger.info("=" * 60)
        self.logger.info("DATABASE STRUCTURE ANALYSIS")
        self.logger.info("=" * 60)
        
        # Check node types
        node_types_query = "CALL db.labels()"
        node_types = self.graph.query(node_types_query)
        
        self.logger.info("Available Node Labels:")
        for label_result in node_types:
            label = label_result['label']
            count_query = f"MATCH (n:{label}) RETURN COUNT(n) as count"
            count_result = self.graph.query(count_query)
            count = count_result[0]['count'] if count_result else 0
            self.logger.info(f"  {label}: {count} nodes")
        
        # Check relationship types
        rel_types_query = "CALL db.relationshipTypes()"
        rel_types = self.graph.query(rel_types_query)
        
        self.logger.info("\nAvailable Relationship Types:")
        for rel_result in rel_types:
            rel_type = rel_result['relationshipType']
            count_query = f"MATCH ()-[r:{rel_type}]->() RETURN COUNT(r) as count"
            count_result = self.graph.query(count_query)
            count = count_result[0]['count'] if count_result else 0
            self.logger.info(f"  {rel_type}: {count} relationships")
    
    def analyze_test_data_structure(self):
        """Analyze the test data structure"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("TEST DATA STRUCTURE ANALYSIS")
        self.logger.info("=" * 60)
        
        # Check test data counts
        test_data_queries = [
            ("Test Sites", "MATCH (s:Site) WHERE s.source = 'test_data' RETURN COUNT(s) as count"),
            ("Test Buildings", "MATCH (b:Building) WHERE b.source = 'test_data' RETURN COUNT(b) as count"),
            ("Test Floors", "MATCH (f:Floor) WHERE f.source = 'test_data' RETURN COUNT(f) as count"),
            ("Test Areas", "MATCH (a:Area) WHERE a.source = 'test_data' RETURN COUNT(a) as count"),
            ("Test Facilities", "MATCH (f:Facility) WHERE f.source = 'test_data' RETURN COUNT(f) as count"),
            ("Test EHS Metrics", "MATCH (m:EHSMetric) WHERE m.source = 'test_data' RETURN COUNT(m) as count"),
            ("Test Incidents", "MATCH (i:Incident) WHERE i.source = 'test_data' RETURN COUNT(i) as count")
        ]
        
        for name, query in test_data_queries:
            result = self.graph.query(query)
            count = result[0]['count'] if result else 0
            status = "✓" if count > 0 else "✗"
            self.logger.info(f"  {status} {name}: {count}")
        
        # Show site hierarchy
        sites_query = """
        MATCH (s:Site) WHERE s.source = 'test_data'
        RETURN s.name as site_name, s.code as site_code
        """
        sites = self.graph.query(sites_query)
        
        self.logger.info(f"\nTest Sites Created:")
        for site in sites:
            self.logger.info(f"  - {site['site_name']} ({site['site_code']})")
    
    def analyze_ehs_metrics(self):
        """Analyze EHS metrics in detail"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EHS METRICS DETAILED ANALYSIS")
        self.logger.info("=" * 60)
        
        # Get metrics summary by category
        metrics_summary_query = """
        MATCH (m:EHSMetric) WHERE m.source = 'test_data'
        RETURN m.category as category, 
               COUNT(m) as count,
               COLLECT(DISTINCT m.metric_name) as metric_names
        ORDER BY category
        """
        
        metrics_summary = self.graph.query(metrics_summary_query)
        
        self.logger.info("Metrics by Category:")
        for metric_cat in metrics_summary:
            category = metric_cat['category']
            count = metric_cat['count']
            names = metric_cat['metric_names']
            self.logger.info(f"  {category}: {count} records")
            for name in names:
                self.logger.info(f"    - {name}")
        
        # Show date range of metrics
        date_range_query = """
        MATCH (m:EHSMetric) WHERE m.source = 'test_data'
        RETURN MIN(m.recorded_date) as earliest_date,
               MAX(m.recorded_date) as latest_date,
               COUNT(DISTINCT m.recorded_date) as unique_dates
        """
        
        date_result = self.graph.query(date_range_query)
        if date_result:
            date_info = date_result[0]
            self.logger.info(f"\nMetrics Date Range:")
            self.logger.info(f"  Earliest: {date_info['earliest_date']}")
            self.logger.info(f"  Latest: {date_info['latest_date']}")
            self.logger.info(f"  Unique Dates: {date_info['unique_dates']}")
    
    def analyze_metric_relationships(self):
        """Analyze how metrics are related to facilities/areas"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("METRIC RELATIONSHIPS ANALYSIS")
        self.logger.info("=" * 60)
        
        # Check MEASURED_AT relationships
        measured_at_query = """
        MATCH (m:EHSMetric)-[r:MEASURED_AT]->(a:Area)
        WHERE m.source = 'test_data'
        RETURN COUNT(r) as measured_at_count
        """
        
        result = self.graph.query(measured_at_query)
        measured_at_count = result[0]['measured_at_count'] if result else 0
        self.logger.info(f"MEASURED_AT relationships: {measured_at_count}")
        
        # Show sample metric-area relationships
        sample_relationships_query = """
        MATCH (m:EHSMetric)-[:MEASURED_AT]->(a:Area)
        WHERE m.source = 'test_data'
        RETURN m.metric_name as metric_name,
               m.category as category,
               m.value as value,
               m.unit as unit,
               m.recorded_date as date,
               a.name as area_name,
               a.type as area_type
        ORDER BY m.recorded_date DESC
        LIMIT 10
        """
        
        relationships = self.graph.query(sample_relationships_query)
        
        self.logger.info("\nSample Metric-Area Relationships:")
        for rel in relationships[:5]:  # Show first 5
            self.logger.info(f"  {rel['metric_name']} ({rel['category']})")
            self.logger.info(f"    Value: {rel['value']} {rel['unit']}")
            self.logger.info(f"    Date: {rel['date']}")
            self.logger.info(f"    Area: {rel['area_name']} ({rel['area_type']})")
            self.logger.info("")
    
    def analyze_facility_metrics_connection(self):
        """Analyze how facilities connect to metrics"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("FACILITY-METRICS CONNECTION ANALYSIS")
        self.logger.info("=" * 60)
        
        # Check if facilities have metrics through areas
        facility_metrics_query = """
        MATCH (f:Facility)-[:LOCATED_IN]->(a:Area)<-[:MEASURED_AT]-(m:EHSMetric)
        WHERE f.source = 'test_data' AND m.source = 'test_data'
        RETURN f.name as facility_name,
               COUNT(m) as metric_count,
               COLLECT(DISTINCT m.category) as categories
        ORDER BY metric_count DESC
        LIMIT 10
        """
        
        facility_metrics = self.graph.query(facility_metrics_query)
        
        self.logger.info("Facilities with Metrics (Top 10):")
        for fm in facility_metrics:
            self.logger.info(f"  {fm['facility_name']}: {fm['metric_count']} metrics")
            self.logger.info(f"    Categories: {', '.join(fm['categories'])}")
    
    def show_recent_metrics_sample(self):
        """Show sample of recent metrics data"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("RECENT METRICS SAMPLE")
        self.logger.info("=" * 60)
        
        # Get recent metrics from the last 7 days
        recent_metrics_query = """
        MATCH (m:EHSMetric)
        WHERE m.source = 'test_data' 
          AND m.recorded_date >= date() - duration('P7D')
        RETURN m.metric_name as name,
               m.category as category,
               m.value as value,
               m.unit as unit,
               m.target_value as target,
               m.recorded_date as date,
               m.site_code as site,
               m.area_name as area
        ORDER BY m.recorded_date DESC, m.category, m.metric_name
        LIMIT 20
        """
        
        recent_metrics = self.graph.query(recent_metrics_query)
        
        if recent_metrics:
            self.logger.info("Recent Metrics (Last 7 Days):")
            current_category = None
            for metric in recent_metrics:
                if metric['category'] != current_category:
                    current_category = metric['category']
                    self.logger.info(f"\n  === {current_category.upper()} ===")
                
                self.logger.info(f"  {metric['name']}")
                self.logger.info(f"    Value: {metric['value']} {metric['unit']} (Target: {metric['target']})")
                self.logger.info(f"    Date: {metric['date']} | Site: {metric['site']} | Area: {metric['area']}")
        else:
            self.logger.warning("No recent metrics found in the last 7 days")
            
            # Get any metrics
            any_metrics_query = """
            MATCH (m:EHSMetric)
            WHERE m.source = 'test_data'
            RETURN m.metric_name as name,
                   m.category as category,
                   m.value as value,
                   m.unit as unit,
                   m.recorded_date as date,
                   m.site_code as site,
                   m.area_name as area
            ORDER BY m.recorded_date DESC
            LIMIT 10
            """
            
            any_metrics = self.graph.query(any_metrics_query)
            if any_metrics:
                self.logger.info("Sample of Any Available Metrics:")
                for metric in any_metrics:
                    self.logger.info(f"  {metric['name']} ({metric['category']})")
                    self.logger.info(f"    Value: {metric['value']} {metric['unit']}")
                    self.logger.info(f"    Date: {metric['date']} | Site: {metric['site']}")
    
    def analyze_dashboard_compatibility(self):
        """Analyze compatibility with dashboard expectations"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("DASHBOARD COMPATIBILITY ANALYSIS")
        self.logger.info("=" * 60)
        
        # Check if there's a MetricData node type (what dashboard might expect)
        metricdata_query = "MATCH (m:MetricData) RETURN COUNT(m) as count"
        try:
            metricdata_result = self.graph.query(metricdata_query)
            metricdata_count = metricdata_result[0]['count'] if metricdata_result else 0
            self.logger.info(f"MetricData nodes: {metricdata_count}")
        except:
            self.logger.info("MetricData nodes: 0 (label doesn't exist)")
        
        # Check EHSMetric structure
        ehsmetric_structure_query = """
        MATCH (m:EHSMetric) 
        WHERE m.source = 'test_data'
        RETURN DISTINCT keys(m) as properties
        LIMIT 1
        """
        
        structure_result = self.graph.query(ehsmetric_structure_query)
        if structure_result:
            properties = structure_result[0]['properties']
            self.logger.info(f"\nEHSMetric Node Properties:")
            for prop in sorted(properties):
                self.logger.info(f"  - {prop}")
        
        # Check if there are Facility nodes and their structure
        facility_structure_query = """
        MATCH (f:Facility) 
        WHERE f.source = 'test_data'
        RETURN DISTINCT keys(f) as properties
        LIMIT 1
        """
        
        facility_structure = self.graph.query(facility_structure_query)
        if facility_structure:
            properties = facility_structure[0]['properties']
            self.logger.info(f"\nFacility Node Properties:")
            for prop in sorted(properties):
                self.logger.info(f"  - {prop}")
        
        # Show facility to metrics path
        facility_metrics_path_query = """
        MATCH path = (f:Facility)-[:LOCATED_IN]->(a:Area)<-[:MEASURED_AT]-(m:EHSMetric)
        WHERE f.source = 'test_data' AND m.source = 'test_data'
        RETURN f.name as facility_name,
               a.name as area_name,
               m.metric_name as metric_name,
               m.value as metric_value
        LIMIT 5
        """
        
        path_results = self.graph.query(facility_metrics_path_query)
        self.logger.info(f"\nFacility->Area->Metrics Path Examples:")
        for path in path_results:
            self.logger.info(f"  {path['facility_name']} -> {path['area_name']} -> {path['metric_name']} ({path['metric_value']})")
    
    def provide_dashboard_recommendations(self):
        """Provide recommendations for dashboard integration"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("DASHBOARD INTEGRATION RECOMMENDATIONS")
        self.logger.info("=" * 60)
        
        recommendations = [
            "1. METRIC ACCESS:",
            "   - Metrics are stored as EHSMetric nodes, not MetricData",
            "   - Query: MATCH (m:EHSMetric) WHERE m.source = 'test_data'",
            "",
            "2. FACILITY-METRICS RELATIONSHIP:",
            "   - Path: Facility -> LOCATED_IN -> Area <- MEASURED_AT <- EHSMetric",
            "   - Direct facility metrics query needed",
            "",
            "3. RECENT DATA QUERY:",
            "   - Use m.recorded_date for filtering recent data",
            "   - Date format appears to be standard Neo4j date type",
            "",
            "4. AGGREGATION PATTERNS:",
            "   - Group by m.category for different metric types",
            "   - Use m.site_code, m.area_name for location grouping",
            "",
            "5. SUGGESTED DASHBOARD QUERIES:",
            "   a) Get facility metrics:",
            "      MATCH (f:Facility)-[:LOCATED_IN]->(a:Area)<-[:MEASURED_AT]-(m:EHSMetric)",
            "      WHERE f.name = $facilityName",
            "      RETURN m.category, m.metric_name, m.value, m.unit, m.recorded_date",
            "",
            "   b) Get recent site metrics:",
            "      MATCH (m:EHSMetric)",
            "      WHERE m.site_code = $siteCode AND m.recorded_date >= date() - duration('P30D')",
            "      RETURN m.category, AVG(m.value) as avg_value",
            "",
            "6. POTENTIAL ISSUES:",
            "   - Dashboard might be looking for 'MetricData' instead of 'EHSMetric'",
            "   - Property names might not match dashboard expectations",
            "   - Date filtering logic might need adjustment"
        ]
        
        for rec in recommendations:
            self.logger.info(rec)
    
    def run_analysis(self):
        """Run complete analysis"""
        try:
            self.logger.info("STARTING NEO4J METRIC DATA ANALYSIS")
            self.logger.info("=" * 80)
            
            if not self.connect_to_neo4j():
                return False
            
            # Run all analysis methods
            analysis_methods = [
                self.analyze_database_structure,
                self.analyze_test_data_structure,
                self.analyze_ehs_metrics,
                self.analyze_metric_relationships,
                self.analyze_facility_metrics_connection,
                self.show_recent_metrics_sample,
                self.analyze_dashboard_compatibility,
                self.provide_dashboard_recommendations
            ]
            
            for method in analysis_methods:
                try:
                    method()
                except Exception as e:
                    self.logger.error(f"Error in {method.__name__}: {e}")
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("ANALYSIS COMPLETED")
            self.logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return False

def main():
    """Main execution function"""
    tester = MetricQueryTester()
    success = tester.run_analysis()
    
    if success:
        print("\n✓ Neo4j metric analysis completed!")
    else:
        print("\n✗ Analysis failed. Check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
