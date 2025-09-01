#!/usr/bin/env python3
"""
Comprehensive Dashboard Requirements Verification Script
========================================================

This script verifies if the EHS Dashboard implementation matches the requirements exactly by checking:
1. What the dashboard currently displays for each category (electricity, water, waste)
2. Whether it shows: goals, facts, risks, recommendations  
3. Where the data is coming from (Neo4j vs generated)
4. Whether 6 months of historical data exists in Neo4j for each site
5. Whether the Risk Assessment Agent stores results in Neo4j
6. Whether risks and recommendations are associated with sites in Neo4j

Usage: python3 verify_dashboard_requirements.py
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import requests
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/tmp/verification.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DashboardVerifier:
    """Comprehensive dashboard requirements verification"""
    
    def __init__(self):
        """Initialize the verifier with environment variables"""
        load_dotenv()
        
        # Neo4j connection
        self.neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD')
        
        # Dashboard API endpoint
        self.dashboard_api = os.getenv('DASHBOARD_API_URL', 'http://localhost:8000')
        
        # Risk Assessment Agent endpoint  
        self.risk_agent_api = os.getenv('RISK_AGENT_API_URL', 'http://localhost:8001')
        
        self.driver = None
        self.verification_results = {
            'timestamp': datetime.now().isoformat(),
            'dashboard_display': {},
            'data_sources': {},
            'historical_data': {},
            'risk_assessment': {},
            'neo4j_integration': {},
            'overall_compliance': False
        }
        
    def connect_to_neo4j(self) -> bool:
        """Establish Neo4j connection"""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri, 
                auth=(self.neo4j_user, self.neo4j_password)
            )
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            logger.info("Successfully connected to Neo4j")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False
    
    def verify_dashboard_display(self):
        """Verify what the dashboard currently displays for each category"""
        logger.info("Verifying dashboard display content...")
        
        categories = ['electricity', 'water', 'waste']
        expected_components = ['goals', 'facts', 'risks', 'recommendations']
        
        for category in categories:
            category_results = {
                'accessible': False,
                'components_found': {},
                'data_structure': None,
                'errors': []
            }
            
            try:
                # Test dashboard endpoint for each category
                response = requests.get(
                    f"{self.dashboard_api}/api/dashboard/{category}",
                    timeout=10
                )
                
                if response.status_code == 200:
                    category_results['accessible'] = True
                    data = response.json()
                    category_results['data_structure'] = data
                    
                    # Check for expected components
                    for component in expected_components:
                        if component in data:
                            category_results['components_found'][component] = {
                                'present': True,
                                'data_type': type(data[component]).__name__,
                                'sample_data': str(data[component])[:200] if data[component] else None
                            }
                        else:
                            category_results['components_found'][component] = {
                                'present': False,
                                'data_type': None,
                                'sample_data': None
                            }
                    
                    logger.info(f"Dashboard {category} category verified successfully")
                else:
                    category_results['errors'].append(f"HTTP {response.status_code}: {response.text}")
                    logger.warning(f"Dashboard {category} returned status {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                category_results['errors'].append(f"Request failed: {str(e)}")
                logger.error(f"Failed to access dashboard {category}: {e}")
            except Exception as e:
                category_results['errors'].append(f"Unexpected error: {str(e)}")
                logger.error(f"Unexpected error verifying {category}: {e}")
            
            self.verification_results['dashboard_display'][category] = category_results
    
    def verify_data_sources(self):
        """Verify where the data is coming from (Neo4j vs generated)"""
        logger.info("Verifying data sources...")
        
        if not self.driver:
            logger.error("No Neo4j connection available")
            return
        
        categories = ['electricity', 'water', 'waste']
        
        for category in categories:
            source_results = {
                'neo4j_data_exists': False,
                'neo4j_record_count': 0,
                'generated_data_indicators': [],
                'data_freshness': None
            }
            
            try:
                with self.driver.session() as session:
                    # Check for consumption data in Neo4j
                    consumption_query = """
                    MATCH (s:Site)-[r:HAS_CONSUMPTION]->(c:Consumption {type: $category})
                    RETURN count(c) as count, max(c.timestamp) as latest_timestamp
                    """
                    result = session.run(consumption_query, category=category)
                    record = result.single()
                    
                    if record and record['count'] > 0:
                        source_results['neo4j_data_exists'] = True
                        source_results['neo4j_record_count'] = record['count']
                        if record['latest_timestamp']:
                            source_results['data_freshness'] = record['latest_timestamp']
                    
                    # Check for goals data
                    goals_query = """
                    MATCH (s:Site)-[r:HAS_GOAL]->(g:Goal {category: $category})
                    RETURN count(g) as goal_count
                    """
                    goal_result = session.run(goals_query, category=category)
                    goal_record = goal_result.single()
                    source_results['neo4j_goals_count'] = goal_record['goal_count'] if goal_record else 0
                    
                    # Check for risk data
                    risk_query = """
                    MATCH (s:Site)-[r:HAS_RISK]->(risk:Risk)
                    WHERE risk.category = $category OR risk.affects_category = $category
                    RETURN count(risk) as risk_count
                    """
                    risk_result = session.run(risk_query, category=category)
                    risk_record = risk_result.single()
                    source_results['neo4j_risks_count'] = risk_record['risk_count'] if risk_record else 0
                    
                    # Check for recommendations
                    rec_query = """
                    MATCH (s:Site)-[r:HAS_RECOMMENDATION]->(rec:Recommendation)
                    WHERE rec.category = $category OR rec.applies_to_category = $category
                    RETURN count(rec) as rec_count
                    """
                    rec_result = session.run(rec_query, category=category)
                    rec_record = rec_result.single()
                    source_results['neo4j_recommendations_count'] = rec_record['rec_count'] if rec_record else 0
                    
            except Exception as e:
                source_results['error'] = str(e)
                logger.error(f"Error checking Neo4j data for {category}: {e}")
            
            # Check dashboard response for generated data indicators
            try:
                response = requests.get(f"{self.dashboard_api}/api/dashboard/{category}", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    
                    # Look for indicators of generated data
                    generated_indicators = []
                    data_str = json.dumps(data).lower()
                    
                    if 'mock' in data_str or 'generated' in data_str or 'sample' in data_str:
                        generated_indicators.append("Contains mock/generated/sample keywords")
                    
                    # Check for suspiciously round numbers or patterns
                    if 'facts' in data and isinstance(data['facts'], list):
                        for fact in data['facts']:
                            if isinstance(fact, dict) and 'value' in fact:
                                try:
                                    value = float(fact['value'])
                                    if value == int(value) and value % 100 == 0:
                                        generated_indicators.append("Contains suspiciously round numbers")
                                        break
                                except:
                                    pass
                    
                    source_results['generated_data_indicators'] = generated_indicators
                    
            except Exception as e:
                logger.error(f"Error checking for generated data indicators in {category}: {e}")
            
            self.verification_results['data_sources'][category] = source_results
    
    def verify_historical_data(self):
        """Verify whether 6 months of historical data exists in Neo4j for each site"""
        logger.info("Verifying historical data requirements...")
        
        if not self.driver:
            logger.error("No Neo4j connection available")
            return
        
        six_months_ago = datetime.now() - timedelta(days=180)
        categories = ['electricity', 'water', 'waste']
        
        historical_results = {
            'sites_checked': [],
            'categories_coverage': {},
            'compliance_summary': {}
        }
        
        try:
            with self.driver.session() as session:
                # Get all sites
                sites_result = session.run("MATCH (s:Site) RETURN s.site_id as site_id, s.name as name")
                sites = [{'site_id': record['site_id'], 'name': record['name']} for record in sites_result]
                historical_results['sites_checked'] = sites
                
                for category in categories:
                    category_coverage = {}
                    
                    for site in sites:
                        site_coverage = {
                            'has_6_months_data': False,
                            'earliest_timestamp': None,
                            'latest_timestamp': None,
                            'record_count': 0,
                            'data_points_per_month': {}
                        }
                        
                        # Check consumption data for this site and category
                        historical_query = """
                        MATCH (s:Site {site_id: $site_id})-[r:HAS_CONSUMPTION]->(c:Consumption {type: $category})
                        WHERE c.timestamp >= $six_months_ago
                        RETURN count(c) as count, 
                               min(c.timestamp) as earliest, 
                               max(c.timestamp) as latest,
                               collect(c.timestamp) as timestamps
                        """
                        
                        result = session.run(
                            historical_query, 
                            site_id=site['site_id'], 
                            category=category,
                            six_months_ago=six_months_ago.isoformat()
                        )
                        record = result.single()
                        
                        if record and record['count'] > 0:
                            site_coverage['record_count'] = record['count']
                            site_coverage['earliest_timestamp'] = record['earliest']
                            site_coverage['latest_timestamp'] = record['latest']
                            
                            # Check if we have at least 6 months of data
                            if record['earliest']:
                                earliest = datetime.fromisoformat(record['earliest'].replace('Z', '+00:00'))
                                if earliest <= six_months_ago:
                                    site_coverage['has_6_months_data'] = True
                            
                            # Analyze data distribution by month
                            if record['timestamps']:
                                monthly_counts = {}
                                for timestamp in record['timestamps']:
                                    if timestamp:
                                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                        month_key = dt.strftime('%Y-%m')
                                        monthly_counts[month_key] = monthly_counts.get(month_key, 0) + 1
                                site_coverage['data_points_per_month'] = monthly_counts
                        
                        category_coverage[site['site_id']] = site_coverage
                    
                    historical_results['categories_coverage'][category] = category_coverage
                
                # Calculate compliance summary
                for category in categories:
                    compliant_sites = 0
                    total_sites = len(sites)
                    
                    for site in sites:
                        site_data = historical_results['categories_coverage'][category].get(site['site_id'], {})
                        if site_data.get('has_6_months_data', False):
                            compliant_sites += 1
                    
                    historical_results['compliance_summary'][category] = {
                        'compliant_sites': compliant_sites,
                        'total_sites': total_sites,
                        'compliance_percentage': (compliant_sites / total_sites * 100) if total_sites > 0 else 0
                    }
                    
        except Exception as e:
            historical_results['error'] = str(e)
            logger.error(f"Error verifying historical data: {e}")
        
        self.verification_results['historical_data'] = historical_results
    
    def verify_risk_assessment_agent(self):
        """Verify whether the Risk Assessment Agent stores results in Neo4j"""
        logger.info("Verifying Risk Assessment Agent integration...")
        
        risk_assessment_results = {
            'agent_accessible': False,
            'stores_in_neo4j': False,
            'recent_assessments': [],
            'neo4j_risk_records': 0,
            'agent_response_structure': None
        }
        
        # Test Risk Assessment Agent accessibility
        try:
            response = requests.get(f"{self.risk_agent_api}/health", timeout=10)
            if response.status_code == 200:
                risk_assessment_results['agent_accessible'] = True
                logger.info("Risk Assessment Agent is accessible")
            else:
                logger.warning(f"Risk Assessment Agent returned status {response.status_code}")
        except Exception as e:
            logger.error(f"Risk Assessment Agent not accessible: {e}")
            risk_assessment_results['agent_error'] = str(e)
        
        # Check Neo4j for risk assessment data
        if self.driver:
            try:
                with self.driver.session() as session:
                    # Count risk records
                    risk_count_query = "MATCH (r:Risk) RETURN count(r) as count"
                    result = session.run(risk_count_query)
                    record = result.single()
                    risk_assessment_results['neo4j_risk_records'] = record['count'] if record else 0
                    
                    # Get recent risk assessments
                    recent_risks_query = """
                    MATCH (s:Site)-[rel:HAS_RISK]->(r:Risk)
                    RETURN s.site_id as site_id, r.risk_type as risk_type, 
                           r.severity as severity, r.created_at as created_at,
                           r.assessment_id as assessment_id
                    ORDER BY r.created_at DESC
                    LIMIT 10
                    """
                    risks_result = session.run(recent_risks_query)
                    recent_risks = []
                    for record in risks_result:
                        recent_risks.append({
                            'site_id': record['site_id'],
                            'risk_type': record['risk_type'],
                            'severity': record['severity'],
                            'created_at': record['created_at'],
                            'assessment_id': record['assessment_id']
                        })
                    risk_assessment_results['recent_assessments'] = recent_risks
                    
                    if risk_assessment_results['neo4j_risk_records'] > 0:
                        risk_assessment_results['stores_in_neo4j'] = True
                        
            except Exception as e:
                logger.error(f"Error checking Neo4j for risk data: {e}")
                risk_assessment_results['neo4j_error'] = str(e)
        
        # Try to trigger a risk assessment to test integration
        if risk_assessment_results['agent_accessible']:
            try:
                test_payload = {
                    'site_id': 'test-site',
                    'categories': ['electricity', 'water', 'waste']
                }
                response = requests.post(
                    f"{self.risk_agent_api}/assess-risks",
                    json=test_payload,
                    timeout=30
                )
                risk_assessment_results['agent_response_structure'] = {
                    'status_code': response.status_code,
                    'response_data': response.json() if response.status_code == 200 else response.text
                }
            except Exception as e:
                logger.error(f"Error testing risk assessment trigger: {e}")
                risk_assessment_results['assessment_test_error'] = str(e)
        
        self.verification_results['risk_assessment'] = risk_assessment_results
    
    def verify_neo4j_integration(self):
        """Verify whether risks and recommendations are associated with sites in Neo4j"""
        logger.info("Verifying Neo4j integration for risks and recommendations...")
        
        if not self.driver:
            logger.error("No Neo4j connection available")
            return
        
        integration_results = {
            'site_risk_associations': {},
            'site_recommendation_associations': {},
            'categories_coverage': {},
            'relationship_integrity': {}
        }
        
        try:
            with self.driver.session() as session:
                # Check site-risk associations
                site_risk_query = """
                MATCH (s:Site)-[r:HAS_RISK]->(risk:Risk)
                RETURN s.site_id as site_id, 
                       collect({
                           risk_type: risk.risk_type,
                           category: risk.category,
                           severity: risk.severity,
                           created_at: risk.created_at
                       }) as risks
                """
                site_risks = session.run(site_risk_query)
                for record in site_risks:
                    integration_results['site_risk_associations'][record['site_id']] = record['risks']
                
                # Check site-recommendation associations  
                site_rec_query = """
                MATCH (s:Site)-[r:HAS_RECOMMENDATION]->(rec:Recommendation)
                RETURN s.site_id as site_id,
                       collect({
                           recommendation_type: rec.recommendation_type,
                           category: rec.category,
                           priority: rec.priority,
                           created_at: rec.created_at
                       }) as recommendations
                """
                site_recs = session.run(site_rec_query)
                for record in site_recs:
                    integration_results['site_recommendation_associations'][record['site_id']] = record['recommendations']
                
                # Check coverage by category
                categories = ['electricity', 'water', 'waste']
                for category in categories:
                    category_coverage = {}
                    
                    # Risks per category
                    risk_coverage_query = """
                    MATCH (s:Site)-[r:HAS_RISK]->(risk:Risk)
                    WHERE risk.category = $category OR risk.affects_category = $category
                    RETURN s.site_id as site_id, count(risk) as risk_count
                    """
                    risk_coverage = session.run(risk_coverage_query, category=category)
                    category_coverage['risks'] = {record['site_id']: record['risk_count'] for record in risk_coverage}
                    
                    # Recommendations per category
                    rec_coverage_query = """
                    MATCH (s:Site)-[r:HAS_RECOMMENDATION]->(rec:Recommendation)  
                    WHERE rec.category = $category OR rec.applies_to_category = $category
                    RETURN s.site_id as site_id, count(rec) as rec_count
                    """
                    rec_coverage = session.run(rec_coverage_query, category=category)
                    category_coverage['recommendations'] = {record['site_id']: record['rec_count'] for record in rec_coverage}
                    
                    integration_results['categories_coverage'][category] = category_coverage
                
                # Check relationship integrity
                integrity_checks = {}
                
                # Check for orphaned risks (not associated with sites)
                orphaned_risks_query = """
                MATCH (r:Risk)
                WHERE NOT (r)<-[:HAS_RISK]-(:Site)
                RETURN count(r) as orphaned_count
                """
                result = session.run(orphaned_risks_query)
                record = result.single()
                integrity_checks['orphaned_risks'] = record['orphaned_count'] if record else 0
                
                # Check for orphaned recommendations
                orphaned_recs_query = """
                MATCH (r:Recommendation)
                WHERE NOT (r)<-[:HAS_RECOMMENDATION]-(:Site)
                RETURN count(r) as orphaned_count
                """
                result = session.run(orphaned_recs_query)
                record = result.single()
                integrity_checks['orphaned_recommendations'] = record['orphaned_count'] if record else 0
                
                # Check for sites without risks or recommendations
                sites_without_risks_query = """
                MATCH (s:Site)
                WHERE NOT (s)-[:HAS_RISK]->(:Risk)
                RETURN collect(s.site_id) as sites_without_risks
                """
                result = session.run(sites_without_risks_query)
                record = result.single()
                integrity_checks['sites_without_risks'] = record['sites_without_risks'] if record else []
                
                sites_without_recs_query = """
                MATCH (s:Site)
                WHERE NOT (s)-[:HAS_RECOMMENDATION]->(:Recommendation)
                RETURN collect(s.site_id) as sites_without_recs
                """
                result = session.run(sites_without_recs_query)
                record = result.single()
                integrity_checks['sites_without_recommendations'] = record['sites_without_recs'] if record else []
                
                integration_results['relationship_integrity'] = integrity_checks
                
        except Exception as e:
            integration_results['error'] = str(e)
            logger.error(f"Error verifying Neo4j integration: {e}")
        
        self.verification_results['neo4j_integration'] = integration_results
    
    def calculate_overall_compliance(self):
        """Calculate overall compliance with requirements"""
        logger.info("Calculating overall compliance...")
        
        compliance_checks = []
        issues = []
        
        # Check 1: Dashboard displays all required components
        dashboard_compliant = True
        categories = ['electricity', 'water', 'waste']
        expected_components = ['goals', 'facts', 'risks', 'recommendations']
        
        for category in categories:
            if category in self.verification_results['dashboard_display']:
                category_data = self.verification_results['dashboard_display'][category]
                if not category_data.get('accessible', False):
                    dashboard_compliant = False
                    issues.append(f"Dashboard {category} category not accessible")
                    continue
                
                for component in expected_components:
                    if category_data.get('components_found', {}).get(component, {}).get('present', False):
                        continue
                    else:
                        dashboard_compliant = False
                        issues.append(f"Dashboard {category} missing {component} component")
            else:
                dashboard_compliant = False
                issues.append(f"Dashboard {category} category not tested")
        
        compliance_checks.append({
            'requirement': 'Dashboard displays goals, facts, risks, recommendations for all categories',
            'compliant': dashboard_compliant,
            'weight': 25
        })
        
        # Check 2: Data comes from Neo4j (not generated)
        data_source_compliant = True
        for category in categories:
            if category in self.verification_results['data_sources']:
                source_data = self.verification_results['data_sources'][category]
                if not source_data.get('neo4j_data_exists', False):
                    data_source_compliant = False
                    issues.append(f"{category} data not found in Neo4j")
                if source_data.get('generated_data_indicators', []):
                    data_source_compliant = False
                    issues.append(f"{category} shows signs of generated data")
        
        compliance_checks.append({
            'requirement': 'Data comes from Neo4j, not generated',
            'compliant': data_source_compliant,
            'weight': 25
        })
        
        # Check 3: 6 months of historical data exists
        historical_compliant = True
        if 'compliance_summary' in self.verification_results['historical_data']:
            for category in categories:
                if category in self.verification_results['historical_data']['compliance_summary']:
                    compliance_pct = self.verification_results['historical_data']['compliance_summary'][category].get('compliance_percentage', 0)
                    if compliance_pct < 100:
                        historical_compliant = False
                        issues.append(f"{category} historical data incomplete ({compliance_pct:.1f}% of sites)")
        else:
            historical_compliant = False
            issues.append("Historical data verification failed")
        
        compliance_checks.append({
            'requirement': '6 months of historical data exists for all sites and categories',
            'compliant': historical_compliant,
            'weight': 25
        })
        
        # Check 4: Risk Assessment Agent stores results in Neo4j
        risk_agent_compliant = (
            self.verification_results['risk_assessment'].get('agent_accessible', False) and
            self.verification_results['risk_assessment'].get('stores_in_neo4j', False)
        )
        if not risk_agent_compliant:
            if not self.verification_results['risk_assessment'].get('agent_accessible', False):
                issues.append("Risk Assessment Agent not accessible")
            if not self.verification_results['risk_assessment'].get('stores_in_neo4j', False):
                issues.append("Risk Assessment Agent does not store results in Neo4j")
        
        compliance_checks.append({
            'requirement': 'Risk Assessment Agent stores results in Neo4j',
            'compliant': risk_agent_compliant,
            'weight': 25
        })
        
        # Calculate weighted compliance score
        total_weight = sum(check['weight'] for check in compliance_checks)
        weighted_score = sum(check['weight'] for check in compliance_checks if check['compliant'])
        overall_compliance_pct = (weighted_score / total_weight * 100) if total_weight > 0 else 0
        
        self.verification_results['overall_compliance'] = {
            'compliant': overall_compliance_pct == 100,
            'compliance_percentage': overall_compliance_pct,
            'checks': compliance_checks,
            'issues': issues,
            'summary': f"{len([c for c in compliance_checks if c['compliant']])}/{len(compliance_checks)} requirements met"
        }
        
        logger.info(f"Overall compliance: {overall_compliance_pct:.1f}% ({self.verification_results['overall_compliance']['summary']})")
    
    def generate_report(self):
        """Generate comprehensive verification report"""
        report_path = '/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/tmp/verification_report.json'
        
        try:
            with open(report_path, 'w') as f:
                json.dump(self.verification_results, f, indent=2, default=str)
            logger.info(f"Verification report saved to {report_path}")
            
            # Generate summary report
            summary_path = '/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/tmp/verification_summary.txt'
            with open(summary_path, 'w') as f:
                f.write("EHS DASHBOARD REQUIREMENTS VERIFICATION SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Verification completed: {self.verification_results['timestamp']}\n\n")
                
                if 'overall_compliance' in self.verification_results:
                    compliance = self.verification_results['overall_compliance']
                    f.write(f"OVERALL COMPLIANCE: {compliance['compliance_percentage']:.1f}%\n")
                    f.write(f"Status: {'COMPLIANT' if compliance['compliant'] else 'NON-COMPLIANT'}\n")
                    f.write(f"Summary: {compliance['summary']}\n\n")
                    
                    f.write("REQUIREMENT BREAKDOWN:\n")
                    for i, check in enumerate(compliance['checks'], 1):
                        status = "✓ PASS" if check['compliant'] else "✗ FAIL"
                        f.write(f"{i}. [{status}] {check['requirement']}\n")
                    
                    if compliance['issues']:
                        f.write("\nISSUES IDENTIFIED:\n")
                        for i, issue in enumerate(compliance['issues'], 1):
                            f.write(f"{i}. {issue}\n")
                
                f.write(f"\nDetailed results saved to: {report_path}\n")
            
            logger.info(f"Summary report saved to {summary_path}")
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
    
    def run_verification(self):
        """Run complete verification process"""
        logger.info("Starting comprehensive dashboard requirements verification...")
        
        # Connect to Neo4j
        if not self.connect_to_neo4j():
            logger.warning("Continuing verification without Neo4j connection")
        
        try:
            # Run all verification steps
            self.verify_dashboard_display()
            self.verify_data_sources()
            self.verify_historical_data()
            self.verify_risk_assessment_agent()
            self.verify_neo4j_integration()
            
            # Calculate overall compliance
            self.calculate_overall_compliance()
            
            # Generate reports
            self.generate_report()
            
            logger.info("Verification completed successfully")
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
        
        finally:
            if self.driver:
                self.driver.close()
                logger.info("Neo4j connection closed")

def main():
    """Main function"""
    print("EHS Dashboard Requirements Verification")
    print("=" * 40)
    print()
    
    verifier = DashboardVerifier()
    verifier.run_verification()
    
    print()
    print("Verification completed. Check the following files for results:")
    print("- Detailed report: /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/tmp/verification_report.json")
    print("- Summary report: /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/tmp/verification_summary.txt")
    print("- Logs: /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/tmp/verification.log")

if __name__ == "__main__":
    main()