#!/usr/bin/env python3
"""
Location Hierarchy Migration Validation Script

This script validates the results of the location hierarchy migration,
checking for data integrity, structure consistency, and generating 
comprehensive validation reports.

Features:
- Validates hierarchy structure completeness
- Checks constraint and index integrity  
- Validates facility mappings
- Reports data inconsistencies
- Generates detailed validation reports
- Performance metrics analysis

Author: AI Assistant
Date: 2025-08-30
"""

import sys
import os
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

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

@dataclass
class ValidationResult:
    """Data class for validation results"""
    check_name: str
    status: str  # 'PASS', 'FAIL', 'WARNING'
    message: str
    details: Dict[str, Any] = None
    count: int = 0

@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    validation_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    overall_status: str = 'IN_PROGRESS'
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    warning_checks: int = 0
    results: List[ValidationResult] = None
    
    def __post_init__(self):
        if self.results is None:
            self.results = []

class LocationMigrationValidator:
    """Main class for validating location hierarchy migration"""
    
    def __init__(self, neo4j_uri: str, neo4j_username: str, neo4j_password: str, neo4j_database: str):
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.neo4j_database = neo4j_database
        self.graph = None
        
        self.validation_id = f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger = self._setup_logging()
        
        self.report = ValidationReport(
            validation_id=self.validation_id,
            started_at=datetime.now()
        )
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_filename = f"validation_{self.validation_id}.log"
        log_path = os.path.join(os.path.dirname(__file__), log_filename)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup file handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Setup console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Create logger
        logger = logging.getLogger(f"MigrationValidator_{self.validation_id}")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def connect_to_neo4j(self) -> bool:
        """Establish connection to Neo4j database"""
        try:
            self.logger.info(f"Connecting to Neo4j at {self.neo4j_uri}")
            
            self.graph = Neo4jGraph(
                url=self.neo4j_uri,
                username=self.neo4j_username,
                password=self.neo4j_password,
                database=self.neo4j_database
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
    
    def add_validation_result(self, result: ValidationResult):
        """Add a validation result to the report"""
        self.report.results.append(result)
        self.report.total_checks += 1
        
        if result.status == 'PASS':
            self.report.passed_checks += 1
        elif result.status == 'FAIL':
            self.report.failed_checks += 1
        elif result.status == 'WARNING':
            self.report.warning_checks += 1
    
    def validate_hierarchy_structure(self):
        """Validate the completeness of hierarchy structure"""
        self.logger.info("Validating hierarchy structure...")
        
        # Check if all hierarchy levels exist
        hierarchy_levels = ['Site', 'Building', 'Floor', 'Area']
        
        for level in hierarchy_levels:
            try:
                count_query = f"MATCH (n:{level}) RETURN COUNT(n) as count"
                result = self.graph.query(count_query)
                count = result[0]['count'] if result else 0
                
                if count > 0:
                    self.add_validation_result(ValidationResult(
                        check_name=f"{level}_nodes_exist",
                        status='PASS',
                        message=f"{level} nodes found in database",
                        count=count
                    ))
                    self.logger.info(f"✓ Found {count} {level} nodes")
                else:
                    self.add_validation_result(ValidationResult(
                        check_name=f"{level}_nodes_exist",
                        status='FAIL',
                        message=f"No {level} nodes found in database",
                        count=0
                    ))
                    self.logger.error(f"✗ No {level} nodes found")
                    
            except Exception as e:
                self.add_validation_result(ValidationResult(
                    check_name=f"{level}_nodes_exist",
                    status='FAIL',
                    message=f"Error checking {level} nodes: {str(e)}",
                    count=0
                ))
        
        # Check hierarchy relationships
        hierarchy_path_query = """
        MATCH path = (s:Site)-[:CONTAINS]->(b:Building)-[:CONTAINS]->(f:Floor)-[:CONTAINS]->(a:Area)
        RETURN COUNT(path) as complete_paths,
               COUNT(DISTINCT s) as sites_with_complete_paths,
               COUNT(DISTINCT b) as buildings_with_complete_paths,
               COUNT(DISTINCT f) as floors_with_complete_paths,
               COUNT(DISTINCT a) as areas_in_complete_paths
        """
        
        try:
            result = self.graph.query(hierarchy_path_query)
            if result:
                data = result[0]
                complete_paths = data['complete_paths']
                
                if complete_paths > 0:
                    self.add_validation_result(ValidationResult(
                        check_name="complete_hierarchy_paths",
                        status='PASS',
                        message=f"Found {complete_paths} complete hierarchy paths",
                        details=data,
                        count=complete_paths
                    ))
                    self.logger.info(f"✓ Found {complete_paths} complete Site->Building->Floor->Area paths")
                else:
                    self.add_validation_result(ValidationResult(
                        check_name="complete_hierarchy_paths",
                        status='FAIL',
                        message="No complete hierarchy paths found",
                        details=data,
                        count=0
                    ))
                    self.logger.error("✗ No complete Site->Building->Floor->Area paths found")
                    
        except Exception as e:
            self.add_validation_result(ValidationResult(
                check_name="complete_hierarchy_paths",
                status='FAIL',
                message=f"Error checking hierarchy paths: {str(e)}"
            ))
    
    def validate_constraints_and_indexes(self):
        """Validate that required constraints and indexes exist"""
        self.logger.info("Validating constraints and indexes...")
        
        expected_constraints = [
            'site_name_unique',
            'site_code_unique',
            'building_site_name_unique',
            'floor_building_name_unique',
            'area_floor_name_unique'
        ]
        
        expected_indexes = [
            'site_code_index',
            'building_site_code_index', 
            'floor_building_id_index',
            'area_floor_id_index'
        ]
        
        # Check constraints
        try:
            constraints_query = "SHOW CONSTRAINTS YIELD name, type, entityType, properties"
            constraint_results = self.graph.query(constraints_query)
            
            existing_constraints = [c['name'] for c in constraint_results if c['name']]
            
            for expected in expected_constraints:
                if expected in existing_constraints:
                    self.add_validation_result(ValidationResult(
                        check_name=f"constraint_{expected}",
                        status='PASS',
                        message=f"Constraint {expected} exists"
                    ))
                    self.logger.info(f"✓ Constraint {expected} exists")
                else:
                    self.add_validation_result(ValidationResult(
                        check_name=f"constraint_{expected}",
                        status='FAIL',
                        message=f"Constraint {expected} missing"
                    ))
                    self.logger.error(f"✗ Constraint {expected} missing")
                    
        except Exception as e:
            self.add_validation_result(ValidationResult(
                check_name="constraints_check",
                status='FAIL',
                message=f"Error checking constraints: {str(e)}"
            ))
        
        # Check indexes
        try:
            indexes_query = "SHOW INDEXES YIELD name, type, entityType, properties"
            index_results = self.graph.query(indexes_query)
            
            existing_indexes = [i['name'] for i in index_results if i['name']]
            
            for expected in expected_indexes:
                if expected in existing_indexes:
                    self.add_validation_result(ValidationResult(
                        check_name=f"index_{expected}",
                        status='PASS',
                        message=f"Index {expected} exists"
                    ))
                    self.logger.info(f"✓ Index {expected} exists")
                else:
                    self.add_validation_result(ValidationResult(
                        check_name=f"index_{expected}",
                        status='WARNING',
                        message=f"Index {expected} missing (performance may be affected)"
                    ))
                    self.logger.warning(f"⚠ Index {expected} missing")
                    
        except Exception as e:
            self.add_validation_result(ValidationResult(
                check_name="indexes_check",
                status='FAIL',
                message=f"Error checking indexes: {str(e)}"
            ))
    
    def validate_facility_mappings(self):
        """Validate facility mappings to location hierarchy"""
        self.logger.info("Validating facility mappings...")
        
        # Check facilities with location mappings
        mapped_facilities_query = """
        MATCH (f:Facility)-[:LOCATED_IN]->(a:Area)
        MATCH path = (s:Site)-[:CONTAINS]->(b:Building)-[:CONTAINS]->(fl:Floor)-[:CONTAINS]->(a)
        RETURN f.name as facility_name, 
               s.name as site_name, s.code as site_code,
               b.name as building_name,
               fl.name as floor_name,
               a.name as area_name
        """
        
        try:
            mapped_results = self.graph.query(mapped_facilities_query)
            mapped_count = len(mapped_results)
            
            if mapped_count > 0:
                self.add_validation_result(ValidationResult(
                    check_name="facilities_mapped_to_areas",
                    status='PASS',
                    message=f"{mapped_count} facilities properly mapped to areas",
                    count=mapped_count,
                    details={"mapped_facilities": [r['facility_name'] for r in mapped_results]}
                ))
                self.logger.info(f"✓ {mapped_count} facilities properly mapped to areas")
                
                for result in mapped_results:
                    self.logger.info(f"  {result['facility_name']} -> {result['site_name']}/{result['building_name']}/{result['floor_name']}/{result['area_name']}")
            else:
                self.add_validation_result(ValidationResult(
                    check_name="facilities_mapped_to_areas",
                    status='WARNING',
                    message="No facilities mapped to areas found",
                    count=0
                ))
                self.logger.warning("⚠ No facilities mapped to areas found")
                
        except Exception as e:
            self.add_validation_result(ValidationResult(
                check_name="facilities_mapped_to_areas",
                status='FAIL',
                message=f"Error checking facility mappings: {str(e)}"
            ))
        
        # Check unmapped facilities
        unmapped_facilities_query = """
        MATCH (f:Facility)
        WHERE NOT EXISTS((f)-[:LOCATED_IN]->(:Area))
        RETURN f.name as facility_name, properties(f) as props
        """
        
        try:
            unmapped_results = self.graph.query(unmapped_facilities_query)
            unmapped_count = len(unmapped_results)
            
            if unmapped_count == 0:
                self.add_validation_result(ValidationResult(
                    check_name="no_unmapped_facilities",
                    status='PASS',
                    message="All facilities are properly mapped",
                    count=0
                ))
                self.logger.info("✓ All facilities are properly mapped")
            else:
                self.add_validation_result(ValidationResult(
                    check_name="no_unmapped_facilities", 
                    status='WARNING',
                    message=f"{unmapped_count} facilities are not mapped to locations",
                    count=unmapped_count,
                    details={"unmapped_facilities": [r['facility_name'] for r in unmapped_results]}
                ))
                self.logger.warning(f"⚠ {unmapped_count} facilities are not mapped to locations:")
                
                for result in unmapped_results:
                    self.logger.warning(f"  - {result['facility_name']}")
                    
        except Exception as e:
            self.add_validation_result(ValidationResult(
                check_name="no_unmapped_facilities",
                status='FAIL',
                message=f"Error checking unmapped facilities: {str(e)}"
            ))
    
    def validate_data_consistency(self):
        """Validate data consistency and integrity"""
        self.logger.info("Validating data consistency...")
        
        # Check for orphaned buildings (buildings without sites)
        orphaned_buildings_query = """
        MATCH (b:Building)
        WHERE NOT EXISTS((s:Site)-[:CONTAINS]->(b))
        RETURN COUNT(b) as orphaned_buildings, COLLECT(b.name) as building_names
        """
        
        try:
            result = self.graph.query(orphaned_buildings_query)
            if result:
                orphaned_count = result[0]['orphaned_buildings']
                building_names = result[0]['building_names']
                
                if orphaned_count == 0:
                    self.add_validation_result(ValidationResult(
                        check_name="no_orphaned_buildings",
                        status='PASS',
                        message="No orphaned buildings found",
                        count=0
                    ))
                    self.logger.info("✓ No orphaned buildings found")
                else:
                    self.add_validation_result(ValidationResult(
                        check_name="no_orphaned_buildings",
                        status='FAIL',
                        message=f"{orphaned_count} orphaned buildings found",
                        count=orphaned_count,
                        details={"orphaned_buildings": building_names}
                    ))
                    self.logger.error(f"✗ {orphaned_count} orphaned buildings found: {building_names}")
                    
        except Exception as e:
            self.add_validation_result(ValidationResult(
                check_name="no_orphaned_buildings",
                status='FAIL',
                message=f"Error checking orphaned buildings: {str(e)}"
            ))
        
        # Check for orphaned floors
        orphaned_floors_query = """
        MATCH (f:Floor)
        WHERE NOT EXISTS((b:Building)-[:CONTAINS]->(f))
        RETURN COUNT(f) as orphaned_floors, COLLECT(f.name) as floor_names
        """
        
        try:
            result = self.graph.query(orphaned_floors_query)
            if result:
                orphaned_count = result[0]['orphaned_floors']
                floor_names = result[0]['floor_names']
                
                if orphaned_count == 0:
                    self.add_validation_result(ValidationResult(
                        check_name="no_orphaned_floors",
                        status='PASS',
                        message="No orphaned floors found",
                        count=0
                    ))
                    self.logger.info("✓ No orphaned floors found")
                else:
                    self.add_validation_result(ValidationResult(
                        check_name="no_orphaned_floors",
                        status='FAIL',
                        message=f"{orphaned_count} orphaned floors found",
                        count=orphaned_count,
                        details={"orphaned_floors": floor_names}
                    ))
                    self.logger.error(f"✗ {orphaned_count} orphaned floors found: {floor_names}")
                    
        except Exception as e:
            self.add_validation_result(ValidationResult(
                check_name="no_orphaned_floors",
                status='FAIL',
                message=f"Error checking orphaned floors: {str(e)}"
            ))
        
        # Check for orphaned areas
        orphaned_areas_query = """
        MATCH (a:Area)
        WHERE NOT EXISTS((f:Floor)-[:CONTAINS]->(a))
        RETURN COUNT(a) as orphaned_areas, COLLECT(a.name) as area_names
        """
        
        try:
            result = self.graph.query(orphaned_areas_query)
            if result:
                orphaned_count = result[0]['orphaned_areas']
                area_names = result[0]['area_names']
                
                if orphaned_count == 0:
                    self.add_validation_result(ValidationResult(
                        check_name="no_orphaned_areas",
                        status='PASS',
                        message="No orphaned areas found",
                        count=0
                    ))
                    self.logger.info("✓ No orphaned areas found")
                else:
                    self.add_validation_result(ValidationResult(
                        check_name="no_orphaned_areas",
                        status='FAIL',
                        message=f"{orphaned_count} orphaned areas found",
                        count=orphaned_count,
                        details={"orphaned_areas": area_names}
                    ))
                    self.logger.error(f"✗ {orphaned_count} orphaned areas found: {area_names}")
                    
        except Exception as e:
            self.add_validation_result(ValidationResult(
                check_name="no_orphaned_areas",
                status='FAIL',
                message=f"Error checking orphaned areas: {str(e)}"
            ))
    
    def validate_performance_metrics(self):
        """Validate performance aspects of the hierarchy"""
        self.logger.info("Validating performance metrics...")
        
        # Check index usage statistics (if available)
        try:
            # Check query performance for common operations
            performance_queries = [
                ("site_lookup_by_code", "MATCH (s:Site {code: 'ALG001'}) RETURN s"),
                ("building_lookup_by_site", "MATCH (s:Site {code: 'ALG001'})-[:CONTAINS]->(b:Building) RETURN b"),
                ("full_hierarchy_traversal", "MATCH (s:Site)-[:CONTAINS]->(b:Building)-[:CONTAINS]->(f:Floor)-[:CONTAINS]->(a:Area) RETURN s, b, f, a LIMIT 10")
            ]
            
            for query_name, query in performance_queries:
                try:
                    # Run query and measure execution (basic timing)
                    start_time = datetime.now()
                    result = self.graph.query(query)
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    
                    if duration < 1.0:  # Less than 1 second is good
                        self.add_validation_result(ValidationResult(
                            check_name=f"performance_{query_name}",
                            status='PASS',
                            message=f"Query '{query_name}' executed in {duration:.3f}s",
                            details={"execution_time_seconds": duration, "result_count": len(result)}
                        ))
                        self.logger.info(f"✓ Query '{query_name}' executed in {duration:.3f}s")
                    else:
                        self.add_validation_result(ValidationResult(
                            check_name=f"performance_{query_name}",
                            status='WARNING',
                            message=f"Query '{query_name}' took {duration:.3f}s (may need optimization)",
                            details={"execution_time_seconds": duration, "result_count": len(result)}
                        ))
                        self.logger.warning(f"⚠ Query '{query_name}' took {duration:.3f}s")
                        
                except Exception as e:
                    self.add_validation_result(ValidationResult(
                        check_name=f"performance_{query_name}",
                        status='FAIL',
                        message=f"Query '{query_name}' failed: {str(e)}"
                    ))
                    
        except Exception as e:
            self.add_validation_result(ValidationResult(
                check_name="performance_validation",
                status='FAIL',
                message=f"Error during performance validation: {str(e)}"
            ))
    
    def generate_validation_report(self) -> str:
        """Generate a comprehensive validation report"""
        self.report.completed_at = datetime.now()
        
        # Determine overall status
        if self.report.failed_checks > 0:
            self.report.overall_status = 'FAILED'
        elif self.report.warning_checks > 0:
            self.report.overall_status = 'PASSED_WITH_WARNINGS'
        else:
            self.report.overall_status = 'PASSED'
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("LOCATION HIERARCHY MIGRATION VALIDATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Validation ID: {self.report.validation_id}")
        report_lines.append(f"Started: {self.report.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Completed: {self.report.completed_at.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Duration: {(self.report.completed_at - self.report.started_at).total_seconds():.2f} seconds")
        report_lines.append(f"Overall Status: {self.report.overall_status}")
        report_lines.append("")
        
        # Summary
        report_lines.append("VALIDATION SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Checks: {self.report.total_checks}")
        report_lines.append(f"Passed: {self.report.passed_checks}")
        report_lines.append(f"Failed: {self.report.failed_checks}")
        report_lines.append(f"Warnings: {self.report.warning_checks}")
        
        success_rate = (self.report.passed_checks / self.report.total_checks * 100) if self.report.total_checks > 0 else 0
        report_lines.append(f"Success Rate: {success_rate:.1f}%")
        report_lines.append("")
        
        # Detailed results by category
        categories = {
            'Structure': ['hierarchy_structure', 'complete_hierarchy_paths'],
            'Constraints': [name for name in [r.check_name for r in self.report.results] if 'constraint_' in name],
            'Indexes': [name for name in [r.check_name for r in self.report.results] if 'index_' in name],
            'Mappings': ['facilities_mapped_to_areas', 'no_unmapped_facilities'],
            'Consistency': ['no_orphaned_buildings', 'no_orphaned_floors', 'no_orphaned_areas'],
            'Performance': [name for name in [r.check_name for r in self.report.results] if 'performance_' in name]
        }
        
        for category, check_names in categories.items():
            category_results = [r for r in self.report.results if r.check_name in check_names or any(check in r.check_name for check in check_names)]
            
            if category_results:
                report_lines.append(f"{category.upper()} VALIDATION RESULTS")
                report_lines.append("-" * 40)
                
                for result in category_results:
                    status_symbol = "✓" if result.status == "PASS" else ("⚠" if result.status == "WARNING" else "✗")
                    report_lines.append(f"{status_symbol} {result.check_name}: {result.message}")
                    
                    if result.count > 0:
                        report_lines.append(f"   Count: {result.count}")
                    
                    if result.details:
                        for key, value in result.details.items():
                            if isinstance(value, (list, dict)):
                                report_lines.append(f"   {key}: {json.dumps(value, indent=2) if len(str(value)) > 100 else str(value)}")
                            else:
                                report_lines.append(f"   {key}: {value}")
                
                report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("-" * 40)
        
        failed_results = [r for r in self.report.results if r.status == 'FAIL']
        warning_results = [r for r in self.report.results if r.status == 'WARNING']
        
        if not failed_results and not warning_results:
            report_lines.append("✓ Migration validation passed successfully. No issues found.")
        else:
            if failed_results:
                report_lines.append("CRITICAL ISSUES (must be addressed):")
                for result in failed_results:
                    report_lines.append(f"  - {result.message}")
                report_lines.append("")
            
            if warning_results:
                report_lines.append("WARNINGS (should be reviewed):")
                for result in warning_results:
                    report_lines.append(f"  - {result.message}")
                report_lines.append("")
        
        report_lines.append("=" * 80)
        report_lines.append("END OF VALIDATION REPORT")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def run_validation(self) -> bool:
        """Run the complete validation process"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("STARTING LOCATION HIERARCHY MIGRATION VALIDATION")
            self.logger.info("=" * 80)
            self.logger.info(f"Validation ID: {self.validation_id}")
            
            # Connect to Neo4j
            if not self.connect_to_neo4j():
                return False
            
            # Run validation checks
            validation_steps = [
                ("Hierarchy Structure", self.validate_hierarchy_structure),
                ("Constraints and Indexes", self.validate_constraints_and_indexes), 
                ("Facility Mappings", self.validate_facility_mappings),
                ("Data Consistency", self.validate_data_consistency),
                ("Performance Metrics", self.validate_performance_metrics)
            ]
            
            for step_name, step_function in validation_steps:
                self.logger.info(f"\n--- {step_name} ---")
                try:
                    step_function()
                    self.logger.info(f"✓ {step_name} validation completed")
                except Exception as e:
                    self.logger.error(f"✗ {step_name} validation failed: {e}")
                    self.add_validation_result(ValidationResult(
                        check_name=f"{step_name.lower().replace(' ', '_')}_validation",
                        status='FAIL',
                        message=f"{step_name} validation failed: {str(e)}"
                    ))
            
            # Generate and save report
            report_text = self.generate_validation_report()
            
            report_filename = f"validation_report_{self.validation_id}.txt"
            report_path = os.path.join(os.path.dirname(__file__), report_filename)
            
            with open(report_path, 'w') as f:
                f.write(report_text)
            
            # Save JSON report for programmatic access
            json_report_filename = f"validation_report_{self.validation_id}.json"
            json_report_path = os.path.join(os.path.dirname(__file__), json_report_filename)
            
            with open(json_report_path, 'w') as f:
                json.dump(asdict(self.report), f, indent=2, default=str)
            
            # Display report
            print(report_text)
            
            self.logger.info(f"\nReports saved:")
            self.logger.info(f"  Text: {report_path}")
            self.logger.info(f"  JSON: {json_report_path}")
            
            return self.report.overall_status in ['PASSED', 'PASSED_WITH_WARNINGS']
            
        except Exception as e:
            self.logger.error(f"Validation process failed: {e}")
            return False

def load_config():
    """Load configuration from environment"""
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(env_path)
    
    return {
        'neo4j_uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        'neo4j_username': os.getenv('NEO4J_USERNAME', 'neo4j'),
        'neo4j_password': os.getenv('NEO4J_PASSWORD', 'EhsAI2024!'),
        'neo4j_database': os.getenv('NEO4J_DATABASE', 'neo4j')
    }

def main():
    """Main execution function"""
    try:
        # Load configuration
        config = load_config()
        
        # Create validator
        validator = LocationMigrationValidator(**config)
        
        # Run validation
        success = validator.run_validation()
        
        if success:
            print("\n✓ Migration validation completed successfully!")
            sys.exit(0)
        else:
            print("\n✗ Migration validation found issues. Check the reports for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()