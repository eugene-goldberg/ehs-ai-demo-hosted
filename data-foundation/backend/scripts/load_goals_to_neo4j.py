#!/usr/bin/env python3
"""
Load EHS Goals into Neo4j

This script loads EHS (Environmental Health & Safety) goals from the EHSGoalsConfig
into Neo4j as Goal nodes with relationships to Site nodes. The goals are loaded
according to the simplified dashboard action plan specification.

Goal Node Structure:
- category: "electricity|water|waste" 
- target_value: float (reduction percentage)
- unit: string
- period: "yearly"
- target_date: date (2025-12-31)
- created_at: datetime

Relationships:
- (:Goal)-[:APPLIES_TO]->(:Site)

Category Mapping:
- CO2 goals → "electricity" category
- Water goals → "water" category
- Waste goals → "waste" category

Created: 2025-09-01
Version: 1.0.0
"""

import os
import sys
import logging
import json
from datetime import datetime, date
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

try:
    from dotenv import load_dotenv
    from neo4j import GraphDatabase, Transaction
    from neo4j.exceptions import ServiceUnavailable, ClientError
    from config.ehs_goals_config import EHSGoalsConfig, SiteLocation, EHSCategory
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure you're running this script from the backend directory with the virtual environment activated")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('load_goals_to_neo4j.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EHSGoalsLoader:
    """
    Loads EHS goals from configuration into Neo4j database
    
    Creates Goal nodes with proper category mapping and relationships to Site nodes.
    """
    
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """
        Initialize the goals loader
        
        Args:
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
            database: Database name (default: "neo4j")
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver = None
        self.goals_config = EHSGoalsConfig()
        
        # Category mapping from EHS categories to simplified dashboard categories
        self.category_mapping = {
            EHSCategory.CO2: "electricity",      # CO2 emissions from electricity consumption
            EHSCategory.WATER: "water",          # Water consumption
            EHSCategory.WASTE: "waste"           # Waste generation
        }
        
        # Site mapping from EHS site locations to Neo4j site IDs
        self.site_mapping = {
            SiteLocation.ALGONQUIN: "algonquin_il",
            SiteLocation.HOUSTON: "houston_tx"
        }
        
        # Target date for all goals (end of 2025)
        self.target_date = date(2025, 12, 31)
        
    def connect(self) -> None:
        """Establish connection to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            # Test connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            logger.info(f"Successfully connected to Neo4j at {self.uri}")
        except ServiceUnavailable:
            logger.error(f"Unable to connect to Neo4j at {self.uri}")
            raise
        except Exception as e:
            logger.error(f"Error connecting to Neo4j: {e}")
            raise

    def close(self) -> None:
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def verify_sites_exist(self) -> bool:
        """
        Verify that the required Site nodes exist in the database
        
        Returns:
            bool: True if both sites exist, False otherwise
        """
        logger.info("Verifying Site nodes exist...")
        
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (s:Site)
                WHERE s.id IN ['algonquin_il', 'houston_tx']
                RETURN s.id as site_id, s.name as site_name
                ORDER BY s.id
            """)
            
            sites = list(result)
            
        if len(sites) < 2:
            logger.error("Required Site nodes not found. Expected sites: algonquin_il, houston_tx")
            logger.error(f"Found sites: {[site['site_id'] for site in sites]}")
            return False
        
        logger.info(f"Found required sites:")
        for site in sites:
            logger.info(f"  - {site['site_id']}: {site['site_name']}")
        
        return True

    def create_goal_constraints(self) -> None:
        """Create constraints and indexes for Goal nodes"""
        logger.info("Creating Goal node constraints and indexes...")
        
        constraints = [
            "CREATE CONSTRAINT goal_id_unique IF NOT EXISTS FOR (g:Goal) REQUIRE g.id IS UNIQUE",
            "CREATE INDEX goal_category_index IF NOT EXISTS FOR (g:Goal) ON (g.category)",
            "CREATE INDEX goal_target_date_index IF NOT EXISTS FOR (g:Goal) ON (g.target_date)",
            "CREATE INDEX goal_period_index IF NOT EXISTS FOR (g:Goal) ON (g.period)"
        ]
        
        with self.driver.session(database=self.database) as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.info(f"Created: {constraint.split('FOR')[0].strip()}")
                except ClientError as e:
                    if "equivalent" not in str(e).lower() and "already exists" not in str(e).lower():
                        logger.warning(f"Failed to create constraint: {e}")

    def load_goals(self) -> Dict[str, Any]:
        """
        Load all EHS goals into Neo4j as Goal nodes
        
        Returns:
            Dict: Summary of loaded goals
        """
        logger.info("Loading EHS goals into Neo4j...")
        
        # Get all goals from configuration
        all_goals = self.goals_config.get_all_goals()
        
        if not all_goals:
            logger.error("No goals found in configuration")
            return {"error": "No goals found in configuration"}
        
        logger.info(f"Found {len(all_goals)} goals to load")
        
        loaded_goals = []
        
        with self.driver.session(database=self.database) as session:
            for goal in all_goals:
                try:
                    result = session.execute_write(self._create_goal_tx, goal)
                    loaded_goals.append(result)
                    logger.info(f"Loaded goal: {result['site_name']} - {result['category']} ({result['target_value']}% reduction)")
                except Exception as e:
                    logger.error(f"Failed to load goal for {goal.site.value} - {goal.category.value}: {e}")
        
        logger.info(f"Successfully loaded {len(loaded_goals)} goals")
        
        return {
            "total_goals_loaded": len(loaded_goals),
            "goals": loaded_goals,
            "timestamp": datetime.now().isoformat()
        }

    def _create_goal_tx(self, tx: Transaction, goal) -> Dict[str, Any]:
        """
        Transaction to create a Goal node and its relationship to Site
        
        Args:
            tx: Neo4j transaction
            goal: EHSGoal object from configuration
            
        Returns:
            Dict: Information about the created goal
        """
        # Map EHS categories to dashboard categories
        dashboard_category = self.category_mapping.get(goal.category)
        if not dashboard_category:
            raise ValueError(f"Unknown category mapping for {goal.category}")
        
        # Map site location to Neo4j site ID
        neo4j_site_id = self.site_mapping.get(goal.site)
        if not neo4j_site_id:
            raise ValueError(f"Unknown site mapping for {goal.site}")
        
        # Generate unique goal ID
        goal_id = f"goal_{neo4j_site_id}_{dashboard_category}_{goal.target_year}"
        
        # Create Goal node
        goal_query = """
        CREATE (g:Goal {
            id: $goal_id,
            category: $category,
            target_value: $target_value,
            unit: $unit,
            period: "yearly",
            target_date: date($target_date),
            created_at: datetime(),
            updated_at: datetime(),
            description: $description,
            baseline_year: $baseline_year,
            target_year: $target_year,
            site_id: $site_id
        })
        RETURN g
        """
        
        goal_params = {
            'goal_id': goal_id,
            'category': dashboard_category,
            'target_value': goal.reduction_percentage,
            'unit': goal.unit,
            'target_date': self.target_date.strftime('%Y-%m-%d'),
            'description': goal.description,
            'baseline_year': goal.baseline_year,
            'target_year': goal.target_year,
            'site_id': neo4j_site_id
        }
        
        result = tx.run(goal_query, goal_params)
        goal_node = result.single()['g']
        
        # Create relationship to Site
        relationship_query = """
        MATCH (s:Site {id: $site_id}), (g:Goal {id: $goal_id})
        MERGE (g)-[:APPLIES_TO]->(s)
        RETURN s.name as site_name
        """
        
        relationship_params = {
            'site_id': neo4j_site_id,
            'goal_id': goal_id
        }
        
        rel_result = tx.run(relationship_query, relationship_params)
        site_name = rel_result.single()['site_name']
        
        return {
            'goal_id': goal_id,
            'site_id': neo4j_site_id,
            'site_name': site_name,
            'category': dashboard_category,
            'target_value': goal.reduction_percentage,
            'unit': goal.unit,
            'description': goal.description,
            'target_date': str(self.target_date)
        }

    def verify_goals_loaded(self) -> Dict[str, Any]:
        """
        Verify that goals were loaded correctly
        
        Returns:
            Dict: Verification results
        """
        logger.info("Verifying loaded goals...")
        
        with self.driver.session(database=self.database) as session:
            # Count goals by category and site
            result = session.run("""
                MATCH (g:Goal)-[:APPLIES_TO]->(s:Site)
                RETURN s.id as site_id, s.name as site_name, 
                       g.category as category, g.target_value as target_value,
                       g.unit as unit, g.description as description
                ORDER BY s.id, g.category
            """)
            
            goals = []
            for record in result:
                goals.append(dict(record))
            
            # Summary statistics
            summary = session.run("""
                MATCH (g:Goal)-[:APPLIES_TO]->(s:Site)
                RETURN count(g) as total_goals,
                       count(DISTINCT s.id) as sites_with_goals,
                       collect(DISTINCT g.category) as categories,
                       avg(g.target_value) as avg_target_value
            """)
            
            stats = dict(summary.single())
        
        verification_result = {
            'verification_timestamp': datetime.now().isoformat(),
            'total_goals_found': len(goals),
            'goals_by_site_and_category': goals,
            'summary_statistics': stats,
            'expected_goals': 6,  # 2 sites × 3 categories each
            'verification_passed': len(goals) == 6 and stats['sites_with_goals'] == 2
        }
        
        if verification_result['verification_passed']:
            logger.info("✅ Goal verification PASSED")
            logger.info(f"   - Total goals loaded: {verification_result['total_goals_found']}")
            logger.info(f"   - Sites with goals: {stats['sites_with_goals']}")
            logger.info(f"   - Categories: {stats['categories']}")
            logger.info(f"   - Average target value: {stats['avg_target_value']:.1f}%")
        else:
            logger.warning("⚠️ Goal verification FAILED")
            logger.warning(f"   - Expected 6 goals, found {verification_result['total_goals_found']}")
            logger.warning(f"   - Expected 2 sites, found {stats['sites_with_goals']}")
        
        return verification_result

    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report
        
        Returns:
            Dict: Complete summary report
        """
        logger.info("Generating summary report...")
        
        with self.driver.session(database=self.database) as session:
            # Detailed goals information
            detailed_goals = session.run("""
                MATCH (g:Goal)-[:APPLIES_TO]->(s:Site)
                RETURN {
                    goal_id: g.id,
                    site: {id: s.id, name: s.name},
                    category: g.category,
                    target_value: g.target_value,
                    unit: g.unit,
                    period: g.period,
                    target_date: toString(g.target_date),
                    created_at: toString(g.created_at),
                    description: g.description,
                    baseline_year: g.baseline_year,
                    target_year: g.target_year
                } as goal_info
                ORDER BY s.id, g.category
            """).data()
            
            # Category breakdown
            category_breakdown = session.run("""
                MATCH (g:Goal)-[:APPLIES_TO]->(s:Site)
                RETURN g.category as category,
                       count(*) as goal_count,
                       avg(g.target_value) as avg_target,
                       min(g.target_value) as min_target,
                       max(g.target_value) as max_target,
                       collect(DISTINCT s.name) as sites
                ORDER BY g.category
            """).data()
            
            # Site breakdown
            site_breakdown = session.run("""
                MATCH (s:Site)<-[:APPLIES_TO]-(g:Goal)
                RETURN s.id as site_id, s.name as site_name,
                       count(g) as goal_count,
                       collect({category: g.category, target: g.target_value}) as goals
                ORDER BY s.id
            """).data()
        
        report = {
            'report_generated_at': datetime.now().isoformat(),
            'script_version': '1.0.0',
            'data_source': 'EHSGoalsConfig',
            'target_date': str(self.target_date),
            'summary': {
                'total_goals_loaded': len([goal['goal_info'] for goal in detailed_goals]),
                'categories_covered': len(category_breakdown),
                'sites_covered': len(site_breakdown),
                'category_mapping_used': {str(k): v for k, v in self.category_mapping.items()},
                'site_mapping_used': {str(k): v for k, v in self.site_mapping.items()}
            },
            'goals_detail': [goal['goal_info'] for goal in detailed_goals],
            'category_breakdown': category_breakdown,
            'site_breakdown': site_breakdown,
            'neo4j_relationships': {
                'goal_to_site': '(:Goal)-[:APPLIES_TO]->(:Site)',
                'relationship_count': len(detailed_goals)
            }
        }
        
        return report

    def run_goals_load(self) -> None:
        """Execute the complete goals loading process"""
        try:
            logger.info("="*80)
            logger.info("STARTING EHS GOALS LOAD TO NEO4J")
            logger.info("="*80)
            
            # Step 1: Connect to database
            self.connect()
            
            # Step 2: Verify required sites exist
            if not self.verify_sites_exist():
                logger.error("Cannot proceed without required Site nodes")
                return
            
            # Step 3: Create constraints and indexes
            self.create_goal_constraints()
            
            # Step 4: Load goals
            load_result = self.load_goals()
            
            if "error" in load_result:
                logger.error(f"Goals loading failed: {load_result['error']}")
                return
            
            # Step 5: Verify goals were loaded correctly
            verification_result = self.verify_goals_loaded()
            
            # Step 6: Generate comprehensive report
            summary_report = self.generate_summary_report()
            
            # Save report to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"ehs_goals_load_report_{timestamp}.json"
            
            with open(report_filename, 'w') as f:
                json.dump(summary_report, f, indent=2, default=str)
            
            # Print final summary
            logger.info("="*80)
            logger.info("EHS GOALS LOAD COMPLETED SUCCESSFULLY!")
            logger.info("="*80)
            logger.info(f"Goals loaded: {load_result['total_goals_loaded']}")
            logger.info(f"Verification: {'PASSED' if verification_result['verification_passed'] else 'FAILED'}")
            logger.info(f"Report saved: {report_filename}")
            
            print("\n" + "="*60)
            print("EHS GOALS LOAD SUMMARY")
            print("="*60)
            print(f"Total Goals Loaded: {load_result['total_goals_loaded']}")
            print(f"Categories: {', '.join(summary_report['category_breakdown'][i]['category'] for i in range(len(summary_report['category_breakdown'])))}")
            print(f"Sites: {len(summary_report['site_breakdown'])}")
            print(f"Target Date: {self.target_date}")
            print(f"Verification: {'✅ PASSED' if verification_result['verification_passed'] else '❌ FAILED'}")
            
            print(f"\nGoals by Site:")
            for site in summary_report['site_breakdown']:
                print(f"  {site['site_name']} ({site['site_id']}):")
                for goal in site['goals']:
                    print(f"    - {goal['category']}: {goal['target']}% reduction")
            
            print(f"\nReport Details:")
            print(f"  Log File: load_goals_to_neo4j.log")
            print(f"  Report File: {report_filename}")
            print("="*60)
            
        except Exception as e:
            logger.error(f"Error during goals load: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        finally:
            self.close()


def main():
    """Main entry point"""
    # Get Neo4j configuration from environment
    config = {
        'uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        'username': os.getenv('NEO4J_USERNAME', 'neo4j'),
        'password': os.getenv('NEO4J_PASSWORD', 'EhsAI2024!'),
        'database': os.getenv('NEO4J_DATABASE', 'neo4j')
    }
    
    # Validate configuration
    if not all([config['uri'], config['username'], config['password']]):
        logger.error("Missing Neo4j configuration. Please check your .env file.")
        logger.error("Required environment variables: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD")
        sys.exit(1)
    
    logger.info("Neo4j Configuration:")
    logger.info(f"  URI: {config['uri']}")
    logger.info(f"  Database: {config['database']}")
    logger.info(f"  Username: {config['username']}")
    
    # Create and run goals loader
    loader = EHSGoalsLoader(**config)
    loader.run_goals_load()


if __name__ == "__main__":
    main()