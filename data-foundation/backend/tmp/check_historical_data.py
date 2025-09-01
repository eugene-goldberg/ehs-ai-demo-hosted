#!/usr/bin/env python3
"""
Neo4j Historical Data Checker

This script checks what historical environmental data exists in Neo4j for both sites.
It analyzes ElectricityConsumption, WaterConsumption, and WasteGeneration nodes
to determine date ranges, site coverage, and data completeness.

Created: 2025-08-31
Version: 1.0.0
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
import json

# Add the project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

try:
    from neo4j import GraphDatabase, Transaction
    from neo4j.exceptions import ServiceUnavailable, ClientError
except ImportError:
    print("Error: neo4j package not installed. Please install with: pip install neo4j")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/tmp/historical_data_check.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HistoricalDataChecker:
    """Checks historical environmental data in Neo4j"""
    
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
    
    def check_electricity_consumption_data(self) -> Dict[str, Any]:
        """Check electricity consumption data by site"""
        query = """
        MATCH (e:ElectricityConsumption)
        RETURN 
            e.facility_id as facility_id,
            e.date as date,
            e.consumption_kwh as consumption_kwh,
            e.cost_usd as cost_usd
        ORDER BY e.facility_id, e.date
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            records = list(result)
        
        # Process data by facility
        facility_data = defaultdict(list)
        for record in records:
            facility_id = record['facility_id']
            facility_data[facility_id].append({
                'date': record['date'],
                'consumption_kwh': record['consumption_kwh'],
                'cost_usd': record['cost_usd']
            })
        
        # Analyze data ranges and completeness
        analysis = {}
        for facility_id, data in facility_data.items():
            dates = [item['date'] for item in data]
            if dates:
                min_date = min(dates)
                max_date = max(dates)
                total_records = len(data)
                
                # Calculate expected days between min and max dates
                date_range_days = (max_date - min_date).days + 1
                
                # Calculate months of data (approximate)
                months_of_data = (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month) + 1
                
                analysis[facility_id] = {
                    'min_date': min_date.strftime('%Y-%m-%d'),
                    'max_date': max_date.strftime('%Y-%m-%d'),
                    'total_records': total_records,
                    'date_range_days': date_range_days,
                    'months_of_data': months_of_data,
                    'has_6_months': months_of_data >= 6,
                    'avg_consumption_kwh': sum(item['consumption_kwh'] for item in data) / len(data),
                    'total_cost_usd': sum(item['cost_usd'] for item in data)
                }
        
        return analysis
    
    def check_water_consumption_data(self) -> Dict[str, Any]:
        """Check water consumption data by site"""
        query = """
        MATCH (w:WaterConsumption)
        RETURN 
            w.facility_id as facility_id,
            w.date as date,
            w.consumption_gallons as consumption_gallons,
            w.cost_usd as cost_usd,
            w.source_type as source_type
        ORDER BY w.facility_id, w.date
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            records = list(result)
        
        # Process data by facility
        facility_data = defaultdict(list)
        for record in records:
            facility_id = record['facility_id']
            facility_data[facility_id].append({
                'date': record['date'],
                'consumption_gallons': record['consumption_gallons'],
                'cost_usd': record['cost_usd'],
                'source_type': record['source_type']
            })
        
        # Analyze data ranges and completeness
        analysis = {}
        for facility_id, data in facility_data.items():
            dates = [item['date'] for item in data]
            if dates:
                min_date = min(dates)
                max_date = max(dates)
                total_records = len(data)
                
                # Calculate months of data
                months_of_data = (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month) + 1
                
                # Get source type distribution
                source_types = [item['source_type'] for item in data]
                source_type_counts = {}
                for source in source_types:
                    source_type_counts[source] = source_type_counts.get(source, 0) + 1
                
                analysis[facility_id] = {
                    'min_date': min_date.strftime('%Y-%m-%d'),
                    'max_date': max_date.strftime('%Y-%m-%d'),
                    'total_records': total_records,
                    'months_of_data': months_of_data,
                    'has_6_months': months_of_data >= 6,
                    'avg_consumption_gallons': sum(item['consumption_gallons'] for item in data) / len(data),
                    'total_cost_usd': sum(item['cost_usd'] for item in data),
                    'source_types': source_type_counts
                }
        
        return analysis
    
    def check_waste_generation_data(self) -> Dict[str, Any]:
        """Check waste generation data by site"""
        query = """
        MATCH (wg:WasteGeneration)
        RETURN 
            wg.facility_id as facility_id,
            wg.date as date,
            wg.waste_type as waste_type,
            wg.amount_pounds as amount_pounds,
            wg.disposal_cost_usd as disposal_cost_usd,
            wg.disposal_method as disposal_method
        ORDER BY wg.facility_id, wg.date
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            records = list(result)
        
        # Process data by facility
        facility_data = defaultdict(list)
        for record in records:
            facility_id = record['facility_id']
            facility_data[facility_id].append({
                'date': record['date'],
                'waste_type': record['waste_type'],
                'amount_pounds': record['amount_pounds'],
                'disposal_cost_usd': record['disposal_cost_usd'],
                'disposal_method': record['disposal_method']
            })
        
        # Analyze data ranges and completeness
        analysis = {}
        for facility_id, data in facility_data.items():
            dates = [item['date'] for item in data]
            if dates:
                min_date = min(dates)
                max_date = max(dates)
                total_records = len(data)
                
                # Calculate months of data (waste data is typically weekly, so adjust calculation)
                months_of_data = (max_date.year - min_date.year) * 12 + (max_date.month - min_date.month) + 1
                
                # Get waste type distribution
                waste_types = [item['waste_type'] for item in data]
                waste_type_counts = {}
                waste_type_amounts = {}
                for item in data:
                    wtype = item['waste_type']
                    waste_type_counts[wtype] = waste_type_counts.get(wtype, 0) + 1
                    waste_type_amounts[wtype] = waste_type_amounts.get(wtype, 0) + item['amount_pounds']
                
                # Get disposal method distribution
                disposal_methods = [item['disposal_method'] for item in data]
                disposal_method_counts = {}
                for method in disposal_methods:
                    disposal_method_counts[method] = disposal_method_counts.get(method, 0) + 1
                
                analysis[facility_id] = {
                    'min_date': min_date.strftime('%Y-%m-%d'),
                    'max_date': max_date.strftime('%Y-%m-%d'),
                    'total_records': total_records,
                    'months_of_data': months_of_data,
                    'has_6_months': months_of_data >= 6,
                    'total_amount_pounds': sum(item['amount_pounds'] for item in data),
                    'total_disposal_cost_usd': sum(item['disposal_cost_usd'] for item in data),
                    'waste_types': waste_type_counts,
                    'waste_type_amounts': waste_type_amounts,
                    'disposal_methods': disposal_method_counts
                }
        
        return analysis
    
    def check_facility_info(self) -> Dict[str, Any]:
        """Get facility information"""
        query = """
        MATCH (f:Facility)
        RETURN f.id as id, f.name as name, f.type as type
        ORDER BY f.id
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            records = list(result)
        
        facilities = {}
        for record in records:
            facilities[record['id']] = {
                'name': record['name'],
                'type': record['type']
            }
        
        return facilities
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        logger.info("Generating comprehensive historical data report...")
        
        # Get facility info
        facilities = self.check_facility_info()
        
        # Check all data types
        electricity_data = self.check_electricity_consumption_data()
        water_data = self.check_water_consumption_data()
        waste_data = self.check_waste_generation_data()
        
        # Combine into comprehensive report
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'total_facilities': len(facilities),
            'facilities': facilities,
            'data_analysis': {
                'electricity_consumption': {
                    'facilities_with_data': len(electricity_data),
                    'facility_analysis': electricity_data
                },
                'water_consumption': {
                    'facilities_with_data': len(water_data),
                    'facility_analysis': water_data
                },
                'waste_generation': {
                    'facilities_with_data': len(waste_data),
                    'facility_analysis': waste_data
                }
            },
            'completeness_summary': self._generate_completeness_summary(electricity_data, water_data, waste_data),
            'recommendations': self._generate_recommendations(electricity_data, water_data, waste_data)
        }
        
        return report
    
    def _generate_completeness_summary(self, electricity_data: Dict, water_data: Dict, waste_data: Dict) -> Dict[str, Any]:
        """Generate data completeness summary"""
        all_facility_ids = set()
        all_facility_ids.update(electricity_data.keys())
        all_facility_ids.update(water_data.keys())
        all_facility_ids.update(waste_data.keys())
        
        summary = {
            'total_facilities_with_environmental_data': len(all_facility_ids),
            'facilities_with_6_months_data': {
                'electricity': sum(1 for data in electricity_data.values() if data.get('has_6_months', False)),
                'water': sum(1 for data in water_data.values() if data.get('has_6_months', False)),
                'waste': sum(1 for data in waste_data.values() if data.get('has_6_months', False))
            },
            'facilities_meeting_6_month_requirement': {
                facility_id: {
                    'electricity': electricity_data.get(facility_id, {}).get('has_6_months', False),
                    'water': water_data.get(facility_id, {}).get('has_6_months', False),
                    'waste': waste_data.get(facility_id, {}).get('has_6_months', False),
                    'all_categories': (
                        electricity_data.get(facility_id, {}).get('has_6_months', False) and
                        water_data.get(facility_id, {}).get('has_6_months', False) and
                        waste_data.get(facility_id, {}).get('has_6_months', False)
                    )
                }
                for facility_id in all_facility_ids
            }
        }
        
        return summary
    
    def _generate_recommendations(self, electricity_data: Dict, water_data: Dict, waste_data: Dict) -> List[str]:
        """Generate recommendations based on data analysis"""
        recommendations = []
        
        # Check for facilities with insufficient data
        all_facility_ids = set()
        all_facility_ids.update(electricity_data.keys())
        all_facility_ids.update(water_data.keys())
        all_facility_ids.update(waste_data.keys())
        
        for facility_id in all_facility_ids:
            elec_months = electricity_data.get(facility_id, {}).get('months_of_data', 0)
            water_months = water_data.get(facility_id, {}).get('months_of_data', 0)
            waste_months = waste_data.get(facility_id, {}).get('months_of_data', 0)
            
            if elec_months < 6:
                recommendations.append(f"Facility {facility_id}: Need {6 - elec_months} more months of electricity data")
            if water_months < 6:
                recommendations.append(f"Facility {facility_id}: Need {6 - water_months} more months of water data")
            if waste_months < 6:
                recommendations.append(f"Facility {facility_id}: Need {6 - waste_months} more months of waste data")
        
        # Check for data quality issues
        for facility_id, data in electricity_data.items():
            if data['total_records'] < 30:  # Less than daily data for a month
                recommendations.append(f"Facility {facility_id}: Electricity data appears incomplete (only {data['total_records']} records)")
        
        for facility_id, data in water_data.items():
            if data['total_records'] < 30:  # Less than daily data for a month
                recommendations.append(f"Facility {facility_id}: Water data appears incomplete (only {data['total_records']} records)")
        
        if not recommendations:
            recommendations.append("Data completeness looks good! All facilities have sufficient historical data.")
        
        return recommendations
    
    def save_report(self, report: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save report to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/tmp/historical_data_report_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to: {filename}")
        return filename
    
    def print_summary(self, report: Dict[str, Any]) -> None:
        """Print a human-readable summary of the report"""
        print("\n" + "="*80)
        print("HISTORICAL DATA ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\nReport Generated: {report['report_timestamp']}")
        print(f"Total Facilities: {report['total_facilities']}")
        
        print(f"\nFacilities:")
        for fid, finfo in report['facilities'].items():
            print(f"  {fid}: {finfo['name']} ({finfo['type']})")
        
        print(f"\nData Coverage:")
        data_analysis = report['data_analysis']
        
        print(f"  Electricity Consumption:")
        print(f"    Facilities with data: {data_analysis['electricity_consumption']['facilities_with_data']}")
        for fid, data in data_analysis['electricity_consumption']['facility_analysis'].items():
            print(f"      {fid}: {data['months_of_data']} months ({data['min_date']} to {data['max_date']}) - {data['total_records']} records")
        
        print(f"  Water Consumption:")
        print(f"    Facilities with data: {data_analysis['water_consumption']['facilities_with_data']}")
        for fid, data in data_analysis['water_consumption']['facility_analysis'].items():
            print(f"      {fid}: {data['months_of_data']} months ({data['min_date']} to {data['max_date']}) - {data['total_records']} records")
        
        print(f"  Waste Generation:")
        print(f"    Facilities with data: {data_analysis['waste_generation']['facilities_with_data']}")
        for fid, data in data_analysis['waste_generation']['facility_analysis'].items():
            print(f"      {fid}: {data['months_of_data']} months ({data['min_date']} to {data['max_date']}) - {data['total_records']} records")
        
        print(f"\n6-Month Data Requirement Status:")
        completeness = report['completeness_summary']
        for fid, status in completeness['facilities_meeting_6_month_requirement'].items():
            print(f"  {fid}:")
            print(f"    Electricity: {'✓' if status['electricity'] else '✗'} ({6 if status['electricity'] else 'insufficient'} months)")
            print(f"    Water: {'✓' if status['water'] else '✗'} ({6 if status['water'] else 'insufficient'} months)")
            print(f"    Waste: {'✓' if status['waste'] else '✗'} ({6 if status['waste'] else 'insufficient'} months)")
            print(f"    All Categories: {'✓ COMPLETE' if status['all_categories'] else '✗ INCOMPLETE'}")
        
        print(f"\nRecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "="*80)


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
                    if key == 'NEO4J_URI':
                        config['uri'] = value
                    elif key == 'NEO4J_USERNAME':
                        config['username'] = value
                    elif key == 'NEO4J_PASSWORD':
                        config['password'] = value
    
    return config


def main():
    """Main function"""
    try:
        logger.info("Historical Data Checker starting...")
        
        # Load configuration
        config = load_config()
        logger.info(f"Connecting to Neo4j at {config['uri']}")
        
        # Create and run the checker
        checker = HistoricalDataChecker(
            uri=config['uri'],
            username=config['username'],
            password=config['password']
        )
        
        # Generate comprehensive report
        report = checker.generate_summary_report()
        
        # Save report to file
        report_file = checker.save_report(report)
        
        # Print summary to console
        checker.print_summary(report)
        
        print(f"\nDetailed report saved to: {report_file}")
        print(f"Log file: /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/tmp/historical_data_check.log")
        
        # Close connection
        checker.close()
        
    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()