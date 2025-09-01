#!/usr/bin/env python3
"""
Site Environmental Data Generator and Loader

This script generates and loads 6 months of environmental data (March-August 2025) for
Algonquin IL and Houston TX sites with specific risk-triggering patterns designed for
LLM assessment and analysis.

Risk Patterns:
- Algonquin IL: High Risk (electricity trending UP, low recycling, water spikes)
- Houston TX: Medium Risk (flat electricity trend, improving recycling)

Created: 2025-08-31
Version: 1.0.0
"""

import os
import sys
import logging
import random
import math
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple
import json

# Add the project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

try:
    from neo4j import GraphDatabase, Transaction
    from neo4j.exceptions import ServiceUnavailable, ClientError
    from dotenv import load_dotenv
except ImportError:
    print("Error: required packages not installed. Please install with: pip install neo4j python-dotenv")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('site_environmental_data_load.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SiteEnvironmentalDataLoader:
    """Generates and loads environmental data for specific sites with risk patterns"""
    
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """Initialize Neo4j connection"""
        self.driver = None
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        
        # Site configurations
        self.sites = {
            "algonquin_il": {
                "name": "Algonquin Manufacturing",
                "location": "Algonquin, IL",
                "state": "Illinois",
                "zip": "60102",
                "country": "USA",
                "facility_type": "Manufacturing",
                "employee_count": 245,
                "operating_hours": 16,  # 2 shifts
                "risk_profile": "HIGH",
                # Electricity: Start 45K, trend UP 8% over 6 months (should be DOWN 15%)
                "electricity_base": 45000,
                "electricity_trend": 0.08,  # 8% increase over period
                "electricity_target_change": -0.15,  # Target: 15% decrease
                # Water: 150K baseline, 20% summer spike
                "water_base": 150000,
                "water_summer_spike": 0.20,
                # Waste: 8K lbs, 25% recycling (target 50%)
                "waste_base": 8000,
                "recycling_rate": 0.25,
                "recycling_target": 0.50
            },
            "houston_tx": {
                "name": "Houston Processing Center",
                "location": "Houston, TX",
                "state": "Texas",
                "zip": "77001",
                "country": "USA",
                "facility_type": "Processing",
                "employee_count": 380,
                "operating_hours": 24,  # 3 shifts
                "risk_profile": "MEDIUM",
                # Electricity: Start 65K, flat trend (should be DOWN 18%)
                "electricity_base": 65000,
                "electricity_trend": 0.0,  # Flat
                "electricity_target_change": -0.18,  # Target: 18% decrease
                # Water: 200K baseline, 30% summer spike
                "water_base": 200000,
                "water_summer_spike": 0.30,
                # Waste: 12K lbs, improving recycling 30% -> 40%
                "waste_base": 12000,
                "recycling_rate": 0.30,  # Starting rate
                "recycling_target": 0.50
            }
        }
        
        # Date range: March 1, 2025 to August 31, 2025 (6 months)
        self.start_date = date(2025, 3, 1)
        self.end_date = date(2025, 8, 31)
        
        # Weather impact factors
        self.weather_factors = {
            3: {"temp_avg": 45, "cooling_factor": 0.1, "heating_factor": 0.8},  # March
            4: {"temp_avg": 58, "cooling_factor": 0.2, "heating_factor": 0.4},  # April
            5: {"temp_avg": 70, "cooling_factor": 0.4, "heating_factor": 0.1},  # May
            6: {"temp_avg": 80, "cooling_factor": 0.8, "heating_factor": 0.0},  # June
            7: {"temp_avg": 85, "cooling_factor": 1.0, "heating_factor": 0.0},  # July
            8: {"temp_avg": 83, "cooling_factor": 0.9, "heating_factor": 0.0},  # August
        }

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

    def create_sites(self) -> None:
        """Create Site nodes for Algonquin IL and Houston TX"""
        logger.info("Creating site nodes...")
        
        with self.driver.session(database=self.database) as session:
            for site_id, config in self.sites.items():
                session.execute_write(self._create_site_node, site_id, config)
        
        logger.info(f"Created {len(self.sites)} site nodes")

    def _create_site_node(self, tx: Transaction, site_id: str, config: Dict[str, Any]) -> None:
        """Create a Site node with full configuration"""
        query = """
        MERGE (s:Site {id: $site_id})
        SET s.name = $name,
            s.location = $location,
            s.state = $state,
            s.zip_code = $zip,
            s.country = $country,
            s.facility_type = $facility_type,
            s.employee_count = $employee_count,
            s.operating_hours_per_day = $operating_hours,
            s.risk_profile = $risk_profile,
            s.electricity_baseline_kwh = $electricity_base,
            s.electricity_target_change = $electricity_target_change,
            s.water_baseline_gallons = $water_base,
            s.waste_baseline_lbs = $waste_base,
            s.recycling_target_rate = $recycling_target,
            s.created_at = datetime(),
            s.updated_at = datetime()
        """
        
        tx.run(query, {
            'site_id': site_id,
            'name': config['name'],
            'location': config['location'],
            'state': config['state'],
            'zip': config['zip'],
            'country': config['country'],
            'facility_type': config['facility_type'],
            'employee_count': config['employee_count'],
            'operating_hours': config['operating_hours'],
            'risk_profile': config['risk_profile'],
            'electricity_base': config['electricity_base'],
            'electricity_target_change': config['electricity_target_change'],
            'water_base': config['water_base'],
            'waste_base': config['waste_base'],
            'recycling_target': config['recycling_target']
        })

    def generate_electricity_data(self) -> None:
        """Generate electricity consumption data with specific patterns"""
        logger.info("Generating electricity consumption data...")
        
        with self.driver.session(database=self.database) as session:
            for site_id, config in self.sites.items():
                session.execute_write(self._create_electricity_data, site_id, config)
        
        logger.info("Electricity data generation completed")

    def _create_electricity_data(self, tx: Transaction, site_id: str, config: Dict[str, Any]) -> None:
        """Create electricity consumption records for a site"""
        base_consumption = config['electricity_base']
        trend = config['electricity_trend']
        days_in_period = (self.end_date - self.start_date).days
        
        current_date = self.start_date
        while current_date <= self.end_date:
            # Calculate days from start for trend application
            days_from_start = (current_date - self.start_date).days
            progress = days_from_start / days_in_period
            
            # Base monthly consumption with trend
            monthly_base = base_consumption * (1 + trend * progress)
            daily_base = monthly_base / 30.44  # Average days per month
            
            # Weather impact
            month = current_date.month
            weather = self.weather_factors.get(month, self.weather_factors[6])
            
            # Apply weather factors based on site location
            if site_id == "houston_tx":
                # Houston gets more AC load in summer
                weather_multiplier = 1.0 + (weather['cooling_factor'] * 0.25)
            else:
                # Algonquin gets heating in spring, AC in summer
                weather_multiplier = 1.0 + (weather['cooling_factor'] * 0.15) + (weather['heating_factor'] * 0.10)
            
            # Daily variations (weekdays vs weekends)
            weekday_factor = 1.0 if current_date.weekday() < 5 else 0.7
            
            # Random daily variation (±10%)
            daily_variation = random.uniform(0.9, 1.1)
            
            # Calculate final consumption
            daily_consumption = daily_base * weather_multiplier * weekday_factor * daily_variation
            
            # Cost calculation (different rates for different sites)
            cost_per_kwh = 0.12 if site_id == "algonquin_il" else 0.11
            daily_cost = daily_consumption * cost_per_kwh
            
            # Equipment issues metadata (for LLM analysis)
            equipment_notes = self._generate_equipment_notes(current_date, site_id, daily_consumption, daily_base)
            
            query = """
            CREATE (ec:ElectricityConsumption {
                id: $id,
                site_id: $site_id,
                date: date($date),
                consumption_kwh: $consumption,
                cost_usd: $cost,
                cost_per_kwh: $cost_per_kwh,
                weather_factor: $weather_factor,
                equipment_notes: $equipment_notes,
                peak_demand_kw: $peak_demand,
                off_peak_consumption_kwh: $off_peak,
                created_at: datetime(),
                updated_at: datetime()
            })
            """
            
            # Peak demand (typically 10-15% of daily consumption as hourly rate)
            peak_demand = daily_consumption * random.uniform(0.10, 0.15)
            off_peak_portion = daily_consumption * random.uniform(0.6, 0.8)
            
            tx.run(query, {
                'id': f'elec_{site_id}_{current_date.strftime("%Y%m%d")}',
                'site_id': site_id,
                'date': current_date.strftime('%Y-%m-%d'),
                'consumption': round(daily_consumption, 2),
                'cost': round(daily_cost, 2),
                'cost_per_kwh': cost_per_kwh,
                'weather_factor': round(weather_multiplier, 3),
                'equipment_notes': equipment_notes,
                'peak_demand': round(peak_demand, 2),
                'off_peak': round(off_peak_portion, 2)
            })
            
            # Create relationship to site
            tx.run("""
                MATCH (s:Site {id: $site_id}), (ec:ElectricityConsumption {id: $elec_id})
                MERGE (s)-[:HAS_ELECTRICITY_CONSUMPTION]->(ec)
            """, {'site_id': site_id, 'elec_id': f'elec_{site_id}_{current_date.strftime("%Y%m%d")}'})
            
            current_date += timedelta(days=1)

    def _generate_equipment_notes(self, date: date, site_id: str, consumption: float, baseline: float) -> str:
        """Generate equipment-related notes for LLM analysis"""
        notes = []
        
        # High consumption alerts
        if consumption > baseline * 1.2:
            if site_id == "algonquin_il":
                equipment_issues = [
                    "HVAC system running inefficiently - requires maintenance",
                    "Compressor unit #2 showing elevated power draw",
                    "Lighting system upgrade delayed - old fluorescents still in use",
                    "Machine shop equipment running extended hours due to backlog"
                ]
            else:  # houston_tx
                equipment_issues = [
                    "Cooling tower pumps working overtime due to heat",
                    "Process chiller #1 needs refrigerant service",
                    "Warehouse lighting left on overnight - review scheduling",
                    "Emergency generator testing consuming additional power"
                ]
            notes.append(random.choice(equipment_issues))
        
        # Seasonal notes
        if date.month in [6, 7, 8]:  # Summer months
            if site_id == "houston_tx":
                notes.append("Peak summer cooling loads - AC systems at maximum capacity")
            else:
                notes.append("Summer production increase driving higher equipment utilization")
        elif date.month in [3, 4]:  # Spring months
            notes.append("Spring maintenance activities requiring additional equipment runtime")
        
        # Weekend/holiday patterns
        if date.weekday() >= 5:  # Weekend
            notes.append("Reduced weekend operations - lower baseline consumption")
        
        # Random operational notes (20% chance)
        if random.random() < 0.2:
            operational_notes = [
                "Energy efficiency audit scheduled for next quarter",
                "LED lighting retrofit project under evaluation",
                "Smart HVAC controls implementation planned",
                "Solar panel installation feasibility study in progress",
                "Variable frequency drives installed on select motors"
            ]
            notes.append(random.choice(operational_notes))
        
        return " | ".join(notes) if notes else "Normal operations"

    def generate_water_data(self) -> None:
        """Generate water consumption data with summer spikes"""
        logger.info("Generating water consumption data...")
        
        with self.driver.session(database=self.database) as session:
            for site_id, config in self.sites.items():
                session.execute_write(self._create_water_data, site_id, config)
        
        logger.info("Water data generation completed")

    def _create_water_data(self, tx: Transaction, site_id: str, config: Dict[str, Any]) -> None:
        """Create water consumption records for a site"""
        base_consumption = config['water_base']
        summer_spike = config['water_summer_spike']
        
        current_date = self.start_date
        while current_date <= self.end_date:
            # Monthly baseline
            daily_base = base_consumption / 30.44
            
            # Summer spike (June, July, August)
            if current_date.month in [6, 7, 8]:
                spike_multiplier = 1.0 + summer_spike
                spike_notes = f"Summer cooling tower demand increased by {summer_spike*100:.0f}%"
            else:
                spike_multiplier = 1.0
                spike_notes = "Normal seasonal usage"
            
            # Weekly patterns (higher on weekdays)
            weekday_factor = 1.0 if current_date.weekday() < 5 else 0.6
            
            # Daily variation
            daily_variation = random.uniform(0.85, 1.15)
            
            # Calculate final consumption
            daily_consumption = daily_base * spike_multiplier * weekday_factor * daily_variation
            
            # Cost calculation
            cost_per_gallon = 0.004 if site_id == "algonquin_il" else 0.0035
            daily_cost = daily_consumption * cost_per_gallon
            
            # Water quality and source information
            source_type = "Municipal" if random.random() < 0.8 else "Well"
            quality_rating = self._get_water_quality_rating(current_date, site_id)
            
            # Usage breakdown
            process_usage = daily_consumption * random.uniform(0.6, 0.7)
            cooling_usage = daily_consumption * random.uniform(0.2, 0.3)
            domestic_usage = daily_consumption - process_usage - cooling_usage
            
            query = """
            CREATE (wc:WaterConsumption {
                id: $id,
                site_id: $site_id,
                date: date($date),
                consumption_gallons: $consumption,
                cost_usd: $cost,
                cost_per_gallon: $cost_per_gallon,
                source_type: $source_type,
                quality_rating: $quality_rating,
                process_usage_gallons: $process_usage,
                cooling_usage_gallons: $cooling_usage,
                domestic_usage_gallons: $domestic_usage,
                seasonal_notes: $seasonal_notes,
                created_at: datetime(),
                updated_at: datetime()
            })
            """
            
            tx.run(query, {
                'id': f'water_{site_id}_{current_date.strftime("%Y%m%d")}',
                'site_id': site_id,
                'date': current_date.strftime('%Y-%m-%d'),
                'consumption': round(daily_consumption, 2),
                'cost': round(daily_cost, 2),
                'cost_per_gallon': cost_per_gallon,
                'source_type': source_type,
                'quality_rating': quality_rating,
                'process_usage': round(process_usage, 2),
                'cooling_usage': round(cooling_usage, 2),
                'domestic_usage': round(domestic_usage, 2),
                'seasonal_notes': spike_notes
            })
            
            # Create relationship to site
            tx.run("""
                MATCH (s:Site {id: $site_id}), (wc:WaterConsumption {id: $water_id})
                MERGE (s)-[:HAS_WATER_CONSUMPTION]->(wc)
            """, {'site_id': site_id, 'water_id': f'water_{site_id}_{current_date.strftime("%Y%m%d")}'})
            
            current_date += timedelta(days=1)

    def _get_water_quality_rating(self, date: date, site_id: str) -> str:
        """Generate water quality rating based on date and site"""
        # Simulate occasional quality issues
        if random.random() < 0.05:  # 5% chance of quality issues
            return random.choice(['C', 'D'])
        elif random.random() < 0.15:  # 15% chance of good quality
            return 'A'
        else:  # 80% chance of standard quality
            return 'B'

    def generate_waste_data(self) -> None:
        """Generate waste data with recycling patterns"""
        logger.info("Generating waste generation data...")
        
        with self.driver.session(database=self.database) as session:
            for site_id, config in self.sites.items():
                session.execute_write(self._create_waste_data, site_id, config)
        
        logger.info("Waste data generation completed")

    def _create_waste_data(self, tx: Transaction, site_id: str, config: Dict[str, Any]) -> None:
        """Create waste generation records for a site"""
        base_waste = config['waste_base']
        recycling_rate = config['recycling_rate']
        days_in_period = (self.end_date - self.start_date).days
        
        # Houston improves recycling over time, Algonquin stays low
        recycling_improvement = 0.10 if site_id == "houston_tx" else 0.0
        
        current_date = self.start_date
        # Generate weekly data (every Monday)
        while current_date <= self.end_date:
            if current_date.weekday() == 0:  # Monday
                # Weekly waste generation
                weekly_base = base_waste / 4.33  # Average weeks per month
                
                # Seasonal variation (more waste in summer due to increased activity)
                if current_date.month in [6, 7, 8]:
                    seasonal_multiplier = 1.15
                else:
                    seasonal_multiplier = random.uniform(0.95, 1.05)
                
                # Calculate progress for recycling improvement
                progress = (current_date - self.start_date).days / days_in_period
                current_recycling_rate = recycling_rate + (recycling_improvement * progress)
                
                # Total waste generation
                total_waste = weekly_base * seasonal_multiplier * random.uniform(0.9, 1.1)
                
                # Waste type breakdown
                waste_types = {
                    'recyclable': total_waste * current_recycling_rate,
                    'non_hazardous': total_waste * (0.8 - current_recycling_rate),
                    'hazardous': total_waste * 0.15,
                    'organic': total_waste * 0.05
                }
                
                for waste_type, amount in waste_types.items():
                    if amount > 0:
                        # Disposal cost varies by waste type
                        cost_multipliers = {
                            'recyclable': -0.50,  # Revenue from recycling
                            'non_hazardous': 0.75,
                            'hazardous': 3.50,
                            'organic': 0.25
                        }
                        
                        disposal_cost = amount * cost_multipliers[waste_type]
                        
                        # Disposal method
                        disposal_methods = {
                            'recyclable': 'Recycling Center',
                            'non_hazardous': 'Landfill',
                            'hazardous': 'Incineration',
                            'organic': 'Composting'
                        }
                        
                        # Contractor information
                        contractors = {
                            'algonquin_il': 'Waste Management Inc.',
                            'houston_tx': 'Republic Services'
                        }
                        
                        # Performance notes
                        performance_notes = self._generate_waste_performance_notes(
                            current_date, site_id, waste_type, current_recycling_rate, config['recycling_target']
                        )
                        
                        query = """
                        CREATE (wg:WasteGeneration {
                            id: $id,
                            site_id: $site_id,
                            date: date($date),
                            waste_type: $waste_type,
                            amount_pounds: $amount,
                            disposal_method: $disposal_method,
                            disposal_cost_usd: $disposal_cost,
                            contractor: $contractor,
                            recycling_rate_achieved: $recycling_rate,
                            recycling_target: $recycling_target,
                            performance_notes: $performance_notes,
                            created_at: datetime(),
                            updated_at: datetime()
                        })
                        """
                        
                        tx.run(query, {
                            'id': f'waste_{site_id}_{waste_type}_{current_date.strftime("%Y%m%d")}',
                            'site_id': site_id,
                            'date': current_date.strftime('%Y-%m-%d'),
                            'waste_type': waste_type.title(),
                            'amount': round(amount, 2),
                            'disposal_method': disposal_methods[waste_type],
                            'disposal_cost': round(disposal_cost, 2),
                            'contractor': contractors[site_id],
                            'recycling_rate': round(current_recycling_rate, 3),
                            'recycling_target': config['recycling_target'],
                            'performance_notes': performance_notes
                        })
                        
                        # Create relationship to site
                        tx.run("""
                            MATCH (s:Site {id: $site_id}), (wg:WasteGeneration {id: $waste_id})
                            MERGE (s)-[:GENERATES_WASTE]->(wg)
                        """, {'site_id': site_id, 'waste_id': f'waste_{site_id}_{waste_type}_{current_date.strftime("%Y%m%d")}'})
            
            current_date += timedelta(days=1)

    def _generate_waste_performance_notes(self, date: date, site_id: str, waste_type: str, 
                                        current_rate: float, target_rate: float) -> str:
        """Generate performance notes for waste management"""
        notes = []
        
        # Performance against target
        if waste_type == 'recyclable':
            if current_rate < target_rate * 0.7:
                notes.append(f"BELOW TARGET: Recycling rate {current_rate:.1%} significantly below target {target_rate:.1%}")
            elif current_rate < target_rate:
                notes.append(f"Recycling rate {current_rate:.1%} approaching target {target_rate:.1%}")
            else:
                notes.append(f"EXCEEDS TARGET: Recycling rate {current_rate:.1%} above target {target_rate:.1%}")
        
        # Site-specific issues
        if site_id == "algonquin_il":
            recycling_issues = [
                "Employee training on sorting protocols needed",
                "Recycling bins insufficient for production floor",
                "Contamination issues with cardboard sorting",
                "Lack of dedicated recycling collection area"
            ]
            if random.random() < 0.3:
                notes.append(random.choice(recycling_issues))
        else:  # houston_tx
            recycling_improvements = [
                "New recycling education program showing results",
                "Improved sorting stations installed in break areas",
                "Partnership with local recycling facility expanding",
                "Waste audit identifying additional recycling opportunities"
            ]
            if random.random() < 0.3:
                notes.append(random.choice(recycling_improvements))
        
        # Hazardous waste specific notes
        if waste_type == 'hazardous':
            hazmat_notes = [
                "Proper containment and labeling protocols followed",
                "Quarterly hazmat training completed",
                "EPA reporting requirements met",
                "Waste minimization opportunities identified"
            ]
            notes.append(random.choice(hazmat_notes))
        
        return " | ".join(notes) if notes else "Standard waste management protocols followed"

    def create_environmental_targets(self) -> None:
        """Create environmental performance targets for each site"""
        logger.info("Creating environmental targets...")
        
        with self.driver.session(database=self.database) as session:
            for site_id, config in self.sites.items():
                session.execute_write(self._create_targets, site_id, config)
        
        logger.info("Environmental targets created")

    def _create_targets(self, tx: Transaction, site_id: str, config: Dict[str, Any]) -> None:
        """Create environmental targets for a site"""
        targets = [
            {
                'type': 'electricity_reduction',
                'target_value': abs(config['electricity_target_change']),
                'target_unit': 'percent_reduction',
                'deadline': '2025-12-31',
                'status': 'in_progress',
                'description': f"Reduce electricity consumption by {abs(config['electricity_target_change'])*100:.0f}% through efficiency improvements"
            },
            {
                'type': 'recycling_rate',
                'target_value': config['recycling_target'],
                'target_unit': 'percentage',
                'deadline': '2025-12-31',
                'status': 'in_progress',
                'description': f"Achieve {config['recycling_target']*100:.0f}% recycling rate for all waste streams"
            },
            {
                'type': 'water_efficiency',
                'target_value': 0.10,
                'target_unit': 'percent_reduction',
                'deadline': '2025-12-31',
                'status': 'planning',
                'description': "Reduce water consumption per unit of production by 10%"
            }
        ]
        
        for i, target in enumerate(targets):
            query = """
            CREATE (et:EnvironmentalTarget {
                id: $id,
                site_id: $site_id,
                target_type: $target_type,
                target_value: $target_value,
                target_unit: $target_unit,
                deadline: date($deadline),
                status: $status,
                description: $description,
                created_at: datetime(),
                updated_at: datetime()
            })
            """
            
            tx.run(query, {
                'id': f'target_{site_id}_{i}',
                'site_id': site_id,
                'target_type': target['type'],
                'target_value': target['target_value'],
                'target_unit': target['target_unit'],
                'deadline': target['deadline'],
                'status': target['status'],
                'description': target['description']
            })
            
            # Create relationship to site
            tx.run("""
                MATCH (s:Site {id: $site_id}), (et:EnvironmentalTarget {id: $target_id})
                MERGE (s)-[:HAS_TARGET]->(et)
            """, {'site_id': site_id, 'target_id': f'target_{site_id}_{i}'})

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of data created"""
        logger.info("Generating summary report...")
        
        with self.driver.session(database=self.database) as session:
            # Count records by type
            counts = {}
            
            for record_type in ['Site', 'ElectricityConsumption', 'WaterConsumption', 'WasteGeneration', 'EnvironmentalTarget']:
                result = session.run(f"MATCH (n:{record_type}) RETURN count(n) as count")
                counts[record_type] = result.single()['count']
            
            # Get date range of data
            result = session.run("""
                MATCH (ec:ElectricityConsumption)
                RETURN min(ec.date) as start_date, max(ec.date) as end_date
            """)
            date_range = result.single()
            
            # Sample data for verification
            sample_data = {}
            for site_id in self.sites.keys():
                result = session.run("""
                    MATCH (s:Site {id: $site_id})
                    OPTIONAL MATCH (s)-[:HAS_ELECTRICITY_CONSUMPTION]->(ec:ElectricityConsumption)
                    OPTIONAL MATCH (s)-[:HAS_WATER_CONSUMPTION]->(wc:WaterConsumption)
                    OPTIONAL MATCH (s)-[:GENERATES_WASTE]->(wg:WasteGeneration)
                    RETURN s.name as site_name,
                           count(DISTINCT ec) as electricity_records,
                           count(DISTINCT wc) as water_records,
                           count(DISTINCT wg) as waste_records,
                           avg(ec.consumption_kwh) as avg_electricity,
                           avg(wc.consumption_gallons) as avg_water
                """, {'site_id': site_id})
                
                record = result.single()
                sample_data[site_id] = dict(record) if record else {}
        
        report = {
            'generation_date': datetime.now().isoformat(),
            'data_period': {
                'start_date': str(self.start_date),
                'end_date': str(self.end_date),
                'duration_days': (self.end_date - self.start_date).days
            },
            'record_counts': counts,
            'date_range_verification': {
                'start_date': str(date_range['start_date']) if date_range['start_date'] else None,
                'end_date': str(date_range['end_date']) if date_range['end_date'] else None
            },
            'site_data_summary': sample_data,
            'risk_patterns': {
                'algonquin_il': {
                    'risk_level': 'HIGH',
                    'electricity_pattern': 'Trending UP 8% (target: DOWN 15%)',
                    'water_pattern': '20% summer spike',
                    'waste_pattern': '25% recycling rate (target: 50%)'
                },
                'houston_tx': {
                    'risk_level': 'MEDIUM',
                    'electricity_pattern': 'Flat trend (target: DOWN 18%)',
                    'water_pattern': '30% summer spike',
                    'waste_pattern': 'Improving recycling 30% -> 40%'
                }
            }
        }
        
        return report

    def run_data_load(self) -> None:
        """Execute the complete data loading process"""
        try:
            logger.info("Starting site environmental data load process...")
            
            self.connect()
            
            # Step 1: Create sites
            self.create_sites()
            
            # Step 2: Generate electricity data
            self.generate_electricity_data()
            
            # Step 3: Generate water data
            self.generate_water_data()
            
            # Step 4: Generate waste data
            self.generate_waste_data()
            
            # Step 5: Create environmental targets
            self.create_environmental_targets()
            
            # Step 6: Generate summary report
            report = self.generate_summary_report()
            
            # Save report to file
            report_file = f'environmental_data_load_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Data load completed successfully!")
            logger.info(f"Summary report saved to: {report_file}")
            logger.info(f"Total records created: {sum(report['record_counts'].values())}")
            
            # Print key metrics
            print("\n" + "="*80)
            print("ENVIRONMENTAL DATA LOAD SUMMARY")
            print("="*80)
            print(f"Data Period: {report['data_period']['start_date']} to {report['data_period']['end_date']}")
            print(f"Duration: {report['data_period']['duration_days']} days")
            print(f"\nRecords Created:")
            for record_type, count in report['record_counts'].items():
                print(f"  {record_type}: {count:,}")
            print(f"\nTotal Records: {sum(report['record_counts'].values()):,}")
            
            print(f"\nRisk Patterns Implemented:")
            for site_id, pattern in report['risk_patterns'].items():
                site_name = self.sites[site_id]['name']
                print(f"  {site_name} ({pattern['risk_level']} RISK):")
                print(f"    - Electricity: {pattern['electricity_pattern']}")
                print(f"    - Water: {pattern['water_pattern']}")
                print(f"    - Waste: {pattern['waste_pattern']}")
            
            print("="*80)
            
        except Exception as e:
            logger.error(f"Error during data load: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            self.close()


def main():
    """Main entry point"""
    # Configuration from environment
    config = {
        'uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        'username': os.getenv('NEO4J_USERNAME', 'neo4j'),
        'password': os.getenv('NEO4J_PASSWORD', 'EhsAI2024!'),
        'database': os.getenv('NEO4J_DATABASE', 'neo4j')
    }
    
    # Validate configuration
    if not all([config['uri'], config['username'], config['password']]):
        logger.error("Missing Neo4j configuration. Please check your .env file.")
        sys.exit(1)
    
    # Create and run data loader
    loader = SiteEnvironmentalDataLoader(**config)
    loader.run_data_load()


if __name__ == "__main__":
    main()