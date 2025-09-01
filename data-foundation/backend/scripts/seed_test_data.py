#!/usr/bin/env python3
"""
Location Hierarchy Test Data Seeding Script

This script creates comprehensive sample data for testing the location hierarchy
migration, including sites, buildings, floors, areas, facilities, EHS metrics, 
and incident data.

Features:
- Creates Algonquin and Houston test sites with realistic structure
- Populates EHS metrics (safety, environmental, health)
- Creates sample incident data
- Links all data through proper relationships
- Supports cleanup of existing test data
- Comprehensive logging and error handling

Author: AI Assistant
Date: 2025-08-30
"""

import sys
import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import random
from uuid import uuid4

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
class SeedingConfig:
    """Configuration for test data seeding"""
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str
    cleanup_existing: bool = True
    seed_ehs_metrics: bool = True
    seed_incidents: bool = True
    metrics_date_range_days: int = 90

class TestDataSeeder:
    """Main class for seeding test data"""
    
    def __init__(self, config: SeedingConfig):
        self.config = config
        self.graph = None
        self.seeding_id = f"seed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger = self._setup_logging()
        
        # Test data templates
        self.site_data = self._get_site_data_templates()
        self.metrics_templates = self._get_metrics_templates()
        self.incident_templates = self._get_incident_templates()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_filename = f"seed_data_{self.seeding_id}.log"
        log_path = os.path.join(os.path.dirname(__file__), log_filename)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        logger = logging.getLogger(f"TestDataSeeder_{self.seeding_id}")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def connect_to_neo4j(self) -> bool:
        """Establish connection to Neo4j database"""
        try:
            self.logger.info(f"Connecting to Neo4j at {self.config.neo4j_uri}")
            
            self.graph = Neo4jGraph(
                url=self.config.neo4j_uri,
                username=self.config.neo4j_username,
                password=self.config.neo4j_password,
                database=self.config.neo4j_database
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
    
    def _get_site_data_templates(self) -> Dict[str, Any]:
        """Get comprehensive site data templates"""
        return {
            "algonquin": {
                "site": {
                    "name": "Algonquin Manufacturing Site",
                    "code": "ALG001",
                    "address": "2350 Millennium Drive, Algonquin, IL 60102",
                    "type": "Manufacturing",
                    "established": "2018-03-15",
                    "employees": 485,
                    "operating_hours": "24/7",
                    "certifications": ["ISO 14001", "OSHA VPP", "ISO 45001"]
                },
                "buildings": [
                    {
                        "name": "Main Manufacturing Building",
                        "code": "ALG001-MFG",
                        "type": "Manufacturing",
                        "area_sqft": 125000,
                        "built_year": 2018,
                        "floors": [
                            {
                                "name": "Production Floor",
                                "level": 0,
                                "area_sqft": 85000,
                                "ceiling_height": 30,
                                "areas": [
                                    {
                                        "name": "Production Line A",
                                        "type": "Production",
                                        "area_sqft": 15000,
                                        "capacity_units_per_hour": 250,
                                        "primary_process": "Assembly"
                                    },
                                    {
                                        "name": "Production Line B", 
                                        "type": "Production",
                                        "area_sqft": 15000,
                                        "capacity_units_per_hour": 300,
                                        "primary_process": "Assembly"
                                    },
                                    {
                                        "name": "Quality Control Lab",
                                        "type": "QC",
                                        "area_sqft": 8000,
                                        "testing_capacity": 500,
                                        "certification": "ISO 17025"
                                    },
                                    {
                                        "name": "Raw Material Staging",
                                        "type": "Storage",
                                        "area_sqft": 12000,
                                        "storage_capacity_tons": 200
                                    },
                                    {
                                        "name": "Maintenance Workshop",
                                        "type": "Maintenance",
                                        "area_sqft": 6000,
                                        "equipment_count": 25
                                    }
                                ]
                            },
                            {
                                "name": "Mezzanine Level",
                                "level": 1,
                                "area_sqft": 25000,
                                "ceiling_height": 12,
                                "areas": [
                                    {
                                        "name": "Plant Office",
                                        "type": "Office",
                                        "area_sqft": 8000,
                                        "workstations": 45
                                    },
                                    {
                                        "name": "Conference Rooms",
                                        "type": "Meeting",
                                        "area_sqft": 3000,
                                        "rooms": 6
                                    },
                                    {
                                        "name": "Break Room",
                                        "type": "Common",
                                        "area_sqft": 2500,
                                        "capacity": 80
                                    },
                                    {
                                        "name": "Training Center",
                                        "type": "Training",
                                        "area_sqft": 4000,
                                        "capacity": 60
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "name": "Warehouse",
                        "code": "ALG001-WH",
                        "type": "Warehouse",
                        "area_sqft": 95000,
                        "built_year": 2019,
                        "floors": [
                            {
                                "name": "Main Warehouse Floor",
                                "level": 0,
                                "area_sqft": 90000,
                                "ceiling_height": 35,
                                "areas": [
                                    {
                                        "name": "Raw Materials Storage",
                                        "type": "Storage",
                                        "area_sqft": 30000,
                                        "storage_capacity_tons": 1500,
                                        "temperature_controlled": False
                                    },
                                    {
                                        "name": "Finished Goods Storage",
                                        "type": "Storage",
                                        "area_sqft": 35000,
                                        "storage_capacity_tons": 800,
                                        "temperature_controlled": True
                                    },
                                    {
                                        "name": "Shipping & Receiving",
                                        "type": "Logistics",
                                        "area_sqft": 15000,
                                        "dock_doors": 12,
                                        "trucks_per_day": 25
                                    },
                                    {
                                        "name": "Chemical Storage",
                                        "type": "Hazmat",
                                        "area_sqft": 5000,
                                        "hazmat_certified": True,
                                        "ventilation_type": "Specialized"
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "name": "Utility Building",
                        "code": "ALG001-UTIL",
                        "type": "Utility",
                        "area_sqft": 8000,
                        "built_year": 2018,
                        "floors": [
                            {
                                "name": "Utility Floor",
                                "level": 0,
                                "area_sqft": 8000,
                                "areas": [
                                    {
                                        "name": "Electrical Room",
                                        "type": "Electrical",
                                        "area_sqft": 2000,
                                        "voltage_primary": 13800,
                                        "backup_generator": True
                                    },
                                    {
                                        "name": "Compressor Room",
                                        "type": "Mechanical",
                                        "area_sqft": 2500,
                                        "air_pressure_psi": 125
                                    },
                                    {
                                        "name": "Water Treatment",
                                        "type": "Environmental",
                                        "area_sqft": 2000,
                                        "treatment_capacity_gpd": 50000
                                    }
                                ]
                            }
                        ]
                    }
                ]
            },
            "houston": {
                "site": {
                    "name": "Houston Corporate Campus",
                    "code": "HOU001",
                    "address": "1200 Smith Street, Houston, TX 77002",
                    "type": "Corporate",
                    "established": "2020-01-10",
                    "employees": 280,
                    "operating_hours": "8AM-6PM",
                    "certifications": ["LEED Gold", "Energy Star"]
                },
                "buildings": [
                    {
                        "name": "Corporate Tower",
                        "code": "HOU001-CT",
                        "type": "Office",
                        "area_sqft": 180000,
                        "built_year": 2020,
                        "floors": [
                            {
                                "name": "Ground Floor Lobby",
                                "level": 0,
                                "area_sqft": 15000,
                                "ceiling_height": 20,
                                "areas": [
                                    {
                                        "name": "Main Reception",
                                        "type": "Reception",
                                        "area_sqft": 3000,
                                        "reception_desks": 3
                                    },
                                    {
                                        "name": "Security Office",
                                        "type": "Security",
                                        "area_sqft": 800,
                                        "monitoring_stations": 4
                                    },
                                    {
                                        "name": "Visitor Center",
                                        "type": "Reception",
                                        "area_sqft": 2000,
                                        "visitor_capacity": 50
                                    },
                                    {
                                        "name": "Building Services",
                                        "type": "Maintenance",
                                        "area_sqft": 1500
                                    }
                                ]
                            },
                            {
                                "name": "Second Floor",
                                "level": 2,
                                "area_sqft": 20000,
                                "areas": [
                                    {
                                        "name": "IT Department",
                                        "type": "Office",
                                        "area_sqft": 8000,
                                        "workstations": 45
                                    },
                                    {
                                        "name": "Data Center",
                                        "type": "DataCenter",
                                        "area_sqft": 3000,
                                        "server_racks": 24,
                                        "temperature_controlled": True
                                    },
                                    {
                                        "name": "Conference Center",
                                        "type": "Meeting",
                                        "area_sqft": 6000,
                                        "rooms": 8,
                                        "video_conferencing": True
                                    }
                                ]
                            },
                            {
                                "name": "Executive Floor",
                                "level": 15,
                                "area_sqft": 18000,
                                "areas": [
                                    {
                                        "name": "Executive Offices",
                                        "type": "Office",
                                        "area_sqft": 10000,
                                        "private_offices": 12
                                    },
                                    {
                                        "name": "Boardroom",
                                        "type": "Meeting",
                                        "area_sqft": 2000,
                                        "capacity": 20
                                    },
                                    {
                                        "name": "Executive Assistant Area",
                                        "type": "Office", 
                                        "area_sqft": 1500,
                                        "workstations": 8
                                    }
                                ]
                            }
                        ]
                    },
                    {
                        "name": "Parking Garage",
                        "code": "HOU001-PG",
                        "type": "Parking",
                        "area_sqft": 150000,
                        "built_year": 2020,
                        "floors": [
                            {
                                "name": "Ground Level",
                                "level": 0,
                                "area_sqft": 50000,
                                "areas": [
                                    {
                                        "name": "Visitor Parking",
                                        "type": "Parking",
                                        "area_sqft": 15000,
                                        "parking_spaces": 100
                                    },
                                    {
                                        "name": "EV Charging Station",
                                        "type": "Utility",
                                        "area_sqft": 2000,
                                        "charging_ports": 20
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        }
    
    def _get_metrics_templates(self) -> Dict[str, Any]:
        """Get EHS metrics templates"""
        return {
            "safety": [
                {
                    "metric_name": "Total Recordable Injury Rate",
                    "metric_code": "TRIR",
                    "category": "Safety",
                    "unit": "per 100 FTE",
                    "frequency": "monthly",
                    "target_value": 1.0,
                    "calculation": "incidents per 200,000 hours worked"
                },
                {
                    "metric_name": "Lost Time Injury Rate",
                    "metric_code": "LTIR", 
                    "category": "Safety",
                    "unit": "per 100 FTE",
                    "frequency": "monthly",
                    "target_value": 0.5,
                    "calculation": "lost time injuries per 200,000 hours worked"
                },
                {
                    "metric_name": "Near Miss Reports",
                    "metric_code": "NMR",
                    "category": "Safety",
                    "unit": "count",
                    "frequency": "monthly",
                    "target_value": 50,
                    "calculation": "total near miss reports submitted"
                },
                {
                    "metric_name": "Safety Training Hours",
                    "metric_code": "STH",
                    "category": "Safety",
                    "unit": "hours per employee",
                    "frequency": "monthly",
                    "target_value": 4.0,
                    "calculation": "total safety training hours / total employees"
                }
            ],
            "environmental": [
                {
                    "metric_name": "Water Usage",
                    "metric_code": "WU",
                    "category": "Environmental",
                    "unit": "gallons per unit produced",
                    "frequency": "monthly",
                    "target_value": 50.0,
                    "calculation": "total water consumed / units produced"
                },
                {
                    "metric_name": "Energy Consumption",
                    "metric_code": "EC",
                    "category": "Environmental",
                    "unit": "kWh per unit produced",
                    "frequency": "monthly",
                    "target_value": 2.5,
                    "calculation": "total energy used / units produced"
                },
                {
                    "metric_name": "Waste Diversion Rate",
                    "metric_code": "WDR",
                    "category": "Environmental", 
                    "unit": "percentage",
                    "frequency": "monthly",
                    "target_value": 85.0,
                    "calculation": "recycled waste / total waste * 100"
                },
                {
                    "metric_name": "CO2 Emissions",
                    "metric_code": "CO2",
                    "category": "Environmental",
                    "unit": "tons CO2e per unit produced",
                    "frequency": "monthly",
                    "target_value": 0.1,
                    "calculation": "total CO2 equivalent emissions / units produced"
                }
            ],
            "health": [
                {
                    "metric_name": "Employee Health Index",
                    "metric_code": "EHI",
                    "category": "Health",
                    "unit": "score (0-100)",
                    "frequency": "quarterly",
                    "target_value": 85.0,
                    "calculation": "composite score based on health indicators"
                },
                {
                    "metric_name": "Air Quality Index",
                    "metric_code": "AQI",
                    "category": "Health",
                    "unit": "AQI score",
                    "frequency": "daily",
                    "target_value": 50.0,
                    "calculation": "composite air quality measurement"
                },
                {
                    "metric_name": "Ergonomic Assessments Completed",
                    "metric_code": "EAC",
                    "category": "Health",
                    "unit": "percentage of workstations",
                    "frequency": "monthly",
                    "target_value": 100.0,
                    "calculation": "assessed workstations / total workstations * 100"
                }
            ]
        }
    
    def _get_incident_templates(self) -> List[Dict[str, Any]]:
        """Get incident data templates"""
        return [
            {
                "incident_type": "Slip/Trip/Fall",
                "severity": "Minor",
                "description_template": "Employee slipped on {surface} in {area}",
                "root_causes": ["Wet floor", "Uneven surface", "Inadequate lighting", "Improper footwear"],
                "corrective_actions": ["Install non-slip materials", "Improve drainage", "Add warning signs", "Enhance lighting"]
            },
            {
                "incident_type": "Chemical Exposure",
                "severity": "Moderate", 
                "description_template": "Employee exposed to {chemical} during {activity}",
                "root_causes": ["PPE failure", "Ventilation inadequate", "Procedure not followed", "Equipment malfunction"],
                "corrective_actions": ["Replace PPE", "Improve ventilation", "Retrain personnel", "Equipment maintenance"]
            },
            {
                "incident_type": "Equipment Injury",
                "severity": "Moderate",
                "description_template": "Employee injured by {equipment} during {operation}",
                "root_causes": ["Machine guard missing", "Lockout/tagout not followed", "Training inadequate", "Maintenance overdue"],
                "corrective_actions": ["Install guards", "Enhance LOTO procedures", "Additional training", "Preventive maintenance"]
            },
            {
                "incident_type": "Environmental Release",
                "severity": "Minor",
                "description_template": "Small release of {substance} from {source}",
                "root_causes": ["Seal failure", "Overfilling", "Corrosion", "Human error"],
                "corrective_actions": ["Replace seals", "Install level indicators", "Inspection program", "Procedure update"]
            }
        ]
    
    def cleanup_existing_test_data(self) -> bool:
        """Clean up any existing test data"""
        try:
            self.logger.info("Cleaning up existing test data...")
            
            cleanup_queries = [
                # Remove test incidents
                "MATCH (i:Incident) WHERE i.source = 'test_data' DETACH DELETE i",
                
                # Remove test metrics
                "MATCH (m:EHSMetric) WHERE m.source = 'test_data' DETACH DELETE m",
                
                # Remove test facilities mapped to areas
                "MATCH (f:Facility)-[:LOCATED_IN]->(:Area) WHERE f.source = 'test_data' DETACH DELETE f",
                
                # Remove test areas
                "MATCH (a:Area) WHERE a.source = 'test_data' DETACH DELETE a",
                
                # Remove test floors
                "MATCH (f:Floor) WHERE f.source = 'test_data' DETACH DELETE f",
                
                # Remove test buildings
                "MATCH (b:Building) WHERE b.source = 'test_data' DETACH DELETE b",
                
                # Remove test sites
                "MATCH (s:Site) WHERE s.source = 'test_data' DETACH DELETE s"
            ]
            
            for query in cleanup_queries:
                result = self.graph.query(query)
                self.logger.debug(f"Executed cleanup: {query}")
            
            self.logger.info("Test data cleanup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup existing test data: {e}")
            return False
    
    def create_test_hierarchy(self) -> bool:
        """Create test site hierarchy"""
        try:
            self.logger.info("Creating test site hierarchy...")
            
            for site_key, site_info in self.site_data.items():
                site = site_info["site"]
                
                # Create site
                site_query = """
                MERGE (s:Site {code: $code})
                SET s.name = $name,
                    s.address = $address,
                    s.type = $type,
                    s.established = date($established),
                    s.employees = $employees,
                    s.operating_hours = $operating_hours,
                    s.certifications = $certifications,
                    s.source = 'test_data',
                    s.created_at = datetime(),
                    s.updated_at = datetime()
                RETURN elementId(s) as site_id
                """
                
                site_result = self.graph.query(site_query, site)
                site_id = site_result[0]['site_id']
                
                self.logger.info(f"Created test site: {site['name']}")
                
                # Create buildings
                for building_data in site_info["buildings"]:
                    building_query = """
                    MATCH (s:Site {code: $site_code})
                    MERGE (b:Building {site_code: $site_code, name: $name})
                    SET b.code = $code,
                        b.type = $type,
                        b.area_sqft = $area_sqft,
                        b.built_year = $built_year,
                        b.source = 'test_data',
                        b.created_at = datetime(),
                        b.updated_at = datetime()
                    MERGE (s)-[:CONTAINS]->(b)
                    RETURN elementId(b) as building_id
                    """
                    
                    building_params = building_data.copy()
                    building_params['site_code'] = site['code']
                    
                    building_result = self.graph.query(building_query, building_params)
                    building_id = building_result[0]['building_id']
                    
                    self.logger.info(f"  Created building: {building_data['name']}")
                    
                    # Create floors
                    for floor_data in building_data.get("floors", []):
                        floor_query = """
                        MATCH (b:Building {site_code: $site_code, name: $building_name})
                        MERGE (f:Floor {building_id: elementId(b), name: $name})
                        SET f.level = $level,
                            f.area_sqft = $area_sqft,
                            f.ceiling_height = $ceiling_height,
                            f.source = 'test_data',
                            f.created_at = datetime(),
                            f.updated_at = datetime()
                        MERGE (b)-[:CONTAINS]->(f)
                        RETURN elementId(f) as floor_id
                        """
                        
                        floor_params = {
                            'site_code': site['code'],
                            'building_name': building_data['name'],
                            'name': floor_data['name'],
                            'level': floor_data['level'],
                            'area_sqft': floor_data.get('area_sqft', 0),
                            'ceiling_height': floor_data.get('ceiling_height', 10)
                        }
                        
                        floor_result = self.graph.query(floor_query, floor_params)
                        floor_id = floor_result[0]['floor_id']
                        
                        self.logger.info(f"    Created floor: {floor_data['name']}")
                        
                        # Create areas
                        for area_data in floor_data.get("areas", []):
                            area_query = """
                            MATCH (b:Building {site_code: $site_code, name: $building_name})
                            MATCH (f:Floor {building_id: elementId(b), name: $floor_name})
                            MERGE (a:Area {floor_id: elementId(f), name: $name})
                            SET a.type = $type,
                                a.area_sqft = $area_sqft,
                                a.source = 'test_data',
                                a.created_at = datetime(),
                                a.updated_at = datetime()
                            """
                            
                            # Add all additional properties from area_data
                            for key, value in area_data.items():
                                if key not in ['name', 'type', 'area_sqft']:
                                    area_query += f", a.{key} = ${key}"
                            
                            area_query += " MERGE (f)-[:CONTAINS]->(a)"
                            
                            area_params = {
                                'site_code': site['code'],
                                'building_name': building_data['name'],
                                'floor_name': floor_data['name'],
                                'name': area_data['name'],
                                'type': area_data['type'],
                                'area_sqft': area_data.get('area_sqft', 0)
                            }
                            
                            # Add additional properties
                            for key, value in area_data.items():
                                if key not in ['name', 'type', 'area_sqft']:
                                    area_params[key] = value
                            
                            self.graph.query(area_query, area_params)
                            
                            self.logger.info(f"      Created area: {area_data['name']}")
                            
                            # Create test facilities in some areas
                            if area_data['type'] in ['Production', 'Storage', 'Office']:
                                facility_name = f"{area_data['name']} Facility"
                                facility_query = """
                                MATCH (a:Area {name: $area_name})
                                WHERE a.source = 'test_data'
                                MERGE (f:Facility {name: $facility_name})
                                SET f.type = $area_type,
                                    f.capacity = $capacity,
                                    f.source = 'test_data',
                                    f.created_at = datetime(),
                                    f.updated_at = datetime()
                                MERGE (f)-[:LOCATED_IN]->(a)
                                """
                                
                                capacity = area_data.get('capacity_units_per_hour', area_data.get('storage_capacity_tons', 100))
                                
                                facility_params = {
                                    'area_name': area_data['name'],
                                    'facility_name': facility_name,
                                    'area_type': area_data['type'],
                                    'capacity': capacity
                                }
                                
                                self.graph.query(facility_query, facility_params)
                                self.logger.info(f"        Created facility: {facility_name}")
            
            self.logger.info("Test site hierarchy creation completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create test hierarchy: {e}")
            return False
    
    def create_test_metrics(self) -> bool:
        """Create test EHS metrics data"""
        if not self.config.seed_ehs_metrics:
            self.logger.info("Skipping EHS metrics creation (disabled)")
            return True
            
        try:
            self.logger.info("Creating test EHS metrics...")
            
            # Get all areas to attach metrics to
            areas_query = """
            MATCH (s:Site)-[:CONTAINS]->(b:Building)-[:CONTAINS]->(f:Floor)-[:CONTAINS]->(a:Area)
            WHERE s.source = 'test_data'
            RETURN s.code as site_code, b.name as building_name, 
                   f.name as floor_name, a.name as area_name,
                   elementId(a) as area_id
            """
            
            areas = self.graph.query(areas_query)
            
            # Create metrics for each area
            for area in areas:
                self.logger.info(f"Creating metrics for area: {area['area_name']}")
                
                for category, metrics_list in self.metrics_templates.items():
                    for metric_template in metrics_list:
                        
                        # Generate historical data
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=self.config.metrics_date_range_days)
                        
                        current_date = start_date
                        while current_date <= end_date:
                            # Generate realistic values with some variance
                            target = metric_template['target_value']
                            if category == 'safety':
                                # Safety metrics should generally be below targets
                                value = max(0, target * (0.3 + random.random() * 0.7))
                            elif category == 'environmental':
                                # Environmental metrics vary around targets
                                value = target * (0.8 + random.random() * 0.4)
                            else:
                                # Health metrics generally aim to meet or exceed targets
                                value = target * (0.9 + random.random() * 0.2)
                            
                            metric_query = """
                            MATCH (a:Area) WHERE elementId(a) = $area_id
                            CREATE (m:EHSMetric {
                                metric_name: $metric_name,
                                metric_code: $metric_code,
                                category: $category,
                                value: $value,
                                unit: $unit,
                                target_value: $target_value,
                                recorded_date: date($recorded_date),
                                site_code: $site_code,
                                building_name: $building_name,
                                floor_name: $floor_name,
                                area_name: $area_name,
                                frequency: $frequency,
                                calculation: $calculation,
                                source: 'test_data',
                                created_at: datetime()
                            })
                            CREATE (m)-[:MEASURED_AT]->(a)
                            """
                            
                            metric_params = {
                                'area_id': area['area_id'],
                                'metric_name': metric_template['metric_name'],
                                'metric_code': metric_template['metric_code'],
                                'category': metric_template['category'],
                                'value': round(value, 2),
                                'unit': metric_template['unit'],
                                'target_value': metric_template['target_value'],
                                'recorded_date': current_date.strftime('%Y-%m-%d'),
                                'site_code': area['site_code'],
                                'building_name': area['building_name'],
                                'floor_name': area['floor_name'],
                                'area_name': area['area_name'],
                                'frequency': metric_template['frequency'],
                                'calculation': metric_template['calculation']
                            }
                            
                            self.graph.query(metric_query, metric_params)
                            
                            # Increment date based on frequency
                            if metric_template['frequency'] == 'daily':
                                current_date += timedelta(days=1)
                            elif metric_template['frequency'] == 'monthly':
                                current_date += timedelta(days=30)
                            elif metric_template['frequency'] == 'quarterly':
                                current_date += timedelta(days=90)
                            else:
                                current_date += timedelta(days=30)  # Default monthly
            
            self.logger.info("Test EHS metrics creation completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create test metrics: {e}")
            return False
    
    def create_test_incidents(self) -> bool:
        """Create test incident data"""
        if not self.config.seed_incidents:
            self.logger.info("Skipping incident creation (disabled)")
            return True
            
        try:
            self.logger.info("Creating test incident data...")
            
            # Get all areas to create incidents for
            areas_query = """
            MATCH (s:Site)-[:CONTAINS]->(b:Building)-[:CONTAINS]->(f:Floor)-[:CONTAINS]->(a:Area)
            WHERE s.source = 'test_data'
            RETURN s.code as site_code, b.name as building_name, 
                   f.name as floor_name, a.name as area_name,
                   a.type as area_type, elementId(a) as area_id
            """
            
            areas = self.graph.query(areas_query)
            
            # Create incidents for manufacturing areas (higher frequency) and office areas (lower frequency)
            for area in areas:
                area_type = area['area_type']
                
                # Determine incident frequency based on area type
                if area_type in ['Production', 'Manufacturing', 'Storage', 'Hazmat']:
                    incident_count = random.randint(2, 8)  # More incidents in high-risk areas
                elif area_type in ['Office', 'Meeting', 'Reception']:
                    incident_count = random.randint(0, 2)  # Fewer incidents in office areas
                else:
                    incident_count = random.randint(0, 4)  # Medium risk areas
                
                for _ in range(incident_count):
                    # Select random incident template
                    incident_template = random.choice(self.incident_templates)
                    
                    # Generate incident date within the last 90 days
                    incident_date = datetime.now() - timedelta(days=random.randint(1, 90))
                    
                    # Generate incident details
                    incident_id = f"INC-{datetime.now().strftime('%Y%m')}-{str(uuid4())[:8].upper()}"
                    
                    # Customize description based on area
                    description_vars = {
                        'surface': random.choice(['wet floor', 'oily surface', 'uneven ground']),
                        'area': area['area_name'],
                        'chemical': random.choice(['cleaning solvent', 'lubricant', 'coolant']),
                        'activity': random.choice(['maintenance', 'cleaning', 'operation']),
                        'equipment': random.choice(['conveyor', 'press', 'forklift', 'pump']),
                        'operation': random.choice(['startup', 'maintenance', 'cleaning']),
                        'substance': random.choice(['oil', 'coolant', 'water']),
                        'source': random.choice(['tank', 'pipe', 'valve'])
                    }
                    
                    description = incident_template['description_template'].format(**description_vars)
                    root_cause = random.choice(incident_template['root_causes'])
                    corrective_action = random.choice(incident_template['corrective_actions'])
                    
                    # Create incident
                    incident_query = """
                    MATCH (a:Area) WHERE elementId(a) = $area_id
                    CREATE (i:Incident {
                        incident_id: $incident_id,
                        incident_type: $incident_type,
                        severity: $severity,
                        description: $description,
                        root_cause: $root_cause,
                        corrective_action: $corrective_action,
                        status: $status,
                        incident_date: date($incident_date),
                        reported_date: date($reported_date),
                        site_code: $site_code,
                        building_name: $building_name,
                        floor_name: $floor_name,
                        area_name: $area_name,
                        area_type: $area_type,
                        injured_person_count: $injured_person_count,
                        days_lost: $days_lost,
                        cost_estimate: $cost_estimate,
                        source: 'test_data',
                        created_at: datetime(),
                        updated_at: datetime()
                    })
                    CREATE (i)-[:OCCURRED_AT]->(a)
                    """
                    
                    # Generate additional incident details
                    injured_count = 1 if incident_template['severity'] in ['Moderate', 'Severe'] else 0
                    days_lost = random.randint(0, 14) if incident_template['severity'] == 'Moderate' else (random.randint(15, 60) if incident_template['severity'] == 'Severe' else 0)
                    cost_estimate = random.randint(100, 10000) if incident_template['severity'] != 'Minor' else random.randint(0, 500)
                    
                    incident_params = {
                        'area_id': area['area_id'],
                        'incident_id': incident_id,
                        'incident_type': incident_template['incident_type'],
                        'severity': incident_template['severity'],
                        'description': description,
                        'root_cause': root_cause,
                        'corrective_action': corrective_action,
                        'status': random.choice(['Open', 'In Progress', 'Closed']),
                        'incident_date': incident_date.strftime('%Y-%m-%d'),
                        'reported_date': (incident_date + timedelta(days=random.randint(0, 2))).strftime('%Y-%m-%d'),
                        'site_code': area['site_code'],
                        'building_name': area['building_name'],
                        'floor_name': area['floor_name'],
                        'area_name': area['area_name'],
                        'area_type': area['area_type'],
                        'injured_person_count': injured_count,
                        'days_lost': days_lost,
                        'cost_estimate': cost_estimate
                    }
                    
                    self.graph.query(incident_query, incident_params)
                
                if incident_count > 0:
                    self.logger.info(f"Created {incident_count} incidents for area: {area['area_name']}")
            
            self.logger.info("Test incident data creation completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create test incidents: {e}")
            return False
    
    def validate_test_data(self) -> bool:
        """Validate the created test data"""
        try:
            self.logger.info("Validating test data creation...")
            
            validation_queries = [
                ("Test Sites", "MATCH (s:Site) WHERE s.source = 'test_data' RETURN COUNT(s) as count"),
                ("Test Buildings", "MATCH (b:Building) WHERE b.source = 'test_data' RETURN COUNT(b) as count"),
                ("Test Floors", "MATCH (f:Floor) WHERE f.source = 'test_data' RETURN COUNT(f) as count"),
                ("Test Areas", "MATCH (a:Area) WHERE a.source = 'test_data' RETURN COUNT(a) as count"),
                ("Test Facilities", "MATCH (f:Facility) WHERE f.source = 'test_data' RETURN COUNT(f) as count"),
                ("Test Metrics", "MATCH (m:EHSMetric) WHERE m.source = 'test_data' RETURN COUNT(m) as count"),
                ("Test Incidents", "MATCH (i:Incident) WHERE i.source = 'test_data' RETURN COUNT(i) as count")
            ]
            
            validation_passed = True
            for name, query in validation_queries:
                result = self.graph.query(query)
                count = result[0]['count'] if result else 0
                
                if count > 0:
                    self.logger.info(f"✓ {name}: {count} created")
                else:
                    self.logger.warning(f"⚠ {name}: {count} created")
                    if name in ['Test Sites', 'Test Buildings', 'Test Areas']:  # Critical elements
                        validation_passed = False
            
            # Check hierarchy relationships
            hierarchy_query = """
            MATCH path = (s:Site)-[:CONTAINS]->(b:Building)-[:CONTAINS]->(f:Floor)-[:CONTAINS]->(a:Area)
            WHERE s.source = 'test_data'
            RETURN COUNT(path) as complete_paths
            """
            
            result = self.graph.query(hierarchy_query)
            complete_paths = result[0]['complete_paths'] if result else 0
            
            if complete_paths > 0:
                self.logger.info(f"✓ Complete hierarchy paths: {complete_paths}")
            else:
                self.logger.error("✗ No complete hierarchy paths found")
                validation_passed = False
            
            return validation_passed
            
        except Exception as e:
            self.logger.error(f"Test data validation failed: {e}")
            return False
    
    def run_seeding(self) -> bool:
        """Run the complete test data seeding process"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("STARTING TEST DATA SEEDING")
            self.logger.info("=" * 80)
            self.logger.info(f"Seeding ID: {self.seeding_id}")
            
            # Connect to Neo4j
            if not self.connect_to_neo4j():
                return False
            
            seeding_steps = [
                ("Cleanup", self.cleanup_existing_test_data),
                ("Hierarchy Creation", self.create_test_hierarchy),
                ("EHS Metrics", self.create_test_metrics),
                ("Incidents", self.create_test_incidents),
                ("Validation", self.validate_test_data)
            ]
            
            for step_name, step_function in seeding_steps:
                self.logger.info(f"\n--- {step_name} ---")
                
                try:
                    success = step_function()
                    if success:
                        self.logger.info(f"✓ {step_name} completed successfully")
                    else:
                        self.logger.error(f"✗ {step_name} failed")
                        return False
                        
                except Exception as e:
                    self.logger.error(f"✗ {step_name} failed with exception: {e}")
                    return False
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("TEST DATA SEEDING COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Seeding process failed: {e}")
            return False

def load_config() -> SeedingConfig:
    """Load seeding configuration"""
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(env_path)
    
    return SeedingConfig(
        neo4j_uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        neo4j_username=os.getenv('NEO4J_USERNAME', 'neo4j'),
        neo4j_password=os.getenv('NEO4J_PASSWORD', 'EhsAI2024!'),
        neo4j_database=os.getenv('NEO4J_DATABASE', 'neo4j'),
        cleanup_existing='--cleanup' in sys.argv or '--clean' in sys.argv,
        seed_ehs_metrics='--no-metrics' not in sys.argv,
        seed_incidents='--no-incidents' not in sys.argv,
        metrics_date_range_days=int(os.getenv('METRICS_DATE_RANGE_DAYS', '90'))
    )

def main():
    """Main execution function"""
    try:
        # Load configuration
        config = load_config()
        
        # Create seeder
        seeder = TestDataSeeder(config)
        
        # Run seeding
        success = seeder.run_seeding()
        
        if success:
            print("\n✓ Test data seeding completed successfully!")
            print("\nUsage:")
            print("  python3 validate_migration.py  # Validate the test data")
            sys.exit(0)
        else:
            print("\n✗ Test data seeding failed. Check the log file for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠ Seeding interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Seeding failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()