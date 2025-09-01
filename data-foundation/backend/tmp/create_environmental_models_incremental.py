#!/usr/bin/env python3
"""
Neo4j Environmental Data Models Script (Incremental version)

This script creates the environmental data models needed for LLM assessment in Neo4j,
but creates data incrementally to avoid transaction conflicts.

Created: 2025-08-30
Version: 1.0.1
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
        logging.FileHandler('tmp/environmental_models_creation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EnvironmentalModelsCreator:
    """Creates Neo4j environmental data models for EHS system"""
    
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
    
    def create_constraints(self) -> None:
        """Create uniqueness constraints for environmental nodes"""
        constraints = [
            # ElectricityConsumption constraints
            "CREATE CONSTRAINT electricity_consumption_id IF NOT EXISTS FOR (e:ElectricityConsumption) REQUIRE e.id IS UNIQUE",
            
            # WaterConsumption constraints
            "CREATE CONSTRAINT water_consumption_id IF NOT EXISTS FOR (w:WaterConsumption) REQUIRE w.id IS UNIQUE",
            
            # WasteGeneration constraints
            "CREATE CONSTRAINT waste_generation_id IF NOT EXISTS FOR (wg:WasteGeneration) REQUIRE wg.id IS UNIQUE",
            
            # EnvironmentalRisk constraints
            "CREATE CONSTRAINT environmental_risk_id IF NOT EXISTS FOR (er:EnvironmentalRisk) REQUIRE er.id IS UNIQUE",
            
            # EnvironmentalRecommendation constraints
            "CREATE CONSTRAINT environmental_recommendation_id IF NOT EXISTS FOR (rec:EnvironmentalRecommendation) REQUIRE rec.id IS UNIQUE",
            
            # EnvironmentalKPI constraints
            "CREATE CONSTRAINT environmental_kpi_id IF NOT EXISTS FOR (kpi:EnvironmentalKPI) REQUIRE kpi.id IS UNIQUE",
            
            # ComplianceRecord constraints
            "CREATE CONSTRAINT compliance_record_id IF NOT EXISTS FOR (cr:ComplianceRecord) REQUIRE cr.id IS UNIQUE"
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.info(f"Created constraint: {constraint.split('FOR')[1].split('REQUIRE')[0].strip()}")
                except ClientError as e:
                    if "already exists" in str(e).lower():
                        logger.info(f"Constraint already exists: {constraint.split('FOR')[1].split('REQUIRE')[0].strip()}")
                    else:
                        logger.error(f"Error creating constraint: {e}")
                        raise
    
    def create_indexes(self) -> None:
        """Create indexes for better query performance"""
        indexes = [
            # Date-based indexes for time series data
            "CREATE INDEX electricity_date_idx IF NOT EXISTS FOR (e:ElectricityConsumption) ON (e.date)",
            "CREATE INDEX water_date_idx IF NOT EXISTS FOR (w:WaterConsumption) ON (w.date)",
            "CREATE INDEX waste_date_idx IF NOT EXISTS FOR (wg:WasteGeneration) ON (wg.date)",
            
            # Facility relationship indexes
            "CREATE INDEX electricity_facility_idx IF NOT EXISTS FOR (e:ElectricityConsumption) ON (e.facility_id)",
            "CREATE INDEX water_facility_idx IF NOT EXISTS FOR (w:WaterConsumption) ON (w.facility_id)",
            "CREATE INDEX waste_facility_idx IF NOT EXISTS FOR (wg:WasteGeneration) ON (wg.facility_id)",
            
            # Risk and compliance indexes
            "CREATE INDEX risk_severity_idx IF NOT EXISTS FOR (er:EnvironmentalRisk) ON (er.severity)",
            "CREATE INDEX compliance_status_idx IF NOT EXISTS FOR (cr:ComplianceRecord) ON (cr.status)",
            
            # KPI indexes
            "CREATE INDEX kpi_type_idx IF NOT EXISTS FOR (kpi:EnvironmentalKPI) ON (kpi.type)",
            "CREATE INDEX kpi_date_idx IF NOT EXISTS FOR (kpi:EnvironmentalKPI) ON (kpi.date)",
            
            # Recommendation indexes
            "CREATE INDEX recommendation_status_idx IF NOT EXISTS FOR (rec:EnvironmentalRecommendation) ON (rec.status)",
            "CREATE INDEX recommendation_priority_idx IF NOT EXISTS FOR (rec:EnvironmentalRecommendation) ON (rec.priority)"
        ]
        
        with self.driver.session() as session:
            for index in indexes:
                try:
                    session.run(index)
                    index_name = index.split('FOR')[1].split('ON')[0].strip()
                    logger.info(f"Created index: {index_name}")
                except ClientError as e:
                    if "already exists" in str(e).lower():
                        logger.info(f"Index already exists: {index.split('FOR')[1].split('ON')[0].strip()}")
                    else:
                        logger.error(f"Error creating index: {e}")
                        raise
    
    def clear_existing_environmental_data(self) -> None:
        """Clear any existing environmental data to avoid conflicts"""
        environmental_labels = [
            'ElectricityConsumption',
            'WaterConsumption', 
            'WasteGeneration',
            'EnvironmentalRisk',
            'EnvironmentalRecommendation',
            'EnvironmentalKPI',
            'ComplianceRecord'
        ]

        with self.driver.session() as session:
            for label in environmental_labels:
                result = session.run(f'MATCH (n:{label}) DETACH DELETE n RETURN count(n) as deleted')
                deleted = result.single()['deleted']
                if deleted > 0:
                    logger.info(f'Cleared {deleted} existing {label} nodes')
    
    def create_electricity_data(self, facility_ids: List[str]) -> int:
        """Create electricity consumption data"""
        logger.info("Creating electricity consumption data...")
        base_date = datetime.now() - timedelta(days=30)
        created_count = 0
        
        with self.driver.session() as session:
            for i, facility_id in enumerate(facility_ids[:3]):  # Limit to 3 facilities
                for day in range(30):
                    date = base_date + timedelta(days=day)
                    consumption = 1000 + (i * 200) + (day * 10) + (day % 7 * 50)
                    cost = consumption * 0.12
                    
                    query = """
                    MERGE (e:ElectricityConsumption {id: $id})
                    SET e.facility_id = $facility_id,
                        e.date = date($date),
                        e.consumption_kwh = $consumption,
                        e.cost_usd = $cost,
                        e.meter_reading = $meter_reading,
                        e.peak_demand_kw = $peak_demand,
                        e.created_at = datetime(),
                        e.updated_at = datetime()
                    """
                    
                    try:
                        session.run(query, {
                            'id': f'elec_{facility_id}_{date.strftime("%Y%m%d")}',
                            'facility_id': facility_id,
                            'date': date.strftime('%Y-%m-%d'),
                            'consumption': consumption,
                            'cost': round(cost, 2),
                            'meter_reading': consumption * (day + 1),
                            'peak_demand': consumption * 0.8
                        })
                        created_count += 1
                    except Exception as e:
                        logger.warning(f"Skipped duplicate electricity record: {e}")
        
        logger.info(f"Created {created_count} electricity consumption records")
        return created_count
    
    def create_water_data(self, facility_ids: List[str]) -> int:
        """Create water consumption data"""
        logger.info("Creating water consumption data...")
        base_date = datetime.now() - timedelta(days=30)
        created_count = 0
        
        with self.driver.session() as session:
            for i, facility_id in enumerate(facility_ids[:3]):
                for day in range(30):
                    date = base_date + timedelta(days=day)
                    consumption = 500 + (i * 100) + (day * 5) + (day % 7 * 25)
                    cost = consumption * 0.003
                    
                    query = """
                    MERGE (w:WaterConsumption {id: $id})
                    SET w.facility_id = $facility_id,
                        w.date = date($date),
                        w.consumption_gallons = $consumption,
                        w.cost_usd = $cost,
                        w.source_type = $source_type,
                        w.quality_rating = $quality_rating,
                        w.created_at = datetime(),
                        w.updated_at = datetime()
                    """
                    
                    try:
                        session.run(query, {
                            'id': f'water_{facility_id}_{date.strftime("%Y%m%d")}',
                            'facility_id': facility_id,
                            'date': date.strftime('%Y-%m-%d'),
                            'consumption': consumption,
                            'cost': round(cost, 2),
                            'source_type': ['Municipal', 'Well', 'Recycled'][i % 3],
                            'quality_rating': 'A' if day % 5 == 0 else 'B'
                        })
                        created_count += 1
                    except Exception as e:
                        logger.warning(f"Skipped duplicate water record: {e}")
        
        logger.info(f"Created {created_count} water consumption records")
        return created_count
    
    def create_waste_data(self, facility_ids: List[str]) -> int:
        """Create waste generation data"""
        logger.info("Creating waste generation data...")
        base_date = datetime.now() - timedelta(days=30)
        waste_types = ['Hazardous', 'Non-Hazardous', 'Recyclable', 'Organic']
        created_count = 0
        
        with self.driver.session() as session:
            for i, facility_id in enumerate(facility_ids[:3]):
                for day in range(0, 30, 7):  # Weekly waste data
                    date = base_date + timedelta(days=day)
                    
                    for waste_type in waste_types:
                        amount = 100 + (i * 50) + (day * 2) + (hash(waste_type) % 50)
                        disposal_cost = amount * (2.5 if waste_type == 'Hazardous' else 0.5)
                        
                        query = """
                        MERGE (wg:WasteGeneration {id: $id})
                        SET wg.facility_id = $facility_id,
                            wg.date = date($date),
                            wg.waste_type = $waste_type,
                            wg.amount_pounds = $amount,
                            wg.disposal_method = $disposal_method,
                            wg.disposal_cost_usd = $disposal_cost,
                            wg.contractor = $contractor,
                            wg.created_at = datetime(),
                            wg.updated_at = datetime()
                        """
                        
                        try:
                            session.run(query, {
                                'id': f'waste_{facility_id}_{waste_type}_{date.strftime("%Y%m%d")}',
                                'facility_id': facility_id,
                                'date': date.strftime('%Y-%m-%d'),
                                'waste_type': waste_type,
                                'amount': amount,
                                'disposal_method': 'Incineration' if waste_type == 'Hazardous' else 'Landfill',
                                'disposal_cost': round(disposal_cost, 2),
                                'contractor': f'Waste Management Co. {(i + 1)}'
                            })
                            created_count += 1
                        except Exception as e:
                            logger.warning(f"Skipped duplicate waste record: {e}")
        
        logger.info(f"Created {created_count} waste generation records")
        return created_count
    
    def create_risk_data(self, facility_ids: List[str]) -> int:
        """Create environmental risk data"""
        logger.info("Creating environmental risk data...")
        risk_types = [
            'Air Quality Violation',
            'Water Contamination',
            'Soil Contamination',
            'Noise Pollution',
            'Chemical Spill Risk',
            'Waste Storage Overflow'
        ]
        severities = ['Low', 'Medium', 'High', 'Critical']
        created_count = 0
        
        with self.driver.session() as session:
            for i, facility_id in enumerate(facility_ids[:3]):
                for j, risk_type in enumerate(risk_types):
                    severity = severities[j % len(severities)]
                    probability = ['Low', 'Medium', 'High'][i % 3]
                    impact_score = {'Low': 2, 'Medium': 5, 'High': 8, 'Critical': 10}[severity]
                    
                    query = """
                    MERGE (er:EnvironmentalRisk {id: $id})
                    SET er.facility_id = $facility_id,
                        er.risk_type = $risk_type,
                        er.description = $description,
                        er.severity = $severity,
                        er.probability = $probability,
                        er.impact_score = $impact_score,
                        er.mitigation_status = $mitigation_status,
                        er.identified_date = date($identified_date),
                        er.last_assessment_date = date($last_assessment_date),
                        er.created_at = datetime(),
                        er.updated_at = datetime()
                    """
                    
                    try:
                        session.run(query, {
                            'id': f'risk_{facility_id}_{j}',
                            'facility_id': facility_id,
                            'risk_type': risk_type,
                            'description': f'Potential {risk_type.lower()} at facility {facility_id}',
                            'severity': severity,
                            'probability': probability,
                            'impact_score': impact_score,
                            'mitigation_status': ['Planned', 'In Progress', 'Completed'][i % 3],
                            'identified_date': (datetime.now() - timedelta(days=30 + j)).strftime('%Y-%m-%d'),
                            'last_assessment_date': (datetime.now() - timedelta(days=j)).strftime('%Y-%m-%d')
                        })
                        created_count += 1
                    except Exception as e:
                        logger.warning(f"Skipped duplicate risk record: {e}")
        
        logger.info(f"Created {created_count} environmental risk records")
        return created_count
    
    def create_recommendation_data(self, facility_ids: List[str]) -> int:
        """Create environmental recommendation data"""
        logger.info("Creating environmental recommendation data...")
        recommendations = [
            {
                'type': 'Energy Efficiency',
                'title': 'Install LED Lighting',
                'description': 'Replace fluorescent lights with LED fixtures to reduce energy consumption by 30%'
            },
            {
                'type': 'Water Conservation',
                'title': 'Install Low-Flow Fixtures',
                'description': 'Upgrade to low-flow faucets and toilets to reduce water consumption'
            },
            {
                'type': 'Waste Reduction',
                'title': 'Implement Recycling Program',
                'description': 'Establish comprehensive recycling program to divert 50% of waste from landfills'
            },
            {
                'type': 'Air Quality',
                'title': 'Upgrade Ventilation System',
                'description': 'Install advanced air filtration to improve indoor air quality'
            }
        ]
        
        priorities = ['Low', 'Medium', 'High']
        statuses = ['Proposed', 'Under Review', 'Approved', 'Implemented']
        created_count = 0
        
        with self.driver.session() as session:
            for i, facility_id in enumerate(facility_ids[:3]):
                for j, rec in enumerate(recommendations):
                    estimated_cost = 5000 + (j * 2000) + (i * 1000)
                    estimated_savings = estimated_cost * 0.3 * (j + 1)
                    
                    query = """
                    MERGE (rec:EnvironmentalRecommendation {id: $id})
                    SET rec.facility_id = $facility_id,
                        rec.type = $type,
                        rec.title = $title,
                        rec.description = $description,
                        rec.priority = $priority,
                        rec.status = $status,
                        rec.estimated_cost_usd = $estimated_cost,
                        rec.estimated_savings_usd = $estimated_savings,
                        rec.implementation_timeline_days = $timeline,
                        rec.created_date = date($created_date),
                        rec.target_completion_date = date($target_date),
                        rec.created_at = datetime(),
                        rec.updated_at = datetime()
                    """
                    
                    try:
                        session.run(query, {
                            'id': f'rec_{facility_id}_{j}',
                            'facility_id': facility_id,
                            'type': rec['type'],
                            'title': rec['title'],
                            'description': rec['description'],
                            'priority': priorities[j % len(priorities)],
                            'status': statuses[i % len(statuses)],
                            'estimated_cost': estimated_cost,
                            'estimated_savings': round(estimated_savings, 2),
                            'timeline': 30 + (j * 15),
                            'created_date': (datetime.now() - timedelta(days=15 + j)).strftime('%Y-%m-%d'),
                            'target_date': (datetime.now() + timedelta(days=60 + j * 15)).strftime('%Y-%m-%d')
                        })
                        created_count += 1
                    except Exception as e:
                        logger.warning(f"Skipped duplicate recommendation record: {e}")
        
        logger.info(f"Created {created_count} environmental recommendation records")
        return created_count
    
    def create_kpi_data(self, facility_ids: List[str]) -> int:
        """Create environmental KPI data"""
        logger.info("Creating environmental KPI data...")
        kpi_types = [
            'Energy Efficiency',
            'Water Usage', 
            'Waste Diversion Rate',
            'Carbon Footprint',
            'Compliance Score'
        ]
        
        base_date = datetime.now() - timedelta(days=90)
        created_count = 0
        
        with self.driver.session() as session:
            for i, facility_id in enumerate(facility_ids[:3]):
                for j, kpi_type in enumerate(kpi_types):
                    for month in range(3):  # Last 3 months
                        date = base_date + timedelta(days=month * 30)
                        
                        # Generate realistic KPI values
                        if kpi_type == 'Energy Efficiency':
                            value = 85 + (i * 2) + (month * 1) + (j % 3)
                            target = 90
                            unit = 'percentage'
                        elif kpi_type == 'Water Usage':
                            value = 1000 - (i * 50) - (month * 20)
                            target = 800
                            unit = 'gallons_per_day'
                        elif kpi_type == 'Waste Diversion Rate':
                            value = 60 + (i * 5) + (month * 2)
                            target = 75
                            unit = 'percentage'
                        elif kpi_type == 'Carbon Footprint':
                            value = 500 - (i * 25) - (month * 10)
                            target = 400
                            unit = 'tons_co2_equivalent'
                        else:  # Compliance Score
                            value = 92 + (i * 1) + (month * 0.5)
                            target = 95
                            unit = 'percentage'
                        
                        variance = ((value - target) / target) * 100
                        status = 'Above Target' if value >= target else 'Below Target'
                        
                        # Use timestamp to make IDs unique
                        unique_id = f'kpi_{facility_id}_{kpi_type.replace(" ", "_")}_{date.strftime("%Y%m")}_{int(datetime.now().timestamp())}'
                        
                        query = """
                        MERGE (kpi:EnvironmentalKPI {id: $id})
                        SET kpi.facility_id = $facility_id,
                            kpi.type = $type,
                            kpi.value = $value,
                            kpi.target_value = $target,
                            kpi.unit = $unit,
                            kpi.date = date($date),
                            kpi.performance_status = $status,
                            kpi.variance_percentage = $variance,
                            kpi.created_at = datetime(),
                            kpi.updated_at = datetime()
                        """
                        
                        try:
                            session.run(query, {
                                'id': unique_id,
                                'facility_id': facility_id,
                                'type': kpi_type,
                                'value': round(value, 2),
                                'target': target,
                                'unit': unit,
                                'date': date.strftime('%Y-%m-%d'),
                                'status': status,
                                'variance': round(variance, 2)
                            })
                            created_count += 1
                        except Exception as e:
                            logger.warning(f"Skipped duplicate KPI record: {e}")
        
        logger.info(f"Created {created_count} environmental KPI records")
        return created_count
    
    def create_compliance_data(self, facility_ids: List[str]) -> int:
        """Create compliance record data"""
        logger.info("Creating compliance record data...")
        regulations = [
            'Clean Air Act',
            'Clean Water Act',
            'Resource Conservation and Recovery Act',
            'Toxic Substances Control Act',
            'Occupational Safety and Health Act'
        ]
        
        statuses = ['Compliant', 'Non-Compliant', 'Under Review']
        created_count = 0
        
        with self.driver.session() as session:
            for i, facility_id in enumerate(facility_ids[:3]):
                for j, regulation in enumerate(regulations):
                    status = statuses[j % len(statuses)]
                    compliance_date = datetime.now() - timedelta(days=30 + j * 10)
                    expiry_date = datetime.now() + timedelta(days=365 - j * 30)
                    inspection_date = datetime.now() - timedelta(days=15 + j * 5)
                    fine_amount = 5000 if status == 'Non-Compliant' else 0
                    
                    query = """
                    MERGE (cr:ComplianceRecord {id: $id})
                    SET cr.facility_id = $facility_id,
                        cr.regulation_name = $regulation,
                        cr.regulation_code = $reg_code,
                        cr.status = $status,
                        cr.compliance_date = date($compliance_date),
                        cr.expiry_date = date($expiry_date),
                        cr.inspector_name = $inspector,
                        cr.inspection_date = date($inspection_date),
                        cr.findings = $findings,
                        cr.corrective_actions = $corrective_actions,
                        cr.fine_amount_usd = $fine_amount,
                        cr.created_at = datetime(),
                        cr.updated_at = datetime()
                    """
                    
                    try:
                        session.run(query, {
                            'id': f'comp_{facility_id}_{j}',
                            'facility_id': facility_id,
                            'regulation': regulation,
                            'reg_code': f'{regulation.split()[0].upper()}-{2024}-{j+1:03d}',
                            'status': status,
                            'compliance_date': compliance_date.strftime('%Y-%m-%d'),
                            'expiry_date': expiry_date.strftime('%Y-%m-%d'),
                            'inspector': f'Inspector {chr(65 + j)}. Smith',
                            'inspection_date': inspection_date.strftime('%Y-%m-%d'),
                            'findings': f'Routine inspection findings for {regulation}' if status == 'Compliant' else f'Violation found in {regulation} compliance',
                            'corrective_actions': 'None required' if status == 'Compliant' else f'Must address {regulation.split()[0].lower()} violations within 30 days',
                            'fine_amount': fine_amount
                        })
                        created_count += 1
                    except Exception as e:
                        logger.warning(f"Skipped duplicate compliance record: {e}")
        
        logger.info(f"Created {created_count} compliance records")
        return created_count
    
    def create_relationships(self, facility_ids: List[str]) -> None:
        """Create relationships between environmental data and facilities"""
        logger.info("Creating relationships...")
        relationships = [
            # Connect consumption data to facilities
            """
            MATCH (f:Facility), (e:ElectricityConsumption)
            WHERE f.id = e.facility_id
            MERGE (f)-[:HAS_ELECTRICITY_CONSUMPTION]->(e)
            """,
            
            """
            MATCH (f:Facility), (w:WaterConsumption)
            WHERE f.id = w.facility_id
            MERGE (f)-[:HAS_WATER_CONSUMPTION]->(w)
            """,
            
            """
            MATCH (f:Facility), (wg:WasteGeneration)
            WHERE f.id = wg.facility_id
            MERGE (f)-[:GENERATES_WASTE]->(wg)
            """,
            
            # Connect risks to facilities
            """
            MATCH (f:Facility), (er:EnvironmentalRisk)
            WHERE f.id = er.facility_id
            MERGE (f)-[:HAS_ENVIRONMENTAL_RISK]->(er)
            """,
            
            # Connect recommendations to facilities
            """
            MATCH (f:Facility), (rec:EnvironmentalRecommendation)
            WHERE f.id = rec.facility_id
            MERGE (f)-[:HAS_RECOMMENDATION]->(rec)
            """,
            
            # Connect KPIs to facilities
            """
            MATCH (f:Facility), (kpi:EnvironmentalKPI)
            WHERE f.id = kpi.facility_id
            MERGE (f)-[:HAS_ENVIRONMENTAL_KPI]->(kpi)
            """,
            
            # Connect compliance records to facilities
            """
            MATCH (f:Facility), (cr:ComplianceRecord)
            WHERE f.id = cr.facility_id
            MERGE (f)-[:HAS_COMPLIANCE_RECORD]->(cr)
            """,
        ]
        
        with self.driver.session() as session:
            for relationship in relationships:
                try:
                    result = session.run(relationship)
                    summary = result.consume()
                    logger.info(f"Created/verified {summary.counters.relationships_created} relationships")
                except Exception as e:
                    logger.error(f"Error creating relationship: {e}")
    
    def get_existing_facilities(self) -> List[str]:
        """Get existing facility IDs from the database"""
        with self.driver.session() as session:
            result = session.run("MATCH (f:Facility) RETURN f.id as id LIMIT 10")
            facility_ids = [record['id'] for record in result if record['id'] is not None]
            
            if not facility_ids:
                # Create sample facilities if none exist
                logger.warning("No facilities found. Creating sample facilities.")
                facility_ids = self.create_sample_facilities(session)
            
            return facility_ids
    
    def create_sample_facilities(self, session) -> List[str]:
        """Create sample facilities for testing"""
        facilities = [
            {'id': 'FAC_001', 'name': 'Manufacturing Plant Alpha', 'type': 'Manufacturing'},
            {'id': 'FAC_002', 'name': 'Distribution Center Beta', 'type': 'Warehouse'},
            {'id': 'FAC_003', 'name': 'Office Complex Gamma', 'type': 'Office'}
        ]
        
        facility_ids = []
        for facility in facilities:
            query = """
            MERGE (f:Facility {id: $id})
            SET f.name = $name,
                f.type = $type,
                f.created_at = datetime(),
                f.updated_at = datetime()
            """
            session.run(query, facility)
            facility_ids.append(facility['id'])
            logger.info(f"Created sample facility: {facility['name']}")
        
        return facility_ids
    
    def run_data_verification(self) -> Dict[str, int]:
        """Verify that data was created correctly"""
        logger.info("Running data verification...")
        
        verification_queries = {
            'ElectricityConsumption': "MATCH (e:ElectricityConsumption) RETURN count(e) as count",
            'WaterConsumption': "MATCH (w:WaterConsumption) RETURN count(w) as count",
            'WasteGeneration': "MATCH (wg:WasteGeneration) RETURN count(wg) as count",
            'EnvironmentalRisk': "MATCH (er:EnvironmentalRisk) RETURN count(er) as count",
            'EnvironmentalRecommendation': "MATCH (rec:EnvironmentalRecommendation) RETURN count(rec) as count",
            'EnvironmentalKPI': "MATCH (kpi:EnvironmentalKPI) RETURN count(kpi) as count",
            'ComplianceRecord': "MATCH (cr:ComplianceRecord) RETURN count(cr) as count",
            'Facilities': "MATCH (f:Facility) RETURN count(f) as count",
            'Relationships': "MATCH ()-[r]->() WHERE type(r) CONTAINS 'ENVIRONMENTAL' OR type(r) CONTAINS 'ELECTRICITY' OR type(r) CONTAINS 'WATER' OR type(r) CONTAINS 'WASTE' OR type(r) CONTAINS 'COMPLIANCE' OR type(r) CONTAINS 'RECOMMENDATION' RETURN count(r) as count"
        }
        
        results = {}
        with self.driver.session() as session:
            for label, query in verification_queries.items():
                result = session.run(query)
                count = result.single()['count']
                results[label] = count
                logger.info(f"{label}: {count} records")
        
        return results
    
    def run(self) -> None:
        """Main execution method"""
        try:
            logger.info("Starting Neo4j Environmental Models Creation")
            
            # Clear existing data first
            logger.info("Clearing existing environmental data...")
            self.clear_existing_environmental_data()
            
            # Create constraints
            logger.info("Creating constraints...")
            self.create_constraints()
            
            # Create indexes
            logger.info("Creating indexes...")
            self.create_indexes()
            
            # Get facilities
            facility_ids = self.get_existing_facilities()
            logger.info(f"Found {len(facility_ids)} facilities: {facility_ids}")
            
            # Create sample data incrementally
            logger.info("Creating sample data incrementally...")
            
            electricity_count = self.create_electricity_data(facility_ids)
            water_count = self.create_water_data(facility_ids)
            waste_count = self.create_waste_data(facility_ids)
            risk_count = self.create_risk_data(facility_ids)
            recommendation_count = self.create_recommendation_data(facility_ids)
            kpi_count = self.create_kpi_data(facility_ids)
            compliance_count = self.create_compliance_data(facility_ids)
            
            # Create relationships
            self.create_relationships(facility_ids)
            
            # Verify data creation
            results = self.run_data_verification()
            
            logger.info("Environmental models creation completed successfully!")
            logger.info(f"Data summary: {json.dumps(results, indent=2)}")
            
        except Exception as e:
            logger.error(f"Failed to create environmental models: {e}")
            raise
        finally:
            self.close()


def load_config() -> Dict[str, str]:
    """Load Neo4j configuration from environment or defaults"""
    config = {
        'uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        'username': os.getenv('NEO4J_USERNAME', 'neo4j'),
        'password': os.getenv('NEO4J_PASSWORD', 'password')
    }
    
    # Try to load from .env file if it exists
    env_path = '.env'
    if os.path.exists(env_path):
        logger.info(f"Loading configuration from {env_path}")
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    
                    # Map .env keys to config keys
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
        logger.info("Neo4j Environmental Models Creator starting...")
        
        # Load configuration
        config = load_config()
        logger.info(f"Connecting to Neo4j at {config['uri']}")
        
        # Create and run the environmental models creator
        creator = EnvironmentalModelsCreator(
            uri=config['uri'],
            username=config['username'],
            password=config['password']
        )
        
        creator.run()
        
        print("\n" + "="*50)
        print("Environmental Models Creation Completed!")
        print("="*50)
        print("The following environmental data models have been created:")
        print("- ElectricityConsumption")
        print("- WaterConsumption")
        print("- WasteGeneration")
        print("- EnvironmentalRisk")
        print("- EnvironmentalRecommendation")
        print("- EnvironmentalKPI")
        print("- ComplianceRecord")
        print("\nSample data has been created for testing.")
        print("Check the log file 'tmp/environmental_models_creation.log' for details.")
        
    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
