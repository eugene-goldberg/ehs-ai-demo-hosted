#!/usr/bin/env python3
"""
Facility Setup Script for EHS AI Demo

This script sets up the facility nodes in Neo4j database for the EHS AI Demo.
It creates the 5 facilities (FAC001-FAC005) with proper addresses, metadata,
and relationships. The script is idempotent and safe to run multiple times.

Features:
- Creates facility nodes with complete metadata
- Sets up proper addresses and contact information
- Creates necessary indexes for performance
- Supports both creating and updating facilities
- Comprehensive error handling and logging
- Safe to run multiple times (idempotent)

Author: EHS AI Demo Team
Created: 2025-08-28
Version: 1.0.0
"""

import os
import sys
import logging
import time
import traceback
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date
from pathlib import Path
from dataclasses import dataclass, asdict, field
from contextlib import contextmanager

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import TransientError, ServiceUnavailable, DatabaseError
    from dotenv import load_dotenv
    from data_generation.utils.data_utils import FacilityType, get_facility_profile
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure you're running this script in the virtual environment with required packages installed")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('facility_setup.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class FacilityData:
    """Data class for facility information"""
    # Required fields (no defaults)
    facility_id: str
    name: str
    facility_type: str
    size: str
    employees: int
    operating_hours: int
    street_address: str
    city: str
    state: str
    zip_code: str
    phone: str
    email: str
    manager_name: str
    established_date: date
    square_footage: int
    safety_officer: str
    
    # Optional fields (with defaults)
    country: str = "USA"
    annual_revenue: Optional[int] = None
    environmental_permits: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    shift_count: int = 1
    operates_weekends: bool = False
    seasonal_variations: bool = False
    
    def to_neo4j_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Neo4j insertion"""
        return {
            'facility_id': self.facility_id,
            'name': self.name,
            'facility_type': self.facility_type,
            'size': self.size,
            'employees': self.employees,
            'operating_hours': self.operating_hours,
            'street_address': self.street_address,
            'city': self.city,
            'state': self.state,
            'zip_code': self.zip_code,
            'country': self.country,
            'phone': self.phone,
            'email': self.email,
            'manager_name': self.manager_name,
            'established_date': self.established_date.isoformat(),
            'square_footage': self.square_footage,
            'annual_revenue': self.annual_revenue,
            'safety_officer': self.safety_officer,
            'environmental_permits': self.environmental_permits,
            'certifications': self.certifications,
            'shift_count': self.shift_count,
            'operates_weekends': self.operates_weekends,
            'seasonal_variations': self.seasonal_variations,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }


class FacilitySetupManager:
    """Manager class for setting up facilities in Neo4j"""
    
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j", 
                 max_retries: int = 3, retry_delay: float = 1.0):
        """Initialize the facility setup manager"""
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.driver = None
        self.connection_verified = False
        
        # Facility data
        self.facilities = self._create_facility_data()
        
        # Setup tracking
        self.stats = {
            'facilities_created': 0,
            'facilities_updated': 0,
            'indexes_created': 0,
            'errors': []
        }
    
    def _create_facility_data(self) -> List[FacilityData]:
        """Create the facility data based on the facilities from generate_historical_data.py"""
        facilities = [
            FacilityData(
                facility_id="FAC001",
                name="Manufacturing Plant A",
                facility_type=FacilityType.MANUFACTURING.value,
                size="large",
                employees=500,
                operating_hours=24,
                street_address="1250 Industrial Blvd",
                city="Detroit",
                state="MI",
                zip_code="48201",
                phone="(313) 555-0101",
                email="operations@mfg-plant-a.com",
                manager_name="Sarah Johnson",
                established_date=date(2010, 3, 15),
                square_footage=450000,
                safety_officer="Michael Chen",
                annual_revenue=85000000,
                environmental_permits=["EPA-MI-001", "MDEQ-AIR-2024"],
                certifications=["ISO 14001", "OHSAS 18001", "ISO 9001"],
                shift_count=3,
                operates_weekends=True,
                seasonal_variations=True
            ),
            FacilityData(
                facility_id="FAC002",
                name="Office Complex B",
                facility_type=FacilityType.OFFICE.value,
                size="medium",
                employees=200,
                operating_hours=12,
                street_address="800 Corporate Dr, Suite 200",
                city="Austin",
                state="TX",
                zip_code="73301",
                phone="(512) 555-0202",
                email="admin@office-complex-b.com",
                manager_name="David Rodriguez",
                established_date=date(2015, 8, 20),
                square_footage=75000,
                safety_officer="Lisa Thompson",
                annual_revenue=12000000,
                environmental_permits=["ENERGY-STAR-2024"],
                certifications=["LEED Gold", "ISO 9001"],
                shift_count=1,
                operates_weekends=False,
                seasonal_variations=False
            ),
            FacilityData(
                facility_id="FAC003",
                name="Warehouse C",
                facility_type=FacilityType.WAREHOUSE.value,
                size="large",
                employees=50,
                operating_hours=16,
                street_address="5500 Logistics Parkway",
                city="Memphis",
                state="TN",
                zip_code="38118",
                phone="(901) 555-0303",
                email="operations@warehouse-c.com",
                manager_name="Robert Kim",
                established_date=date(2018, 1, 10),
                square_footage=320000,
                safety_officer="Jennifer Martinez",
                annual_revenue=8500000,
                environmental_permits=["TDEC-SW-2024"],
                certifications=["ISO 9001", "C-TPAT"],
                shift_count=2,
                operates_weekends=True,
                seasonal_variations=True
            ),
            FacilityData(
                facility_id="FAC004",
                name="Data Center D",
                facility_type=FacilityType.MANUFACTURING.value,  # Data centers map to manufacturing in the enum
                size="medium",
                employees=30,
                operating_hours=24,
                street_address="2400 Technology Center Dr",
                city="Ashburn",
                state="VA",
                zip_code="20147",
                phone="(703) 555-0404",
                email="ops@datacenter-d.com",
                manager_name="Amanda Foster",
                established_date=date(2020, 6, 5),
                square_footage=125000,
                safety_officer="James Wilson",
                annual_revenue=25000000,
                environmental_permits=["EPA-VA-DC-001", "ENERGY-STAR-2024"],
                certifications=["ISO 27001", "SOC 2", "PCI DSS", "LEED Silver"],
                shift_count=3,
                operates_weekends=True,
                seasonal_variations=False
            ),
            FacilityData(
                facility_id="FAC005",
                name="Research Lab E",
                facility_type=FacilityType.MANUFACTURING.value,  # Labs map to manufacturing in the enum
                size="small",
                employees=75,
                operating_hours=12,
                street_address="150 Innovation Way",
                city="Cambridge",
                state="MA",
                zip_code="02139",
                phone="(617) 555-0505",
                email="research@lab-e.com",
                manager_name="Dr. Elena Vasquez",
                established_date=date(2019, 11, 12),
                square_footage=85000,
                safety_officer="Dr. Kevin Park",
                annual_revenue=18000000,
                environmental_permits=["EPA-MA-LAB-001", "MHSP-CHEM-2024"],
                certifications=["ISO 17025", "Good Laboratory Practice", "ISO 14001"],
                shift_count=1,
                operates_weekends=False,
                seasonal_variations=False
            )
        ]
        
        return facilities
    
    def connect(self) -> bool:
        """Establish connection to Neo4j database with retry logic"""
        for attempt in range(self.max_retries):
            try:
                self.driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.username, self.password),
                    max_connection_lifetime=30 * 60,  # 30 minutes
                    max_connection_pool_size=50,
                    connection_acquisition_timeout=60  # 60 seconds
                )
                
                # Verify connection
                with self.driver.session(database=self.database) as session:
                    result = session.run("RETURN 1 as test")
                    test_value = result.single()["test"]
                    if test_value == 1:
                        self.connection_verified = True
                        logger.info(f"Successfully connected to Neo4j at {self.uri} (attempt {attempt + 1})")
                        return True
                        
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"Failed to connect to Neo4j after {self.max_retries} attempts: {e}")
                    return False
        
        return False
    
    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    @contextmanager
    def session_scope(self):
        """Context manager for Neo4j sessions with error handling"""
        session = None
        try:
            session = self.driver.session(database=self.database)
            yield session
        except Exception as e:
            logger.error(f"Session error: {e}")
            raise
        finally:
            if session:
                session.close()
    
    def execute_query_with_retry(self, query: str, parameters: Dict = None, session=None) -> Any:
        """Execute Neo4j query with retry logic"""
        if parameters is None:
            parameters = {}
        
        for attempt in range(self.max_retries):
            try:
                if session:
                    result = session.run(query, parameters)
                else:
                    with self.session_scope() as session:
                        result = session.run(query, parameters)
                
                # Consume result to ensure query execution
                result_data = list(result)
                return result_data
                
            except (TransientError, ServiceUnavailable) as e:
                logger.warning(f"Transient error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    logger.error(f"Query failed after {self.max_retries} attempts: {e}")
                    raise
            except Exception as e:
                logger.error(f"Query execution error: {e}")
                logger.error(f"Query: {query[:200]}...")
                raise
    
    def create_indexes(self) -> bool:
        """Create performance indexes for Facility nodes"""
        indexes = [
            # Primary unique constraint
            "CREATE CONSTRAINT facility_id_unique IF NOT EXISTS "
            "FOR (f:Facility) REQUIRE f.facility_id IS UNIQUE",
            
            # Index for facility name searches
            "CREATE INDEX facility_name IF NOT EXISTS "
            "FOR (f:Facility) ON (f.name)",
            
            # Index for facility type filtering
            "CREATE INDEX facility_type IF NOT EXISTS "
            "FOR (f:Facility) ON (f.facility_type)",
            
            # Index for location-based queries
            "CREATE INDEX facility_location IF NOT EXISTS "
            "FOR (f:Facility) ON (f.city, f.state)",
            
            # Index for size-based filtering
            "CREATE INDEX facility_size IF NOT EXISTS "
            "FOR (f:Facility) ON (f.size)",
            
            # Full text search index for facility names and addresses
            "CREATE FULLTEXT INDEX facility_search IF NOT EXISTS "
            "FOR (f:Facility) ON EACH [f.name, f.street_address, f.city]"
        ]
        
        created_count = 0
        try:
            for index_query in indexes:
                try:
                    logger.info(f"Creating index/constraint: {index_query[:60]}...")
                    self.execute_query_with_retry(index_query)
                    created_count += 1
                    time.sleep(0.1)  # Brief pause between index operations
                except Exception as e:
                    if "already exists" in str(e).lower():
                        logger.info("Index/constraint already exists, skipping")
                    else:
                        logger.warning(f"Error creating index/constraint: {e}")
                        
            self.stats['indexes_created'] = created_count
            logger.info(f"Processed {len(indexes)} index/constraint operations ({created_count} created)")
            return True
            
        except Exception as e:
            logger.error(f"Error in index creation process: {e}")
            return False
    
    def facility_exists(self, facility_id: str) -> bool:
        """Check if a facility already exists"""
        query = "MATCH (f:Facility {facility_id: $facility_id}) RETURN count(f) as count"
        try:
            result = self.execute_query_with_retry(query, {'facility_id': facility_id})
            return result[0]['count'] > 0 if result else False
        except Exception as e:
            logger.error(f"Error checking facility existence: {e}")
            return False
    
    def create_facility(self, facility: FacilityData) -> bool:
        """Create a single facility in Neo4j"""
        try:
            facility_data = facility.to_neo4j_dict()
            
            # Check if facility already exists
            if self.facility_exists(facility.facility_id):
                logger.info(f"Facility {facility.facility_id} already exists, updating...")
                return self.update_facility(facility)
            
            # Create facility node
            create_query = """
            CREATE (f:Facility)
            SET f = $facility_data
            RETURN f.facility_id as id, f.name as name
            """
            
            result = self.execute_query_with_retry(create_query, {'facility_data': facility_data})
            
            if result:
                created_facility = result[0]
                logger.info(f"Created facility: {created_facility['id']} - {created_facility['name']}")
                self.stats['facilities_created'] += 1
                return True
            else:
                logger.error(f"Failed to create facility {facility.facility_id}")
                return False
                
        except Exception as e:
            error_msg = f"Error creating facility {facility.facility_id}: {e}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)
            return False
    
    def update_facility(self, facility: FacilityData) -> bool:
        """Update an existing facility in Neo4j"""
        try:
            facility_data = facility.to_neo4j_dict()
            # Update the updated_at timestamp
            facility_data['updated_at'] = datetime.now().isoformat()
            
            # Update facility node (preserve original created_at)
            update_query = """
            MATCH (f:Facility {facility_id: $facility_id})
            SET f += $facility_data,
                f.created_at = COALESCE(f.created_at, $facility_data.created_at)
            RETURN f.facility_id as id, f.name as name
            """
            
            result = self.execute_query_with_retry(update_query, {
                'facility_id': facility.facility_id,
                'facility_data': facility_data
            })
            
            if result:
                updated_facility = result[0]
                logger.info(f"Updated facility: {updated_facility['id']} - {updated_facility['name']}")
                self.stats['facilities_updated'] += 1
                return True
            else:
                logger.error(f"Failed to update facility {facility.facility_id}")
                return False
                
        except Exception as e:
            error_msg = f"Error updating facility {facility.facility_id}: {e}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)
            return False
    
    def setup_all_facilities(self) -> bool:
        """Set up all facilities in the database"""
        try:
            logger.info(f"Setting up {len(self.facilities)} facilities...")
            
            success_count = 0
            for facility in self.facilities:
                if self.create_facility(facility):
                    success_count += 1
                else:
                    logger.error(f"Failed to set up facility {facility.facility_id}")
            
            logger.info(f"Successfully set up {success_count}/{len(self.facilities)} facilities")
            return success_count == len(self.facilities)
            
        except Exception as e:
            error_msg = f"Error in facility setup process: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.stats['errors'].append(error_msg)
            return False
    
    def verify_setup(self) -> Dict[str, Any]:
        """Verify that all facilities were set up correctly"""
        try:
            verification_query = """
            MATCH (f:Facility)
            RETURN 
                count(f) as total_facilities,
                collect(f.facility_id) as facility_ids,
                collect(DISTINCT f.facility_type) as facility_types,
                collect(DISTINCT f.size) as facility_sizes,
                avg(f.employees) as avg_employees,
                sum(f.square_footage) as total_square_footage
            """
            
            result = self.execute_query_with_retry(verification_query)
            
            if result:
                stats = result[0]
                
                # Check if all expected facilities are present
                expected_ids = {f.facility_id for f in self.facilities}
                actual_ids = set(stats['facility_ids'])
                missing_ids = expected_ids - actual_ids
                
                verification_results = {
                    'verification_time': datetime.now().isoformat(),
                    'total_facilities': stats['total_facilities'],
                    'expected_facilities': len(self.facilities),
                    'facility_ids': stats['facility_ids'],
                    'facility_types': stats['facility_types'],
                    'facility_sizes': stats['facility_sizes'],
                    'avg_employees': round(stats['avg_employees'], 1) if stats['avg_employees'] else 0,
                    'total_square_footage': stats['total_square_footage'] or 0,
                    'missing_facilities': list(missing_ids),
                    'setup_complete': len(missing_ids) == 0 and stats['total_facilities'] == len(self.facilities)
                }
                
                return verification_results
            else:
                return {'error': 'No verification data returned'}
                
        except Exception as e:
            logger.error(f"Error during setup verification: {e}")
            return {'error': str(e)}
    
    def generate_setup_report(self) -> str:
        """Generate a comprehensive setup report"""
        verification = self.verify_setup()
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FACILITY SETUP REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Database: {self.uri}/{self.database}")
        report_lines.append("")
        
        # Setup summary
        report_lines.append("SETUP SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Facilities Created: {self.stats['facilities_created']}")
        report_lines.append(f"Facilities Updated: {self.stats['facilities_updated']}")
        report_lines.append(f"Indexes Created: {self.stats['indexes_created']}")
        report_lines.append(f"Errors Encountered: {len(self.stats['errors'])}")
        report_lines.append("")
        
        # Verification results
        if 'error' not in verification:
            report_lines.append("VERIFICATION RESULTS")
            report_lines.append("-" * 40)
            report_lines.append(f"Total Facilities in DB: {verification['total_facilities']}")
            report_lines.append(f"Expected Facilities: {verification['expected_facilities']}")
            report_lines.append(f"Setup Complete: {'✓ YES' if verification['setup_complete'] else '✗ NO'}")
            
            if verification['missing_facilities']:
                report_lines.append(f"Missing Facilities: {', '.join(verification['missing_facilities'])}")
            
            report_lines.append("")
            report_lines.append("FACILITY DETAILS:")
            for facility_id in verification['facility_ids']:
                facility = next((f for f in self.facilities if f.facility_id == facility_id), None)
                if facility:
                    report_lines.append(f"  {facility_id}: {facility.name} ({facility.facility_type})")
            
            report_lines.append("")
            report_lines.append("STATISTICS:")
            report_lines.append(f"  Facility Types: {', '.join(verification['facility_types'])}")
            report_lines.append(f"  Facility Sizes: {', '.join(verification['facility_sizes'])}")
            report_lines.append(f"  Average Employees: {verification['avg_employees']}")
            report_lines.append(f"  Total Square Footage: {verification['total_square_footage']:,}")
        else:
            report_lines.append("VERIFICATION ERROR")
            report_lines.append("-" * 40)
            report_lines.append(f"Error: {verification['error']}")
        
        report_lines.append("")
        
        # Error details
        if self.stats['errors']:
            report_lines.append("ERRORS ENCOUNTERED")
            report_lines.append("-" * 40)
            for i, error in enumerate(self.stats['errors'], 1):
                report_lines.append(f"  {i}. {error}")
        else:
            report_lines.append("No errors encountered during setup.")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def run_setup(self) -> bool:
        """Run the complete facility setup process"""
        try:
            logger.info("Starting facility setup process...")
            
            # Step 1: Create indexes
            logger.info("Step 1: Creating database indexes...")
            if not self.create_indexes():
                logger.warning("Some indexes may not have been created properly")
            
            # Step 2: Set up facilities
            logger.info("Step 2: Setting up facilities...")
            setup_success = self.setup_all_facilities()
            
            # Step 3: Verify setup
            logger.info("Step 3: Verifying setup...")
            verification = self.verify_setup()
            
            # Step 4: Generate report
            logger.info("Step 4: Generating setup report...")
            report = self.generate_setup_report()
            
            # Print report
            print("\n" + report)
            
            # Save report to file
            report_file = f"facility_setup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            logger.info(f"Setup report saved to: {report_file}")
            
            success = setup_success and verification.get('setup_complete', False)
            
            if success:
                logger.info("Facility setup completed successfully!")
            else:
                logger.error("Facility setup completed with issues. Please check the report.")
            
            return success
            
        except Exception as e:
            logger.error(f"Critical error in facility setup: {e}")
            logger.error(traceback.format_exc())
            return False


def main():
    """Main execution function"""
    try:
        logger.info("=" * 80)
        logger.info("EHS AI DEMO - FACILITY SETUP")
        logger.info("=" * 80)
        
        # Get Neo4j credentials from environment
        uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        username = os.getenv('NEO4J_USERNAME', 'neo4j')
        password = os.getenv('NEO4J_PASSWORD', 'EhsAI2024!')
        database = os.getenv('NEO4J_DATABASE', 'neo4j')
        
        if not password or password == 'EhsAI2024!':
            logger.warning("Using default Neo4j password. Consider setting NEO4J_PASSWORD environment variable.")
        
        logger.info(f"Connecting to Neo4j at: {uri}")
        logger.info(f"Database: {database}")
        logger.info("=" * 80)
        
        # Initialize setup manager
        setup_manager = FacilitySetupManager(
            uri=uri,
            username=username,
            password=password,
            database=database
        )
        
        # Connect to database
        if not setup_manager.connect():
            logger.error("Failed to establish connection to Neo4j database")
            sys.exit(1)
        
        # Run setup
        success = setup_manager.run_setup()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up
        if 'setup_manager' in locals():
            setup_manager.close()


if __name__ == '__main__':
    main()