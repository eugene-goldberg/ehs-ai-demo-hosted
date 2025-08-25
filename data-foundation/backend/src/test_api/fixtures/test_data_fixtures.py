"""
Comprehensive Test Data Fixtures for EHS AI Demo

This module provides realistic test data fixtures for all features including:
- Sample documents (utility bills, water bills, waste manifests, permits)
- User data with different roles
- Facility data with various configurations
- Billing periods with different scenarios
- Pro-rating test cases
- Rejection scenarios
- Edge cases and performance test data

Author: EHS AI Demo Team
Created: 2025-08-23
"""

import json
import random
from datetime import datetime, timedelta, date
from decimal import Decimal
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import faker

# Initialize Faker for realistic data generation
fake = faker.Faker()

class DocumentType(Enum):
    UTILITY_BILL = "utility_bill"
    WATER_BILL = "water_bill"
    WASTE_MANIFEST = "waste_manifest"
    PERMIT = "permit"

class UserRole(Enum):
    ADMIN = "admin"
    FACILITY_MANAGER = "facility_manager"
    ACCOUNTANT = "accountant"
    VIEWER = "viewer"

class FacilityType(Enum):
    MANUFACTURING = "manufacturing"
    WAREHOUSE = "warehouse"
    OFFICE = "office"
    LABORATORY = "laboratory"
    MIXED_USE = "mixed_use"

class BillStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ALLOCATED = "allocated"

class RejectionReason(Enum):
    QUALITY_ISSUE = "quality_issue"
    DUPLICATE = "duplicate"
    INVALID_PERIOD = "invalid_period"
    MISSING_DATA = "missing_data"
    INCORRECT_FORMAT = "incorrect_format"

@dataclass
class TestUser:
    """Test user fixture"""
    id: str
    username: str
    email: str
    first_name: str
    last_name: str
    role: UserRole
    created_at: datetime
    is_active: bool = True
    facilities_access: List[str] = None
    
    def __post_init__(self):
        if self.facilities_access is None:
            self.facilities_access = []

@dataclass
class TestFacility:
    """Test facility fixture"""
    id: str
    name: str
    code: str
    facility_type: FacilityType
    address: Dict[str, str]
    square_footage: int
    employee_count: int
    created_at: datetime
    is_active: bool = True
    cost_center: str = ""
    manager_id: Optional[str] = None

@dataclass
class TestUtilityBill:
    """Test utility bill fixture"""
    id: str
    facility_id: str
    document_type: str = "utility_bill"
    utility_type: str = "electricity"  # electricity, gas, steam
    vendor: str = ""
    account_number: str = ""
    billing_period_start: date = None
    billing_period_end: date = None
    usage_amount: Decimal = Decimal('0')
    usage_unit: str = "kWh"
    total_cost: Decimal = Decimal('0')
    demand_charges: Decimal = Decimal('0')
    energy_charges: Decimal = Decimal('0')
    taxes_fees: Decimal = Decimal('0')
    rate_schedule: str = ""
    meter_readings: Dict[str, Any] = None
    created_at: datetime = None
    status: BillStatus = BillStatus.PENDING
    rejection_reason: Optional[RejectionReason] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.meter_readings is None:
            self.meter_readings = {}

@dataclass
class TestWaterBill:
    """Test water bill fixture"""
    id: str
    facility_id: str
    document_type: str = "water_bill"
    vendor: str = ""
    account_number: str = ""
    billing_period_start: date = None
    billing_period_end: date = None
    water_usage_gallons: Decimal = Decimal('0')
    sewer_usage_gallons: Decimal = Decimal('0')
    water_charges: Decimal = Decimal('0')
    sewer_charges: Decimal = Decimal('0')
    service_charges: Decimal = Decimal('0')
    total_cost: Decimal = Decimal('0')
    meter_readings: Dict[str, Any] = None
    created_at: datetime = None
    status: BillStatus = BillStatus.PENDING
    rejection_reason: Optional[RejectionReason] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.meter_readings is None:
            self.meter_readings = {}

@dataclass
class TestWasteManifest:
    """Test waste manifest fixture"""
    id: str
    facility_id: str
    document_type: str = "waste_manifest"
    manifest_number: str = ""
    waste_type: str = ""  # hazardous, non-hazardous, recycling
    waste_description: str = ""
    quantity: Decimal = Decimal('0')
    unit: str = "pounds"
    transporter: str = ""
    disposal_facility: str = ""
    disposal_method: str = ""
    pickup_date: date = None
    disposal_date: date = None
    cost: Decimal = Decimal('0')
    epa_codes: List[str] = None
    created_at: datetime = None
    status: BillStatus = BillStatus.PENDING
    rejection_reason: Optional[RejectionReason] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.epa_codes is None:
            self.epa_codes = []

@dataclass
class TestPermit:
    """Test permit fixture"""
    id: str
    facility_id: str
    document_type: str = "permit"
    permit_number: str = ""
    permit_type: str = ""  # air, water, waste, construction
    issuing_authority: str = ""
    issue_date: date = None
    expiration_date: date = None
    annual_fee: Decimal = Decimal('0')
    renewal_fee: Decimal = Decimal('0')
    status: str = "active"  # active, expired, pending, revoked
    conditions: List[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.conditions is None:
            self.conditions = []

@dataclass
class TestAllocationScenario:
    """Test allocation scenario fixture"""
    id: str
    name: str
    description: str
    facilities: List[str]
    allocation_method: str  # square_footage, employee_count, usage_based, custom
    weights: Dict[str, Decimal]
    billing_period_start: date
    billing_period_end: date
    total_amount: Decimal
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class TestDataFixtures:
    """Main class for generating comprehensive test data fixtures"""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional seed for reproducible data"""
        if seed:
            fake.seed_instance(seed)
            random.seed(seed)
        
        self._users = []
        self._facilities = []
        self._utility_bills = []
        self._water_bills = []
        self._waste_manifests = []
        self._permits = []
        self._allocation_scenarios = []
    
    # User Generation Methods
    def generate_user(self, role: UserRole = None, **kwargs) -> TestUser:
        """Generate a single test user"""
        if role is None:
            role = random.choice(list(UserRole))
        
        user_id = kwargs.get('id', str(uuid.uuid4()))
        first_name = kwargs.get('first_name', fake.first_name())
        last_name = kwargs.get('last_name', fake.last_name())
        username = kwargs.get('username', f"{first_name.lower()}.{last_name.lower()}")
        email = kwargs.get('email', f"{username}@{fake.domain_name()}")
        
        return TestUser(
            id=user_id,
            username=username,
            email=email,
            first_name=first_name,
            last_name=last_name,
            role=role,
            created_at=kwargs.get('created_at', fake.date_time_between(start_date='-2y', end_date='now')),
            is_active=kwargs.get('is_active', True),
            facilities_access=kwargs.get('facilities_access', [])
        )
    
    def generate_users_batch(self, count: int = 10) -> List[TestUser]:
        """Generate a batch of test users with various roles"""
        users = []
        for _ in range(count):
            users.append(self.generate_user())
        self._users.extend(users)
        return users
    
    # Facility Generation Methods
    def generate_facility(self, facility_type: FacilityType = None, **kwargs) -> TestFacility:
        """Generate a single test facility"""
        if facility_type is None:
            facility_type = random.choice(list(FacilityType))
        
        facility_id = kwargs.get('id', str(uuid.uuid4()))
        name = kwargs.get('name', f"{fake.company()} {facility_type.value.replace('_', ' ').title()}")
        code = kwargs.get('code', f"{fake.lexify('???').upper()}{fake.numerify('###')}")
        
        # Size varies by facility type
        size_ranges = {
            FacilityType.MANUFACTURING: (50000, 500000),
            FacilityType.WAREHOUSE: (100000, 1000000),
            FacilityType.OFFICE: (5000, 50000),
            FacilityType.LABORATORY: (10000, 100000),
            FacilityType.MIXED_USE: (20000, 200000)
        }
        
        min_size, max_size = size_ranges.get(facility_type, (10000, 100000))
        square_footage = kwargs.get('square_footage', random.randint(min_size, max_size))
        
        # Employee count roughly correlates with size
        employee_count = kwargs.get('employee_count', max(1, square_footage // random.randint(200, 800)))
        
        address = kwargs.get('address', {
            'street': fake.street_address(),
            'city': fake.city(),
            'state': fake.state_abbr(),
            'zip_code': fake.zipcode(),
            'country': 'USA'
        })
        
        return TestFacility(
            id=facility_id,
            name=name,
            code=code,
            facility_type=facility_type,
            address=address,
            square_footage=square_footage,
            employee_count=employee_count,
            created_at=kwargs.get('created_at', fake.date_time_between(start_date='-3y', end_date='-1y')),
            is_active=kwargs.get('is_active', True),
            cost_center=kwargs.get('cost_center', f"CC-{fake.numerify('####')}"),
            manager_id=kwargs.get('manager_id')
        )
    
    def generate_facilities_batch(self, count: int = 5) -> List[TestFacility]:
        """Generate a batch of test facilities"""
        facilities = []
        for _ in range(count):
            facilities.append(self.generate_facility())
        self._facilities.extend(facilities)
        return facilities
    
    # Utility Bill Generation Methods
    def generate_utility_bill(self, facility_id: str = None, **kwargs) -> TestUtilityBill:
        """Generate a single utility bill"""
        if facility_id is None and self._facilities:
            facility_id = random.choice(self._facilities).id
        elif facility_id is None:
            facility_id = str(uuid.uuid4())
        
        utility_types = ['electricity', 'natural_gas', 'steam', 'propane']
        utility_type = kwargs.get('utility_type', random.choice(utility_types))
        
        # Generate realistic billing period
        end_date = kwargs.get('billing_period_end', 
                             fake.date_between(start_date='-1y', end_date='today'))
        start_date = kwargs.get('billing_period_start',
                               end_date - timedelta(days=random.randint(28, 35)))
        
        # Generate usage based on facility size and type
        base_usage = random.randint(5000, 50000)
        usage_amount = kwargs.get('usage_amount', Decimal(str(base_usage)))
        
        # Generate costs
        rate_per_unit = Decimal(str(random.uniform(0.08, 0.25)))
        energy_charges = kwargs.get('energy_charges', usage_amount * rate_per_unit)
        demand_charges = kwargs.get('demand_charges', energy_charges * Decimal('0.3'))
        taxes_fees = kwargs.get('taxes_fees', energy_charges * Decimal('0.15'))
        total_cost = kwargs.get('total_cost', energy_charges + demand_charges + taxes_fees)
        
        vendors = ['Pacific Gas & Electric', 'ConEd', 'Duke Energy', 'Southern Company', 'ComEd']
        vendor = kwargs.get('vendor', random.choice(vendors))
        
        return TestUtilityBill(
            id=kwargs.get('id', str(uuid.uuid4())),
            facility_id=facility_id,
            utility_type=utility_type,
            vendor=vendor,
            account_number=kwargs.get('account_number', fake.numerify('####-####-####')),
            billing_period_start=start_date,
            billing_period_end=end_date,
            usage_amount=usage_amount,
            usage_unit=kwargs.get('usage_unit', 'kWh' if utility_type == 'electricity' else 'therms'),
            total_cost=total_cost,
            demand_charges=demand_charges,
            energy_charges=energy_charges,
            taxes_fees=taxes_fees,
            rate_schedule=kwargs.get('rate_schedule', f"{random.choice(['A1', 'A6', 'E19', 'E20'])}"),
            meter_readings=kwargs.get('meter_readings', {
                'previous': random.randint(100000, 999999),
                'current': random.randint(1000000, 9999999),
                'multiplier': random.choice([1, 10, 100])
            }),
            status=kwargs.get('status', BillStatus.PENDING),
            rejection_reason=kwargs.get('rejection_reason')
        )
    
    def generate_utility_bills_batch(self, count: int = 20) -> List[TestUtilityBill]:
        """Generate a batch of utility bills"""
        bills = []
        for _ in range(count):
            bills.append(self.generate_utility_bill())
        self._utility_bills.extend(bills)
        return bills
    
    # Water Bill Generation Methods
    def generate_water_bill(self, facility_id: str = None, **kwargs) -> TestWaterBill:
        """Generate a single water bill"""
        if facility_id is None and self._facilities:
            facility_id = random.choice(self._facilities).id
        elif facility_id is None:
            facility_id = str(uuid.uuid4())
        
        # Generate billing period
        end_date = kwargs.get('billing_period_end',
                             fake.date_between(start_date='-1y', end_date='today'))
        start_date = kwargs.get('billing_period_start',
                               end_date - timedelta(days=random.randint(28, 35)))
        
        # Generate usage
        water_usage = kwargs.get('water_usage_gallons', Decimal(str(random.randint(1000, 50000))))
        sewer_usage = kwargs.get('sewer_usage_gallons', water_usage * Decimal('0.8'))
        
        # Generate costs
        water_rate = Decimal(str(random.uniform(0.003, 0.012)))  # per gallon
        sewer_rate = Decimal(str(random.uniform(0.004, 0.015)))  # per gallon
        
        water_charges = kwargs.get('water_charges', water_usage * water_rate)
        sewer_charges = kwargs.get('sewer_charges', sewer_usage * sewer_rate)
        service_charges = kwargs.get('service_charges', Decimal(str(random.uniform(25, 100))))
        total_cost = kwargs.get('total_cost', water_charges + sewer_charges + service_charges)
        
        vendors = ['City Water Department', 'Municipal Water District', 'Regional Water Authority', 'Aqua America']
        vendor = kwargs.get('vendor', random.choice(vendors))
        
        return TestWaterBill(
            id=kwargs.get('id', str(uuid.uuid4())),
            facility_id=facility_id,
            vendor=vendor,
            account_number=kwargs.get('account_number', fake.numerify('####-####')),
            billing_period_start=start_date,
            billing_period_end=end_date,
            water_usage_gallons=water_usage,
            sewer_usage_gallons=sewer_usage,
            water_charges=water_charges,
            sewer_charges=sewer_charges,
            service_charges=service_charges,
            total_cost=total_cost,
            meter_readings=kwargs.get('meter_readings', {
                'water_previous': random.randint(10000, 99999),
                'water_current': random.randint(100000, 999999),
                'sewer_calculated': True
            }),
            status=kwargs.get('status', BillStatus.PENDING),
            rejection_reason=kwargs.get('rejection_reason')
        )
    
    def generate_water_bills_batch(self, count: int = 15) -> List[TestWaterBill]:
        """Generate a batch of water bills"""
        bills = []
        for _ in range(count):
            bills.append(self.generate_water_bill())
        self._water_bills.extend(bills)
        return bills
    
    # Waste Manifest Generation Methods
    def generate_waste_manifest(self, facility_id: str = None, **kwargs) -> TestWasteManifest:
        """Generate a single waste manifest"""
        if facility_id is None and self._facilities:
            facility_id = random.choice(self._facilities).id
        elif facility_id is None:
            facility_id = str(uuid.uuid4())
        
        waste_types = ['hazardous', 'non-hazardous', 'recycling', 'universal']
        waste_type = kwargs.get('waste_type', random.choice(waste_types))
        
        # Generate waste-specific data
        waste_descriptions = {
            'hazardous': ['Used oil', 'Solvent waste', 'Paint waste', 'Chemical residue'],
            'non-hazardous': ['Office waste', 'Cardboard', 'Metal scrap', 'Wood waste'],
            'recycling': ['Paper', 'Plastic', 'Aluminum cans', 'Glass bottles'],
            'universal': ['Fluorescent bulbs', 'Batteries', 'Electronics', 'Mercury devices']
        }
        
        waste_description = kwargs.get('waste_description', 
                                     random.choice(waste_descriptions.get(waste_type, ['General waste'])))
        
        # EPA codes for hazardous waste
        epa_codes = kwargs.get('epa_codes', [])
        if waste_type == 'hazardous' and not epa_codes:
            codes = ['D001', 'D002', 'D008', 'F003', 'F005', 'U002', 'K001']
            epa_codes = random.sample(codes, random.randint(1, 3))
        
        pickup_date = kwargs.get('pickup_date',
                                fake.date_between(start_date='-6m', end_date='today'))
        disposal_date = kwargs.get('disposal_date',
                                  pickup_date + timedelta(days=random.randint(1, 30)))
        
        quantity = kwargs.get('quantity', Decimal(str(random.randint(50, 5000))))
        cost_per_unit = Decimal(str(random.uniform(0.50, 5.00)))
        cost = kwargs.get('cost', quantity * cost_per_unit)
        
        transporters = ['Waste Management', 'Republic Services', 'Clean Harbors', 'Stericycle']
        disposal_facilities = ['Regional Landfill', 'Incineration Facility', 'Recycling Center', 'Treatment Plant']
        
        return TestWasteManifest(
            id=kwargs.get('id', str(uuid.uuid4())),
            facility_id=facility_id,
            manifest_number=kwargs.get('manifest_number', fake.bothify('?####????').upper()),
            waste_type=waste_type,
            waste_description=waste_description,
            quantity=quantity,
            unit=kwargs.get('unit', random.choice(['pounds', 'gallons', 'cubic_yards', 'tons'])),
            transporter=kwargs.get('transporter', random.choice(transporters)),
            disposal_facility=kwargs.get('disposal_facility', random.choice(disposal_facilities)),
            disposal_method=kwargs.get('disposal_method', 
                                     random.choice(['landfill', 'incineration', 'recycling', 'treatment'])),
            pickup_date=pickup_date,
            disposal_date=disposal_date,
            cost=cost,
            epa_codes=epa_codes,
            status=kwargs.get('status', BillStatus.PENDING),
            rejection_reason=kwargs.get('rejection_reason')
        )
    
    def generate_waste_manifests_batch(self, count: int = 10) -> List[TestWasteManifest]:
        """Generate a batch of waste manifests"""
        manifests = []
        for _ in range(count):
            manifests.append(self.generate_waste_manifest())
        self._waste_manifests.extend(manifests)
        return manifests
    
    # Permit Generation Methods
    def generate_permit(self, facility_id: str = None, **kwargs) -> TestPermit:
        """Generate a single permit"""
        if facility_id is None and self._facilities:
            facility_id = random.choice(self._facilities).id
        elif facility_id is None:
            facility_id = str(uuid.uuid4())
        
        permit_types = ['air', 'water', 'waste', 'construction', 'operating']
        permit_type = kwargs.get('permit_type', random.choice(permit_types))
        
        # Generate dates
        issue_date = kwargs.get('issue_date',
                               fake.date_between(start_date='-5y', end_date='-1y'))
        expiration_date = kwargs.get('expiration_date',
                                    issue_date + timedelta(days=random.randint(365, 1825)))
        
        # Generate fees
        annual_fee = kwargs.get('annual_fee', Decimal(str(random.randint(500, 10000))))
        renewal_fee = kwargs.get('renewal_fee', annual_fee * Decimal('0.75'))
        
        # Generate conditions
        condition_templates = [
            'Monitor emissions monthly',
            'Submit annual compliance report',
            'Maintain records for 5 years',
            'Install monitoring equipment',
            'Conduct quarterly inspections'
        ]
        conditions = kwargs.get('conditions', random.sample(condition_templates, random.randint(2, 4)))
        
        authorities = ['EPA', 'State Environmental Agency', 'Local Air District', 'Water Board']
        
        return TestPermit(
            id=kwargs.get('id', str(uuid.uuid4())),
            facility_id=facility_id,
            permit_number=kwargs.get('permit_number', f"{permit_type.upper()}-{fake.numerify('######')}"),
            permit_type=permit_type,
            issuing_authority=kwargs.get('issuing_authority', random.choice(authorities)),
            issue_date=issue_date,
            expiration_date=expiration_date,
            annual_fee=annual_fee,
            renewal_fee=renewal_fee,
            status=kwargs.get('status', random.choice(['active', 'pending', 'expired'])),
            conditions=conditions
        )
    
    def generate_permits_batch(self, count: int = 8) -> List[TestPermit]:
        """Generate a batch of permits"""
        permits = []
        for _ in range(count):
            permits.append(self.generate_permit())
        self._permits.extend(permits)
        return permits
    
    # Allocation Scenario Generation Methods
    def generate_allocation_scenario(self, facilities: List[str] = None, **kwargs) -> TestAllocationScenario:
        """Generate an allocation scenario"""
        if facilities is None and self._facilities:
            facilities = random.sample([f.id for f in self._facilities], 
                                     random.randint(2, min(len(self._facilities), 5)))
        elif facilities is None:
            facilities = [str(uuid.uuid4()) for _ in range(random.randint(2, 4))]
        
        allocation_methods = ['square_footage', 'employee_count', 'usage_based', 'custom']
        allocation_method = kwargs.get('allocation_method', random.choice(allocation_methods))
        
        # Generate weights for facilities
        if allocation_method == 'custom':
            weights = {}
            total_weight = Decimal('100')
            remaining = total_weight
            for i, facility_id in enumerate(facilities):
                if i == len(facilities) - 1:
                    weights[facility_id] = remaining
                else:
                    weight = Decimal(str(random.uniform(10, float(remaining - (len(facilities) - i - 1) * 10))))
                    weights[facility_id] = weight
                    remaining -= weight
        else:
            weights = {facility_id: Decimal('0') for facility_id in facilities}
        
        # Generate billing period
        end_date = kwargs.get('billing_period_end', fake.date_between(start_date='-3m', end_date='today'))
        start_date = kwargs.get('billing_period_start', end_date - timedelta(days=random.randint(28, 35)))
        
        total_amount = kwargs.get('total_amount', Decimal(str(random.randint(5000, 50000))))
        
        scenario_names = [
            'Monthly Utility Allocation',
            'Quarterly Water Distribution',
            'Annual Waste Cost Share',
            'Pro-rated Electricity Bill',
            'Shared Service Allocation'
        ]
        
        return TestAllocationScenario(
            id=kwargs.get('id', str(uuid.uuid4())),
            name=kwargs.get('name', random.choice(scenario_names)),
            description=kwargs.get('description', f"Allocation using {allocation_method.replace('_', ' ')} method"),
            facilities=facilities,
            allocation_method=allocation_method,
            weights=weights,
            billing_period_start=start_date,
            billing_period_end=end_date,
            total_amount=total_amount
        )
    
    def generate_allocation_scenarios_batch(self, count: int = 5) -> List[TestAllocationScenario]:
        """Generate a batch of allocation scenarios"""
        scenarios = []
        for _ in range(count):
            scenarios.append(self.generate_allocation_scenario())
        self._allocation_scenarios.extend(scenarios)
        return scenarios
    
    # Edge Cases and Special Scenarios
    def generate_edge_case_bills(self) -> Dict[str, List[Any]]:
        """Generate edge case bills for boundary testing"""
        edge_cases = {
            'zero_usage': [],
            'negative_charges': [],
            'extreme_dates': [],
            'missing_data': [],
            'boundary_values': []
        }
        
        # Zero usage bills
        for _ in range(3):
            bill = self.generate_utility_bill(
                usage_amount=Decimal('0'),
                energy_charges=Decimal('0'),
                total_cost=Decimal('25.00')  # Only service charge
            )
            edge_cases['zero_usage'].append(bill)
        
        # Bills with adjustments (negative charges)
        for _ in range(2):
            bill = self.generate_utility_bill(
                energy_charges=Decimal('-100.00'),
                total_cost=Decimal('50.00')  # Net positive after adjustments
            )
            edge_cases['negative_charges'].append(bill)
        
        # Extreme dates
        future_bill = self.generate_utility_bill(
            billing_period_start=date.today() + timedelta(days=1),
            billing_period_end=date.today() + timedelta(days=31)
        )
        old_bill = self.generate_utility_bill(
            billing_period_start=date(2020, 1, 1),
            billing_period_end=date(2020, 1, 31)
        )
        edge_cases['extreme_dates'].extend([future_bill, old_bill])
        
        # Missing/incomplete data
        incomplete_bill = self.generate_utility_bill(
            vendor="",
            account_number="",
            usage_amount=Decimal('0')
        )
        edge_cases['missing_data'].append(incomplete_bill)
        
        # Boundary values
        max_usage_bill = self.generate_utility_bill(
            usage_amount=Decimal('999999999.99'),
            total_cost=Decimal('999999999.99')
        )
        min_usage_bill = self.generate_utility_bill(
            usage_amount=Decimal('0.01'),
            total_cost=Decimal('0.01')
        )
        edge_cases['boundary_values'].extend([max_usage_bill, min_usage_bill])
        
        return edge_cases
    
    def generate_rejection_scenarios(self) -> Dict[RejectionReason, List[Any]]:
        """Generate bills with various rejection scenarios"""
        rejection_cases = {}
        
        for reason in RejectionReason:
            cases = []
            
            if reason == RejectionReason.QUALITY_ISSUE:
                # Poor quality scans, unreadable data
                for _ in range(3):
                    bill = self.generate_utility_bill(
                        status=BillStatus.REJECTED,
                        rejection_reason=reason,
                        vendor="[UNREADABLE]",
                        usage_amount=Decimal('0')
                    )
                    cases.append(bill)
            
            elif reason == RejectionReason.DUPLICATE:
                # Same bill submitted twice
                original_bill = self.generate_utility_bill()
                duplicate_bill = self.generate_utility_bill(
                    facility_id=original_bill.facility_id,
                    account_number=original_bill.account_number,
                    billing_period_start=original_bill.billing_period_start,
                    billing_period_end=original_bill.billing_period_end,
                    status=BillStatus.REJECTED,
                    rejection_reason=reason
                )
                cases.extend([original_bill, duplicate_bill])
            
            elif reason == RejectionReason.INVALID_PERIOD:
                # Bills with invalid date ranges
                invalid_bill = self.generate_utility_bill(
                    billing_period_start=date(2025, 1, 15),
                    billing_period_end=date(2025, 1, 10),  # End before start
                    status=BillStatus.REJECTED,
                    rejection_reason=reason
                )
                cases.append(invalid_bill)
            
            elif reason == RejectionReason.MISSING_DATA:
                # Bills missing critical information
                missing_data_bill = self.generate_utility_bill(
                    account_number="",
                    usage_amount=Decimal('0'),
                    total_cost=Decimal('0'),
                    status=BillStatus.REJECTED,
                    rejection_reason=reason
                )
                cases.append(missing_data_bill)
            
            elif reason == RejectionReason.INCORRECT_FORMAT:
                # Bills with formatting issues
                format_bill = self.generate_water_bill(
                    water_usage_gallons=Decimal('-1000'),  # Negative usage
                    status=BillStatus.REJECTED,
                    rejection_reason=reason
                )
                cases.append(format_bill)
            
            rejection_cases[reason] = cases
        
        return rejection_cases
    
    def generate_performance_test_data(self, large_dataset_size: int = 1000) -> Dict[str, List[Any]]:
        """Generate large datasets for performance testing"""
        performance_data = {
            'large_facilities': [],
            'large_bills': [],
            'large_manifests': [],
            'large_permits': []
        }
        
        # Generate large facility dataset
        performance_data['large_facilities'] = self.generate_facilities_batch(large_dataset_size // 10)
        
        # Generate large bill datasets
        performance_data['large_bills'] = self.generate_utility_bills_batch(large_dataset_size)
        
        # Generate large manifest dataset
        performance_data['large_manifests'] = self.generate_waste_manifests_batch(large_dataset_size // 2)
        
        # Generate large permit dataset
        performance_data['large_permits'] = self.generate_permits_batch(large_dataset_size // 5)
        
        return performance_data
    
    # Pro-rating Test Cases
    def generate_prorating_scenarios(self) -> List[TestAllocationScenario]:
        """Generate specific pro-rating test scenarios"""
        scenarios = []
        
        # Partial month scenario (15 days)
        partial_month = self.generate_allocation_scenario(
            name="Partial Month Pro-rating",
            billing_period_start=date(2024, 6, 15),
            billing_period_end=date(2024, 6, 30),
            total_amount=Decimal('2000.00')
        )
        scenarios.append(partial_month)
        
        # Mid-month facility addition
        mid_month_addition = self.generate_allocation_scenario(
            name="Mid-month Facility Addition",
            billing_period_start=date(2024, 7, 1),
            billing_period_end=date(2024, 7, 31),
            total_amount=Decimal('3500.00')
        )
        scenarios.append(mid_month_addition)
        
        # Leap year February
        leap_year = self.generate_allocation_scenario(
            name="Leap Year February",
            billing_period_start=date(2024, 2, 1),
            billing_period_end=date(2024, 2, 29),
            total_amount=Decimal('2800.00')
        )
        scenarios.append(leap_year)
        
        return scenarios
    
    # Data Serialization and Validation
    def to_dict(self, obj: Any) -> Dict[str, Any]:
        """Convert test object to dictionary for API testing"""
        if hasattr(obj, '__dict__'):
            result = {}
            for key, value in obj.__dict__.items():
                if isinstance(value, (datetime, date)):
                    result[key] = value.isoformat()
                elif isinstance(value, Decimal):
                    result[key] = str(value)
                elif isinstance(value, Enum):
                    result[key] = value.value
                elif isinstance(value, list):
                    result[key] = [self.to_dict(item) if hasattr(item, '__dict__') else item for item in value]
                elif isinstance(value, dict):
                    result[key] = {k: self.to_dict(v) if hasattr(v, '__dict__') else v for k, v in value.items()}
                else:
                    result[key] = value
            return result
        return obj
    
    def to_json(self, obj: Any) -> str:
        """Convert test object to JSON string"""
        return json.dumps(self.to_dict(obj), indent=2, default=str)
    
    def validate_bill_data(self, bill: Union[TestUtilityBill, TestWaterBill, TestWasteManifest]) -> Dict[str, List[str]]:
        """Validate bill data and return validation errors"""
        errors = {'errors': [], 'warnings': []}
        
        # Common validations
        if not bill.facility_id:
            errors['errors'].append("Missing facility_id")
        
        if hasattr(bill, 'billing_period_start') and hasattr(bill, 'billing_period_end'):
            if bill.billing_period_start and bill.billing_period_end:
                if bill.billing_period_start >= bill.billing_period_end:
                    errors['errors'].append("Billing period start date must be before end date")
        
        if hasattr(bill, 'total_cost') and bill.total_cost < 0:
            errors['warnings'].append("Negative total cost detected")
        
        # Specific validations
        if isinstance(bill, TestUtilityBill):
            if bill.usage_amount < 0:
                errors['errors'].append("Usage amount cannot be negative")
            if not bill.vendor:
                errors['warnings'].append("Missing vendor information")
        
        elif isinstance(bill, TestWaterBill):
            if bill.water_usage_gallons < 0:
                errors['errors'].append("Water usage cannot be negative")
            if bill.sewer_usage_gallons > bill.water_usage_gallons:
                errors['warnings'].append("Sewer usage exceeds water usage")
        
        elif isinstance(bill, TestWasteManifest):
            if bill.quantity <= 0:
                errors['errors'].append("Waste quantity must be positive")
            if bill.waste_type == 'hazardous' and not bill.epa_codes:
                errors['warnings'].append("Hazardous waste should have EPA codes")
        
        return errors
    
    def get_all_data(self) -> Dict[str, List[Any]]:
        """Get all generated test data"""
        return {
            'users': self._users,
            'facilities': self._facilities,
            'utility_bills': self._utility_bills,
            'water_bills': self._water_bills,
            'waste_manifests': self._waste_manifests,
            'permits': self._permits,
            'allocation_scenarios': self._allocation_scenarios
        }
    
    def clear_all_data(self):
        """Clear all generated test data"""
        self._users.clear()
        self._facilities.clear()
        self._utility_bills.clear()
        self._water_bills.clear()
        self._waste_manifests.clear()
        self._permits.clear()
        self._allocation_scenarios.clear()
    
    def generate_complete_test_suite(self, seed: int = 42) -> Dict[str, Any]:
        """Generate a complete test suite with all types of data"""
        if seed:
            fake.seed_instance(seed)
            random.seed(seed)
        
        # Generate base data
        users = self.generate_users_batch(15)
        facilities = self.generate_facilities_batch(8)
        
        # Assign facility access to users
        for user in users:
            if user.role in [UserRole.ADMIN]:
                user.facilities_access = [f.id for f in facilities]
            elif user.role in [UserRole.FACILITY_MANAGER]:
                user.facilities_access = random.sample([f.id for f in facilities], 
                                                      random.randint(1, 3))
        
        # Generate documents
        utility_bills = self.generate_utility_bills_batch(50)
        water_bills = self.generate_water_bills_batch(30)
        waste_manifests = self.generate_waste_manifests_batch(20)
        permits = self.generate_permits_batch(15)
        
        # Generate allocation scenarios
        allocation_scenarios = self.generate_allocation_scenarios_batch(10)
        prorating_scenarios = self.generate_prorating_scenarios()
        
        # Generate edge cases and special scenarios
        edge_cases = self.generate_edge_case_bills()
        rejection_scenarios = self.generate_rejection_scenarios()
        
        return {
            'basic_data': {
                'users': users,
                'facilities': facilities,
                'utility_bills': utility_bills,
                'water_bills': water_bills,
                'waste_manifests': waste_manifests,
                'permits': permits,
                'allocation_scenarios': allocation_scenarios
            },
            'special_scenarios': {
                'prorating': prorating_scenarios,
                'edge_cases': edge_cases,
                'rejections': rejection_scenarios
            },
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'seed': seed,
                'total_records': {
                    'users': len(users),
                    'facilities': len(facilities),
                    'utility_bills': len(utility_bills),
                    'water_bills': len(water_bills),
                    'waste_manifests': len(waste_manifests),
                    'permits': len(permits),
                    'allocation_scenarios': len(allocation_scenarios) + len(prorating_scenarios)
                }
            }
        }

# Convenience functions for quick access
def create_sample_data(seed: int = None) -> TestDataFixtures:
    """Create a fixtures instance with sample data"""
    fixtures = TestDataFixtures(seed=seed)
    fixtures.generate_users_batch(10)
    fixtures.generate_facilities_batch(5)
    fixtures.generate_utility_bills_batch(20)
    fixtures.generate_water_bills_batch(15)
    fixtures.generate_waste_manifests_batch(10)
    fixtures.generate_permits_batch(8)
    return fixtures

def create_minimal_data() -> TestDataFixtures:
    """Create minimal test data for basic testing"""
    fixtures = TestDataFixtures(seed=123)
    fixtures.generate_users_batch(3)
    fixtures.generate_facilities_batch(2)
    fixtures.generate_utility_bills_batch(5)
    return fixtures

# Example usage for API testing
if __name__ == "__main__":
    # Create comprehensive test data
    fixtures = TestDataFixtures(seed=42)
    complete_suite = fixtures.generate_complete_test_suite()
    
    # Print summary
    print("Generated comprehensive test data suite:")
    print(json.dumps(complete_suite['metadata'], indent=2))
    
    # Example: Get a specific bill for API testing
    sample_bill = fixtures.generate_utility_bill()
    print("\nSample utility bill for API testing:")
    print(fixtures.to_json(sample_bill))
    
    # Example: Validate data
    validation_result = fixtures.validate_bill_data(sample_bill)
    print(f"\nValidation result: {validation_result}")