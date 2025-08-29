# Facility Setup Script

This script sets up the facility nodes in Neo4j database for the EHS AI Demo project.

## Overview

The `setup_facilities.py` script creates 5 facilities (FAC001-FAC005) in the Neo4j database with complete metadata including:

- **Facility Information**: ID, name, type, size, operating hours
- **Address Details**: Street address, city, state, ZIP code, country
- **Contact Information**: Phone, email, manager name, safety officer
- **Operational Metadata**: Established date, square footage, annual revenue
- **EHS Information**: Environmental permits, certifications
- **Operational Characteristics**: Shift count, weekend operations, seasonal variations

## Facilities Created

1. **FAC001** - Manufacturing Plant A (Detroit, MI) - Large, 500 employees, 24-hour operations
2. **FAC002** - Office Complex B (Austin, TX) - Medium, 200 employees, 12-hour operations  
3. **FAC003** - Warehouse C (Memphis, TN) - Large, 50 employees, 16-hour operations
4. **FAC004** - Data Center D (Ashburn, VA) - Medium, 30 employees, 24-hour operations
5. **FAC005** - Research Lab E (Cambridge, MA) - Small, 75 employees, 12-hour operations

## Prerequisites

1. **Neo4j Database**: Ensure Neo4j is running and accessible
2. **Python Environment**: Python 3.8+ with required packages
3. **Environment Variables**: Configure Neo4j connection details

### Required Python Packages

```bash
pip install neo4j python-dotenv
```

### Environment Variables

Create a `.env` file or set these environment variables:

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

## Usage

### Basic Usage

```bash
# Run from the backend directory
cd /path/to/backend
source venv/bin/activate
python3 src/data_generation/setup_facilities.py
```

### Command Line Options

```bash
# Show help
python3 src/data_generation/setup_facilities.py --help

# Dry run (test without connecting to Neo4j)
python3 src/data_generation/setup_facilities.py --dry-run

# Verbose output
python3 src/data_generation/setup_facilities.py --verbose

# Custom Neo4j connection
python3 src/data_generation/setup_facilities.py --uri bolt://custom:7687 --username admin --password secret
```

## Features

### Idempotent Operations
- Safe to run multiple times
- Updates existing facilities instead of creating duplicates
- Preserves original creation timestamps

### Comprehensive Indexing
- Creates performance indexes for efficient queries
- Unique constraints to prevent data duplication
- Full-text search capabilities

### Error Handling
- Robust error handling and retry logic
- Detailed logging to file and console
- Connection validation and graceful degradation

### Reporting
- Comprehensive setup reports
- Verification of all facilities
- Statistics and performance metrics
- Report saved to timestamped files

## Database Schema

### Facility Node Properties

```cypher
(:Facility {
  facility_id: "FAC001",
  name: "Manufacturing Plant A",
  facility_type: "manufacturing",
  size: "large",
  employees: 500,
  operating_hours: 24,
  street_address: "1250 Industrial Blvd",
  city: "Detroit",
  state: "MI",
  zip_code: "48201",
  country: "USA",
  phone: "(313) 555-0101",
  email: "operations@mfg-plant-a.com",
  manager_name: "Sarah Johnson",
  established_date: "2010-03-15",
  square_footage: 450000,
  annual_revenue: 85000000,
  safety_officer: "Michael Chen",
  environmental_permits: ["EPA-MI-001", "MDEQ-AIR-2024"],
  certifications: ["ISO 14001", "OHSAS 18001", "ISO 9001"],
  shift_count: 3,
  operates_weekends: true,
  seasonal_variations: true,
  created_at: "2025-08-28T10:00:00Z",
  updated_at: "2025-08-28T10:00:00Z"
})
```

### Indexes and Constraints

- **Unique Constraint**: `facility_id` (prevents duplicates)
- **Property Indexes**: `name`, `facility_type`, `size`, `city`, `state`
- **Compound Index**: `(city, state)` for location queries
- **Full-Text Index**: `name`, `street_address`, `city` for search

## Integration

### With Historical Data Generator

This script should be run before the historical data generator (`generate_historical_data.py`) to ensure facility nodes exist for metric relationships.

```bash
# Recommended sequence:
python3 src/data_generation/setup_facilities.py
python3 src/data_generation/generate_historical_data.py
```

### With Neo4j Loader

The facilities created by this script are referenced by the Neo4j loader for creating relationships between facilities and historical metrics.

## Troubleshooting

### Connection Issues
- Verify Neo4j is running: `systemctl status neo4j` or check Neo4j Browser
- Check firewall settings and port 7687 accessibility
- Validate credentials in `.env` file

### Permission Errors
- Ensure Neo4j user has CREATE and INDEX privileges
- Check database access permissions
- Verify authentication credentials

### Import Errors
- Ensure you're in the correct Python virtual environment
- Install required packages: `pip install neo4j python-dotenv`
- Check Python path and module imports

### Data Issues
- Review the setup report for detailed error information
- Check Neo4j logs for database-specific errors
- Use `--verbose` flag for detailed logging

## Logs and Reports

### Log Files
- `facility_setup.log` - Detailed execution log
- `facility_setup_report_TIMESTAMP.txt` - Setup report

### Report Contents
- Setup summary (created/updated facilities, indexes)
- Verification results (completeness check)
- Performance statistics
- Error details (if any)

## Development

### Extending Facility Data
To add new facilities or modify existing ones, edit the `_create_facility_data()` method in the `FacilitySetupManager` class.

### Adding New Properties
1. Update the `FacilityData` dataclass
2. Modify the `to_neo4j_dict()` method
3. Update indexes if needed for new searchable properties

### Testing
```python
# Test without Neo4j connection
from data_generation.setup_facilities import FacilitySetupManager

setup_manager = FacilitySetupManager('test://localhost', 'test', 'test')
print(f"Created {len(setup_manager.facilities)} facilities")
```

## Version History

- **v1.0.0** (2025-08-28): Initial version with 5 facilities
  - Manufacturing Plant A (Detroit, MI)
  - Office Complex B (Austin, TX)
  - Warehouse C (Memphis, TN)  
  - Data Center D (Ashburn, VA)
  - Research Lab E (Cambridge, MA)