# Location Hierarchy Migration Scripts

This directory contains executable Python migration scripts for implementing the Neo4j location hierarchy transformation plan for EHS data management.

## Scripts Overview

### 1. migrate_location_hierarchy.py
**Main migration script** that implements the data transformation plan.

**Features:**
- Creates hierarchical location structure (Site → Building → Floor → Area)
- Cleans up duplicate facility data
- Maps existing facilities to the new hierarchy
- Includes rollback functionality
- Comprehensive error handling and logging
- Progress tracking

**Usage:**
```bash
# Run migration
python3 migrate_location_hierarchy.py

# Dry run (test without changes)
python3 migrate_location_hierarchy.py --dry-run
```

**Output Files:**
- `location_migration_YYYYMMDD_HHMMSS.log` - Detailed execution log
- `location_backup_YYYYMMDD_HHMMSS.json` - Backup of existing data
- `migration_state_YYYYMMDD_HHMMSS.json` - Migration state tracking

### 2. validate_migration.py
**Validation script** that checks migration results and generates comprehensive reports.

**Features:**
- Validates hierarchy structure completeness
- Checks constraint and index integrity
- Validates facility mappings
- Reports data inconsistencies
- Generates detailed validation reports
- Performance metrics analysis

**Usage:**
```bash
# Run validation
python3 validate_migration.py
```

**Output Files:**
- `validation_YYYYMMDD_HHMMSS.log` - Validation execution log
- `validation_report_YYYYMMDD_HHMMSS.txt` - Human-readable report
- `validation_report_YYYYMMDD_HHMMSS.json` - Machine-readable report

### 3. seed_test_data.py
**Test data seeding script** that creates comprehensive sample data for testing.

**Features:**
- Creates Algonquin and Houston test sites with realistic structure
- Populates EHS metrics (safety, environmental, health)
- Creates sample incident data
- Links all data through proper relationships
- Supports cleanup of existing test data

**Usage:**
```bash
# Seed all test data (default)
python3 seed_test_data.py

# Clean existing test data first
python3 seed_test_data.py --cleanup

# Seed without EHS metrics
python3 seed_test_data.py --no-metrics

# Seed without incidents
python3 seed_test_data.py --no-incidents
```

**Output Files:**
- `seed_data_YYYYMMDD_HHMMSS.log` - Seeding execution log

## Migration Workflow

### Standard Migration Process
```bash
# 1. Seed test data (optional, for testing)
python3 seed_test_data.py --cleanup

# 2. Run the migration
python3 migrate_location_hierarchy.py

# 3. Validate results
python3 validate_migration.py
```

### Testing Migration Process
```bash
# 1. Test with dry run
python3 migrate_location_hierarchy.py --dry-run

# 2. If dry run looks good, run actual migration
python3 migrate_location_hierarchy.py

# 3. Validate results
python3 validate_migration.py
```

## Configuration

All scripts use the same Neo4j connection configuration from the parent directory's `.env` file:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=EhsAI2024!
NEO4J_DATABASE=neo4j
```

## Data Structure Created

### Hierarchy Structure
```
Site (Algonquin Manufacturing Site, Houston Corporate Campus)
├── Building (Main Manufacturing, Warehouse, Corporate Tower, etc.)
    ├── Floor (Production Floor, Mezzanine Level, Executive Floor, etc.)
        ├── Area (Production Line A, Quality Control Lab, Executive Offices, etc.)
            ├── Facility (Production Line A Facility, etc.)
```

### EHS Metrics Categories
- **Safety:** TRIR, LTIR, Near Miss Reports, Safety Training Hours
- **Environmental:** Water Usage, Energy Consumption, Waste Diversion Rate, CO2 Emissions  
- **Health:** Employee Health Index, Air Quality Index, Ergonomic Assessments

### Sample Incidents
- Slip/Trip/Fall incidents
- Chemical Exposure incidents
- Equipment Injury incidents
- Environmental Release incidents

## Error Handling & Recovery

### Migration Rollback
If migration fails, automatic rollback can be triggered:
- Removes created hierarchy nodes
- Drops created constraints
- Preserves original data

### Manual Rollback
```bash
# Connect to Neo4j and run:
# MATCH (s:Site) WHERE s.source = 'migration' DETACH DELETE s
# MATCH (b:Building) WHERE s.source = 'migration' DETACH DELETE b
# (etc. for Floor, Area nodes)
```

### Test Data Cleanup
```bash
# Remove all test data
python3 seed_test_data.py --cleanup
```

## Validation Checks

The validation script performs these checks:

1. **Structure Validation:**
   - All hierarchy levels exist (Site, Building, Floor, Area)
   - Complete hierarchy paths exist
   - Proper relationship structure

2. **Constraint Validation:**
   - Uniqueness constraints exist
   - Performance indexes exist

3. **Data Consistency:**
   - No orphaned nodes
   - All facilities properly mapped
   - Data integrity maintained

4. **Performance Validation:**
   - Query performance acceptable
   - Index usage statistics

## Troubleshooting

### Common Issues

**Connection Failed:**
```bash
# Check Neo4j is running
sudo systemctl status neo4j
# or
docker ps | grep neo4j
```

**Migration Fails:**
- Check log files for detailed error messages
- Verify database permissions
- Ensure sufficient disk space

**Validation Failures:**
- Review validation report for specific issues
- Check constraint and index creation
- Verify data relationships

### Log Analysis
```bash
# View recent logs
tail -f location_migration_YYYYMMDD_HHMMSS.log

# Search for errors
grep -i "error\|failed" location_migration_YYYYMMDD_HHMMSS.log
```

## Requirements

- Python 3.8+
- Neo4j 4.4+
- Required Python packages:
  - langchain-neo4j
  - python-dotenv
  - neo4j

Install dependencies:
```bash
pip install langchain-neo4j python-dotenv neo4j
```

## Support

For issues or questions:
1. Check the log files for detailed error information
2. Review the validation report for data integrity issues
3. Verify Neo4j connection and permissions
4. Ensure all required dependencies are installed