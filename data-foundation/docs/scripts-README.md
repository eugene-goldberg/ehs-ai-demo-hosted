# Data Foundation Scripts

This directory contains utility scripts for the EHS AI Platform data foundation component.

## Scripts

### test_neo4j_connection.py
Tests the Neo4j database connection and creates initial EHS schema indexes.

**Usage:**
```bash
python scripts/test_neo4j_connection.py
```

## Script Organization Rules

All utility scripts for the data-foundation project should be placed in this directory, organized by purpose:

- **Testing scripts**: `test_*.py`
- **Setup/initialization scripts**: `setup_*.py` or `init_*.py`
- **Data migration scripts**: `migrate_*.py`
- **Utility/helper scripts**: `util_*.py`
- **Maintenance scripts**: `maintain_*.py`

## Development Guidelines

1. All scripts should include proper documentation in docstrings
2. Use consistent error handling and logging
3. Include usage examples in script headers
4. Make scripts executable when appropriate (`chmod +x`)
5. Add new scripts to this README with description and usage