# Neo4j Comprehensive Backup and Restore System

This directory contains comprehensive backup and restore tools for Neo4j databases, specifically designed for the EHS AI Demo project.

## Overview

The backup and restore system provides:
- **Complete database backup** with all nodes, relationships, properties, and labels
- **Multiple backup formats** (JSON and Cypher scripts)
- **Data validation** and integrity checks
- **Flexible restore options** with selective restoration capabilities
- **Comprehensive logging** and progress tracking

## Files

- `backup_neo4j_full.py` - Comprehensive backup script
- `restore_neo4j_full.py` - Comprehensive restore script
- `NEO4J_BACKUP_RESTORE_README.md` - This documentation

## Prerequisites

1. **Environment Variables**: Ensure your `.env` file contains:
   ```
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=EhsAI2024!
   NEO4J_DATABASE=neo4j
   ```

2. **Python Dependencies**: The scripts require:
   ```
   neo4j
   python-dotenv
   ```

3. **Python Virtual Environment**: Use the project's virtual environment:
   ```bash
   source venv/bin/activate  # or activate your project's venv
   ```

## Backup Usage

### Basic Backup

```bash
cd /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/scripts
python3 backup_neo4j_full.py
```

### What Gets Backed Up

The backup script captures:

1. **All Nodes**:
   - Site, Facility, Building nodes
   - ElectricityConsumption, WaterConsumption, WasteGeneration data
   - Goal, EnvironmentalTarget nodes
   - RiskAssessment, Recommendation nodes
   - All other nodes in the database

2. **All Relationships**:
   - Location hierarchies (CONTAINS, LOCATED_AT, etc.)
   - Consumption relationships (HAS_CONSUMPTION, etc.)
   - Goal relationships (HAS_GOAL, TARGETS, etc.)
   - All other relationships with properties

3. **Database Metadata**:
   - Schema information (labels, relationship types, property keys)
   - Database statistics
   - Index and constraint definitions
   - Backup timestamp and version info

### Backup Output

The backup creates a timestamped directory with:

```
backups/neo4j_backup_20250905_143022/
├── neo4j_backup_20250905_143022.json           # Uncompressed JSON backup
├── neo4j_backup_20250905_143022.json.gz        # Compressed JSON backup
├── neo4j_restore_20250905_143022.cypher        # Cypher restore script
├── backup_metadata_20250905_143022.json        # Backup metadata
└── backup_20250905_143022.log                  # Detailed backup log
```

## Restore Usage

### Basic Restore (Add to Existing Data)

```bash
cd /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/scripts
python3 restore_neo4j_full.py backups/neo4j_backup_20250905_143022/
```

### Full Restore (Clear Database First)

⚠️ **WARNING**: This will delete all existing data!

```bash
python3 restore_neo4j_full.py backups/neo4j_backup_20250905_143022/ --clear
```

### Restore with Constraints and Indexes

```bash
python3 restore_neo4j_full.py backups/neo4j_backup_20250905_143022/ --clear --constraints --indexes
```

### Command Line Options

```bash
python3 restore_neo4j_full.py <backup_path> [OPTIONS]

Arguments:
  backup_path           Path to backup file or directory

Options:
  --clear              Clear existing database before restore
  --constraints        Restore constraints (experimental)
  --indexes           Restore indexes (experimental)
  --force             Skip confirmation prompts
```

## Usage Examples

### 1. Daily Backup

```bash
#!/bin/bash
# daily_backup.sh
cd /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/scripts
source ../venv/bin/activate
python3 backup_neo4j_full.py
```

### 2. Backup Before Major Changes

```bash
# Before running migrations or major data changes
python3 backup_neo4j_full.py
echo "Backup completed. Proceeding with changes..."
```

### 3. Restore from Specific Backup

```bash
# List available backups
ls -la backups/

# Restore from specific backup (replace timestamp)
python3 restore_neo4j_full.py backups/neo4j_backup_20250905_143022/
```

### 4. Full Database Reset and Restore

```bash
# Complete database reset (BE CAREFUL!)
python3 restore_neo4j_full.py backups/neo4j_backup_20250905_143022/ --clear --force
```

## Backup File Formats

### JSON Format

The JSON backup contains:

```json
{
  "metadata": {
    "backup_timestamp": "20250905_143022",
    "backup_version": "1.0.0",
    "database_info": { ... },
    "schema_info": { ... },
    "statistics": { ... }
  },
  "nodes": [
    {
      "internal_id": 123,
      "labels": ["Site", "Location"],
      "properties": {
        "site_id": "ALG001",
        "name": "Algonquin IL",
        ...
      }
    },
    ...
  ],
  "relationships": [
    {
      "internal_id": 456,
      "start_node_id": 123,
      "end_node_id": 789,
      "type": "CONTAINS",
      "properties": { ... }
    },
    ...
  ]
}
```

### Cypher Format

The Cypher backup is a executable script:

```cypher
// Neo4j Database Restore Script
// Generated: 20250905_143022
// Total Nodes: 1250
// Total Relationships: 3400

// === NODES ===
CREATE (n123:Site:Location {site_id: "ALG001", name: "Algonquin IL", ...});
CREATE (n789:Facility {facility_id: "ALG001-F1", name: "Main Building", ...});
...

// === RELATIONSHIPS ===
MATCH (a), (b) WHERE id(a) = 123 AND id(b) = 789 CREATE (a)-[r:CONTAINS {...}]->(b);
...
```

## Data Validation

Both backup and restore processes include validation:

### Backup Validation
- Checks data structure integrity
- Validates all required fields are present
- Ensures referential integrity between nodes and relationships

### Restore Validation
- Compares restored data counts with original backup
- Verifies all nodes and relationships were created
- Reports any discrepancies or errors

## Troubleshooting

### Common Issues

1. **Connection Failed**
   ```
   Error: Failed to connect to Neo4j
   ```
   - Check if Neo4j is running
   - Verify connection details in `.env` file
   - Test connection: `python3 -c "from neo4j import GraphDatabase; print('OK')"`

2. **Permission Denied**
   ```
   Error: Permission denied
   ```
   - Make scripts executable: `chmod +x *.py`
   - Check file permissions in backup directory

3. **Out of Memory**
   ```
   Error: Memory allocation failed
   ```
   - Large databases may need smaller batch sizes
   - Consider using compressed JSON backups only
   - Increase available memory for Python process

4. **Backup Validation Failed**
   ```
   Warning: Node count mismatch
   ```
   - Check Neo4j logs for errors during backup
   - Verify database is not being modified during backup
   - Re-run backup with smaller batch sizes

### Debugging

Enable debug logging by modifying the logging level:

```python
# In backup_neo4j_full.py or restore_neo4j_full.py
logging.basicConfig(
    level=logging.DEBUG,  # Changed from INFO to DEBUG
    ...
)
```

## Performance Considerations

### Large Databases

For databases with >100k nodes:
- Backup may take several minutes to hours
- Consider running during low-usage periods
- Monitor disk space for backup files
- Use compressed JSON format to save space

### Batch Sizes

The scripts use configurable batch sizes:
- **Backup**: 1000 nodes/relationships per batch
- **Restore**: 100 nodes/relationships per batch

To modify batch sizes, edit the variables in the scripts:

```python
# In backup script
batch_size = 1000  # Increase for faster backup, decrease for lower memory usage

# In restore script
batch_size = 100   # Increase for faster restore, decrease for lower memory usage
```

## Security Considerations

1. **Backup Files**: Contain all database data including sensitive information
2. **Storage**: Store backups in secure locations with appropriate access controls
3. **Network**: Use encrypted connections to Neo4j (consider neo4j+s:// for SSL)
4. **Credentials**: Ensure `.env` file is not included in version control

## Integration with CI/CD

### Automated Testing Backups

```bash
# In your test setup
python3 scripts/backup_neo4j_full.py
# Run tests
# Restore from backup if needed
python3 scripts/restore_neo4j_full.py backups/latest/ --clear
```

### Production Deployment

```bash
# Before deployment
python3 scripts/backup_neo4j_full.py
# Deploy changes
# If rollback needed:
# python3 scripts/restore_neo4j_full.py backups/latest/ --clear
```

## Advanced Usage

### Programmatic Usage

You can also use the backup/restore classes in your own scripts:

```python
from backup_neo4j_full import Neo4jBackupManager
from restore_neo4j_full import Neo4jRestoreManager

# Create backup
backup_manager = Neo4jBackupManager()
success = backup_manager.run_backup()

# Restore backup
restore_manager = Neo4jRestoreManager(
    backup_path="backups/neo4j_backup_20250905_143022/",
    clear_database=True
)
success = restore_manager.run_restore()
```

### Custom Backup Filtering

To modify the backup scripts for selective backups (e.g., only specific node types), modify the queries in the backup methods:

```python
# Example: Only backup Site and Facility nodes
query = """
MATCH (n)
WHERE any(label IN labels(n) WHERE label IN ['Site', 'Facility'])
RETURN id(n) as node_id, labels(n) as labels, properties(n) as properties
SKIP $offset LIMIT $limit
"""
```

## Maintenance

### Backup Retention

Consider implementing backup retention policies:

```bash
# Delete backups older than 30 days
find backups/ -type d -name "neo4j_backup_*" -mtime +30 -exec rm -rf {} \;
```

### Backup Verification

Regularly verify backup integrity:

```bash
# Test restore on a test database
python3 restore_neo4j_full.py backups/latest/ --clear
```

## Support

For issues or questions:
1. Check the log files in the backup directory
2. Review this documentation
3. Check Neo4j connection and database status
4. Verify environment variables and dependencies

## Version History

- **v1.0.0** (2025-09-05): Initial comprehensive backup and restore system
  - Complete node and relationship backup
  - JSON and Cypher export formats
  - Data validation and integrity checks
  - Flexible restore options