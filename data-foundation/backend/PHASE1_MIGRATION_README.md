# Phase 1 Database Schema Migration

## Overview

The `migrate_phase1_schema.py` script creates all necessary constraints, indexes, and node types for Phase 1 features including:

- **Audit Trail Schema**: Document provenance tracking with original_filename and source_file_path properties
- **Pro-Rating Schema**: Monthly usage allocations with MonthlyUsageAllocation nodes and relationships
- **Rejection Tracking Schema**: Document rejection workflow with status tracking and User relationships

## Prerequisites

### Required Dependencies

Make sure you have the following Python packages installed:

```bash
pip3 install -r requirements.txt
```

Key dependencies for the migration:
- `langchain-neo4j==0.4.0` (Neo4j integration)
- `neo4j` (Neo4j Python driver) 
- `python-dotenv` (environment variable loading)

### Neo4j Database

Ensure your Neo4j database is running and accessible with credentials configured in `.env`:

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=EhsAI2024!
NEO4J_DATABASE=neo4j
```

## Usage

### Basic Migration

Run the complete Phase 1 schema migration:

```bash
python3 migrate_phase1_schema.py
```

### Virtual Environment Setup (Recommended)

If dependencies are not installed system-wide:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows

# Install dependencies
pip3 install -r requirements.txt

# Run migration
python3 migrate_phase1_schema.py

# Deactivate when done
deactivate
```

## What the Migration Does

### Audit Trail Schema
- Creates unique constraint on `Document.original_filename`
- Creates index on `Document.source_file_path`
- Migrates existing documents to include audit properties
- Validates all documents have required audit fields

### Pro-Rating Schema
- Creates unique constraint on `MonthlyUsageAllocation.allocation_id`
- Creates indexes for year/month queries, allocation percentage, and usage amounts
- Sets up schema for creating allocation nodes and relationships
- Validates schema structure

### Rejection Tracking Schema
- Creates indexes for document status, rejection reason, timestamps, and user IDs
- Creates unique constraint on `User.user_id`
- Migrates existing documents to include status properties (defaults to PROCESSING)
- Validates schema and document properties

## Migration Output

The script provides detailed logging including:

- Connection status to Neo4j
- Progress for each schema migration
- Validation results for each component
- Overall migration summary
- Timestamped log file: `phase1_migration_YYYYMMDD_HHMMSS.log`

### Success Output Example:
```
Phase 1 Database Schema Migration Started
Log file: phase1_migration_20250823_141530.log
Loaded Neo4j credentials for database: neo4j
Connecting to Neo4j at bolt://localhost:7687
Neo4j connection successful

=== Starting Audit Trail Schema Migration ===
Creating complete audit trail schema
Creating constraints and indexes for audit trail properties
Constraints and indexes created successfully
Adding audit trail properties to existing ProcessedDocument nodes
Updated 0 documents with audit trail properties
Validating audit trail properties
Audit Trail Migration Result: {'constraints_created': True, 'indexes_created': True, 'documents_migrated': 0, 'validation_passed': True}
✓ Audit Trail Schema migration completed successfully

=== Starting Pro-Rating Schema Migration ===
Creating complete pro-rating allocation schema
Creating constraints and indexes for pro-rating schema
Constraints and indexes created successfully
Validating pro-rating schema
Pro-Rating Migration Result: {'constraints_created': True, 'indexes_created': True, 'schema_validation_passed': True}
✓ Pro-Rating Schema migration completed successfully

=== Starting Rejection Tracking Schema Migration ===
Creating complete rejection tracking schema
Creating constraints and indexes for rejection tracking schema
Constraints and indexes created successfully
Adding rejection tracking properties to existing ProcessedDocument nodes
Updated 0 documents with rejection tracking properties
Validating rejection tracking schema
Rejection Tracking Migration Result: {'constraints_created': True, 'indexes_created': True, 'documents_migrated': 0, 'schema_validation_passed': True}
✓ Rejection Tracking Schema migration completed successfully

=== Migration Summary ===
Audit Trail Schema: ✓ SUCCESS
Pro-Rating Schema: ✓ SUCCESS
Rejection Tracking Schema: ✓ SUCCESS
Overall: 3/3 migrations successful
🎉 Phase 1 migration completed successfully!

✓ Phase 1 schema migration completed successfully!
```

## Error Handling

The script gracefully handles errors:

- Connection failures to Neo4j
- Import errors for missing dependencies
- Schema creation failures
- Individual migration component failures

Check the generated log file for detailed error information if any migration fails.

## Validation

After migration, the script validates:

- All constraints and indexes were created successfully
- Existing documents were properly migrated
- Schema structure meets requirements
- Database connectivity and query functionality

## Files Created

The migration creates the following database objects:

### Constraints
- `unique_original_filename` on Document.original_filename
- `unique_allocation_id` on MonthlyUsageAllocation.allocation_id
- `unique_user_id` on User.user_id

### Indexes
- `idx_source_file_path` on Document.source_file_path
- `idx_allocation_year_month` on MonthlyUsageAllocation (year, month)
- `idx_allocation_percentage` on MonthlyUsageAllocation.allocation_percentage
- `idx_allocated_usage` on MonthlyUsageAllocation.allocated_usage
- `idx_document_status` on Document.status
- `idx_rejection_reason` on Document.rejection_reason
- `idx_rejected_at` on Document.rejected_at
- `idx_rejected_by_user_id` on Document.rejected_by_user_id

### Node Properties Added
- Document nodes: `original_filename`, `source_file_path`, `status`, `rejection_reason`, etc.
- MonthlyUsageAllocation nodes: Complete allocation tracking properties
- User nodes: User management for rejection tracking

## Troubleshooting

### Import Errors
```bash
# Install missing Neo4j driver
pip3 install neo4j

# Install missing langchain-neo4j
pip3 install langchain-neo4j==0.4.0
```

### Connection Errors
- Verify Neo4j is running: `sudo systemctl status neo4j` (Linux) or check Neo4j Desktop
- Check `.env` file credentials
- Ensure Neo4j is listening on the configured port (default: 7687)

### Permission Errors  
- Ensure the Neo4j user has admin permissions
- Check database write permissions
- Verify constraint creation permissions

## Schema Files

The migration uses these schema classes:
- `/src/phase1_enhancements/audit_trail_schema.py`
- `/src/phase1_enhancements/prorating_schema.py`  
- `/src/phase1_enhancements/rejection_tracking_schema.py`

Each file contains detailed methods for creating, validating, and managing their respective schemas.