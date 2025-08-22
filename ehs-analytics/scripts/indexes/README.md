# EHS Analytics Index Creation Scripts

This directory contains scripts for creating and managing Neo4j indexes for the EHS Analytics system. These indexes are essential for optimal query performance and support both vector search (RAG) and traditional text search capabilities.

## Scripts Overview

### 1. create_vector_indexes.py
Creates vector indexes for semantic search and RAG operations.

**Features:**
- Document content, title, and summary vector indexes
- DocumentChunk content vector indexes with cosine and euclidean similarity
- EHS entity (Equipment, Permit, Facility) description vector indexes
- OpenAI text-embedding-ada-002 compatibility (1536 dimensions)
- Idempotent operations with existence checks

**Usage:**
```bash
# Run directly
python3 create_vector_indexes.py

# Or as module
python3 -m scripts.indexes.create_vector_indexes
```

### 2. create_fulltext_indexes.py
Creates fulltext indexes for traditional text search across EHS entities.

**Features:**
- Facility name, address, and description indexes
- Permit description and compliance requirement indexes
- Equipment model and specification indexes
- Document content and metadata indexes
- Custom analyzer configuration for EHS terminology
- Multi-entity specialized indexes for compliance and environmental terms

**Usage:**
```bash
# Run directly
python3 create_fulltext_indexes.py

# Or as module
python3 -m scripts.indexes.create_fulltext_indexes
```

### 3. create_all_indexes.py
Orchestrates the creation of all indexes in the correct order with comprehensive monitoring.

**Features:**
- Runs vector and fulltext index creation in optimal sequence
- Database connection verification and optimization settings
- Progress tracking and performance monitoring
- Index health verification and status reporting
- Comprehensive execution summary with timing and statistics

**Usage:**
```bash
# Run the complete index setup
python3 create_all_indexes.py

# This is the recommended way to set up all indexes
```

### 4. populate_embeddings.py
Generates embeddings for existing documents and creates document chunks.

**Features:**
- Batch processing for efficient OpenAI API usage
- Automatic document chunking for large texts
- Progress tracking with tqdm progress bars
- Error recovery and retry mechanisms
- Rate limiting to respect OpenAI API limits
- Token counting and text truncation for model limits

**Usage:**
```bash
# Generate embeddings for all documents
python3 populate_embeddings.py

# With custom batch size
EMBEDDING_BATCH_SIZE=25 python3 populate_embeddings.py
```

## Environment Variables

The scripts use the following environment variables:

### Required Variables
```bash
# Neo4j Connection
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=EhsAI2024!

# OpenAI API (for populate_embeddings.py)
OPENAI_API_KEY=your_openai_api_key
```

### Optional Configuration
```bash
# Embedding generation settings
EMBEDDING_BATCH_SIZE=50          # Number of documents per batch
EMBEDDING_RATE_DELAY=0.1         # Delay between API calls (seconds)
EMBEDDING_CHUNK_SIZE=1000        # Characters per document chunk
```

## Execution Order

For a fresh setup, run the scripts in this order:

1. **create_all_indexes.py** - Creates all necessary indexes
2. **populate_embeddings.py** - Generates embeddings for existing data

```bash
# Complete setup sequence
python3 create_all_indexes.py
python3 populate_embeddings.py
```

## Index Types Created

### Vector Indexes (Semantic Search)
- `ehs_document_content_vector_idx` - Document content embeddings
- `ehs_document_title_vector_idx` - Document title embeddings  
- `ehs_document_summary_vector_idx` - Document summary embeddings
- `ehs_chunk_content_vector_idx` - Chunk content embeddings (cosine)
- `ehs_chunk_euclidean_vector_idx` - Chunk content embeddings (euclidean)
- `ehs_equipment_desc_vector_idx` - Equipment description embeddings
- `ehs_permit_desc_vector_idx` - Permit description embeddings
- `ehs_facility_desc_vector_idx` - Facility description embeddings

### Fulltext Indexes (Text Search)
- `ehs_facility_name_fulltext_idx` - Facility names and addresses
- `ehs_facility_description_fulltext_idx` - Facility descriptions and types
- `ehs_permit_description_fulltext_idx` - Permit types and descriptions
- `ehs_permit_compliance_fulltext_idx` - Compliance requirements and conditions
- `ehs_equipment_model_fulltext_idx` - Equipment models and types
- `ehs_equipment_specs_fulltext_idx` - Equipment specifications
- `ehs_document_content_fulltext_idx` - Document content and titles
- `ehs_document_metadata_fulltext_idx` - Document metadata and classification
- `ehs_chunk_content_fulltext_idx` - Document chunk content
- `ehs_compliance_terms_fulltext_idx` - Cross-entity compliance terms
- `ehs_environmental_terms_fulltext_idx` - Environmental and safety terms

## Performance Considerations

### Vector Indexes
- Use cosine similarity for most semantic search operations
- Euclidean distance available for specialized use cases
- Embeddings are 1536-dimensional (OpenAI text-embedding-ada-002)
- Indexes support efficient k-NN search for RAG operations

### Fulltext Indexes
- Use standard-folding analyzer for case-insensitive search
- Eventually consistent for better write performance
- Optimized for EHS terminology and compliance language
- Support phrase and boolean queries

## Monitoring and Verification

All scripts include comprehensive logging and verification:

- **Connection Testing** - Verifies Neo4j connectivity before operations
- **Index Status Checking** - Confirms indexes are online and functional
- **Performance Metrics** - Tracks creation time and response times
- **Error Handling** - Graceful handling of existing indexes and failures
- **Progress Tracking** - Real-time progress with detailed logging

## Troubleshooting

### Common Issues

1. **Connection Errors**
   - Verify Neo4j is running and accessible
   - Check NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD
   - Ensure network connectivity to Neo4j instance

2. **Index Creation Failures**
   - Check Neo4j version compatibility (requires 5.0+ for vector indexes)
   - Verify sufficient memory for index creation
   - Review Neo4j logs for detailed error messages

3. **Embedding Generation Issues**
   - Verify OPENAI_API_KEY is valid and has sufficient credits
   - Check rate limits and adjust EMBEDDING_RATE_DELAY
   - Monitor token usage for large documents

4. **Performance Issues**
   - Reduce batch sizes for memory-constrained environments
   - Increase delays for rate-limited APIs
   - Check database memory settings

### Log Analysis

Scripts provide detailed logging at INFO level. For debugging:

```bash
# Run with debug logging
export PYTHONPATH=/path/to/ehs-analytics
python3 -c "
import logging
logging.basicConfig(level=logging.DEBUG)
exec(open('create_all_indexes.py').read())
"
```

## Integration with EHS Analytics

These indexes are designed to support:

- **RAG Operations** - Vector search for document retrieval
- **Hybrid Search** - Combining vector and fulltext search
- **Query Routing** - Optimized query execution paths
- **Compliance Monitoring** - Efficient regulatory document search
- **Risk Assessment** - Fast access to safety and environmental data

The indexes integrate seamlessly with the EHS Analytics query routing and retrieval systems defined in the main application.