#!/usr/bin/env python3
"""
Populate Embeddings for Existing EHS Documents

This script generates embeddings for existing documents in the Neo4j database
to support vector search and RAG operations in the EHS Analytics system.

Features:
- Generate embeddings for existing Document and DocumentChunk nodes
- Batch processing for efficiency and API rate limiting
- Progress tracking with detailed logging
- Error recovery and retry mechanisms
- Support for different embedding models
- Chunking strategy for large documents
- Metadata extraction and enhancement
"""

import logging
import os
import sys
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import openai
from neo4j import GraphDatabase, Driver
from dotenv import load_dotenv
import tiktoken
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model: str = "text-embedding-ada-002"
    dimension: int = 1536
    batch_size: int = 100
    max_tokens: int = 8191
    rate_limit_delay: float = 0.1
    retry_attempts: int = 3
    retry_delay: float = 1.0
    chunk_size: int = 1000
    chunk_overlap: int = 200


class EmbeddingPopulator:
    """Generates and populates embeddings for EHS documents."""
    
    def __init__(self, uri: str, username: str, password: str, config: EmbeddingConfig = None):
        """Initialize with Neo4j and OpenAI configurations."""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.config = config or EmbeddingConfig()
        
        # Initialize OpenAI
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
            
        # Initialize tokenizer for token counting
        self.tokenizer = tiktoken.encoding_for_model(self.config.model)
        
    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer."""
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            # Fallback estimation: ~4 chars per token
            return len(text) // 4
            
    def truncate_text(self, text: str, max_tokens: int = None) -> str:
        """Truncate text to fit within token limits."""
        max_tokens = max_tokens or self.config.max_tokens
        
        if self.count_tokens(text) <= max_tokens:
            return text
            
        # Binary search for optimal truncation
        left, right = 0, len(text)
        while left < right:
            mid = (left + right + 1) // 2
            truncated = text[:mid]
            if self.count_tokens(truncated) <= max_tokens:
                left = mid
            else:
                right = mid - 1
                
        return text[:left]
        
    async def generate_embedding(self, text: str, retry_count: int = 0) -> Optional[List[float]]:
        """Generate embedding for a single text with retry logic."""
        try:
            # Truncate text if necessary
            truncated_text = self.truncate_text(text)
            
            if not truncated_text.strip():
                logger.warning("Empty text provided for embedding")
                return None
                
            # Generate embedding
            response = await openai.Embedding.acreate(
                model=self.config.model,
                input=truncated_text
            )
            
            embedding = response['data'][0]['embedding']
            
            # Add rate limiting delay
            await asyncio.sleep(self.config.rate_limit_delay)
            
            return embedding
            
        except Exception as e:
            if retry_count < self.config.retry_attempts:
                logger.warning(f"Embedding generation failed (attempt {retry_count + 1}): {e}")
                await asyncio.sleep(self.config.retry_delay * (retry_count + 1))
                return await self.generate_embedding(text, retry_count + 1)
            else:
                logger.error(f"Embedding generation failed after {self.config.retry_attempts} attempts: {e}")
                return None
                
    def get_documents_without_embeddings(self) -> List[Dict[str, Any]]:
        """Get all documents that don't have embeddings."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (d:Document)
                WHERE d.content_embedding IS NULL 
                   OR d.title_embedding IS NULL 
                   OR d.summary_embedding IS NULL
                RETURN d.id as id, 
                       d.title as title, 
                       d.content as content, 
                       d.summary as summary,
                       d.document_type as document_type,
                       d.content_embedding as content_embedding,
                       d.title_embedding as title_embedding,
                       d.summary_embedding as summary_embedding
                ORDER BY d.created_date DESC
            """)
            
            documents = []
            for record in result:
                doc = dict(record)
                documents.append(doc)
                
            logger.info(f"Found {len(documents)} documents needing embeddings")
            return documents
            
    def get_chunks_without_embeddings(self) -> List[Dict[str, Any]]:
        """Get all document chunks that don't have embeddings."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:DocumentChunk)
                WHERE c.content_embedding IS NULL
                RETURN c.id as id,
                       c.content as content,
                       c.chunk_index as chunk_index,
                       c.section_title as section_title
                ORDER BY c.chunk_index
            """)
            
            chunks = []
            for record in result:
                chunk = dict(record)
                chunks.append(chunk)
                
            logger.info(f"Found {len(chunks)} chunks needing embeddings")
            return chunks
            
    def create_document_chunks(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks for a large document."""
        content = document.get("content", "")
        if not content or len(content) < self.config.chunk_size:
            return []
            
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(content):
            end = start + self.config.chunk_size
            
            # Try to break at word boundary
            if end < len(content):
                while end > start and content[end] not in [' ', '\n', '.', '!', '?']:
                    end -= 1
                if end == start:  # No good break point found
                    end = start + self.config.chunk_size
                    
            chunk_content = content[start:end].strip()
            
            if chunk_content:
                chunks.append({
                    "document_id": document["id"],
                    "content": chunk_content,
                    "chunk_index": chunk_index,
                    "start_position": start,
                    "end_position": end
                })
                chunk_index += 1
                
            start = end - self.config.chunk_overlap if end < len(content) else end
            
        logger.info(f"Created {len(chunks)} chunks for document {document['id']}")
        return chunks
        
    def save_document_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Save document chunks to Neo4j."""
        if not chunks:
            return 0
            
        with self.driver.session() as session:
            # Create chunks
            for chunk in chunks:
                session.run("""
                    MATCH (d:Document {id: $document_id})
                    CREATE (c:DocumentChunk {
                        id: $document_id + '_chunk_' + $chunk_index,
                        content: $content,
                        chunk_index: $chunk_index,
                        start_position: $start_position,
                        end_position: $end_position,
                        created_date: datetime()
                    })
                    CREATE (d)-[:HAS_CHUNK]->(c)
                """, 
                    document_id=chunk["document_id"],
                    content=chunk["content"],
                    chunk_index=str(chunk["chunk_index"]),
                    start_position=chunk["start_position"],
                    end_position=chunk["end_position"]
                )
                
            logger.info(f"Saved {len(chunks)} chunks to Neo4j")
            return len(chunks)
            
    async def process_documents_batch(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch of documents for embedding generation."""
        results = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "chunks_created": 0,
            "errors": []
        }
        
        for doc in tqdm(documents, desc="Processing documents"):
            try:
                doc_id = doc["id"]
                updates = {}
                
                # Generate content embedding if missing
                if not doc.get("content_embedding") and doc.get("content"):
                    content_embedding = await self.generate_embedding(doc["content"])
                    if content_embedding:
                        updates["content_embedding"] = content_embedding
                        
                # Generate title embedding if missing
                if not doc.get("title_embedding") and doc.get("title"):
                    title_embedding = await self.generate_embedding(doc["title"])
                    if title_embedding:
                        updates["title_embedding"] = title_embedding
                        
                # Generate summary embedding if missing
                if not doc.get("summary_embedding") and doc.get("summary"):
                    summary_embedding = await self.generate_embedding(doc["summary"])
                    if summary_embedding:
                        updates["summary_embedding"] = summary_embedding
                        
                # Update document with embeddings
                if updates:
                    with self.driver.session() as session:
                        update_query = "MATCH (d:Document {id: $doc_id}) SET "
                        update_clauses = []
                        
                        for key, value in updates.items():
                            update_clauses.append(f"d.{key} = ${key}")
                            
                        update_query += ", ".join(update_clauses)
                        
                        session.run(update_query, doc_id=doc_id, **updates)
                        
                # Create chunks for large documents
                if doc.get("content") and len(doc["content"]) > self.config.chunk_size:
                    chunks = self.create_document_chunks(doc)
                    if chunks:
                        chunks_saved = self.save_document_chunks(chunks)
                        results["chunks_created"] += chunks_saved
                        
                results["successful"] += 1
                
            except Exception as e:
                error_msg = f"Failed to process document {doc.get('id', 'unknown')}: {e}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
                results["failed"] += 1
                
            results["processed"] += 1
            
        return results
        
    async def process_chunks_batch(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch of chunks for embedding generation."""
        results = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "errors": []
        }
        
        for chunk in tqdm(chunks, desc="Processing chunks"):
            try:
                chunk_id = chunk["id"]
                content = chunk.get("content", "")
                
                if not content:
                    logger.warning(f"Empty content for chunk {chunk_id}")
                    continue
                    
                # Generate embedding
                embedding = await self.generate_embedding(content)
                
                if embedding:
                    # Update chunk with embedding
                    with self.driver.session() as session:
                        session.run("""
                            MATCH (c:DocumentChunk {id: $chunk_id})
                            SET c.content_embedding = $embedding
                        """, chunk_id=chunk_id, embedding=embedding)
                        
                    results["successful"] += 1
                else:
                    results["failed"] += 1
                    
            except Exception as e:
                error_msg = f"Failed to process chunk {chunk.get('id', 'unknown')}: {e}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
                results["failed"] += 1
                
            results["processed"] += 1
            
        return results
        
    async def populate_all_embeddings(self) -> Dict[str, Any]:
        """Populate embeddings for all documents and chunks."""
        logger.info("Starting embedding population for EHS Analytics...")
        
        summary = {
            "start_time": time.time(),
            "documents": {"total": 0, "processed": 0, "successful": 0, "failed": 0},
            "chunks": {"total": 0, "processed": 0, "successful": 0, "failed": 0},
            "chunks_created": 0,
            "success": False,
            "errors": []
        }
        
        try:
            # Phase 1: Process documents
            logger.info("=" * 50)
            logger.info("PHASE 1: Processing Documents")
            logger.info("=" * 50)
            
            documents = self.get_documents_without_embeddings()
            summary["documents"]["total"] = len(documents)
            
            if documents:
                # Process in batches
                for i in range(0, len(documents), self.config.batch_size):
                    batch = documents[i:i + self.config.batch_size]
                    logger.info(f"Processing document batch {i//self.config.batch_size + 1} "
                              f"({len(batch)} documents)")
                    
                    batch_results = await self.process_documents_batch(batch)
                    
                    summary["documents"]["processed"] += batch_results["processed"]
                    summary["documents"]["successful"] += batch_results["successful"]
                    summary["documents"]["failed"] += batch_results["failed"]
                    summary["chunks_created"] += batch_results["chunks_created"]
                    summary["errors"].extend(batch_results["errors"])
                    
            # Phase 2: Process chunks (including newly created ones)
            logger.info("=" * 50)
            logger.info("PHASE 2: Processing Document Chunks")
            logger.info("=" * 50)
            
            chunks = self.get_chunks_without_embeddings()
            summary["chunks"]["total"] = len(chunks)
            
            if chunks:
                # Process in batches
                for i in range(0, len(chunks), self.config.batch_size):
                    batch = chunks[i:i + self.config.batch_size]
                    logger.info(f"Processing chunk batch {i//self.config.batch_size + 1} "
                              f"({len(batch)} chunks)")
                    
                    batch_results = await self.process_chunks_batch(batch)
                    
                    summary["chunks"]["processed"] += batch_results["processed"]
                    summary["chunks"]["successful"] += batch_results["successful"]
                    summary["chunks"]["failed"] += batch_results["failed"]
                    summary["errors"].extend(batch_results["errors"])
                    
            # Calculate success
            total_processed = summary["documents"]["processed"] + summary["chunks"]["processed"]
            total_successful = summary["documents"]["successful"] + summary["chunks"]["successful"]
            
            summary["success"] = (total_processed > 0 and 
                                len(summary["errors"]) == 0 and
                                total_successful == total_processed)
                                
        except Exception as e:
            error_msg = f"Embedding population failed: {e}"
            logger.error(error_msg)
            summary["errors"].append(error_msg)
            
        finally:
            summary["end_time"] = time.time()
            summary["duration"] = summary["end_time"] - summary["start_time"]
            
        return summary


def print_summary(summary: Dict[str, Any]):
    """Print detailed execution summary."""
    logger.info("=" * 70)
    logger.info("EHS ANALYTICS EMBEDDING POPULATION SUMMARY")
    logger.info("=" * 70)
    
    logger.info(f"Execution Time: {summary.get('duration', 0):.2f} seconds")
    logger.info(f"Success: {'✓' if summary.get('success') else '✗'}")
    
    # Document stats
    docs = summary.get("documents", {})
    logger.info(f"Documents - Total: {docs.get('total', 0)}, "
               f"Processed: {docs.get('processed', 0)}, "
               f"Successful: {docs.get('successful', 0)}, "
               f"Failed: {docs.get('failed', 0)}")
    
    # Chunk stats
    chunks = summary.get("chunks", {})
    logger.info(f"Chunks - Total: {chunks.get('total', 0)}, "
               f"Processed: {chunks.get('processed', 0)}, "
               f"Successful: {chunks.get('successful', 0)}, "
               f"Failed: {chunks.get('failed', 0)}")
    
    logger.info(f"New Chunks Created: {summary.get('chunks_created', 0)}")
    
    # Errors
    errors = summary.get("errors", [])
    if errors:
        logger.info(f"Errors Encountered: {len(errors)}")
        for i, error in enumerate(errors[:5], 1):  # Show first 5 errors
            logger.error(f"  {i}. {error}")
        if len(errors) > 5:
            logger.error(f"  ... and {len(errors) - 5} more errors")
    
    logger.info("=" * 70)


async def main():
    """Main function to populate embeddings."""
    # Get configuration from environment
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "EhsAI2024!")
    
    # Create configuration
    config = EmbeddingConfig(
        batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "50")),
        rate_limit_delay=float(os.getenv("EMBEDDING_RATE_DELAY", "0.1")),
        chunk_size=int(os.getenv("EMBEDDING_CHUNK_SIZE", "1000"))
    )
    
    logger.info("Starting EHS Analytics embedding population...")
    logger.info(f"Configuration - Model: {config.model}, "
               f"Batch Size: {config.batch_size}, "
               f"Chunk Size: {config.chunk_size}")
    
    populator = EmbeddingPopulator(uri, username, password, config)
    
    try:
        summary = await populator.populate_all_embeddings()
        print_summary(summary)
        
        if summary.get("success"):
            logger.info("🎉 Embedding population completed successfully!")
            return 0
        else:
            logger.error("❌ Embedding population completed with errors.")
            return 1
            
    except Exception as e:
        logger.error(f"Critical error during embedding population: {e}")
        return 1
        
    finally:
        populator.close()


if __name__ == "__main__":
    # Run the async main function
    sys.exit(asyncio.run(main()))