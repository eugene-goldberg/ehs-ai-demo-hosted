#!/usr/bin/env python3
"""
Simplified Text2Cypher Test Script
"""

import sys
import os
import logging
import asyncio
from datetime import datetime
import traceback
import json

# Add the src directory to path
sys.path.append('/Users/eugene/dev/ai/agentos/ehs-ai-demo/ehs-analytics/src')

from ehs_analytics.retrieval.strategies.text2cypher import Text2CypherRetriever
from ehs_analytics.config import Settings
from neo4j import GraphDatabase

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/eugene/dev/ai/agentos/ehs-ai-demo/ehs-analytics/test_phase2_schema_fix.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def run_test():
    """Run comprehensive Text2Cypher test"""
    logger.info('='*80)
    logger.info('STARTING TEXT2CYPHER RETRIEVER TEST')
    logger.info('='*80)
    
    # Test queries
    test_queries = [
        "Show me all facilities",
        "What equipment do we have?", 
        "List active permits",
        "Show water consumption for all facilities",
        "What are the CO2 emissions?",
        "Show permits expiring in next 30 days"
    ]
    
    try:
        # Initialize settings
        settings = Settings()
        logger.info(f'Connecting to Neo4j at: {settings.NEO4J_URL}')
        
        # Test Neo4j connection first
        driver = GraphDatabase.driver(
            settings.NEO4J_URL,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )
        
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as total")
            total_nodes = result.single()['total']
            logger.info(f'📊 Total nodes in database: {total_nodes}')
            
            # Get schema info
            result = session.run("CALL db.labels()")
            labels = [record['label'] for record in result]
            logger.info(f'📋 Node labels: {labels}')
            
            result = session.run("CALL db.relationshipTypes()")  
            relationships = [record['relationshipType'] for record in result]
            logger.info(f'🔗 Relationship types: {relationships}')
            
        driver.close()
        logger.info('✅ Neo4j connection verified')
        
        # Initialize Text2Cypher retriever
        logger.info('\nInitializing Text2Cypher retriever...')
        retriever = Text2CypherRetriever(
            neo4j_url=settings.NEO4J_URL,
            neo4j_user=settings.NEO4J_USER,
            neo4j_password=settings.NEO4J_PASSWORD,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        await retriever.initialize()
        logger.info('✅ Text2Cypher retriever initialized')
        
        # Test all queries
        results_summary = []
        
        for i, query in enumerate(test_queries):
            logger.info(f'\n{"="*50}')
            logger.info(f'TEST {i+1}/6: {query}')
            logger.info("="*50)
            
            try:
                start_time = datetime.now()
                results = await retriever.retrieve(query)
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # Try to get generated Cypher if available
                cypher_query = getattr(retriever, '_last_generated_cypher', 'Not available')
                logger.info(f'🔍 Generated Cypher: {cypher_query}')
                
                if results:
                    logger.info(f'✅ SUCCESS: Found {len(results)} results ({execution_time:.2f}s)')
                    for j, result in enumerate(results[:3]):
                        logger.info(f'   Result {j+1}: {result}')
                    if len(results) > 3:
                        logger.info(f'   ... and {len(results)-3} more')
                else:
                    logger.warning(f'⚠️ Query executed but returned 0 results ({execution_time:.2f}s)')
                    
                results_summary.append({
                    'query': query,
                    'success': True,
                    'result_count': len(results) if results else 0,
                    'execution_time': execution_time,
                    'cypher': cypher_query
                })
                
            except Exception as e:
                logger.error(f'❌ Query failed: {str(e)}')
                results_summary.append({
                    'query': query,
                    'success': False,
                    'error': str(e),
                    'execution_time': 0
                })
        
        # Final summary
        logger.info('\n' + '='*80)
        logger.info('TEST SUMMARY')
        logger.info('='*80)
        
        successful = sum(1 for r in results_summary if r['success'])
        with_results = sum(1 for r in results_summary if r.get('result_count', 0) > 0)
        
        logger.info(f'📊 Total Queries: {len(test_queries)}')
        logger.info(f'✅ Successful Executions: {successful}')
        logger.info(f'📋 Queries with Results: {with_results}')
        
        if with_results == 0 and successful > 0:
            logger.warning('⚠️ Schema alignment issue: queries execute but return no data')
            logger.info('📝 Check Cypher prompt examples in text2cypher.py')
        elif with_results > 0:
            logger.info('✅ Schema alignment appears correct!')
            
        # Save results
        with open('test_results_phase2_schema_fix.json', 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total': len(test_queries),
                    'successful': successful,
                    'with_results': with_results
                },
                'details': results_summary
            }, f, indent=2)
            
        logger.info('📄 Results saved to test_results_phase2_schema_fix.json')
        logger.info('✅ Test completed successfully!')
        
    except Exception as e:
        logger.error(f'❌ Fatal error: {str(e)}')
        logger.error(traceback.format_exc())

if __name__ == '__main__':
    asyncio.run(run_test())
