#!/usr/bin/env python3
"""
Corrected Text2Cypher Test Script
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_test():
    """Run Text2Cypher retriever test with correct initialization"""
    
    logger.info('='*60)
    logger.info('TEXT2CYPHER RETRIEVER TEST (CORRECTED)')
    logger.info('='*60)
    
    test_queries = [
        "Show me all facilities",
        "What equipment do we have?",
        "List active permits", 
        "Show water consumption for all facilities",
        "What are the CO2 emissions?",
        "Show permits expiring in next 30 days"
    ]
    
    try:
        settings = Settings()
        logger.info(f'Neo4j URI: {settings.neo4j_uri}')
        
        # Create config dictionary for Text2CypherRetriever
        config = {
            'neo4j_uri': settings.neo4j_uri,
            'neo4j_user': settings.neo4j_username,
            'neo4j_password': settings.neo4j_password,
            'openai_api_key': settings.openai_api_key,
            'model_name': 'gpt-3.5-turbo'
        }
        
        logger.info('Initializing Text2Cypher retriever...')
        retriever = Text2CypherRetriever(config)
        await retriever.initialize()
        logger.info('✅ Text2Cypher retriever initialized')
        
        results_summary = []
        successful_queries = 0
        queries_with_results = 0
        
        for i, query in enumerate(test_queries):
            logger.info(f'\n{"="*50}')
            logger.info(f'TEST {i+1}/6: {query}')
            logger.info("="*50)
            
            try:
                start_time = datetime.now()
                results = await retriever.retrieve(query)
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # Check for generated Cypher if available
                cypher_query = getattr(retriever, '_last_generated_cypher', 'Not available')
                
                if results:
                    successful_queries += 1
                    queries_with_results += 1
                    logger.info(f'✅ SUCCESS: Found {len(results)} results ({execution_time:.2f}s)')
                    logger.info(f'Generated Cypher: {cypher_query}')
                    
                    # Show first few results
                    for j, result in enumerate(results[:3]):
                        logger.info(f'   Result {j+1}: {result}')
                    if len(results) > 3:
                        logger.info(f'   ... and {len(results)-3} more')
                        
                else:
                    successful_queries += 1
                    logger.warning(f'⚠️ Query executed but returned 0 results ({execution_time:.2f}s)')
                    logger.info(f'Generated Cypher: {cypher_query}')
                    
                results_summary.append({
                    'query': query,
                    'success': True,
                    'result_count': len(results) if results else 0,
                    'execution_time': execution_time,
                    'cypher': str(cypher_query)
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
        logger.info(f'\n{"="*60}')
        logger.info('TEST RESULTS SUMMARY')
        logger.info('='*60)
        
        logger.info(f'📊 Total Queries: {len(test_queries)}')
        logger.info(f'✅ Successful Executions: {successful_queries}')
        logger.info(f'📋 Queries with Results: {queries_with_results}')
        logger.info(f'❌ Failed Queries: {len(test_queries) - successful_queries}')
        
        # Schema assessment
        if queries_with_results == 0 and successful_queries > 0:
            logger.warning('⚠️ SCHEMA MISMATCH: Queries execute but return no data')
            logger.info('✏️ ACTION NEEDED: Update Cypher prompt examples in text2cypher.py')
            assessment = "SCHEMA_MISMATCH"
        elif queries_with_results > 0:
            logger.info('✅ SCHEMA ALIGNMENT: Queries returning data successfully')
            assessment = "SCHEMA_ALIGNED"
        else:
            logger.error('❌ EXECUTION ERRORS: Multiple query failures')
            assessment = "EXECUTION_ERRORS"
            
        # Save results
        with open('test_results_phase2_schema_fix.json', 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'assessment': assessment,
                'summary': {
                    'total': len(test_queries),
                    'successful': successful_queries,
                    'with_results': queries_with_results
                },
                'details': results_summary
            }, f, indent=2)
            
        logger.info('📄 Results saved to test_results_phase2_schema_fix.json')
        logger.info(f'🔧 Schema Assessment: {assessment}')
        logger.info('✅ Test completed!')
        
        return assessment
        
    except Exception as e:
        logger.error(f'❌ Fatal error: {str(e)}')
        logger.error(traceback.format_exc())
        return "FATAL_ERROR"

if __name__ == '__main__':
    result = asyncio.run(run_test())
    print(f"\nFINAL ASSESSMENT: {result}")
