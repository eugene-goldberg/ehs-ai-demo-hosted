#!/usr/bin/env python3
"""
Final Text2Cypher Test Script with Correct Configuration
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

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/eugene/dev/ai/agentos/ehs-ai-demo/ehs-analytics/test_phase2_schema_fix.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def run_comprehensive_test():
    """Run comprehensive Text2Cypher retriever test"""
    
    logger.info('='*80)
    logger.info('COMPREHENSIVE TEXT2CYPHER RETRIEVER TEST')
    logger.info('='*80)
    logger.info(f'Test started at: {datetime.now()}')
    
    # Test queries covering all major EHS areas
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
        logger.info(f'Connecting to Neo4j at: {settings.neo4j_uri}')
        logger.info(f'Username: {settings.neo4j_username}')
        logger.info(f'Database: {settings.neo4j_database}')
        
        # Step 1: Verify Neo4j connection and schema
        logger.info('\n' + '='*60)
        logger.info('STEP 1: VERIFYING NEO4J CONNECTION AND SCHEMA')
        logger.info('='*60)
        
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password)
        )
        
        with driver.session() as session:
            # Get total node count
            result = session.run("MATCH (n) RETURN count(n) as total")
            total_nodes = result.single()['total']
            logger.info(f'📊 Total nodes in database: {total_nodes}')
            
            # Get node labels
            result = session.run("CALL db.labels()")
            labels = [record['label'] for record in result]
            logger.info(f'📋 Node labels: {labels}')
            
            # Get relationship types
            result = session.run("CALL db.relationshipTypes()")
            relationships = [record['relationshipType'] for record in result]
            logger.info(f'🔗 Relationship types: {relationships}')
            
            # Sample nodes from each label
            for label in labels:
                try:
                    result = session.run(f"MATCH (n:{label}) RETURN n LIMIT 2")
                    nodes = list(result)
                    logger.info(f'📋 Sample {label} nodes ({len(nodes)} shown):')
                    for i, record in enumerate(nodes):
                        node_props = dict(record['n'])
                        logger.info(f'   {i+1}. {node_props}')
                except Exception as e:
                    logger.warning(f'Could not sample {label}: {e}')
        
        driver.close()
        logger.info('✅ Neo4j connection and schema verified')
        
        # Step 2: Initialize Text2Cypher retriever
        logger.info('\n' + '='*60)
        logger.info('STEP 2: INITIALIZING TEXT2CYPHER RETRIEVER')
        logger.info('='*60)
        
        retriever = Text2CypherRetriever(
            neo4j_url=settings.neo4j_uri,
            neo4j_user=settings.neo4j_username,
            neo4j_password=settings.neo4j_password,
            openai_api_key=settings.openai_api_key
        )
        
        await retriever.initialize()
        logger.info('✅ Text2Cypher retriever initialized successfully')
        
        # Step 3: Test all queries
        logger.info('\n' + '='*60)
        logger.info('STEP 3: TESTING QUERIES')
        logger.info('='*60)
        
        test_results = []
        successful_queries = 0
        queries_with_results = 0
        total_execution_time = 0
        
        for i, query in enumerate(test_queries):
            logger.info('\n' + '-'*50)
            logger.info(f'TEST {i+1}/6: {query}')
            logger.info('-'*50)
            
            test_result = {
                'query': query,
                'success': False,
                'result_count': 0,
                'execution_time': 0,
                'cypher': None,
                'error': None,
                'results_sample': []
            }
            
            try:
                start_time = datetime.now()
                results = await retriever.retrieve(query)
                execution_time = (datetime.now() - start_time).total_seconds()
                
                test_result['execution_time'] = execution_time
                total_execution_time += execution_time
                
                # Try to capture generated Cypher
                if hasattr(retriever, '_last_generated_cypher'):
                    test_result['cypher'] = retriever._last_generated_cypher
                    logger.info(f'🔍 Generated Cypher: {test_result["cypher"]}')
                
                if results:
                    test_result['success'] = True
                    test_result['result_count'] = len(results)
                    test_result['results_sample'] = results[:3]  # Store first 3 results
                    
                    successful_queries += 1
                    queries_with_results += 1
                    
                    logger.info(f'✅ SUCCESS: Found {len(results)} results ({execution_time:.2f}s)')
                    for j, result in enumerate(results[:3]):
                        logger.info(f'   Result {j+1}: {result}')
                    if len(results) > 3:
                        logger.info(f'   ... and {len(results)-3} more results')
                        
                else:
                    test_result['success'] = True  # No error, just no results
                    successful_queries += 1
                    logger.warning(f'⚠️ Query executed successfully but returned 0 results ({execution_time:.2f}s)')
                    
            except Exception as e:
                test_result['error'] = str(e)
                logger.error(f'❌ Query failed: {str(e)}')
                logger.error(traceback.format_exc())
                
            test_results.append(test_result)
        
        # Step 4: Generate comprehensive summary
        logger.info('\n' + '='*80)
        logger.info('STEP 4: TEST RESULTS SUMMARY')
        logger.info('='*80)
        
        failed_queries = len(test_queries) - successful_queries
        avg_execution_time = total_execution_time / len(test_queries) if test_queries else 0
        
        logger.info(f'📊 EXECUTION SUMMARY:')
        logger.info(f'   Total Queries: {len(test_queries)}')
        logger.info(f'   ✅ Successful Executions: {successful_queries}')
        logger.info(f'   📋 Queries with Results: {queries_with_results}')
        logger.info(f'   ❌ Failed Queries: {failed_queries}')
        logger.info(f'   ⏱️ Total Execution Time: {total_execution_time:.2f}s')
        logger.info(f'   ⏱️ Average Time per Query: {avg_execution_time:.2f}s')
        
        logger.info('\n📋 DETAILED QUERY RESULTS:')
        for i, result in enumerate(test_results):
            status = '✅' if result['success'] and result['result_count'] > 0 else '⚠️' if result['success'] else '❌'
            result_info = f"{result['result_count']} results" if result['result_count'] > 0 else 'no results'
            time_info = f"{result['execution_time']:.2f}s"
            
            logger.info(f'{status} Query {i+1}: "{result["query"]}" - {result_info} ({time_info})')
            
            if result['cypher']:
                logger.info(f'    Cypher: {result["cypher"]}')
                
            if result['error']:
                logger.info(f'    Error: {result["error"]}')
        
        # Schema alignment assessment
        logger.info('\n🔧 SCHEMA ALIGNMENT ASSESSMENT:')
        if queries_with_results == 0 and successful_queries > 0:
            logger.warning('⚠️ SCHEMA MISMATCH DETECTED:')
            logger.warning('   - All queries executed successfully without errors')
            logger.warning('   - But all queries returned 0 results')
            logger.warning('   - This indicates schema mismatch in Cypher prompt templates')
            logger.info('✏️ FIX NEEDED: Update prompt examples in text2cypher.py to match actual Neo4j schema')
            assessment = "SCHEMA_MISMATCH"
        elif queries_with_results > 0:
            logger.info('✅ SCHEMA ALIGNMENT CORRECT:')
            logger.info('   - Queries are successfully returning data')
            logger.info('   - Cypher generation appears to match database schema')
            assessment = "SCHEMA_ALIGNED"
        else:
            logger.error('❌ MULTIPLE EXECUTION FAILURES:')
            logger.error('   - Multiple queries failed to execute')
            logger.error('   - Review error messages above for debugging')
            assessment = "EXECUTION_ERRORS"
        
        # Performance assessment
        logger.info('\n⚡ PERFORMANCE ASSESSMENT:')
        if avg_execution_time < 2.0:
            logger.info('✅ PERFORMANCE: Excellent (< 2.0s average)')
        elif avg_execution_time < 5.0:
            logger.info('⚠️ PERFORMANCE: Good (< 5.0s average)')
        else:
            logger.warning('⚠️ PERFORMANCE: Needs optimization (> 5.0s average)')
            
        # Save detailed results to JSON
        results_file = '/Users/eugene/dev/ai/agentos/ehs-ai-demo/ehs-analytics/test_results_phase2_schema_fix.json'
        detailed_results = {
            'timestamp': datetime.now().isoformat(),
            'test_info': {
                'neo4j_uri': settings.neo4j_uri,
                'neo4j_database': settings.neo4j_database,
                'total_nodes': total_nodes,
                'node_labels': labels,
                'relationship_types': relationships
            },
            'summary': {
                'total_queries': len(test_queries),
                'successful_executions': successful_queries,
                'queries_with_results': queries_with_results,
                'failed_queries': failed_queries,
                'total_execution_time': total_execution_time,
                'average_execution_time': avg_execution_time,
                'schema_assessment': assessment
            },
            'detailed_results': test_results
        }
        
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
            
        logger.info('\n' + '='*80)
        logger.info('TEST COMPLETED SUCCESSFULLY')
        logger.info('='*80)
        logger.info(f'📄 Detailed results saved to: {results_file}')
        logger.info(f'📊 Schema Assessment: {assessment}')
        logger.info(f'⏱️ Total Test Duration: {total_execution_time:.2f}s')
        logger.info(f'Test completed at: {datetime.now()}')
        
        return assessment, test_results
        
    except Exception as e:
        logger.error(f'❌ FATAL ERROR in test execution: {str(e)}')
        logger.error(traceback.format_exc())
        return "FATAL_ERROR", []

if __name__ == '__main__':
    asyncio.run(run_comprehensive_test())
