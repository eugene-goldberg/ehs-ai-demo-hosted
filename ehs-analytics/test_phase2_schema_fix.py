#!/usr/bin/env python3
"""
Comprehensive Text2Cypher Retriever Test Script
Tests the retriever after schema fixes with detailed logging
"""

import sys
import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any
import traceback
import json

# Add the src directory to path
sys.path.append('/Users/eugene/dev/ai/agentos/ehs-ai-demo/ehs-analytics/src')

from ehs_analytics.retrieval.strategies.text2cypher import Text2CypherRetriever
from ehs_analytics.core.config import EHSSettings
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

class ComprehensiveText2CypherTest:
    """Comprehensive testing of Text2Cypher retriever"""
    
    def __init__(self):
        self.settings = EHSSettings()
        self.retriever = None
        self.test_queries = [
            'Show me all facilities',
            'What equipment do we have?',
            'List active permits', 
            'Show water consumption for all facilities',
            'What are the CO2 emissions?',
            'Show permits expiring in next 30 days'
        ]
        
    async def initialize_retriever(self):
        """Initialize the Text2Cypher retriever"""
        try:
            logger.info('='*60)
            logger.info('INITIALIZING TEXT2CYPHER RETRIEVER')
            logger.info('='*60)
            
            self.retriever = Text2CypherRetriever(
                neo4j_url=self.settings.NEO4J_URL,
                neo4j_user=self.settings.NEO4J_USER,
                neo4j_password=self.settings.NEO4J_PASSWORD,
                openai_api_key=self.settings.OPENAI_API_KEY
            )
            
            # Initialize the retriever (this loads the schema)
            await self.retriever.initialize()
            logger.info('✅ Text2Cypher retriever initialized successfully')
            
            return True
            
        except Exception as e:
            logger.error(f'❌ Failed to initialize Text2Cypher retriever: {str(e)}')
            logger.error(traceback.format_exc())
            return False
    
    def verify_neo4j_connection(self):
        """Verify Neo4j connection and log schema info"""
        try:
            logger.info('')
            logger.info('='*60)
            logger.info('VERIFYING NEO4J CONNECTION AND SCHEMA')
            logger.info('='*60)
            
            driver = GraphDatabase.driver(
                self.settings.NEO4J_URL,
                auth=(self.settings.NEO4J_USER, self.settings.NEO4J_PASSWORD)
            )
            
            with driver.session() as session:
                # Get node count
                result = session.run('MATCH (n) RETURN count(n) as total')
                total_nodes = result.single()['total']
                logger.info(f'📊 Total nodes in database: {total_nodes}')
                
                # Get node labels
                result = session.run('CALL db.labels()')
                labels = [record['label'] for record in result]
                logger.info(f'📋 Node labels: {labels}')
                
                # Get relationship types
                result = session.run('CALL db.relationshipTypes()')
                relationships = [record['relationshipType'] for record in result]
                logger.info(f'🔗 Relationship types: {relationships}')
                
                # Sample data from each label
                for label in labels:
                    try:
                        result = session.run(f'MATCH (n:{label}) RETURN n LIMIT 3')
                        records = list(result)
                        logger.info(f'')
                        logger.info(f'📋 Sample {label} nodes: {len(records)} found')
                        for i, record in enumerate(records[:2]):  # Show first 2
                            node_props = dict(record['n'])
                            logger.info(f'   Node {i+1}: {node_props}')
                    except Exception as e:
                        logger.warning(f'Could not query {label}: {e}')
                        
            driver.close()
            logger.info('✅ Neo4j connection verified successfully')
            return True
            
        except Exception as e:
            logger.error(f'❌ Failed to verify Neo4j connection: {str(e)}')
            return False
    
    async def test_single_query(self, query: str, query_index: int) -> Dict[str, Any]:
        """Test a single query and return detailed results"""
        logger.info('')
        logger.info('-'*50)
        logger.info(f'TEST {query_index + 1}/6: {query}')
        logger.info('-'*50)
        
        test_result = {
            'query': query,
            'success': False,
            'generated_cypher': None,
            'execution_error': None,
            'result_count': 0,
            'results': [],
            'execution_time': 0
        }
        
        try:
            start_time = datetime.now()
            
            # Execute query
            results = await self.retriever.retrieve(query)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            test_result['execution_time'] = execution_time
            
            if hasattr(self.retriever, '_last_generated_cypher'):
                test_result['generated_cypher'] = self.retriever._last_generated_cypher
                logger.info(f'🔍 Generated Cypher: {test_result["generated_cypher"]}')
            
            if results:
                test_result['success'] = True
                test_result['result_count'] = len(results)
                test_result['results'] = results
                
                logger.info(f'✅ Query successful! Found {len(results)} results')
                logger.info(f'⏱️ Execution time: {execution_time:.2f}s')
                
                # Log first few results
                for i, result in enumerate(results[:3]):
                    logger.info(f'   Result {i+1}: {result}')
                    
                if len(results) > 3:
                    logger.info(f'   ... and {len(results) - 3} more results')
                    
            else:
                logger.warning(f'⚠️ Query executed but returned 0 results')
                logger.info(f'⏱️ Execution time: {execution_time:.2f}s')
                test_result['success'] = True  # No error, just empty results
                
        except Exception as e:
            error_msg = str(e)
            test_result['execution_error'] = error_msg
            logger.error(f'❌ Query failed: {error_msg}')
            logger.error(traceback.format_exc())
            
        return test_result
    
    async def run_comprehensive_test(self):
        """Run the complete test suite"""
        logger.info('')
        logger.info('='*80)
        logger.info('STARTING COMPREHENSIVE TEXT2CYPHER RETRIEVER TEST')
        logger.info('='*80)
        logger.info(f'Test started at: {datetime.now()}')
        logger.info(f'Testing {len(self.test_queries)} queries')
        
        # Step 1: Verify Neo4j connection
        if not self.verify_neo4j_connection():
            logger.error('❌ Neo4j connection failed - aborting test')
            return
        
        # Step 2: Initialize retriever  
        if not await self.initialize_retriever():
            logger.error('❌ Retriever initialization failed - aborting test')
            return
            
        # Step 3: Run all test queries
        test_results = []
        successful_queries = 0
        queries_with_results = 0
        
        for i, query in enumerate(self.test_queries):
            result = await self.test_single_query(query, i)
            test_results.append(result)
            
            if not result['execution_error']:
                successful_queries += 1
                if result['result_count'] > 0:
                    queries_with_results += 1
        
        # Step 4: Generate comprehensive summary
        logger.info('')
        logger.info('='*80)
        logger.info('TEST SUMMARY')
        logger.info('='*80)
        
        logger.info(f'📊 Total Queries: {len(self.test_queries)}')
        logger.info(f'✅ Successful Executions: {successful_queries}')
        logger.info(f'📋 Queries with Results: {queries_with_results}')
        logger.info(f'❌ Failed Queries: {len(self.test_queries) - successful_queries}')
        
        # Detailed results
        logger.info('')
        logger.info('📋 DETAILED RESULTS:')
        for i, result in enumerate(test_results):
            status = '✅' if not result['execution_error'] else '❌'
            result_info = f"{result['result_count']} results" if result['result_count'] > 0 else 'no results'
            time_info = f"{result['execution_time']:.2f}s"
            
            logger.info(f'{status} Query {i+1}: "{result["query"]}" - {result_info} ({time_info})')
            
            if result['generated_cypher']:
                logger.info(f'    Cypher: {result["generated_cypher"]}')
                
            if result['execution_error']:
                logger.info(f'    Error: {result["execution_error"]}')
        
        # Performance summary
        total_time = sum(r['execution_time'] for r in test_results if r['execution_time'] > 0)
        avg_time = total_time / len(test_results) if test_results else 0
        
        logger.info('')
        logger.info('⏱️ PERFORMANCE SUMMARY:')
        logger.info(f'Total execution time: {total_time:.2f}s')
        logger.info(f'Average time per query: {avg_time:.2f}s')
        
        # Schema alignment assessment
        logger.info('')
        logger.info('🔧 SCHEMA ALIGNMENT ASSESSMENT:')
        if queries_with_results == 0 and successful_queries > 0:
            logger.warning('⚠️ All queries executed successfully but returned 0 results')
            logger.warning('⚠️ This indicates schema mismatch in Cypher prompt templates')
            logger.info('✏️ Fix needed: Update prompt examples in text2cypher.py')
        elif queries_with_results > 0:
            logger.info('✅ Schema alignment appears correct - queries returning data')
        else:
            logger.error('❌ Multiple execution failures - investigate errors above')
            
        logger.info('')
        logger.info('='*80)
        logger.info('TEST COMPLETED')
        logger.info('='*80)
        logger.info(f'Test completed at: {datetime.now()}')
        
        # Save detailed results to JSON
        results_file = '/Users/eugene/dev/ai/agentos/ehs-ai-demo/ehs-analytics/test_results_phase2_schema_fix.json'
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_queries': len(self.test_queries),
                    'successful_executions': successful_queries, 
                    'queries_with_results': queries_with_results,
                    'failed_queries': len(self.test_queries) - successful_queries,
                    'total_time': total_time,
                    'average_time': avg_time
                },
                'detailed_results': test_results
            }, indent=2)
            
        logger.info(f'📄 Detailed results saved to: {results_file}')

async def main():
    """Main test execution function"""
    try:
        test_runner = ComprehensiveText2CypherTest()
        await test_runner.run_comprehensive_test()
    except Exception as e:
        logger.error(f'Fatal error in test execution: {str(e)}')
        logger.error(traceback.format_exc())

if __name__ == '__main__':
    asyncio.run(main())
