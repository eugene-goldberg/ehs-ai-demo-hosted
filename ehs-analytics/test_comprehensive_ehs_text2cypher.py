#!/usr/bin/env python3
"""
Comprehensive EHS Text2Cypher Retriever Test Suite

This script tests all 7 EHS query types with both base Text2Cypher and enhanced EHS versions:
1. consumption_analysis queries
2. compliance_check queries  
3. risk_assessment queries
4. emission_tracking queries
5. equipment_efficiency queries
6. permit_status queries
7. general_inquiry queries

Uses examples from ehs_examples.py and verifies:
- Query generation works for each type
- Cypher queries are valid
- Results are returned from Neo4j
- Query optimization happens based on intent
"""

import asyncio
import logging
import time
import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import EHS Analytics components
from ehs_analytics.retrieval.strategies.text2cypher import Text2CypherRetriever
from ehs_analytics.retrieval.strategies.ehs_text2cypher import EHSText2CypherRetriever, EHSQueryIntent
from ehs_analytics.retrieval.ehs_examples import (
    get_consumption_analysis_examples,
    get_compliance_check_examples,
    get_risk_assessment_examples,
    get_emission_tracking_examples,
    get_equipment_efficiency_examples,
    get_permit_status_examples,
    get_general_inquiry_examples,
    get_example_summary
)
from ehs_analytics.retrieval.base import QueryType, RetrievalStrategy
from ehs_analytics.utils.logging import get_ehs_logger

# Configure logging for comprehensive output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'test_ehs_text2cypher_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

logger = get_ehs_logger(__name__)


class EHSText2CypherTestSuite:
    """Comprehensive test suite for EHS Text2Cypher retrievers."""
    
    def __init__(self):
        """Initialize the test suite."""
        self.config = self._load_test_config()
        self.base_retriever: Optional[Text2CypherRetriever] = None
        self.ehs_retriever: Optional[EHSText2CypherRetriever] = None
        
        # Test results tracking
        self.test_results = {
            "start_time": None,
            "end_time": None,
            "total_duration_seconds": 0,
            "base_retriever_tests": {},
            "ehs_retriever_tests": {},
            "query_type_summary": {},
            "success_rates": {},
            "errors": [],
            "performance_metrics": {}
        }
        
        # Query type mappings
        self.query_type_mappings = {
            "consumption_analysis": QueryType.CONSUMPTION,
            "compliance_check": QueryType.COMPLIANCE,
            "risk_assessment": QueryType.RISK,
            "emission_tracking": QueryType.EMISSIONS,
            "equipment_efficiency": QueryType.EFFICIENCY,
            "permit_status": QueryType.COMPLIANCE,  # Use compliance for permit queries
            "general_inquiry": QueryType.GENERAL
        }
        
        # EHS intent mappings
        self.ehs_intent_mappings = {
            "consumption_analysis": EHSQueryIntent.CONSUMPTION_ANALYSIS,
            "compliance_check": EHSQueryIntent.COMPLIANCE_CHECK,
            "risk_assessment": EHSQueryIntent.RISK_ASSESSMENT,
            "emission_tracking": EHSQueryIntent.EMISSION_TRACKING,
            "equipment_efficiency": EHSQueryIntent.EQUIPMENT_EFFICIENCY,
            "permit_status": EHSQueryIntent.PERMIT_STATUS,
            "general_inquiry": EHSQueryIntent.GENERAL_INQUIRY
        }
    
    def _load_test_config(self) -> Dict[str, Any]:
        """Load test configuration from environment or defaults."""
        return {
            "neo4j_uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            "neo4j_user": os.getenv("NEO4J_USER", "neo4j"),
            "neo4j_password": os.getenv("NEO4J_PASSWORD", "ehs_analytics_2024!"),
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.0,
            "max_tokens": 2000,
            "test_timeout": 300,  # 5 minutes per test
            "result_limit": 5,    # Limit results for faster testing
            "use_graphrag": True,
            "query_optimization": True,
            "cache_common_queries": False  # Disable caching for testing
        }
    
    async def setup(self) -> bool:
        """Set up the test environment and initialize retrievers."""
        logger.info("Setting up EHS Text2Cypher test environment")
        
        try:
            # Validate configuration
            if not self.config.get("openai_api_key"):
                logger.error("OpenAI API key not configured")
                return False
            
            # Initialize base Text2Cypher retriever
            logger.info("Initializing base Text2Cypher retriever")
            self.base_retriever = Text2CypherRetriever(self.config)
            await self.base_retriever.initialize()
            logger.info("Base Text2Cypher retriever initialized successfully")
            
            # Initialize EHS Text2Cypher retriever
            logger.info("Initializing EHS Text2Cypher retriever")
            self.ehs_retriever = EHSText2CypherRetriever(self.config)
            await self.ehs_retriever.initialize()
            logger.info("EHS Text2Cypher retriever initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup test environment: {e}", exc_info=True)
            self.test_results["errors"].append({
                "phase": "setup",
                "error": str(e),
                "error_type": type(e).__name__
            })
            return False
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive tests for all EHS query types."""
        logger.info("Starting comprehensive EHS Text2Cypher tests")
        self.test_results["start_time"] = datetime.now().isoformat()
        start_time = time.time()
        
        try:
            # Get example summary
            example_summary = get_example_summary()
            logger.info(f"Testing with {example_summary['total_examples']} examples across {len(example_summary['intent_types'])} intent types")
            
            # Test each query type
            for intent_type in example_summary["intent_types"]:
                logger.info(f"\n{'='*60}")
                logger.info(f"Testing {intent_type.upper()} queries")
                logger.info(f"{'='*60}")
                
                await self._test_query_type(intent_type)
            
            # Generate summary
            await self._generate_test_summary()
            
        except Exception as e:
            logger.error(f"Comprehensive test execution failed: {e}", exc_info=True)
            self.test_results["errors"].append({
                "phase": "execution",
                "error": str(e),
                "error_type": type(e).__name__
            })
        
        finally:
            self.test_results["end_time"] = datetime.now().isoformat()
            self.test_results["total_duration_seconds"] = time.time() - start_time
            
            # Save detailed results
            await self._save_test_results()
        
        return self.test_results
    
    async def _test_query_type(self, intent_type: str) -> None:
        """Test a specific query type with all available examples."""
        # Get examples for this intent type
        examples = self._get_examples_for_intent(intent_type)
        
        if not examples:
            logger.warning(f"No examples found for intent type: {intent_type}")
            return
        
        query_type = self.query_type_mappings.get(intent_type, QueryType.GENERAL)
        
        # Initialize results for this query type
        base_results = []
        ehs_results = []
        
        logger.info(f"Testing {len(examples)} examples for {intent_type}")
        
        for i, example in enumerate(examples, 1):
            question = example.get("question", "")
            expected_cypher = example.get("cypher", "")
            description = example.get("description", "")
            
            logger.info(f"\n--- Example {i}/{len(examples)} ---")
            logger.info(f"Question: {question}")
            logger.info(f"Description: {description}")
            
            # Test with base retriever
            base_result = await self._test_single_query(
                self.base_retriever,
                "base",
                question,
                query_type,
                intent_type,
                expected_cypher
            )
            base_results.append(base_result)
            
            # Test with EHS retriever
            ehs_result = await self._test_single_query(
                self.ehs_retriever,
                "ehs",
                question,
                query_type,
                intent_type,
                expected_cypher
            )
            ehs_results.append(ehs_result)
            
            # Compare results
            self._compare_retriever_results(base_result, ehs_result, intent_type, i)
        
        # Store results for this query type
        self.test_results["base_retriever_tests"][intent_type] = base_results
        self.test_results["ehs_retriever_tests"][intent_type] = ehs_results
        
        # Generate query type summary
        self._generate_query_type_summary(intent_type, base_results, ehs_results)
    
    async def _test_single_query(
        self,
        retriever: Any,
        retriever_type: str,
        question: str,
        query_type: QueryType,
        intent_type: str,
        expected_cypher: str
    ) -> Dict[str, Any]:
        """Test a single query with a specific retriever."""
        test_result = {
            "retriever_type": retriever_type,
            "question": question,
            "query_type": query_type.value,
            "intent_type": intent_type,
            "expected_cypher": expected_cypher,
            "success": False,
            "execution_time_ms": 0,
            "result_count": 0,
            "confidence_score": 0.0,
            "generated_cypher": "",
            "error": None,
            "metadata": {}
        }
        
        start_time = time.time()
        
        try:
            logger.info(f"Testing with {retriever_type} retriever...")
            
            # Execute query
            result = await retriever.retrieve(
                query=question,
                query_type=query_type,
                limit=self.config["result_limit"]
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            # Extract results
            test_result.update({
                "success": result.success,
                "execution_time_ms": execution_time,
                "result_count": len(result.data) if result.data else 0,
                "confidence_score": result.metadata.confidence_score if result.metadata else 0.0,
                "generated_cypher": result.metadata.cypher_query if result.metadata else "",
                "error": result.message if not result.success else None,
                "metadata": {
                    "nodes_retrieved": result.metadata.nodes_retrieved if result.metadata else 0,
                    "relationships_retrieved": result.metadata.relationships_retrieved if result.metadata else 0,
                    "strategy": result.metadata.strategy.value if result.metadata else "unknown"
                }
            })
            
            # Log results
            if result.success:
                logger.info(f"✅ {retriever_type.upper()} SUCCESS: {test_result['result_count']} results in {execution_time:.1f}ms")
                logger.info(f"   Generated Cypher: {test_result['generated_cypher'][:100]}...")
                logger.info(f"   Confidence: {test_result['confidence_score']:.2f}")
            else:
                logger.error(f"❌ {retriever_type.upper()} FAILED: {test_result['error']}")
            
            # Validate Cypher query if generated
            if test_result["generated_cypher"]:
                cypher_valid = await self._validate_cypher_query(test_result["generated_cypher"])
                test_result["cypher_valid"] = cypher_valid
                if cypher_valid:
                    logger.info(f"   ✅ Cypher syntax is valid")
                else:
                    logger.warning(f"   ⚠️ Cypher syntax may have issues")
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            test_result.update({
                "execution_time_ms": execution_time,
                "error": str(e),
                "error_type": type(e).__name__
            })
            logger.error(f"❌ {retriever_type.upper()} ERROR: {str(e)}")
        
        return test_result
    
    async def _validate_cypher_query(self, cypher: str) -> bool:
        """Validate Cypher query syntax."""
        try:
            # Basic syntax checks
            if not cypher.strip():
                return False
            
            # Check for basic Cypher keywords
            cypher_upper = cypher.upper()
            has_match = "MATCH" in cypher_upper
            has_return = "RETURN" in cypher_upper
            
            # Basic validation
            if not (has_match or has_return):
                return False
            
            # Try to validate with Neo4j (if connection available)
            if self.base_retriever and self.base_retriever.driver:
                try:
                    with self.base_retriever.driver.session() as session:
                        # Explain query to check syntax without executing
                        result = session.run(f"EXPLAIN {cypher}")
                        return True
                except Exception:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _compare_retriever_results(
        self,
        base_result: Dict[str, Any],
        ehs_result: Dict[str, Any],
        intent_type: str,
        example_num: int
    ) -> None:
        """Compare results between base and EHS retrievers."""
        logger.info(f"\n📊 COMPARISON - Example {example_num}:")
        logger.info(f"   Base Success: {base_result['success']} | EHS Success: {ehs_result['success']}")
        logger.info(f"   Base Time: {base_result['execution_time_ms']:.1f}ms | EHS Time: {ehs_result['execution_time_ms']:.1f}ms")
        logger.info(f"   Base Results: {base_result['result_count']} | EHS Results: {ehs_result['result_count']}")
        logger.info(f"   Base Confidence: {base_result['confidence_score']:.2f} | EHS Confidence: {ehs_result['confidence_score']:.2f}")
        
        # Determine winner
        winner = "tie"
        if ehs_result['success'] and not base_result['success']:
            winner = "ehs"
        elif base_result['success'] and not ehs_result['success']:
            winner = "base"
        elif both_successful := (base_result['success'] and ehs_result['success']):
            # Compare by confidence score and result count
            ehs_score = ehs_result['confidence_score'] + (ehs_result['result_count'] * 0.1)
            base_score = base_result['confidence_score'] + (base_result['result_count'] * 0.1)
            
            if ehs_score > base_score:
                winner = "ehs"
            elif base_score > ehs_score:
                winner = "base"
        
        logger.info(f"   🏆 Winner: {winner.upper()}")
    
    def _get_examples_for_intent(self, intent_type: str) -> List[Dict[str, str]]:
        """Get examples for a specific intent type."""
        example_functions = {
            "consumption_analysis": get_consumption_analysis_examples,
            "compliance_check": get_compliance_check_examples,
            "risk_assessment": get_risk_assessment_examples,
            "emission_tracking": get_emission_tracking_examples,
            "equipment_efficiency": get_equipment_efficiency_examples,
            "permit_status": get_permit_status_examples,
            "general_inquiry": get_general_inquiry_examples
        }
        
        function = example_functions.get(intent_type)
        if function:
            return function()
        else:
            logger.warning(f"No example function found for intent type: {intent_type}")
            return []
    
    def _generate_query_type_summary(
        self,
        intent_type: str,
        base_results: List[Dict[str, Any]],
        ehs_results: List[Dict[str, Any]]
    ) -> None:
        """Generate summary for a specific query type."""
        summary = {
            "intent_type": intent_type,
            "total_examples": len(base_results),
            "base_retriever": self._calculate_retriever_stats(base_results),
            "ehs_retriever": self._calculate_retriever_stats(ehs_results)
        }
        
        self.test_results["query_type_summary"][intent_type] = summary
        
        logger.info(f"\n📈 {intent_type.upper()} SUMMARY:")
        logger.info(f"   Total Examples: {summary['total_examples']}")
        logger.info(f"   Base Success Rate: {summary['base_retriever']['success_rate']:.1%}")
        logger.info(f"   EHS Success Rate: {summary['ehs_retriever']['success_rate']:.1%}")
        logger.info(f"   Base Avg Time: {summary['base_retriever']['avg_execution_time_ms']:.1f}ms")
        logger.info(f"   EHS Avg Time: {summary['ehs_retriever']['avg_execution_time_ms']:.1f}ms")
        logger.info(f"   Base Avg Confidence: {summary['base_retriever']['avg_confidence']:.2f}")
        logger.info(f"   EHS Avg Confidence: {summary['ehs_retriever']['avg_confidence']:.2f}")
    
    def _calculate_retriever_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for a retriever's results."""
        if not results:
            return {
                "success_rate": 0.0,
                "avg_execution_time_ms": 0.0,
                "avg_confidence": 0.0,
                "avg_result_count": 0.0,
                "total_successes": 0,
                "total_failures": 0
            }
        
        successes = [r for r in results if r["success"]]
        failures = [r for r in results if not r["success"]]
        
        return {
            "success_rate": len(successes) / len(results),
            "avg_execution_time_ms": sum(r["execution_time_ms"] for r in results) / len(results),
            "avg_confidence": sum(r["confidence_score"] for r in successes) / len(successes) if successes else 0.0,
            "avg_result_count": sum(r["result_count"] for r in successes) / len(successes) if successes else 0.0,
            "total_successes": len(successes),
            "total_failures": len(failures)
        }
    
    async def _generate_test_summary(self) -> None:
        """Generate overall test summary."""
        logger.info(f"\n{'='*80}")
        logger.info("COMPREHENSIVE TEST SUMMARY")
        logger.info(f"{'='*80}")
        
        # Calculate overall statistics
        base_stats = self._calculate_overall_stats("base")
        ehs_stats = self._calculate_overall_stats("ehs")
        
        self.test_results["success_rates"] = {
            "base_retriever": base_stats,
            "ehs_retriever": ehs_stats
        }
        
        logger.info(f"Test Duration: {self.test_results['total_duration_seconds']:.1f} seconds")
        logger.info(f"Total Query Types Tested: {len(self.test_results['query_type_summary'])}")
        logger.info(f"Total Examples Tested: {base_stats['total_examples']}")
        
        logger.info(f"\n🔵 BASE TEXT2CYPHER RETRIEVER:")
        logger.info(f"   Overall Success Rate: {base_stats['success_rate']:.1%}")
        logger.info(f"   Average Execution Time: {base_stats['avg_execution_time_ms']:.1f}ms")
        logger.info(f"   Average Confidence: {base_stats['avg_confidence']:.2f}")
        logger.info(f"   Total Successes: {base_stats['total_successes']}")
        logger.info(f"   Total Failures: {base_stats['total_failures']}")
        
        logger.info(f"\n🟢 EHS TEXT2CYPHER RETRIEVER:")
        logger.info(f"   Overall Success Rate: {ehs_stats['success_rate']:.1%}")
        logger.info(f"   Average Execution Time: {ehs_stats['avg_execution_time_ms']:.1f}ms")
        logger.info(f"   Average Confidence: {ehs_stats['avg_confidence']:.2f}")
        logger.info(f"   Total Successes: {ehs_stats['total_successes']}")
        logger.info(f"   Total Failures: {ehs_stats['total_failures']}")
        
        # Performance comparison
        if ehs_stats['success_rate'] > base_stats['success_rate']:
            improvement = ehs_stats['success_rate'] - base_stats['success_rate']
            logger.info(f"\n🏆 EHS RETRIEVER WINS with {improvement:.1%} better success rate!")
        elif base_stats['success_rate'] > ehs_stats['success_rate']:
            improvement = base_stats['success_rate'] - ehs_stats['success_rate']
            logger.info(f"\n🏆 BASE RETRIEVER WINS with {improvement:.1%} better success rate!")
        else:
            logger.info(f"\n🤝 TIE - Both retrievers have the same success rate!")
        
        # Query optimization analysis
        self._analyze_query_optimization()
    
    def _calculate_overall_stats(self, retriever_type: str) -> Dict[str, Any]:
        """Calculate overall statistics for a retriever type."""
        all_results = []
        results_key = f"{retriever_type}_retriever_tests"
        
        for intent_type, results in self.test_results[results_key].items():
            all_results.extend(results)
        
        return self._calculate_retriever_stats(all_results)
    
    def _analyze_query_optimization(self) -> None:
        """Analyze query optimization effectiveness."""
        logger.info(f"\n🔧 QUERY OPTIMIZATION ANALYSIS:")
        
        for intent_type in self.test_results["query_type_summary"].keys():
            base_avg_time = self.test_results["query_type_summary"][intent_type]["base_retriever"]["avg_execution_time_ms"]
            ehs_avg_time = self.test_results["query_type_summary"][intent_type]["ehs_retriever"]["avg_execution_time_ms"]
            
            if base_avg_time > 0:
                time_diff_pct = ((ehs_avg_time - base_avg_time) / base_avg_time) * 100
                if time_diff_pct < -10:
                    logger.info(f"   {intent_type}: EHS is {abs(time_diff_pct):.1f}% FASTER ⚡")
                elif time_diff_pct > 10:
                    logger.info(f"   {intent_type}: EHS is {time_diff_pct:.1f}% slower 🐌")
                else:
                    logger.info(f"   {intent_type}: Similar performance 🟰")
            else:
                logger.info(f"   {intent_type}: Unable to compare (base had no valid times)")
    
    async def _save_test_results(self) -> None:
        """Save detailed test results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"ehs_text2cypher_test_results_{timestamp}.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)
            
            logger.info(f"\n💾 Test results saved to: {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save test results: {e}")
    
    async def cleanup(self) -> None:
        """Clean up test resources."""
        logger.info("Cleaning up test resources")
        
        try:
            if self.base_retriever:
                await self.base_retriever.cleanup()
            
            if self.ehs_retriever:
                await self.ehs_retriever.cleanup()
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


async def main():
    """Main test execution function."""
    logger.info("🚀 Starting Comprehensive EHS Text2Cypher Test Suite")
    
    # Create test suite
    test_suite = EHSText2CypherTestSuite()
    
    try:
        # Setup
        if not await test_suite.setup():
            logger.error("❌ Test setup failed")
            return 1
        
        # Run tests
        results = await test_suite.run_comprehensive_tests()
        
        # Print final summary
        base_success_rate = results["success_rates"]["base_retriever"]["success_rate"]
        ehs_success_rate = results["success_rates"]["ehs_retriever"]["success_rate"]
        
        logger.info(f"\n{'='*80}")
        logger.info("🎯 FINAL RESULTS")
        logger.info(f"{'='*80}")
        logger.info(f"Base Text2Cypher Success Rate: {base_success_rate:.1%}")
        logger.info(f"EHS Text2Cypher Success Rate: {ehs_success_rate:.1%}")
        
        if ehs_success_rate >= 0.8:
            logger.info("✅ EHS Text2Cypher retriever is performing well!")
            return_code = 0
        elif ehs_success_rate >= 0.6:
            logger.info("⚠️ EHS Text2Cypher retriever has moderate performance")
            return_code = 0
        else:
            logger.info("❌ EHS Text2Cypher retriever needs improvement")
            return_code = 1
        
        return return_code
        
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}", exc_info=True)
        return 1
    
    finally:
        await test_suite.cleanup()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
