#!/usr/bin/env python3
"""Comprehensive test suite for Phase 2 Text2Cypher implementation."""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ehs_analytics.config import Settings
from ehs_analytics.retrieval.strategies.text2cypher import Text2CypherRetriever
from ehs_analytics.retrieval.base import QueryType

# Test queries organized by category
TEST_QUERIES = {
    "facilities": [
        "Show all facilities",
        "List manufacturing facilities",
        "Find facilities in Texas",
        "What facilities do we have?"
    ],
    "utility_consumption": [
        "Show electricity consumption for all facilities",
        "What is the total water usage?",
        "Find facilities with high energy consumption",
        "Show utility bills from 2024"
    ],
    "equipment": [
        "List all equipment",
        "Show equipment at Main Manufacturing Plant",
        "Find equipment that needs maintenance",
        "What pumps do we have?"
    ],
    "permits": [
        "Show all permits",
        "List active permits",
        "Find permits expiring soon",
        "What environmental permits do we have?"
    ],
    "relationships": [
        "Show facilities and their equipment",
        "Which facilities have utility bills?",
        "Show permits for each facility",
        "List facilities with their consumption data"
    ],
    "aggregations": [
        "What is the total electricity consumption?",
        "Calculate average water usage per facility",
        "Count equipment by facility",
        "Show monthly utility costs"
    ]
}

class ComprehensiveTestRunner:
    """Runs comprehensive tests for Text2Cypher retriever."""
    
    def __init__(self):
        self.settings = Settings()
        self.retriever = None
        self.results = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "queries_with_results": 0,
            "queries_with_cypher": 0,
            "category_results": {},
            "query_details": []
        }
    
    async def setup(self):
        """Initialize the Text2Cypher retriever."""
        config = {
            "neo4j_uri": self.settings.neo4j_uri,
            "neo4j_user": self.settings.neo4j_username,
            "neo4j_password": self.settings.neo4j_password,
            "openai_api_key": self.settings.openai_api_key,
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.0,
            "cypher_validation": True
        }
        
        self.retriever = Text2CypherRetriever(config)
        await self.retriever.initialize()
        print("✅ Text2Cypher retriever initialized\n")
    
    async def test_query(self, query: str, category: str) -> Dict[str, Any]:
        """Test a single query and return detailed results."""
        print(f"📊 Testing: '{query}'")
        
        result_detail = {
            "query": query,
            "category": category,
            "success": False,
            "has_results": False,
            "has_cypher": False,
            "result_count": 0,
            "cypher_query": None,
            "error": None,
            "execution_time_ms": None,
            "confidence_score": None
        }
        
        try:
            # Execute query
            result = await self.retriever.retrieve(
                query=query,
                query_type=QueryType.GENERAL,
                limit=10
            )
            
            # Extract details
            result_detail["success"] = result.success
            result_detail["has_results"] = len(result.data) > 0
            result_detail["result_count"] = len(result.data)
            
            if result.metadata:
                result_detail["cypher_query"] = result.metadata.cypher_query
                result_detail["has_cypher"] = bool(result.metadata.cypher_query)
                result_detail["execution_time_ms"] = result.metadata.execution_time_ms
                result_detail["confidence_score"] = result.metadata.confidence_score
            
            # Print summary
            if result.success:
                print(f"  ✅ Success: {result_detail['result_count']} results")
                if result_detail['cypher_query']:
                    print(f"  📝 Cypher: {result_detail['cypher_query'][:100]}...")
                else:
                    print("  ⚠️  No Cypher query extracted")
            else:
                print(f"  ❌ Failed: {result.message}")
                result_detail["error"] = result.message
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            result_detail["error"] = str(e)
        
        print()
        return result_detail
    
    async def run_category_tests(self, category: str, queries: List[str]):
        """Run all tests for a specific category."""
        print(f"\n{'='*60}")
        print(f"📁 Testing Category: {category.upper()}")
        print(f"{'='*60}\n")
        
        category_stats = {
            "total": len(queries),
            "successful": 0,
            "with_results": 0,
            "with_cypher": 0,
            "avg_execution_time": 0,
            "avg_confidence": 0
        }
        
        execution_times = []
        confidence_scores = []
        
        for query in queries:
            result = await self.test_query(query, category)
            self.results["query_details"].append(result)
            self.results["total_queries"] += 1
            
            if result["success"]:
                self.results["successful_queries"] += 1
                category_stats["successful"] += 1
                
                if result["has_results"]:
                    self.results["queries_with_results"] += 1
                    category_stats["with_results"] += 1
                
                if result["has_cypher"]:
                    self.results["queries_with_cypher"] += 1
                    category_stats["with_cypher"] += 1
                
                if result["execution_time_ms"]:
                    execution_times.append(result["execution_time_ms"])
                
                if result["confidence_score"]:
                    confidence_scores.append(result["confidence_score"])
            else:
                self.results["failed_queries"] += 1
        
        # Calculate averages
        if execution_times:
            category_stats["avg_execution_time"] = sum(execution_times) / len(execution_times)
        
        if confidence_scores:
            category_stats["avg_confidence"] = sum(confidence_scores) / len(confidence_scores)
        
        self.results["category_results"][category] = category_stats
        
        # Print category summary
        print(f"\n📊 Category Summary for {category}:")
        print(f"  - Total queries: {category_stats['total']}")
        print(f"  - Successful: {category_stats['successful']}")
        print(f"  - With results: {category_stats['with_results']}")
        print(f"  - With Cypher: {category_stats['with_cypher']}")
        print(f"  - Avg execution time: {category_stats['avg_execution_time']:.2f}ms")
        print(f"  - Avg confidence: {category_stats['avg_confidence']:.2f}")
    
    async def run_all_tests(self):
        """Run all test categories."""
        await self.setup()
        
        for category, queries in TEST_QUERIES.items():
            await self.run_category_tests(category, queries)
            # Small delay between categories to avoid rate limiting
            await asyncio.sleep(1)
        
        await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources."""
        if self.retriever:
            await self.retriever.cleanup()
    
    def print_final_report(self):
        """Print comprehensive test report."""
        print(f"\n{'='*60}")
        print("📊 FINAL TEST REPORT")
        print(f"{'='*60}\n")
        
        print(f"🔢 Overall Statistics:")
        print(f"  - Total queries tested: {self.results['total_queries']}")
        print(f"  - Successful queries: {self.results['successful_queries']}")
        print(f"  - Failed queries: {self.results['failed_queries']}")
        print(f"  - Queries with results: {self.results['queries_with_results']}")
        print(f"  - Queries with Cypher: {self.results['queries_with_cypher']}")
        
        success_rate = (self.results['successful_queries'] / self.results['total_queries'] * 100) if self.results['total_queries'] > 0 else 0
        result_rate = (self.results['queries_with_results'] / self.results['successful_queries'] * 100) if self.results['successful_queries'] > 0 else 0
        cypher_rate = (self.results['queries_with_cypher'] / self.results['successful_queries'] * 100) if self.results['successful_queries'] > 0 else 0
        
        print(f"\n📈 Success Rates:")
        print(f"  - Query success rate: {success_rate:.1f}%")
        print(f"  - Result retrieval rate: {result_rate:.1f}%")
        print(f"  - Cypher extraction rate: {cypher_rate:.1f}%")
        
        print(f"\n📁 Category Performance:")
        for category, stats in self.results['category_results'].items():
            print(f"\n  {category.upper()}:")
            print(f"    - Success rate: {stats['successful']/stats['total']*100:.1f}%")
            print(f"    - Avg execution time: {stats['avg_execution_time']:.2f}ms")
            print(f"    - Avg confidence: {stats['avg_confidence']:.2f}")
        
        # Identify problematic queries
        failed_queries = [q for q in self.results['query_details'] if not q['success']]
        no_cypher_queries = [q for q in self.results['query_details'] if q['success'] and not q['has_cypher']]
        no_result_queries = [q for q in self.results['query_details'] if q['success'] and not q['has_results']]
        
        if failed_queries:
            print(f"\n❌ Failed Queries ({len(failed_queries)}):")
            for q in failed_queries[:5]:  # Show first 5
                print(f"  - '{q['query']}': {q['error']}")
        
        if no_cypher_queries:
            print(f"\n⚠️  Queries without Cypher extraction ({len(no_cypher_queries)}):")
            for q in no_cypher_queries[:5]:  # Show first 5
                print(f"  - '{q['query']}'")
        
        if no_result_queries:
            print(f"\n⚠️  Queries without results ({len(no_result_queries)}):")
            for q in no_result_queries[:5]:  # Show first 5
                print(f"  - '{q['query']}'")
        
        # Save detailed results
        self.save_results()
        
        # Overall assessment
        print(f"\n{'='*60}")
        if success_rate >= 90 and cypher_rate >= 90:
            print("✅ PHASE 2 TEST SUITE: PASSED")
            print("   Text2Cypher retriever is working excellently!")
        elif success_rate >= 70 and cypher_rate >= 70:
            print("⚠️  PHASE 2 TEST SUITE: PASSED WITH WARNINGS")
            print("   Text2Cypher retriever is functional but needs improvements")
        else:
            print("❌ PHASE 2 TEST SUITE: FAILED")
            print("   Text2Cypher retriever has critical issues")
        print(f"{'='*60}\n")
    
    def save_results(self):
        """Save detailed test results to file."""
        output_file = Path(__file__).parent / "test_results" / f"phase2_comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {output_file}")

async def main():
    """Run comprehensive Phase 2 tests."""
    runner = ComprehensiveTestRunner()
    
    try:
        await runner.run_all_tests()
        runner.print_final_report()
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    print("🚀 Starting Phase 2 Comprehensive Test Suite")
    print(f"   Timestamp: {datetime.now().isoformat()}")
    print()
    
    asyncio.run(main())