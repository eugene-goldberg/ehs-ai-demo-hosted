#!/usr/bin/env python3
"""
Simple test runner for Phase 2 retrievers.
This script can be executed by the test-runner sub-agent.
"""

import asyncio
import sys
import os
import logging
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from test_comprehensive_phase2_retrievers import Phase2RetrieverTestSuite

async def run_tests():
    """Run the comprehensive Phase 2 retriever tests."""
    print("🚀 Starting Phase 2 Retriever Tests")
    print("=" * 50)
    
    try:
        # Initialize test suite
        test_suite = Phase2RetrieverTestSuite()
        
        # Setup retrievers
        print("📋 Setting up retrievers...")
        setup_success = await test_suite.setup_retrievers()
        
        if not setup_success:
            print("❌ Failed to setup retrievers")
            return False
        
        print(f"✅ Initialized {len(test_suite.retrievers)} retrievers")
        
        # Run tests
        print("🧪 Running tests...")
        report = await test_suite.run_comprehensive_tests()
        
        # Print results
        print(f"\n📊 Results: {report.successful_tests}/{report.successful_tests + report.failed_tests} passed")
        print(f"📈 Success Rate: {report.performance_metrics['success_rate']:.1%}")
        
        # Save report
        report_file = test_suite.save_test_report(report)
        print(f"📄 Report saved: {report_file}")
        
        return report.failed_tests == 0
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)
