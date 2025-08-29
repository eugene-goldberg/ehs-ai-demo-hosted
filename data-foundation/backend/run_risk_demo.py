#!/usr/bin/env python3
"""
Quick runner script for the Risk Assessment Workflow Demonstration.

This script provides an easy way to run the comprehensive risk assessment 
demonstration with common configurations.

Usage:
    python3 run_risk_demo.py [options]
    
Examples:
    python3 run_risk_demo.py                    # Full demonstration
    python3 run_risk_demo.py --quick            # Quick demo without Neo4j queries
    python3 run_risk_demo.py --sample-doc       # Use specific sample document
    python3 run_risk_demo.py --save-results     # Save results to file
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(
        description='Quick runner for EHS AI Risk Assessment Demonstration',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--quick', action='store_true', 
                       help='Quick demo (skip Neo4j queries and detailed traces)')
    parser.add_argument('--sample-doc', action='store_true',
                       help='Use electric bill sample document')
    parser.add_argument('--save-results', action='store_true',
                       help='Save results to timestamped JSON file')
    parser.add_argument('--facility-id', type=str, default='DEMO_FACILITY_001',
                       help='Facility ID to use in demonstration')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Build command for the main demo script
    script_path = os.path.join(
        os.path.dirname(__file__), 
        'src', 'workflows', 'demo_risk_assessment_workflow.py'
    )
    
    cmd = [sys.executable, script_path]
    
    # Add arguments based on options
    if args.quick:
        cmd.extend(['--no-neo4j-queries', '--no-traces'])
        print("🚀 Running QUICK demonstration (limited queries and traces)")
    else:
        cmd.extend(['--enable-traces', '--query-neo4j'])
        print("🔬 Running COMPREHENSIVE demonstration (full analysis)")
    
    if args.sample_doc:
        cmd.append('--sample-document')
        print("📄 Using sample document for demonstration")
    
    if args.facility_id != 'DEMO_FACILITY_001':
        cmd.extend(['--facility-id', args.facility_id])
        print(f"🏢 Using facility ID: {args.facility_id}")
    
    if args.save_results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'/tmp/risk_demo_results_{timestamp}.json'
        cmd.extend(['--output-file', output_file])
        print(f"💾 Results will be saved to: {output_file}")
    
    if args.verbose:
        print(f"🔧 Executing command: {' '.join(cmd)}")
    
    print("\n" + "="*80)
    print("EHS AI PLATFORM - RISK ASSESSMENT DEMONSTRATION")
    print("="*80)
    print(f"Started at: {datetime.now()}")
    print()
    
    try:
        # Execute the demonstration script
        result = subprocess.run(cmd, check=True, cwd=os.path.dirname(__file__))
        
        print("\n" + "="*80)
        print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        if args.save_results:
            print(f"📊 Results saved to: {output_file}")
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Demonstration failed with exit code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print(f"\n🛑 Demonstration interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        return 1

if __name__ == '__main__':
    sys.exit(main())