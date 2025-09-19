#!/usr/bin/env python3

import sys
import os
import json

# Add the current directory to Python path
sys.path.insert(0, '.')

from context_retriever import ContextRetriever

def test_recommendations():
    print("Testing fixed get_recommendations_context method...")
    
    # Test both sites
    sites = ['houston_tx', 'algonquin_il']
    
    for site in sites:
        print(f"\n=== Testing site: {site} ===")
        
        try:
            retriever = ContextRetriever()
            result = retriever.get_recommendations_context(site)
            
            print(f"Success! Found {result.get('record_count', 0)} recommendations")
            print(f"Total recommendations in data: {result.get('total_recommendations', 0)}")
            
            if result.get('recommendations'):
                print("\nFirst recommendation:")
                first_rec = result['recommendations'][0]
                print(f"  Title: {first_rec.get('title')}")
                print(f"  Category: {first_rec.get('category')}")
                print(f"  Priority: {first_rec.get('priority')}")
                print(f"  Expected Impact: {first_rec.get('expected_impact')}")
                print(f"  Timeline: {first_rec.get('timeline')}")
                
            # Test with category filter
            print(f"\nTesting with category filter 'electricity'...")
            result_filtered = retriever.get_recommendations_context(site, category='electricity')
            print(f"Found {result_filtered.get('record_count', 0)} electricity recommendations")
            
            retriever.close()
            
        except Exception as e:
            print(f"Error testing site {site}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_recommendations()
