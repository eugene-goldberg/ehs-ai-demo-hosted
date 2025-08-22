#!/usr/bin/env python3
"""
Minimal test to verify the Text2Cypher input key fix from "question" to "query".
"""

import sys
sys.path.insert(0, 'src')

print("=== Minimal Text2Cypher Input Key Test ===")

# Test 1: Check if the input key fix is in place
try:
    from ehs_analytics.retrieval.strategies.text2cypher import Text2CypherRetriever
    print("✅ Text2CypherRetriever imported successfully")
    
    # Look for the fixed input key in the code
    import inspect
    source = inspect.getsource(Text2CypherRetriever)
    
    if '"query":' in source and 'input_text' in source:
        print("✅ Input key fix appears to be implemented (found 'query' and 'input_text')")
    elif '"question":' in source:
        print("❌ Old input key 'question' still found in source code")
    else:
        print("⚠️  Could not determine input key format from source")
        
    print(f"Source length: {len(source)} characters")
    
except Exception as e:
    print(f"❌ Import failed: {e}")

# Test 2: Check GraphCypherQAChain import and usage
try:
    from langchain_community.graphs.graph_document import GraphCypherQAChain
    print("✅ GraphCypherQAChain available (should use 'query' input)")
except ImportError:
    try:
        from langchain.chains import GraphCypherQAChain
        print("⚠️  Using deprecated GraphCypherQAChain import")
    except ImportError:
        print("❌ GraphCypherQAChain not available")

# Test 3: Check the specific line that was fixed
try:
    import inspect
    from ehs_analytics.retrieval.strategies.text2cypher import Text2CypherRetriever
    source_lines = inspect.getsource(Text2CypherRetriever).split('\n')
    
    for i, line in enumerate(source_lines):
        if '"query": input_text' in line:
            print(f"✅ Found fixed input mapping at line {i}: {line.strip()}")
            break
        elif '"question": input_text' in line:
            print(f"❌ Found old input mapping at line {i}: {line.strip()}")
            break
    else:
        print("⚠️  Could not find specific input mapping line")
        
except Exception as e:
    print(f"❌ Source inspection failed: {e}")

print("\n=== Test Complete ===")
print("If the fix is in place, queries should work with the 'query' input key.")
