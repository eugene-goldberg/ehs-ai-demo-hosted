# GraphCypherQAChain Input Keys Debugging - Final Report

## 🎯 Problem Identified

**Issue**: GraphCypherQAChain expects input keys `['name', 'source', 'question']` instead of the expected `['query']`.

## 🔍 Root Cause Analysis

### What We Found

1. **The Cypher prompt template** correctly uses only `{question}` variable
2. **The QA prompt template** correctly uses `{question}` and `{context}` variables  
3. **However**, LangChain's PromptTemplate parser is incorrectly detecting additional variables

### The Real Problem: Neo4j Syntax in Examples

The issue is in these lines in the EHS Cypher prompt (`text2cypher.py` around lines 466 and 471):

```cypher
# Line 466 (problematic):
Cypher: MATCH (f:Facility {name: 'Main Manufacturing Plant'})-[:HAS_EQUIPMENT]->(e:Equipment)

# Line 471 (problematic):  
Cypher: MATCH (e:Emission {source: 'Boiler-01'})
```

**LangChain's PromptTemplate parser** treats `{name: 'value'}` and `{source: 'value'}` as template variables `{name}` and `{source}`, even though they're meant to be literal Neo4j property syntax.

## 🛠️ Solution

**Fix the Neo4j property syntax** by escaping the curly braces in the example queries:

### Current (Broken):
```cypher
MATCH (f:Facility {name: 'Main Manufacturing Plant'})-[:HAS_EQUIPMENT]->(e:Equipment)
MATCH (e:Emission {source: 'Boiler-01'})
```

### Fixed (Working):
```cypher
MATCH (f:Facility {{name: 'Main Manufacturing Plant'}})-[:HAS_EQUIPMENT]->(e:Equipment)
MATCH (e:Emission {{source: 'Boiler-01'}})
```

## 📝 Implementation Steps

1. **Edit** `/Users/eugene/dev/ai/agentos/ehs-ai-demo/ehs-analytics/src/ehs_analytics/retrieval/strategies/text2cypher.py`
2. **Find lines 466 and 471** in the `_build_ehs_cypher_prompt()` method
3. **Replace** `{name:` with `{{name:` and `{source:` with `{{source:`
4. **Test** by creating a chain and checking: `print(chain.input_keys)` - should show `['query']`

## 🧪 Verification

After the fix:
- `chain.input_keys` should return `['query']`
- `chain.invoke({'query': 'your question'})` should work
- No more `KeyError: 'name'` or `KeyError: 'source'` errors

## 📊 Debug Script Results

### Before Fix:
- **Manual regex**: `{'question'}` ✅
- **LangChain parsing**: `['name', 'question', 'source']` ❌
- **Problem lines identified**: Lines 74 and 79 in the generated prompt

### After Fix (Expected):
- **Manual regex**: `{'question'}` ✅  
- **LangChain parsing**: `['question']` ✅
- **Chain input_keys**: `['query']` ✅

## 📁 Debug Files Created

1. `debug_cypher_chain.py` - Basic GraphCypherQAChain inspection
2. `debug_cypher_chain_improved.py` - Advanced analysis with working mocks
3. `debug_cypher_chain_final.py` - Focused debugging on input key mismatch
4. `find_template_vars.py` - Template variable inspection
5. `debug_chain_creation.py` - Chain creation process analysis
6. `debug_langchain_parsing.py` - LangChain parsing behavior analysis
7. `DEBUG_FINDINGS_SUMMARY.md` - This summary document

## ✅ Key Insights

1. **Template variables are parsed by LangChain**, not just by regex
2. **Neo4j property syntax** `{property: value}` conflicts with template syntax `{variable}`
3. **Escaping with double braces** `{{property: value}}` solves the conflict
4. **The issue was not in custom prompts** but in example queries within prompts
5. **GraphCypherQAChain input_keys** are the union of cypher_prompt and qa_prompt variables

This debugging process demonstrates the importance of understanding how different systems parse template syntax and the need to properly escape special characters when mixing syntaxes.