#!/usr/bin/env python3
"""
Improved debugging script to understand GraphCypherQAChain input key requirements.

This script creates a more complete mock to inspect GraphCypherQAChain behavior
and understand why it might expect 'name' and 'source' as input keys.

Uses virtual environment: /Users/eugene/dev/ai/agentos/ehs-ai-demo/ehs-analytics/venv
"""

import sys
import os
import logging
from datetime import datetime
from typing import Dict, Any, List
import traceback

# Setup logging
logging.basicConfig(
    filename='debug_cypher_chain_improved.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

logger = logging.getLogger(__name__)

class CompleteMockGraph:
    """Complete mock graph that implements all required methods."""
    
    def __init__(self):
        self.schema = {
            'node_props': {
                'Facility': ['name', 'location', 'type'],
                'Equipment': ['name', 'type', 'manufacturer'],
                'UtilityBill': ['amount', 'billing_period', 'cost']
            },
            'rel_props': {},
            'relationships': [
                'HAS_EQUIPMENT', 'HAS_UTILITY_BILL', 'LOCATED_AT'
            ]
        }
        
    def get_structured_schema(self):
        """Return structured schema."""
        return self.schema
    
    def refresh_schema(self):
        """Mock schema refresh."""
        pass
    
    def query(self, query: str):
        """Mock query execution."""
        return []
    
    def get_schema(self):
        """Return schema as string."""
        return "Mock Neo4j Schema: Facility, Equipment, UtilityBill nodes"

def test_graphcypherqa_with_working_mock():
    """Test GraphCypherQAChain with a complete mock."""
    logger.info("\n=== TESTING GRAPHCYPHERQACHAIN WITH COMPLETE MOCK ===")
    
    try:
        from langchain.chains import GraphCypherQAChain
        from langchain.prompts import PromptTemplate
        from langchain_openai import ChatOpenAI
        
        # Create components
        graph = CompleteMockGraph()
        llm = ChatOpenAI(api_key="fake-key-for-testing", model_name="gpt-3.5-turbo", temperature=0.0)
        
        logger.info("✅ Created mock graph and LLM")
        
        # Test 1: Basic chain creation
        logger.info("🧪 Test 1: Basic chain creation")
        try:
            chain = GraphCypherQAChain.from_llm(
                llm=llm,
                graph=graph,
                verbose=True
            )
            
            logger.info("✅ Basic GraphCypherQAChain created successfully!")
            logger.info(f"📋 Input keys: {chain.input_keys}")
            logger.info(f"📋 Output keys: {chain.output_keys}")
            
            # Inspect prompts
            if hasattr(chain, 'cypher_prompt'):
                logger.info(f"📋 Cypher prompt input variables: {chain.cypher_prompt.input_variables}")
                logger.info(f"📋 Cypher prompt template preview: {chain.cypher_prompt.template[:200]}...")
            
            if hasattr(chain, 'qa_prompt'):
                logger.info(f"📋 QA prompt input variables: {chain.qa_prompt.input_variables}")
                logger.info(f"📋 QA prompt template preview: {chain.qa_prompt.template[:200]}...")
            
        except Exception as e:
            logger.error(f"❌ Basic chain creation failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return
        
        # Test 2: Custom cypher prompt with different variables
        logger.info("\n🧪 Test 2: Custom cypher prompt with 'question' variable")
        try:
            custom_cypher_prompt = PromptTemplate.from_template(
                "Generate Cypher for: {question}\nCypher:"
            )
            
            chain2 = GraphCypherQAChain.from_llm(
                llm=llm,
                graph=graph,
                cypher_prompt=custom_cypher_prompt,
                verbose=True
            )
            
            logger.info("✅ Chain with 'question' variable created")
            logger.info(f"📋 Input keys: {chain2.input_keys}")
            logger.info(f"📋 Cypher prompt variables: {chain2.cypher_prompt.input_variables}")
            
        except Exception as e:
            logger.error(f"❌ Custom question prompt failed: {e}")
        
        # Test 3: Custom prompt with 'name' and 'source' (the problematic case)
        logger.info("\n🧪 Test 3: Custom cypher prompt with 'name' and 'source' variables")
        try:
            problematic_prompt = PromptTemplate.from_template(
                "Generate Cypher query for entity named '{name}' from source '{source}'\nCypher:"
            )
            
            chain3 = GraphCypherQAChain.from_llm(
                llm=llm,
                graph=graph,
                cypher_prompt=problematic_prompt,
                verbose=True
            )
            
            logger.info("✅ Chain with 'name' and 'source' variables created")
            logger.info(f"📋 Input keys: {chain3.input_keys}")
            logger.info(f"📋 Cypher prompt variables: {chain3.cypher_prompt.input_variables}")
            logger.info("🔍 This explains why your chain expects 'name' and 'source'!")
            
        except Exception as e:
            logger.error(f"❌ Name/source prompt failed: {e}")
        
        # Test 4: Inspect default prompts
        logger.info("\n🧪 Test 4: Inspecting default prompt generation")
        try:
            from langchain_community.chains.graph_qa.cypher import CYPHER_GENERATION_TEMPLATE, CYPHER_QA_TEMPLATE
            logger.info("✅ Found default templates")
            logger.info(f"📋 Default Cypher generation template variables:")
            
            # Parse the default template to see what variables it expects
            default_cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)
            logger.info(f"   Variables: {default_cypher_prompt.input_variables}")
            
            default_qa_prompt = PromptTemplate.from_template(CYPHER_QA_TEMPLATE)
            logger.info(f"📋 Default QA template variables: {default_qa_prompt.input_variables}")
            
        except ImportError as e:
            logger.warning(f"⚠️ Could not import default templates: {e}")
        
        return [chain, chain2, chain3] if 'chain3' in locals() else [chain, chain2] if 'chain2' in locals() else [chain]
        
    except Exception as e:
        logger.error(f"❌ Complete test failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def inspect_existing_code_prompts():
    """Inspect the prompts being used in the existing code."""
    logger.info("\n=== INSPECTING EXISTING CODE PROMPTS ===")
    
    try:
        # Add the project path to import the existing retriever
        sys.path.insert(0, '/Users/eugene/dev/ai/agentos/ehs-ai-demo/ehs-analytics/src')
        
        from ehs_analytics.retrieval.strategies.text2cypher import Text2CypherRetriever
        
        logger.info("✅ Successfully imported existing Text2CypherRetriever")
        
        # Create an instance to inspect the prompt
        config = {
            "neo4j_uri": "bolt://test:7687",
            "neo4j_user": "test",
            "neo4j_password": "test",
            "openai_api_key": "fake-key",
            "model_name": "gpt-3.5-turbo"
        }
        
        retriever = Text2CypherRetriever(config)
        
        # Inspect the EHS cypher prompt
        logger.info("📋 Inspecting _build_ehs_cypher_prompt method")
        
        # Check if we can get the prompt template
        if hasattr(retriever, '_build_ehs_cypher_prompt'):
            prompt_text = retriever._build_ehs_cypher_prompt()
            logger.info(f"📋 EHS prompt length: {len(prompt_text)}")
            
            # Look for template variables in the prompt
            import re
            template_vars = re.findall(r'\{(\w+)\}', prompt_text)
            logger.info(f"📋 Template variables found in EHS prompt: {set(template_vars)}")
            
            # Check specifically for 'name' and 'source'
            if 'name' in template_vars and 'source' in template_vars:
                logger.info("🔍 FOUND THE PROBLEM: EHS prompt contains 'name' and 'source' variables!")
            elif 'question' in template_vars:
                logger.info("✅ EHS prompt uses 'question' variable (correct)")
            else:
                logger.info(f"📋 EHS prompt variables: {template_vars}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to inspect existing code: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_input_combinations_detailed(chains):
    """Test detailed input combinations to understand the behavior."""
    logger.info("\n=== DETAILED INPUT COMBINATION TESTING ===")
    
    if not chains:
        logger.error("❌ No chains to test")
        return
    
    test_cases = [
        {"description": "Standard query input", "input": {"query": "Show all facilities"}},
        {"description": "Question input", "input": {"question": "Show all facilities"}},
        {"description": "Name and source inputs", "input": {"name": "Facility1", "source": "Database"}},
        {"description": "Mixed inputs", "input": {"query": "Show facilities", "context": "Additional context"}},
    ]
    
    for i, chain in enumerate(chains):
        if not chain:
            continue
            
        logger.info(f"\n📋 Testing Chain {i+1}:")
        logger.info(f"   Expected input keys: {chain.input_keys}")
        logger.info(f"   Expected output keys: {chain.output_keys}")
        
        for case in test_cases:
            logger.info(f"\n  🧪 {case['description']}: {case['input']}")
            
            try:
                input_keys_set = set(chain.input_keys)
                test_keys_set = set(case['input'].keys())
                
                if input_keys_set == test_keys_set:
                    logger.info(f"    ✅ Perfect match - this input would work")
                elif input_keys_set.issubset(test_keys_set):
                    logger.info(f"    ✅ All required keys present (+ extras)")
                elif test_keys_set.issubset(input_keys_set):
                    missing = input_keys_set - test_keys_set
                    logger.info(f"    ❌ Missing required keys: {missing}")
                else:
                    extra = test_keys_set - input_keys_set
                    missing = input_keys_set - test_keys_set
                    logger.info(f"    ❌ Key mismatch - missing: {missing}, extra: {extra}")
                
            except Exception as e:
                logger.warning(f"    ⚠️ Error analyzing: {e}")

def provide_debugging_recommendations():
    """Provide specific debugging recommendations."""
    logger.info("\n=== DEBUGGING RECOMMENDATIONS ===")
    
    recommendations = [
        "",
        "🔧 IMMEDIATE DEBUGGING STEPS:",
        "",
        "1. CHECK YOUR PROMPT TEMPLATE:",
        "   - Look at the cypher_prompt in your chain creation",
        "   - Check for variables like '{name}' and '{source}' in the template",
        "   - The input_keys are determined by the prompt template variables",
        "",
        "2. INSPECT YOUR CHAIN CREATION CODE:",
        "   - Find where GraphCypherQAChain.from_llm() is called",
        "   - Check the cypher_prompt parameter",
        "   - Look for PromptTemplate.from_template() calls",
        "",
        "3. VERIFY THE CURRENT CHAIN:",
        "   - Add this debug line after chain creation:",
        "     print(f'Chain input keys: {chain.input_keys}')",
        "     print(f'Cypher prompt variables: {chain.cypher_prompt.input_variables}')",
        "",
        "4. FIX THE ISSUE:",
        "   - If you want 'query' input, use: PromptTemplate.from_template('...{question}...')",
        "   - Standard LangChain pattern uses 'question' variable",
        "   - Check _build_ehs_cypher_prompt() method in your code",
        "",
        "5. COMMON MISTAKE:",
        "   - Custom prompt templates with wrong variable names",
        "   - Copy-paste from examples with different variable names",
        "   - Mixing different chain types (e.g., conversation chain patterns)",
        "",
        "6. QUICK TEST:",
        "   - Create a simple test with basic GraphCypherQAChain",
        "   - No custom prompts to see the default behavior",
        "   - Then gradually add your customizations",
    ]
    
    for rec in recommendations:
        logger.info(rec)

def main():
    """Main debugging function."""
    logger.info(f"🐛 Improved GraphCypherQAChain Debugging Started at {datetime.now()}")
    logger.info(f"🐍 Python version: {sys.version}")
    logger.info(f"📁 Working directory: {os.getcwd()}")
    
    # Test with complete mock
    chains = test_graphcypherqa_with_working_mock()
    
    # Inspect existing code
    inspect_existing_code_prompts()
    
    # Test input combinations
    test_input_combinations_detailed(chains)
    
    # Provide recommendations
    provide_debugging_recommendations()
    
    logger.info(f"\n✅ Improved debugging completed at {datetime.now()}")
    logger.info("📋 Check debug_cypher_chain_improved.log for detailed results")

if __name__ == "__main__":
    main()