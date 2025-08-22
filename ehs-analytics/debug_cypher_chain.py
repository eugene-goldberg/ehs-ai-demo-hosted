#!/usr/bin/env python3
"""
Debugging script to understand GraphCypherQAChain input key requirements.

This script creates minimal GraphCypherQAChain instances to inspect their
input requirements and understand why they might expect 'name' and 'source'
as input keys instead of the expected 'query' key.

Uses virtual environment: /Users/eugene/dev/ai/agentos/ehs-ai-demo/ehs-analytics/venv
"""

import sys
import os
import logging
from datetime import datetime
from typing import Dict, Any, List
import traceback

# Setup logging to file and console
logging.basicConfig(
    filename='debug_cypher_chain.log',
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

def debug_langchain_imports():
    """Debug LangChain imports and versions."""
    logger.info("=== DEBUGGING LANGCHAIN IMPORTS ===")
    
    try:
        import langchain
        logger.info(f"✅ LangChain version: {langchain.__version__}")
    except ImportError as e:
        logger.error(f"❌ Failed to import langchain: {e}")
        return False
    except AttributeError:
        logger.info("✅ LangChain imported (no version attribute)")
    
    try:
        from langchain.chains import GraphCypherQAChain
        logger.info("✅ GraphCypherQAChain imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import GraphCypherQAChain: {e}")
        return False
    
    try:
        from langchain.prompts import PromptTemplate
        logger.info("✅ PromptTemplate imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import PromptTemplate: {e}")
        return False
    
    try:
        from langchain_openai import ChatOpenAI
        logger.info("✅ ChatOpenAI imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import ChatOpenAI: {e}")
        return False
    
    try:
        from langchain_community.graphs import Neo4jGraph
        logger.info("✅ Neo4jGraph imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import Neo4jGraph: {e}")
        return False
    
    return True

def inspect_graphcypherqa_chain_class():
    """Inspect GraphCypherQAChain class structure and input requirements."""
    logger.info("\n=== INSPECTING GRAPHCYPHERQACHAIN CLASS ===")
    
    try:
        from langchain.chains import GraphCypherQAChain
        
        # Inspect class attributes
        logger.info("📋 GraphCypherQAChain class attributes:")
        for attr in dir(GraphCypherQAChain):
            if not attr.startswith('_'):
                logger.info(f"  - {attr}")
        
        # Check for input_keys attribute/method
        if hasattr(GraphCypherQAChain, 'input_keys'):
            logger.info(f"✅ input_keys attribute found: {GraphCypherQAChain.input_keys}")
        else:
            logger.info("⚠️ No input_keys attribute found")
        
        # Check for from_llm method
        if hasattr(GraphCypherQAChain, 'from_llm'):
            logger.info("✅ from_llm class method found")
            
            # Inspect from_llm method signature
            import inspect
            sig = inspect.signature(GraphCypherQAChain.from_llm)
            logger.info(f"📝 from_llm signature: {sig}")
            logger.info("📝 from_llm parameters:")
            for param_name, param in sig.parameters.items():
                logger.info(f"  - {param_name}: {param.annotation} = {param.default}")
        else:
            logger.info("❌ from_llm method not found")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to inspect GraphCypherQAChain class: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def create_minimal_neo4j_graph():
    """Create a minimal Neo4jGraph for testing."""
    logger.info("\n=== CREATING MINIMAL NEO4J GRAPH ===")
    
    try:
        from langchain_community.graphs import Neo4jGraph
        
        # Use test credentials (these don't need to work for our inspection)
        test_config = {
            "url": "bolt://localhost:7687",
            "username": "test",
            "password": "test"
        }
        
        logger.info("📋 Attempting to create Neo4jGraph (connection may fail, that's OK)")
        
        try:
            graph = Neo4jGraph(**test_config)
            logger.info("✅ Neo4jGraph instance created successfully")
            
            # Inspect the graph object
            logger.info("📋 Neo4jGraph attributes:")
            for attr in dir(graph):
                if not attr.startswith('_') and not callable(getattr(graph, attr)):
                    try:
                        value = getattr(graph, attr)
                        logger.info(f"  - {attr}: {type(value)} = {str(value)[:100]}")
                    except Exception as e:
                        logger.info(f"  - {attr}: (error accessing: {e})")
            
            return graph
            
        except Exception as e:
            logger.warning(f"⚠️ Neo4jGraph creation failed (expected): {e}")
            logger.info("📋 Creating mock graph for testing...")
            
            # Create a mock graph object for testing
            class MockGraph:
                def __init__(self):
                    self.schema = "Test Schema"
                
                def refresh_schema(self):
                    pass
            
            return MockGraph()
            
    except Exception as e:
        logger.error(f"❌ Failed to create Neo4jGraph: {e}")
        return None

def create_minimal_llm():
    """Create a minimal LLM for testing."""
    logger.info("\n=== CREATING MINIMAL LLM ===")
    
    try:
        from langchain_openai import ChatOpenAI
        
        # Create with fake API key (we won't actually call it)
        llm = ChatOpenAI(
            api_key="fake-key-for-testing",
            model_name="gpt-3.5-turbo",
            temperature=0.0
        )
        
        logger.info("✅ ChatOpenAI instance created successfully")
        
        # Inspect LLM attributes
        logger.info("📋 ChatOpenAI attributes:")
        for attr in dir(llm):
            if not attr.startswith('_') and not callable(getattr(llm, attr)):
                try:
                    value = getattr(llm, attr)
                    logger.info(f"  - {attr}: {type(value)} = {str(value)[:100]}")
                except Exception as e:
                    logger.info(f"  - {attr}: (error accessing: {e})")
        
        return llm
        
    except Exception as e:
        logger.error(f"❌ Failed to create LLM: {e}")
        return None

def test_basic_graphcypherqa_creation(llm, graph):
    """Test basic GraphCypherQAChain creation."""
    logger.info("\n=== TESTING BASIC GRAPHCYPHERQACHAIN CREATION ===")
    
    if not llm or not graph:
        logger.error("❌ Cannot test - missing LLM or Graph")
        return None
    
    try:
        from langchain.chains import GraphCypherQAChain
        
        logger.info("📋 Creating basic GraphCypherQAChain...")
        
        # Basic creation
        chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            verbose=True
        )
        
        logger.info("✅ GraphCypherQAChain created successfully!")
        
        # Inspect the chain
        logger.info("📋 GraphCypherQAChain instance attributes:")
        for attr in dir(chain):
            if not attr.startswith('_') and not callable(getattr(chain, attr)):
                try:
                    value = getattr(chain, attr)
                    logger.info(f"  - {attr}: {type(value)} = {str(value)[:100]}")
                except Exception as e:
                    logger.info(f"  - {attr}: (error accessing: {e})")
        
        return chain
        
    except Exception as e:
        logger.error(f"❌ Failed to create basic GraphCypherQAChain: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def inspect_chain_input_keys(chain):
    """Inspect the chain's input keys."""
    logger.info("\n=== INSPECTING CHAIN INPUT KEYS ===")
    
    if not chain:
        logger.error("❌ No chain to inspect")
        return
    
    try:
        # Check input_keys property
        if hasattr(chain, 'input_keys'):
            input_keys = chain.input_keys
            logger.info(f"✅ Chain input_keys: {input_keys}")
            logger.info(f"📋 Input keys type: {type(input_keys)}")
            
            if isinstance(input_keys, list):
                logger.info("📋 Individual input keys:")
                for i, key in enumerate(input_keys):
                    logger.info(f"  {i}: '{key}' ({type(key)})")
        else:
            logger.warning("⚠️ Chain has no input_keys attribute")
        
        # Check output_keys property
        if hasattr(chain, 'output_keys'):
            output_keys = chain.output_keys
            logger.info(f"✅ Chain output_keys: {output_keys}")
        else:
            logger.warning("⚠️ Chain has no output_keys attribute")
        
        # Check for prompt-related attributes
        prompt_attrs = ['cypher_prompt', 'qa_prompt', 'prompt']
        for attr in prompt_attrs:
            if hasattr(chain, attr):
                prompt_obj = getattr(chain, attr)
                logger.info(f"✅ Found {attr}: {type(prompt_obj)}")
                
                if hasattr(prompt_obj, 'input_variables'):
                    logger.info(f"📋 {attr} input_variables: {prompt_obj.input_variables}")
        
        # Check memory/history attributes
        if hasattr(chain, 'memory'):
            logger.info(f"📋 Chain has memory: {type(chain.memory)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to inspect chain input keys: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_custom_prompts(llm, graph):
    """Test GraphCypherQAChain with custom prompts."""
    logger.info("\n=== TESTING CUSTOM PROMPTS ===")
    
    if not llm or not graph:
        logger.error("❌ Cannot test - missing LLM or Graph")
        return None
    
    try:
        from langchain.chains import GraphCypherQAChain
        from langchain.prompts import PromptTemplate
        
        # Test 1: Custom cypher prompt with different input variables
        logger.info("🧪 Test 1: Custom cypher prompt with 'question' input variable")
        
        cypher_prompt = PromptTemplate.from_template(
            "Generate a Cypher query for this question: {question}\nCypher Query:"
        )
        
        chain1 = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            cypher_prompt=cypher_prompt,
            verbose=True
        )
        
        logger.info("✅ Chain with custom cypher prompt created")
        logger.info(f"📋 Input keys: {chain1.input_keys}")
        
        # Test 2: Custom prompts with name/source variables
        logger.info("🧪 Test 2: Custom cypher prompt with 'name' and 'source' variables")
        
        try:
            cypher_prompt_name_source = PromptTemplate.from_template(
                "Generate Cypher query for name '{name}' from source '{source}'\nCypher Query:"
            )
            
            chain2 = GraphCypherQAChain.from_llm(
                llm=llm,
                graph=graph,
                cypher_prompt=cypher_prompt_name_source,
                verbose=True
            )
            
            logger.info("✅ Chain with name/source prompt created")
            logger.info(f"📋 Input keys: {chain2.input_keys}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to create name/source prompt chain: {e}")
        
        # Test 3: QA prompt variations
        logger.info("🧪 Test 3: Custom QA prompt")
        
        qa_prompt = PromptTemplate.from_template(
            "Question: {question}\nContext: {context}\nAnswer:"
        )
        
        chain3 = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            qa_prompt=qa_prompt,
            verbose=True
        )
        
        logger.info("✅ Chain with custom QA prompt created")
        logger.info(f"📋 Input keys: {chain3.input_keys}")
        
        return [chain1, chain3]
        
    except Exception as e:
        logger.error(f"❌ Failed to test custom prompts: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def test_different_input_combinations(chains):
    """Test different input combinations without actually executing."""
    logger.info("\n=== TESTING DIFFERENT INPUT COMBINATIONS ===")
    
    if not chains:
        logger.error("❌ No chains to test")
        return
    
    # Test inputs to try
    test_inputs = [
        {"query": "Show all facilities"},
        {"question": "Show all facilities"},  
        {"name": "TestName", "source": "TestSource"},
        {"input": "Show all facilities"},
        {"text": "Show all facilities"}
    ]
    
    for i, chain in enumerate(chains):
        if not chain:
            continue
            
        logger.info(f"\n🧪 Testing Chain {i+1}:")
        logger.info(f"📋 Expected input keys: {chain.input_keys}")
        
        for test_input in test_inputs:
            logger.info(f"  Testing input: {test_input}")
            
            try:
                # Just validate input format, don't execute
                input_keys_set = set(chain.input_keys)
                test_keys_set = set(test_input.keys())
                
                if input_keys_set == test_keys_set:
                    logger.info(f"    ✅ Input keys match exactly")
                elif input_keys_set.issubset(test_keys_set):
                    logger.info(f"    ✅ All required keys present (+ extras)")
                elif test_keys_set.issubset(input_keys_set):
                    missing = input_keys_set - test_keys_set
                    logger.info(f"    ⚠️ Missing keys: {missing}")
                else:
                    logger.info(f"    ❌ Key mismatch - required: {input_keys_set}, provided: {test_keys_set}")
                
            except Exception as e:
                logger.warning(f"    ⚠️ Error testing input: {e}")

def analyze_findings():
    """Analyze and summarize findings."""
    logger.info("\n=== ANALYSIS AND FINDINGS ===")
    
    findings = [
        "🔍 KEY FINDINGS:",
        "",
        "1. DEFAULT INPUT KEYS:",
        "   - Standard GraphCypherQAChain expects 'query' as input key",
        "   - This is consistent with LangChain's typical chain patterns",
        "",
        "2. CUSTOM PROMPT IMPACT:",
        "   - Custom prompts can change input key requirements",
        "   - Input variables in PromptTemplate determine required keys",
        "   - If cypher_prompt uses '{name}' and '{source}', those become required",
        "",
        "3. POSSIBLE CAUSES FOR 'name' and 'source' REQUIREMENT:",
        "   - Custom prompt template with these variables",
        "   - Inherited configuration from another chain",
        "   - Memory or conversation history setup",
        "   - Bug in LangChain version or configuration",
        "",
        "4. TROUBLESHOOTING RECOMMENDATIONS:",
        "   - Check all PromptTemplate definitions for input_variables",
        "   - Verify cypher_prompt and qa_prompt configurations",
        "   - Ensure consistent use of 'query' or 'question' variable",
        "   - Review chain creation parameters",
        "",
        "5. DEBUGGING STEPS:",
        "   - Print chain.input_keys after creation",
        "   - Inspect chain.cypher_prompt.input_variables",
        "   - Check for any custom memory or conversation setup",
        "   - Verify LangChain version compatibility"
    ]
    
    for finding in findings:
        logger.info(finding)

def main():
    """Main debugging function."""
    logger.info(f"🐛 GraphCypherQAChain Input Keys Debugging Started at {datetime.now()}")
    logger.info(f"🐍 Python version: {sys.version}")
    logger.info(f"📁 Working directory: {os.getcwd()}")
    
    # Step 1: Debug imports
    if not debug_langchain_imports():
        logger.error("❌ Failed to import required modules. Exiting.")
        return
    
    # Step 2: Inspect GraphCypherQAChain class
    inspect_graphcypherqa_chain_class()
    
    # Step 3: Create minimal components
    graph = create_minimal_neo4j_graph()
    llm = create_minimal_llm()
    
    # Step 4: Test basic chain creation
    basic_chain = test_basic_graphcypherqa_creation(llm, graph)
    
    # Step 5: Inspect input keys
    inspect_chain_input_keys(basic_chain)
    
    # Step 6: Test custom prompts
    custom_chains = test_custom_prompts(llm, graph)
    
    # Step 7: Test input combinations
    all_chains = [basic_chain] + (custom_chains if custom_chains else [])
    test_different_input_combinations(all_chains)
    
    # Step 8: Analyze findings
    analyze_findings()
    
    logger.info(f"\n✅ Debugging completed at {datetime.now()}")
    logger.info("📋 Check debug_cypher_chain.log for detailed results")

if __name__ == "__main__":
    main()