#!/usr/bin/env python3
"""
Final debugging script to understand the query/question input key mismatch.

Based on our findings:
- The EHS prompt template uses {question} variable (correct)
- But somehow the chain expects 'name' and 'source' as input keys
- This script will focus on finding where this mismatch occurs.

Uses virtual environment: /Users/eugene/dev/ai/agentos/ehs-ai-demo/ehs-analytics/venv
"""

import sys
import os
import logging
from datetime import datetime
import traceback
import re

# Setup logging
logging.basicConfig(
    filename='debug_cypher_chain_final.log',
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

def analyze_actual_chain_creation():
    """Analyze the actual chain creation process in the existing code."""
    logger.info("=== ANALYZING ACTUAL CHAIN CREATION PROCESS ===")
    
    try:
        # Add the project path
        sys.path.insert(0, '/Users/eugene/dev/ai/agentos/ehs-ai-demo/ehs-analytics/src')
        
        from langchain.chains import GraphCypherQAChain
        from langchain.prompts import PromptTemplate
        from langchain_openai import ChatOpenAI
        from ehs_analytics.retrieval.strategies.text2cypher import Text2CypherRetriever
        
        logger.info("✅ Successfully imported modules")
        
        # Create a retriever instance to analyze its chain creation
        config = {
            "neo4j_uri": "bolt://test:7687",
            "neo4j_user": "test",
            "neo4j_password": "test",
            "openai_api_key": "fake-key",
            "model_name": "gpt-3.5-turbo"
        }
        
        retriever = Text2CypherRetriever(config)
        
        # Get the EHS prompt
        ehs_prompt_text = retriever._build_ehs_cypher_prompt()
        logger.info(f"📋 EHS prompt text length: {len(ehs_prompt_text)}")
        
        # Find all template variables in the prompt
        template_vars = re.findall(r'\{(\w+)\}', ehs_prompt_text)
        unique_vars = set(template_vars)
        logger.info(f"📋 All template variables: {unique_vars}")
        
        # Create the PromptTemplate from the EHS prompt
        logger.info("🧪 Creating PromptTemplate from EHS prompt...")
        try:
            ehs_prompt_template = PromptTemplate.from_template(ehs_prompt_text)
            logger.info(f"✅ PromptTemplate created successfully")
            logger.info(f"📋 PromptTemplate input_variables: {ehs_prompt_template.input_variables}")
        except Exception as e:
            logger.error(f"❌ Failed to create PromptTemplate: {e}")
            return
        
        # Test different QA prompt templates
        logger.info("\n🧪 Testing different QA prompt configurations...")
        
        qa_prompts_to_test = [
            ("Standard", "Answer the user's question based on the database results:\n\nQuestion: {question}\nDatabase Results: {context}\n\nAnswer:"),
            ("With query", "Answer based on results:\n\nQuery: {query}\nResults: {context}\n\nAnswer:"),
            ("With name/source", "Answer for {name} from {source}:\n\nResults: {context}\n\nAnswer:")
        ]
        
        for prompt_name, qa_template in qa_prompts_to_test:
            try:
                qa_prompt = PromptTemplate.from_template(qa_template)
                logger.info(f"📋 {prompt_name} QA prompt variables: {qa_prompt.input_variables}")
                
                # The key insight: when we combine cypher_prompt and qa_prompt,
                # the input_keys are the union of both templates' variables
                combined_vars = set(ehs_prompt_template.input_variables) | set(qa_prompt.input_variables)
                logger.info(f"📋 Combined input variables would be: {combined_vars}")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to test {prompt_name}: {e}")
        
        return ehs_prompt_template
        
    except Exception as e:
        logger.error(f"❌ Failed to analyze chain creation: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def check_existing_chain_initialization():
    """Check how the existing chain is actually initialized."""
    logger.info("\n=== CHECKING EXISTING CHAIN INITIALIZATION ===")
    
    try:
        # Look at the actual initialization code
        text2cypher_file = "/Users/eugene/dev/ai/agentos/ehs-ai-demo/ehs-analytics/src/ehs_analytics/retrieval/strategies/text2cypher.py"
        
        logger.info(f"📋 Reading {text2cypher_file}")
        
        with open(text2cypher_file, 'r') as f:
            content = f.read()
        
        # Find the GraphCypherQAChain.from_llm call
        chain_creation_match = re.search(
            r'GraphCypherQAChain\.from_llm\((.*?)\)',
            content,
            re.DOTALL
        )
        
        if chain_creation_match:
            chain_creation_code = chain_creation_match.group(1)
            logger.info("✅ Found GraphCypherQAChain.from_llm call")
            logger.info(f"📋 Chain creation parameters:\n{chain_creation_code[:500]}...")
            
            # Look for qa_prompt parameter
            if 'qa_prompt' in chain_creation_code:
                logger.info("✅ Found qa_prompt parameter in chain creation")
                
                # Extract the qa_prompt template
                qa_prompt_match = re.search(
                    r'qa_prompt=PromptTemplate\.from_template\("(.*?)"\)',
                    chain_creation_code,
                    re.DOTALL
                )
                
                if qa_prompt_match:
                    qa_template = qa_prompt_match.group(1)
                    logger.info(f"📋 QA template found: {qa_template[:200]}...")
                    
                    # Find variables in the QA template
                    qa_vars = re.findall(r'\{(\w+)\}', qa_template)
                    logger.info(f"📋 QA template variables: {set(qa_vars)}")
                    
                    # This is the smoking gun!
                    if 'name' in qa_vars or 'source' in qa_vars:
                        logger.info("🔍 FOUND THE PROBLEM: QA template contains 'name' or 'source' variables!")
                    
        else:
            logger.warning("⚠️ Could not find GraphCypherQAChain.from_llm call")
        
        # Also check for any other chain setups
        if 'cypher_prompt=' in content:
            logger.info("✅ Found cypher_prompt parameter usage")
        
        if 'qa_prompt=' in content:
            logger.info("✅ Found qa_prompt parameter usage")
            
    except Exception as e:
        logger.error(f"❌ Failed to check chain initialization: {e}")

def simulate_chain_input_key_calculation():
    """Simulate how LangChain calculates input keys."""
    logger.info("\n=== SIMULATING INPUT KEY CALCULATION ===")
    
    try:
        from langchain.prompts import PromptTemplate
        
        # Test scenarios
        scenarios = [
            {
                "name": "Correct setup",
                "cypher_template": "Generate Cypher for: {question}\nCypher:",
                "qa_template": "Question: {question}\nContext: {context}\nAnswer:"
            },
            {
                "name": "Wrong QA template", 
                "cypher_template": "Generate Cypher for: {question}\nCypher:",
                "qa_template": "Name: {name}\nSource: {source}\nContext: {context}\nAnswer:"
            },
            {
                "name": "Wrong Cypher template",
                "cypher_template": "Generate Cypher for {name} from {source}\nCypher:",
                "qa_template": "Question: {question}\nContext: {context}\nAnswer:"
            }
        ]
        
        for scenario in scenarios:
            logger.info(f"\n🧪 Testing scenario: {scenario['name']}")
            
            try:
                cypher_prompt = PromptTemplate.from_template(scenario['cypher_template'])
                qa_prompt = PromptTemplate.from_template(scenario['qa_template'])
                
                cypher_vars = set(cypher_prompt.input_variables)
                qa_vars = set(qa_prompt.input_variables)
                combined_vars = cypher_vars | qa_vars
                
                logger.info(f"📋 Cypher prompt variables: {cypher_vars}")
                logger.info(f"📋 QA prompt variables: {qa_vars}")
                logger.info(f"📋 Combined input keys would be: {combined_vars}")
                
                if 'name' in combined_vars and 'source' in combined_vars:
                    logger.info("🔍 THIS SCENARIO WOULD CAUSE THE 'name' and 'source' INPUT KEY REQUIREMENT!")
                
            except Exception as e:
                logger.warning(f"⚠️ Scenario failed: {e}")
    
    except Exception as e:
        logger.error(f"❌ Simulation failed: {e}")

def provide_solution():
    """Provide the solution based on our analysis."""
    logger.info("\n=== SOLUTION RECOMMENDATION ===")
    
    solutions = [
        "",
        "🔧 SOLUTION STEPS:",
        "",
        "1. IDENTIFY THE PROBLEM:",
        "   - The issue is likely in the QA prompt template, not the Cypher prompt",
        "   - Look for a qa_prompt parameter in GraphCypherQAChain.from_llm()",
        "   - Check if it uses {name} and {source} instead of {question} and {context}",
        "",
        "2. FIND THE PROBLEMATIC CODE:",
        "   - Look in text2cypher.py around line 145-155",
        "   - Find: qa_prompt=PromptTemplate.from_template(...)",
        "   - Check what variables are used in that template",
        "",
        "3. FIX THE TEMPLATE:",
        "   - Change any {name} or {source} to appropriate variables",
        "   - Standard pattern: {question} for the user query, {context} for results",
        "   - Example fix:",
        '     OLD: "Answer for {name} from {source}: {context}"',
        '     NEW: "Answer the question based on results:\\nQuestion: {question}\\nResults: {context}\\nAnswer:"',
        "",
        "4. VERIFY THE FIX:",
        "   - After changing, create the chain and check: print(chain.input_keys)",
        "   - Should show ['query'] for standard GraphCypherQAChain",
        "",
        "5. ALTERNATIVE APPROACH:",
        "   - Use 'question' as input key instead of 'query'",
        "   - Call chain.invoke({'question': query}) instead of {'query': query}",
        "   - But fixing the template is the better long-term solution"
    ]
    
    for solution in solutions:
        logger.info(solution)

def main():
    """Main debugging function."""
    logger.info(f"🐛 Final GraphCypherQAChain Debugging Started at {datetime.now()}")
    
    # Step 1: Analyze the actual chain creation
    analyze_actual_chain_creation()
    
    # Step 2: Check existing initialization
    check_existing_chain_initialization()
    
    # Step 3: Simulate input key calculation
    simulate_chain_input_key_calculation()
    
    # Step 4: Provide solution
    provide_solution()
    
    logger.info(f"\n✅ Final debugging completed at {datetime.now()}")
    logger.info("📋 Check debug_cypher_chain_final.log for detailed results")

if __name__ == "__main__":
    main()