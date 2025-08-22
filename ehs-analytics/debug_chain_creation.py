#!/usr/bin/env python3
"""
Debug the actual chain creation process to understand the input keys mismatch.
"""
import sys
import logging
sys.path.insert(0, '/Users/eugene/dev/ai/agentos/ehs-ai-demo/ehs-analytics/src')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_chain_creation():
    """Debug the actual chain creation in the existing code."""
    try:
        from langchain.chains import GraphCypherQAChain
        from langchain.prompts import PromptTemplate
        from langchain_openai import ChatOpenAI
        from ehs_analytics.retrieval.strategies.text2cypher import Text2CypherRetriever
        
        logger.info("🔍 Debugging chain creation...")
        
        # Create the components exactly as the retriever does
        config = {
            'neo4j_uri': 'bolt://test:7687', 
            'neo4j_user': 'test', 
            'neo4j_password': 'test', 
            'openai_api_key': 'fake-key'
        }
        
        retriever = Text2CypherRetriever(config)
        
        # Get the prompts
        cypher_prompt_text = retriever._build_ehs_cypher_prompt()
        qa_prompt_text = "Answer the user's question based on the database results:\n\nQuestion: {question}\nDatabase Results: {context}\n\nAnswer:"
        
        # Create PromptTemplate objects
        cypher_prompt = PromptTemplate.from_template(cypher_prompt_text)
        qa_prompt = PromptTemplate.from_template(qa_prompt_text)
        
        logger.info(f"📋 Cypher prompt variables: {cypher_prompt.input_variables}")
        logger.info(f"📋 QA prompt variables: {qa_prompt.input_variables}")
        
        # Combine variables (this is what GraphCypherQAChain does)
        combined_vars = set(cypher_prompt.input_variables) | set(qa_prompt.input_variables)
        logger.info(f"📋 Combined variables: {combined_vars}")
        
        if 'name' in combined_vars or 'source' in combined_vars:
            logger.error("🚨 PROBLEM FOUND: 'name' or 'source' in combined variables!")
            
            # Find which prompt contains these variables
            if 'name' in cypher_prompt.input_variables or 'source' in cypher_prompt.input_variables:
                logger.error("❌ Problem is in Cypher prompt")
                # Find the lines
                lines = cypher_prompt_text.split('\n')
                for i, line in enumerate(lines, 1):
                    if '{name}' in line or '{source}' in line:
                        logger.error(f"Line {i}: {line}")
            
            if 'name' in qa_prompt.input_variables or 'source' in qa_prompt.input_variables:
                logger.error("❌ Problem is in QA prompt")
                logger.error(f"QA prompt text: {qa_prompt_text}")
        else:
            logger.info("✅ Prompts look correct individually")
        
        # Try to simulate the problem by examining what happens in our debugging script
        logger.info("\n🧪 Reproducing the debugging script behavior...")
        
        # The debugging script showed different results, let's see if we can reproduce that
        from langchain.prompts import PromptTemplate
        import re
        
        # Check if there's something different when we process the prompt
        vars_direct = re.findall(r'\{(\w+)\}', cypher_prompt_text)
        logger.info(f"📋 Direct regex on cypher_prompt_text: {set(vars_direct)}")
        
        # Check if PromptTemplate.from_template does something different
        template_obj = PromptTemplate.from_template(cypher_prompt_text)
        logger.info(f"📋 PromptTemplate.input_variables: {template_obj.input_variables}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Debug failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    debug_chain_creation()