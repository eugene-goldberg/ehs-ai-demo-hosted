#!/usr/bin/env python3
"""
Debug LangChain PromptTemplate parsing behavior.
"""
import sys
import logging
import re
sys.path.insert(0, '/Users/eugene/dev/ai/agentos/ehs-ai-demo/ehs-analytics/src')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_prompt_parsing():
    """Analyze how LangChain is parsing the prompt template."""
    try:
        from langchain.prompts import PromptTemplate
        from ehs_analytics.retrieval.strategies.text2cypher import Text2CypherRetriever
        
        logger.info("🔍 Analyzing LangChain PromptTemplate parsing...")
        
        # Get the prompt
        config = {'neo4j_uri': 'bolt://test:7687', 'neo4j_user': 'test', 'neo4j_password': 'test', 'openai_api_key': 'fake-key'}
        retriever = Text2CypherRetriever(config)
        cypher_prompt_text = retriever._build_ehs_cypher_prompt()
        
        # Manual regex check
        manual_vars = re.findall(r'\{(\w+)\}', cypher_prompt_text)
        logger.info(f"📋 Manual regex finds: {set(manual_vars)}")
        
        # Check for specific patterns that might confuse LangChain
        logger.info("\n🔍 Checking for problematic patterns in the prompt...")
        
        # Look for double curly braces
        double_braces = re.findall(r'\{\{(\w+)\}\}', cypher_prompt_text)
        if double_braces:
            logger.warning(f"⚠️ Found double braces (Neo4j syntax): {double_braces}")
            logger.info("This might be causing LangChain to interpret Neo4j property syntax as template variables!")
        
        # Look for specific lines with 'name' or 'source'
        lines_with_name_source = []
        for i, line in enumerate(cypher_prompt_text.split('\n'), 1):
            if 'name' in line.lower() or 'source' in line.lower():
                if any(pattern in line for pattern in ['{name', '{source', '{{name', '{{source']):
                    lines_with_name_source.append((i, line))
        
        if lines_with_name_source:
            logger.warning("⚠️ Found lines that might contain name/source patterns:")
            for line_num, line in lines_with_name_source:
                logger.warning(f"Line {line_num}: {line.strip()}")
        
        # Create PromptTemplate and see what it thinks
        try:
            template = PromptTemplate.from_template(cypher_prompt_text)
            logger.info(f"\n📋 LangChain detected variables: {template.input_variables}")
            
            # Try to see what LangChain's parser is doing
            # Check if there's an alternative way to create the template
            try:
                # Manual template creation
                manual_template = PromptTemplate(
                    template=cypher_prompt_text,
                    input_variables=['question']
                )
                logger.info(f"📋 Manual template variables: {manual_template.input_variables}")
                logger.info("✅ Manual template creation works with just 'question'")
                
                # Test formatting with manual template
                test_format = manual_template.format(question="test query")
                logger.info("✅ Manual template formatting works")
                
            except Exception as e:
                logger.error(f"❌ Manual template failed: {e}")
        
        except Exception as e:
            logger.error(f"❌ PromptTemplate creation failed: {e}")
        
        # Test with a simple template to ensure LangChain is working correctly
        logger.info("\n🧪 Testing with simple template...")
        simple_template = PromptTemplate.from_template("Answer this question: {question}")
        logger.info(f"📋 Simple template variables: {simple_template.input_variables}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    analyze_prompt_parsing()