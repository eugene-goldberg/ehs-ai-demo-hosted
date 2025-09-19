#!/usr/bin/env python3
"""
Debug script to investigate citation loss in the chatbot pipeline.
This script tests the complete flow from Neo4j → context_retriever → prompt_augmenter → LLM prompt
to identify exactly where the citations (Smithfield Foods, Archer Daniels Midland, Cleveland-Cliffs, General Motors) are being lost.
"""

import sys
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any

# Add the services path to sys.path
sys.path.append('/home/azureuser/dev/ehs-ai-demo/data-foundation/backend/src/services')

from context_retriever import ContextRetriever
from prompt_augmenter import PromptAugmenter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_neo4j_direct_query():
    """Test direct Neo4j query to see raw recommendation data with supporting_evidence"""
    from neo4j import GraphDatabase
    from dotenv import load_dotenv
    
    load_dotenv()
    
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.getenv('NEO4J_USERNAME', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD', 'password')
    
    print("\n" + "="*80)
    print("STEP 1: DIRECT NEO4J QUERY - RAW DATA FROM DATABASE")
    print("="*80)
    
    try:
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        with driver.session() as session:
            # Query to get raw recommendation nodes with ALL properties
            query = """
            MATCH (r:Recommendation)
            WHERE r.site_id = 'algonquin'
            RETURN r
            ORDER BY r.created_date DESC
            LIMIT 1
            """
            
            result = session.run(query)
            records = list(result)
            
            if not records:
                print("❌ No recommendations found in Neo4j for Algonquin site")
                return None
            
            node = records[0]['r']
            print(f"✅ Found recommendation node for site: {node.get('site_id', 'unknown')}")
            print(f"📅 Created date: {node.get('created_date', 'unknown')}")
            print(f"🔢 Version: {node.get('version', 'unknown')}")
            
            # Show ALL node properties
            print("\n🔍 ALL NODE PROPERTIES:")
            for key, value in dict(node).items():
                if key == 'recommendations':
                    print(f"  {key}: [JSON - will analyze below]")
                else:
                    print(f"  {key}: {value}")
            
            # Parse and analyze the recommendations JSON
            recommendations_json = node.get('recommendations', '[]')
            try:
                if isinstance(recommendations_json, str):
                    recommendations_list = json.loads(recommendations_json)
                else:
                    recommendations_list = recommendations_json
                    
                print(f"\n📋 PARSED RECOMMENDATIONS COUNT: {len(recommendations_list)}")
                
                # Check each recommendation for citations/supporting_evidence
                citations_found = []
                for i, rec in enumerate(recommendations_list, 1):
                    print(f"\n🔍 RECOMMENDATION {i}:")
                    print(f"  Title: {rec.get('title', 'N/A')}")
                    print(f"  Category: {rec.get('category', 'N/A')}")
                    print(f"  Priority: {rec.get('priority', 'N/A')}")
                    
                    # Check for various citation fields
                    citation_fields = [
                        'industry_citations', 'supporting_evidence', 'citations', 
                        'sources', 'references', 'examples'
                    ]
                    
                    for field in citation_fields:
                        if field in rec:
                            value = rec[field]
                            print(f"  ✅ {field}: {value}")
                            citations_found.append((i, field, value))
                        else:
                            print(f"  ❌ {field}: NOT FOUND")
                    
                    # Show all keys in this recommendation
                    print(f"  📄 All keys in recommendation: {list(rec.keys())}")
                
                if citations_found:
                    print(f"\n🎯 CITATIONS FOUND IN RAW DATA:")
                    for rec_num, field, value in citations_found:
                        print(f"  Rec {rec_num} - {field}: {value}")
                else:
                    print(f"\n⚠️  NO CITATIONS FOUND IN ANY RECOMMENDATIONS")
                
                return node, recommendations_list, citations_found
                
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse recommendations JSON: {e}")
                return None
                
    except Exception as e:
        print(f"❌ Error connecting to Neo4j: {e}")
        return None
    finally:
        try:
            driver.close()
        except:
            pass

def test_context_retriever(site='algonquin'):
    """Test the ContextRetriever to see what it returns"""
    print("\n" + "="*80)
    print("STEP 2: CONTEXT RETRIEVER - PROCESSED DATA")
    print("="*80)
    
    try:
        retriever = ContextRetriever()
        context = retriever.get_recommendations_context(site=site)
        
        print(f"✅ ContextRetriever returned data for site: {site}")
        print(f"📊 Summary: {context.get('summary', 'N/A')}")
        
        data = context.get('data', {})
        recommendations = data.get('recommendations', [])
        print(f"📋 Recommendations count: {len(recommendations)}")
        
        # Check each recommendation for citations
        citations_found = []
        for i, rec in enumerate(recommendations, 1):
            print(f"\n🔍 PROCESSED RECOMMENDATION {i}:")
            print(f"  Title: {rec.get('title', 'N/A')}")
            print(f"  Category: {rec.get('category', 'N/A')}")
            print(f"  Priority: {rec.get('priority', 'N/A')}")
            print(f"  Impact: {rec.get('impact', 'N/A')}")
            
            # Check for citation fields
            citation_fields = [
                'industry_citations', 'supporting_evidence', 'citations', 
                'sources', 'references', 'examples'
            ]
            
            for field in citation_fields:
                if field in rec:
                    value = rec[field]
                    print(f"  ✅ {field}: {value}")
                    citations_found.append((i, field, value))
                else:
                    print(f"  ❌ {field}: NOT FOUND")
            
            # Show all keys in processed recommendation
            print(f"  📄 All keys in processed recommendation: {list(rec.keys())}")
        
        if citations_found:
            print(f"\n🎯 CITATIONS FOUND IN PROCESSED DATA:")
            for rec_num, field, value in citations_found:
                print(f"  Rec {rec_num} - {field}: {value}")
        else:
            print(f"\n⚠️  NO CITATIONS FOUND IN PROCESSED DATA")
        
        return context, citations_found
        
    except Exception as e:
        print(f"❌ Error in ContextRetriever: {e}")
        return None, []

def test_prompt_augmenter(context_data, user_query="What recommendations do you have for reducing environmental impact?"):
    """Test the PromptAugmenter to see the final formatted prompt"""
    print("\n" + "="*80)
    print("STEP 3: PROMPT AUGMENTER - FORMATTED CONTEXT FOR LLM")
    print("="*80)
    
    try:
        augmenter = PromptAugmenter()
        formatted_prompt = augmenter.create_augmented_prompt(
            user_query=user_query,
            context_data=context_data,
            intent_type="recommendations"
        )
        
        print(f"✅ PromptAugmenter created formatted prompt")
        print(f"📝 User query: {user_query}")
        print(f"📄 Prompt length: {len(formatted_prompt)} characters")
        
        # Check if citations appear in the final prompt
        citation_keywords = [
            'Smithfield Foods', 'Archer Daniels Midland', 'Cleveland-Cliffs', 
            'General Motors', 'industry_citations', 'supporting_evidence',
            'citations', 'sources', 'references'
        ]
        
        citations_in_prompt = []
        for keyword in citation_keywords:
            if keyword.lower() in formatted_prompt.lower():
                citations_in_prompt.append(keyword)
        
        print(f"\n🔍 CITATION KEYWORDS FOUND IN FINAL PROMPT:")
        if citations_in_prompt:
            for keyword in citations_in_prompt:
                print(f"  ✅ '{keyword}' found")
        else:
            print(f"  ❌ NO CITATION KEYWORDS FOUND")
        
        # Show the relevant sections of the prompt
        print(f"\n📋 FORMATTED CONTEXT SECTION:")
        lines = formatted_prompt.split('\n')
        context_start = -1
        context_end = -1
        
        for i, line in enumerate(lines):
            if 'CONTEXT DATA:' in line:
                context_start = i
            elif 'USER QUESTION:' in line:
                context_end = i
                break
        
        if context_start >= 0:
            context_section = lines[context_start:context_end if context_end > 0 else None]
            for line in context_section:
                print(f"  {line}")
        
        return formatted_prompt, citations_in_prompt
        
    except Exception as e:
        print(f"❌ Error in PromptAugmenter: {e}")
        return None, []

def test_complete_flow():
    """Test the complete flow and generate comprehensive report"""
    print("\n" + "🔥"*80)
    print("COMPREHENSIVE CITATION FLOW DEBUG REPORT")
    print("Investigating where citations are lost in the pipeline")
    print("🔥"*80)
    
    # Step 1: Test direct Neo4j query
    neo4j_result = test_neo4j_direct_query()
    
    # Step 2: Test ContextRetriever
    context_data, context_citations = test_context_retriever('algonquin')
    
    # Step 3: Test PromptAugmenter
    if context_data:
        prompt, prompt_citations = test_prompt_augmenter(context_data)
    else:
        prompt, prompt_citations = None, []
    
    # Generate summary report
    print("\n" + "="*80)
    print("CITATION FLOW ANALYSIS SUMMARY")
    print("="*80)
    
    if neo4j_result:
        node, recs_list, neo4j_citations = neo4j_result
        print(f"✅ Neo4j Raw Data: Found {len(neo4j_citations)} citation references")
        for rec_num, field, value in neo4j_citations:
            print(f"   - Rec {rec_num}: {field} = {value[:100]}{'...' if len(str(value)) > 100 else ''}")
    else:
        print(f"❌ Neo4j Raw Data: No data retrieved")
        neo4j_citations = []
    
    print(f"🔄 Context Retriever: Found {len(context_citations)} citation references")
    for rec_num, field, value in context_citations:
        print(f"   - Rec {rec_num}: {field} = {value[:100]}{'...' if len(str(value)) > 100 else ''}")
    
    print(f"📝 Prompt Augmenter: Found {len(prompt_citations)} citation keywords in final prompt")
    for keyword in prompt_citations:
        print(f"   - '{keyword}' appears in prompt")
    
    # Identify where the loss occurs
    print(f"\n🔍 CITATION LOSS ANALYSIS:")
    if neo4j_citations and not context_citations:
        print(f"❌ CITATIONS LOST IN CONTEXT RETRIEVER")
        print(f"   Citations exist in Neo4j but are not being extracted by ContextRetriever")
    elif context_citations and not prompt_citations:
        print(f"❌ CITATIONS LOST IN PROMPT AUGMENTER")
        print(f"   Citations exist in context but are not appearing in final prompt")
    elif neo4j_citations and context_citations and prompt_citations:
        print(f"✅ CITATIONS PRESERVED THROUGH ENTIRE PIPELINE")
    elif not neo4j_citations:
        print(f"❌ NO CITATIONS IN SOURCE DATA")
        print(f"   The problem is that no citations exist in the Neo4j database")
    else:
        print(f"❓ UNCLEAR CITATION LOSS PATTERN")
    
    # Save detailed report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"/home/azureuser/dev/ehs-ai-demo/data-foundation/backend/tmp/citation_debug_report_{timestamp}.json"
    
    report_data = {
        "timestamp": timestamp,
        "neo4j_raw_data": {
            "found": neo4j_result is not None,
            "citations_count": len(neo4j_citations) if neo4j_result else 0,
            "citations": neo4j_citations if neo4j_result else []
        },
        "context_retriever": {
            "found": context_data is not None,
            "citations_count": len(context_citations),
            "citations": context_citations
        },
        "prompt_augmenter": {
            "found": prompt is not None,
            "citations_count": len(prompt_citations),
            "citations": prompt_citations
        }
    }
    
    try:
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        print(f"\n💾 Detailed report saved to: {report_file}")
    except Exception as e:
        print(f"❌ Failed to save report: {e}")
    
    return report_data

if __name__ == "__main__":
    test_complete_flow()
