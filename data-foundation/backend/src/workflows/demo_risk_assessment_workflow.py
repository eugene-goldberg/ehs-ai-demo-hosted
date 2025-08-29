#!/usr/bin/env python3
"""
EHS AI Platform - Risk Assessment Workflow Demonstration Script

This comprehensive demonstration script showcases the complete risk assessment 
workflow integration, including:

1. Document processing through the enhanced workflow
2. Risk assessment execution with detailed output
3. LangSmith trace monitoring and display
4. Neo4j data storage and retrieval
5. Risk analysis results and recommendations
6. Performance metrics and error handling

The script is designed to be easily understood and executed, providing
clear insights into the integrated EHS AI system capabilities.

Usage:
    python3 demo_risk_assessment_workflow.py [options]
    
Example:
    python3 demo_risk_assessment_workflow.py --sample-document --enable-traces --query-results
    python3 demo_risk_assessment_workflow.py --document-path /path/to/document.pdf --facility-id DEMO_FACILITY_001

Author: EHS AI Development Team
Created: 2025-08-28
"""

import os
import sys
import json
import logging
import argparse
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add the backend src directory to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from workflows.ingestion_workflow_with_risk_assessment import (
    RiskAssessmentIntegratedWorkflow,
    create_risk_integrated_workflow,
    DocumentStateWithRisk
)
from agents.risk_assessment.agent import (
    RiskAssessmentAgent,
    create_risk_assessment_agent,
    RiskLevel,
    AssessmentStatus
)
from shared.common_fn import create_graph_database_connection
from langsmith_config import config as langsmith_config

# Set up logging with detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/demo_risk_assessment.log')
    ]
)

logger = logging.getLogger(__name__)


class RiskAssessmentDemo:
    """
    Comprehensive demonstration of the risk assessment workflow.
    
    This class orchestrates the complete demonstration including document processing,
    risk assessment, result analysis, and data querying.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the demonstration with configuration parameters."""
        self.config = config
        self.workflow = None
        self.risk_agent = None
        self.neo4j_connection = None
        self.demo_results = {}
        self.start_time = datetime.utcnow()
        
        logger.info("=" * 80)
        logger.info("EHS AI PLATFORM - RISK ASSESSMENT WORKFLOW DEMONSTRATION")
        logger.info("=" * 80)
        logger.info(f"Demonstration started at: {self.start_time}")
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize workflow components with error handling."""
        try:
            logger.info("\n🔧 INITIALIZING COMPONENTS")
            logger.info("-" * 50)
            
            # 1. Initialize the integrated workflow
            logger.info("1️⃣  Initializing Risk Assessment Integrated Workflow...")
            self.workflow = create_risk_integrated_workflow(
                llama_parse_api_key=self.config['llama_parse_api_key'],
                neo4j_uri=self.config['neo4j_uri'],
                neo4j_username=self.config['neo4j_username'],
                neo4j_password=self.config['neo4j_password'],
                neo4j_database=self.config['neo4j_database'],
                llm_model=self.config['llm_model'],
                enable_risk_assessment=True,
                enable_phase1_features=True,
                storage_path='/tmp/demo_storage/',
                risk_assessment_methodology='comprehensive'
            )
            logger.info("✅ Risk Assessment Integrated Workflow initialized successfully")
            
            # 2. Initialize standalone risk assessment agent for direct queries
            logger.info("2️⃣  Initializing standalone Risk Assessment Agent...")
            self.risk_agent = create_risk_assessment_agent(
                neo4j_uri=self.config['neo4j_uri'],
                neo4j_username=self.config['neo4j_username'],
                neo4j_password=self.config['neo4j_password'],
                neo4j_database=self.config['neo4j_database'],
                llm_model=self.config['llm_model'],
                enable_langsmith=langsmith_config.is_available,
                risk_assessment_methodology='comprehensive'
            )
            logger.info("✅ Standalone Risk Assessment Agent initialized successfully")
            
            # 3. Initialize Neo4j connection for direct querying
            logger.info("3️⃣  Establishing Neo4j database connection...")
            self.neo4j_connection = create_graph_database_connection(
                self.config['neo4j_uri'],
                self.config['neo4j_username'],
                self.config['neo4j_password'],
                self.config['neo4j_database']
            )
            logger.info("✅ Neo4j connection established successfully")
            
            # 4. Check LangSmith configuration
            if langsmith_config.is_available:
                logger.info("4️⃣  LangSmith tracing is enabled and available")
                logger.info(f"   🔍 Project: {langsmith_config.project_name}")
                logger.info("✅ LangSmith integration ready")
            else:
                logger.warning("4️⃣  LangSmith tracing is not available")
                logger.info("   ℹ️  Demonstration will continue without trace monitoring")
            
            logger.info("\n🎉 All components initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {str(e)}")
            raise RuntimeError(f"Failed to initialize demo components: {str(e)}")
    
    def run_complete_demonstration(self) -> Dict[str, Any]:
        """
        Run the complete risk assessment workflow demonstration.
        
        Returns:
            Comprehensive results dictionary with all demonstration data
        """
        try:
            logger.info("\n" + "=" * 80)
            logger.info("STARTING COMPLETE RISK ASSESSMENT DEMONSTRATION")
            logger.info("=" * 80)
            
            # Step 1: Document Processing
            processing_results = self._demonstrate_document_processing()
            self.demo_results['document_processing'] = processing_results
            
            # Step 2: Risk Assessment Analysis
            risk_results = self._demonstrate_risk_assessment_analysis(processing_results)
            self.demo_results['risk_assessment'] = risk_results
            
            # Step 3: LangSmith Trace Analysis (if enabled)
            if langsmith_config.is_available and self.config.get('show_traces', True):
                trace_results = self._demonstrate_langsmith_traces(processing_results)
                self.demo_results['langsmith_traces'] = trace_results
            
            # Step 4: Neo4j Data Querying
            if self.config.get('query_neo4j', True):
                neo4j_results = self._demonstrate_neo4j_queries(processing_results)
                self.demo_results['neo4j_queries'] = neo4j_results
            
            # Step 5: Performance Analysis
            performance_results = self._analyze_performance()
            self.demo_results['performance_analysis'] = performance_results
            
            # Step 6: Generate Summary Report
            summary_report = self._generate_summary_report()
            self.demo_results['summary_report'] = summary_report
            
            logger.info("\n🎊 DEMONSTRATION COMPLETED SUCCESSFULLY!")
            logger.info(f"Total execution time: {datetime.utcnow() - self.start_time}")
            
            return self.demo_results
            
        except Exception as e:
            logger.error(f"❌ Demonstration failed: {str(e)}")
            self.demo_results['error'] = str(e)
            return self.demo_results
    
    def _demonstrate_document_processing(self) -> Dict[str, Any]:
        """Demonstrate document processing through the integrated workflow."""
        logger.info("\n📄 STEP 1: DOCUMENT PROCESSING DEMONSTRATION")
        logger.info("-" * 60)
        
        try:
            # Get document path
            document_path = self._get_demo_document_path()
            document_id = f"demo_doc_{int(time.time())}"
            
            logger.info(f"📋 Processing Document:")
            logger.info(f"   📂 Path: {document_path}")
            logger.info(f"   🆔 Document ID: {document_id}")
            logger.info(f"   📊 Document Type: {self._detect_document_type(document_path)}")
            
            # Prepare metadata
            metadata = {
                "facility_id": self.config.get('facility_id', 'DEMO_FACILITY_001'),
                "facility_name": "Demo Manufacturing Facility",
                "document_category": "environmental",
                "upload_source": "demonstration_script",
                "business_unit": "Manufacturing Operations",
                "compliance_year": "2024",
                "demo_run": True,
                "upload_timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"   📋 Metadata: {json.dumps(metadata, indent=2)}")
            
            # Start processing timer
            processing_start_time = time.time()
            
            logger.info("\n🔄 Starting Document Processing Workflow...")
            logger.info("   This will include: Validation → Parsing → Extraction → Risk Assessment")
            
            # Process the document through the integrated workflow
            final_state = self.workflow.process_document(
                file_path=document_path,
                document_id=document_id,
                metadata=metadata
            )
            
            processing_time = time.time() - processing_start_time
            
            # Analyze processing results
            results = self._analyze_processing_results(final_state, processing_time)
            
            logger.info("✅ Document processing completed!")
            logger.info(f"   ⏱️  Total processing time: {processing_time:.2f} seconds")
            logger.info(f"   📊 Final status: {final_state.get('status', 'unknown')}")
            logger.info(f"   🎯 Risk assessment status: {final_state.get('risk_assessment_status', 'not performed')}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {str(e)}")
            return {"error": str(e), "processing_time": 0}
    
    def _demonstrate_risk_assessment_analysis(self, processing_results: Dict[str, Any]) -> Dict[str, Any]:
        """Demonstrate detailed risk assessment analysis."""
        logger.info("\n🎯 STEP 2: RISK ASSESSMENT ANALYSIS")
        logger.info("-" * 60)
        
        try:
            if processing_results.get('error'):
                logger.warning("⚠️  Skipping risk assessment due to document processing error")
                return {"error": "Document processing failed", "skipped": True}
            
            document_state = processing_results.get('final_state')
            if not document_state:
                logger.warning("⚠️  No document state available for risk assessment")
                return {"error": "No document state available", "skipped": True}
            
            # Extract risk assessment results from the workflow
            risk_assessment_results = document_state.get('risk_assessment_results', {})
            risk_level = document_state.get('risk_level')
            risk_score = document_state.get('risk_score')
            risk_factors = document_state.get('risk_factors', [])
            recommendations = document_state.get('risk_recommendations', [])
            
            logger.info("📊 Risk Assessment Results:")
            logger.info(f"   🚨 Risk Level: {risk_level or 'Not assessed'}")
            logger.info(f"   📈 Risk Score: {risk_score or 'Not calculated'}")
            logger.info(f"   ⚠️  Risk Factors Identified: {len(risk_factors)}")
            logger.info(f"   💡 Recommendations Generated: {len(recommendations)}")
            
            # Display detailed risk factors
            if risk_factors:
                logger.info("\n🔍 DETAILED RISK FACTORS:")
                for i, factor in enumerate(risk_factors[:3], 1):  # Show first 3 factors
                    logger.info(f"   {i}. {factor.get('name', 'Unnamed Factor')}")
                    logger.info(f"      Category: {factor.get('category', 'Unknown')}")
                    logger.info(f"      Severity: {factor.get('severity', 'N/A')}/10")
                    logger.info(f"      Probability: {factor.get('probability', 'N/A')}")
                    logger.info(f"      Description: {factor.get('description', 'No description')[:100]}...")
                
                if len(risk_factors) > 3:
                    logger.info(f"   ... and {len(risk_factors) - 3} more risk factors")
            
            # Display recommendations
            if recommendations:
                logger.info("\n💡 TOP RECOMMENDATIONS:")
                for i, rec in enumerate(recommendations[:2], 1):  # Show first 2 recommendations
                    logger.info(f"   {i}. {rec.get('title', 'Untitled Recommendation')}")
                    logger.info(f"      Priority: {rec.get('priority', 'Unknown')}")
                    logger.info(f"      Timeline: {rec.get('implementation_timeline', 'Not specified')}")
                    logger.info(f"      Impact: {rec.get('estimated_impact', 'N/A')}/10")
                    logger.info(f"      Description: {rec.get('description', 'No description')[:100]}...")
                
                if len(recommendations) > 2:
                    logger.info(f"   ... and {len(recommendations) - 2} more recommendations")
            
            # Perform additional standalone risk assessment for comparison
            logger.info("\n🔬 PERFORMING STANDALONE RISK ASSESSMENT FOR COMPARISON...")
            standalone_results = self._perform_standalone_risk_assessment(
                processing_results.get('facility_id', 'DEMO_FACILITY_001'),
                document_state
            )
            
            results = {
                "workflow_risk_results": {
                    "risk_level": risk_level,
                    "risk_score": risk_score,
                    "risk_factors_count": len(risk_factors),
                    "recommendations_count": len(recommendations),
                    "risk_factors": risk_factors,
                    "recommendations": recommendations,
                    "processing_time": document_state.get('risk_processing_time', 0)
                },
                "standalone_risk_results": standalone_results,
                "comparison_analysis": self._compare_risk_assessments(
                    document_state, standalone_results
                )
            }
            
            logger.info("✅ Risk assessment analysis completed!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Risk assessment analysis failed: {str(e)}")
            return {"error": str(e)}
    
    def _demonstrate_langsmith_traces(self, processing_results: Dict[str, Any]) -> Dict[str, Any]:
        """Demonstrate LangSmith trace analysis and monitoring."""
        logger.info("\n🔍 STEP 3: LANGSMITH TRACE ANALYSIS")
        logger.info("-" * 60)
        
        try:
            if not langsmith_config.is_available:
                logger.warning("⚠️  LangSmith is not available - skipping trace analysis")
                return {"error": "LangSmith not available", "skipped": True}
            
            logger.info("📊 LangSmith Configuration:")
            logger.info(f"   🔗 Endpoint: {langsmith_config.endpoint}")
            logger.info(f"   📁 Project: {langsmith_config.project_name}")
            logger.info(f"   🔄 Tracing Enabled: {langsmith_config.tracing_enabled}")
            
            # Extract trace information from processing results
            document_state = processing_results.get('final_state', {})
            trace_info = {
                "document_processing_traces": {
                    "document_id": document_state.get('document_id'),
                    "processing_time": document_state.get('processing_time'),
                    "langsmith_session": document_state.get('phase1_processing', {}).get('langsmith_session')
                },
                "risk_assessment_traces": {
                    "risk_assessment_id": document_state.get('risk_assessment_id'),
                    "risk_processing_time": document_state.get('risk_processing_time'),
                    "risk_errors": document_state.get('risk_errors', [])
                }
            }
            
            logger.info("📈 Trace Information:")
            logger.info(f"   📄 Document ID: {trace_info['document_processing_traces']['document_id']}")
            logger.info(f"   🎯 Risk Assessment ID: {trace_info['risk_assessment_traces']['risk_assessment_id']}")
            logger.info(f"   ⏱️  Document Processing Time: {trace_info['document_processing_traces']['processing_time']:.2f}s")
            logger.info(f"   ⏱️  Risk Processing Time: {trace_info['risk_assessment_traces']['risk_processing_time']:.2f}s")
            
            # Provide guidance for viewing traces
            logger.info("\n🌐 LANGSMITH TRACE VIEWING INSTRUCTIONS:")
            logger.info("   1. Open your browser and navigate to: https://smith.langchain.com")
            logger.info(f"   2. Select project: {langsmith_config.project_name}")
            logger.info(f"   3. Look for traces with document ID: {document_state.get('document_id')}")
            logger.info(f"   4. Look for traces with risk assessment ID: {document_state.get('risk_assessment_id')}")
            logger.info("   5. Examine the complete execution flow and performance metrics")
            
            # Generate trace analysis summary
            trace_summary = self._analyze_trace_performance(trace_info)
            
            logger.info("✅ LangSmith trace analysis completed!")
            return {
                "trace_info": trace_info,
                "trace_summary": trace_summary,
                "viewing_instructions": "Check LangSmith dashboard for detailed traces"
            }
            
        except Exception as e:
            logger.error(f"❌ LangSmith trace analysis failed: {str(e)}")
            return {"error": str(e)}
    
    def _demonstrate_neo4j_queries(self, processing_results: Dict[str, Any]) -> Dict[str, Any]:
        """Demonstrate Neo4j data querying and analysis."""
        logger.info("\n💾 STEP 4: NEO4J DATA QUERYING AND ANALYSIS")
        logger.info("-" * 60)
        
        try:
            document_state = processing_results.get('final_state', {})
            document_id = document_state.get('document_id')
            facility_id = processing_results.get('facility_id', 'DEMO_FACILITY_001')
            risk_assessment_id = document_state.get('risk_assessment_id')
            
            logger.info("🔍 Querying Neo4j Database:")
            logger.info(f"   📄 Document ID: {document_id}")
            logger.info(f"   🏢 Facility ID: {facility_id}")
            logger.info(f"   🎯 Risk Assessment ID: {risk_assessment_id}")
            
            results = {}
            
            # Query 1: Document information
            logger.info("\n1️⃣  Querying Document Information...")
            doc_query = """
            MATCH (d:Document {id: $document_id})
            OPTIONAL MATCH (d)-[r]->(n)
            RETURN d, collect(DISTINCT {relationship: type(r), target: labels(n)[0], properties: n}) as relationships
            """
            
            with self.neo4j_connection.session() as session:
                doc_result = session.run(doc_query, document_id=document_id)
                doc_record = doc_result.single()
                
                if doc_record:
                    results['document_info'] = {
                        'properties': dict(doc_record['d']),
                        'relationships': doc_record['relationships'][:5]  # Limit to first 5
                    }
                    logger.info(f"   ✅ Found document with {len(doc_record['relationships'])} relationships")
                else:
                    logger.warning("   ⚠️  Document not found in database")
                    results['document_info'] = None
            
            # Query 2: Risk Assessment data
            if risk_assessment_id:
                logger.info("\n2️⃣  Querying Risk Assessment Data...")
                risk_query = """
                MATCH (ra:RiskAssessment {id: $risk_assessment_id})
                OPTIONAL MATCH (ra)-[:IDENTIFIES]->(rf:RiskFactor)
                OPTIONAL MATCH (ra)-[:RECOMMENDS]->(rec:RiskRecommendation)
                RETURN ra, 
                       collect(DISTINCT rf) as risk_factors,
                       collect(DISTINCT rec) as recommendations
                """
                
                with self.neo4j_connection.session() as session:
                    risk_result = session.run(risk_query, risk_assessment_id=risk_assessment_id)
                    risk_record = risk_result.single()
                    
                    if risk_record:
                        results['risk_assessment_data'] = {
                            'assessment_properties': dict(risk_record['ra']),
                            'risk_factors': [dict(rf) for rf in risk_record['risk_factors']],
                            'recommendations': [dict(rec) for rec in risk_record['recommendations']]
                        }
                        logger.info(f"   ✅ Found risk assessment with {len(risk_record['risk_factors'])} risk factors")
                        logger.info(f"      and {len(risk_record['recommendations'])} recommendations")
                    else:
                        logger.warning("   ⚠️  Risk assessment not found in database")
                        results['risk_assessment_data'] = None
            
            # Query 3: Facility context and related data
            logger.info("\n3️⃣  Querying Facility Context...")
            facility_query = """
            MATCH (f:Facility)
            WHERE f.id = $facility_id OR f.name CONTAINS $facility_search
            OPTIONAL MATCH (f)-[:HAS_DOCUMENT]->(d:Document)
            OPTIONAL MATCH (d)-[:HAS_RISK_ASSESSMENT]->(ra:RiskAssessment)
            RETURN f, 
                   count(DISTINCT d) as document_count,
                   count(DISTINCT ra) as risk_assessment_count,
                   collect(DISTINCT d.id)[..5] as recent_documents
            """
            
            with self.neo4j_connection.session() as session:
                facility_result = session.run(
                    facility_query, 
                    facility_id=facility_id,
                    facility_search="Demo"
                )
                facility_record = facility_result.single()
                
                if facility_record:
                    results['facility_context'] = {
                        'facility_properties': dict(facility_record['f']),
                        'document_count': facility_record['document_count'],
                        'risk_assessment_count': facility_record['risk_assessment_count'],
                        'recent_documents': facility_record['recent_documents']
                    }
                    logger.info(f"   ✅ Found facility with {facility_record['document_count']} documents")
                    logger.info(f"      and {facility_record['risk_assessment_count']} risk assessments")
                else:
                    logger.warning("   ⚠️  Facility not found in database")
                    results['facility_context'] = None
            
            # Query 4: Database statistics
            logger.info("\n4️⃣  Gathering Database Statistics...")
            stats_query = """
            CALL db.labels() YIELD label
            WITH label
            CALL apoc.cypher.run('MATCH (n:`' + label + '`) RETURN count(n) as count', {}) YIELD value
            RETURN label, value.count as count
            ORDER BY value.count DESC
            """
            
            try:
                with self.neo4j_connection.session() as session:
                    stats_result = session.run(stats_query)
                    stats_records = list(stats_result)
                    
                    results['database_stats'] = [
                        {'label': record['label'], 'count': record['count']} 
                        for record in stats_records
                    ]
                    
                    logger.info("   📊 Database Node Statistics:")
                    for stat in results['database_stats'][:10]:  # Show top 10
                        logger.info(f"      {stat['label']}: {stat['count']} nodes")
                        
            except Exception as e:
                logger.warning(f"   ⚠️  Could not gather database statistics: {str(e)}")
                results['database_stats'] = []
            
            logger.info("✅ Neo4j querying completed!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Neo4j querying failed: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze overall performance metrics of the demonstration."""
        logger.info("\n📈 STEP 5: PERFORMANCE ANALYSIS")
        logger.info("-" * 60)
        
        try:
            total_duration = datetime.utcnow() - self.start_time
            
            # Gather performance metrics from different components
            doc_processing = self.demo_results.get('document_processing', {})
            risk_assessment = self.demo_results.get('risk_assessment', {})
            
            performance_metrics = {
                "total_execution_time": total_duration.total_seconds(),
                "document_processing_time": doc_processing.get('processing_time', 0),
                "risk_assessment_time": risk_assessment.get('workflow_risk_results', {}).get('processing_time', 0),
                "components_analyzed": len([k for k in self.demo_results.keys() if not k.endswith('_error')]),
                "errors_encountered": len([k for k in self.demo_results.keys() if k.endswith('_error')]),
                "success_rate": self._calculate_success_rate()
            }
            
            logger.info("⚡ Performance Summary:")
            logger.info(f"   ⏱️  Total Execution Time: {performance_metrics['total_execution_time']:.2f} seconds")
            logger.info(f"   📄 Document Processing Time: {performance_metrics['document_processing_time']:.2f} seconds")
            logger.info(f"   🎯 Risk Assessment Time: {performance_metrics['risk_assessment_time']:.2f} seconds")
            logger.info(f"   🔧 Components Analyzed: {performance_metrics['components_analyzed']}")
            logger.info(f"   ❌ Errors Encountered: {performance_metrics['errors_encountered']}")
            logger.info(f"   ✅ Success Rate: {performance_metrics['success_rate']:.1f}%")
            
            # Performance recommendations
            recommendations = []
            if performance_metrics['document_processing_time'] > 30:
                recommendations.append("Consider optimizing document parsing for large files")
            if performance_metrics['risk_assessment_time'] > 20:
                recommendations.append("Risk assessment could benefit from caching frequently accessed data")
            if performance_metrics['errors_encountered'] > 0:
                recommendations.append("Review error logs for potential improvements")
                
            performance_metrics['recommendations'] = recommendations
            
            if recommendations:
                logger.info("💡 Performance Recommendations:")
                for rec in recommendations:
                    logger.info(f"   • {rec}")
            
            logger.info("✅ Performance analysis completed!")
            return performance_metrics
            
        except Exception as e:
            logger.error(f"❌ Performance analysis failed: {str(e)}")
            return {"error": str(e)}
    
    def _generate_summary_report(self) -> Dict[str, Any]:
        """Generate a comprehensive summary report of the demonstration."""
        logger.info("\n📋 STEP 6: GENERATING SUMMARY REPORT")
        logger.info("-" * 60)
        
        try:
            summary = {
                "demonstration_overview": {
                    "start_time": self.start_time.isoformat(),
                    "end_time": datetime.utcnow().isoformat(),
                    "total_duration": (datetime.utcnow() - self.start_time).total_seconds(),
                    "components_tested": list(self.demo_results.keys())
                },
                "document_processing_summary": self._summarize_document_processing(),
                "risk_assessment_summary": self._summarize_risk_assessment(),
                "data_integration_summary": self._summarize_data_integration(),
                "overall_assessment": self._generate_overall_assessment()
            }
            
            logger.info("📊 DEMONSTRATION SUMMARY REPORT")
            logger.info("=" * 60)
            logger.info(f"🕒 Duration: {summary['demonstration_overview']['total_duration']:.2f} seconds")
            logger.info(f"🔧 Components: {len(summary['demonstration_overview']['components_tested'])}")
            
            # Document Processing Summary
            doc_summary = summary['document_processing_summary']
            logger.info(f"\n📄 Document Processing:")
            logger.info(f"   Status: {doc_summary.get('status', 'Unknown')}")
            logger.info(f"   Processing Time: {doc_summary.get('processing_time', 0):.2f}s")
            
            # Risk Assessment Summary  
            risk_summary = summary['risk_assessment_summary']
            logger.info(f"\n🎯 Risk Assessment:")
            logger.info(f"   Risk Level: {risk_summary.get('risk_level', 'Not assessed')}")
            logger.info(f"   Risk Factors: {risk_summary.get('risk_factors_count', 0)}")
            logger.info(f"   Recommendations: {risk_summary.get('recommendations_count', 0)}")
            
            # Overall Assessment
            overall = summary['overall_assessment']
            logger.info(f"\n✅ Overall Assessment:")
            logger.info(f"   Success Rate: {overall.get('success_rate', 0):.1f}%")
            logger.info(f"   Key Achievements: {len(overall.get('achievements', []))}")
            logger.info(f"   Improvement Areas: {len(overall.get('improvements', []))}")
            
            logger.info("✅ Summary report generated!")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Summary report generation failed: {str(e)}")
            return {"error": str(e)}
    
    # Helper methods
    def _get_demo_document_path(self) -> str:
        """Get the path to a demonstration document."""
        if self.config.get('document_path'):
            return self.config['document_path']
        
        # Use a sample document from the test directory
        sample_docs = [
            '/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/test/test_documents/electricity_bills/electric_bill.pdf',
            '/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/electric_bill.pdf',
            '/Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend/test/test_documents/water_bills/water_bill.pdf'
        ]
        
        for doc_path in sample_docs:
            if os.path.exists(doc_path):
                return doc_path
        
        raise FileNotFoundError("No sample documents found for demonstration")
    
    def _detect_document_type(self, document_path: str) -> str:
        """Detect document type from filename."""
        filename = os.path.basename(document_path).lower()
        if 'electric' in filename or 'electricity' in filename:
            return 'utility_bill'
        elif 'water' in filename:
            return 'water_bill'
        elif 'waste' in filename:
            return 'waste_manifest'
        else:
            return 'unknown'
    
    def _analyze_processing_results(self, final_state: DocumentStateWithRisk, processing_time: float) -> Dict[str, Any]:
        """Analyze document processing results."""
        return {
            "final_state": final_state,
            "processing_time": processing_time,
            "status": final_state.get('status'),
            "document_id": final_state.get('document_id'),
            "facility_id": final_state.get('upload_metadata', {}).get('facility_id'),
            "errors": final_state.get('errors', []),
            "risk_assessment_performed": bool(final_state.get('risk_assessment_id')),
            "success": final_state.get('status') == 'completed' and not final_state.get('errors')
        }
    
    def _perform_standalone_risk_assessment(self, facility_id: str, document_state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform standalone risk assessment for comparison."""
        try:
            assessment_scope = {
                "document_triggered": True,
                "document_id": document_state.get('document_id'),
                "assessment_depth": "comprehensive"
            }
            
            result = self.risk_agent.assess_facility_risk(
                facility_id=facility_id,
                assessment_scope=assessment_scope,
                metadata={"comparison_assessment": True}
            )
            
            return {
                "status": result.get('status'),
                "assessment_id": result.get('assessment_id'),
                "processing_time": result.get('processing_time', 0),
                "errors": result.get('errors', []),
                "risk_assessment": result.get('risk_assessment'),
                "recommendations": result.get('recommendations')
            }
            
        except Exception as e:
            logger.error(f"Standalone risk assessment failed: {str(e)}")
            return {"error": str(e)}
    
    def _compare_risk_assessments(self, workflow_state: Dict[str, Any], standalone_result: Dict[str, Any]) -> Dict[str, Any]:
        """Compare workflow vs standalone risk assessments."""
        return {
            "workflow_processing_time": workflow_state.get('risk_processing_time', 0),
            "standalone_processing_time": standalone_result.get('processing_time', 0),
            "workflow_risk_level": workflow_state.get('risk_level'),
            "standalone_risk_level": standalone_result.get('risk_assessment', {}).get('overall_risk_level'),
            "both_successful": (
                workflow_state.get('risk_assessment_status') == 'completed' and 
                standalone_result.get('status') == 'completed'
            )
        }
    
    def _analyze_trace_performance(self, trace_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trace performance metrics."""
        doc_time = trace_info.get('document_processing_traces', {}).get('processing_time', 0)
        risk_time = trace_info.get('risk_assessment_traces', {}).get('risk_processing_time', 0)
        
        return {
            "total_traced_time": doc_time + risk_time,
            "document_percentage": (doc_time / (doc_time + risk_time)) * 100 if (doc_time + risk_time) > 0 else 0,
            "risk_percentage": (risk_time / (doc_time + risk_time)) * 100 if (doc_time + risk_time) > 0 else 0,
            "trace_quality": "high" if not trace_info.get('risk_assessment_traces', {}).get('risk_errors') else "degraded"
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate of the demonstration."""
        total_components = len(self.demo_results)
        successful_components = len([
            k for k, v in self.demo_results.items() 
            if not (isinstance(v, dict) and v.get('error'))
        ])
        return (successful_components / total_components) * 100 if total_components > 0 else 0
    
    def _summarize_document_processing(self) -> Dict[str, Any]:
        """Summarize document processing results."""
        doc_results = self.demo_results.get('document_processing', {})
        return {
            "status": "success" if doc_results.get('success') else "failed",
            "processing_time": doc_results.get('processing_time', 0),
            "document_id": doc_results.get('document_id'),
            "errors_count": len(doc_results.get('final_state', {}).get('errors', []))
        }
    
    def _summarize_risk_assessment(self) -> Dict[str, Any]:
        """Summarize risk assessment results."""
        risk_results = self.demo_results.get('risk_assessment', {})
        workflow_results = risk_results.get('workflow_risk_results', {})
        return {
            "risk_level": workflow_results.get('risk_level'),
            "risk_score": workflow_results.get('risk_score'),
            "risk_factors_count": workflow_results.get('risk_factors_count', 0),
            "recommendations_count": workflow_results.get('recommendations_count', 0),
            "processing_time": workflow_results.get('processing_time', 0)
        }
    
    def _summarize_data_integration(self) -> Dict[str, Any]:
        """Summarize data integration and storage results."""
        neo4j_results = self.demo_results.get('neo4j_queries', {})
        return {
            "document_stored": bool(neo4j_results.get('document_info')),
            "risk_data_stored": bool(neo4j_results.get('risk_assessment_data')),
            "facility_linked": bool(neo4j_results.get('facility_context')),
            "database_accessible": not bool(neo4j_results.get('error'))
        }
    
    def _generate_overall_assessment(self) -> Dict[str, Any]:
        """Generate overall assessment of the demonstration."""
        achievements = []
        improvements = []
        
        # Check achievements
        if self.demo_results.get('document_processing', {}).get('success'):
            achievements.append("Successfully processed document through complete workflow")
        if self.demo_results.get('risk_assessment', {}).get('workflow_risk_results', {}).get('risk_level'):
            achievements.append("Generated comprehensive risk assessment")
        if self.demo_results.get('neo4j_queries', {}).get('document_info'):
            achievements.append("Successfully stored and retrieved data from Neo4j")
        if self.demo_results.get('langsmith_traces', {}).get('trace_info'):
            achievements.append("Enabled comprehensive tracing and monitoring")
        
        # Check for improvements
        if any('error' in str(v) for v in self.demo_results.values()):
            improvements.append("Address identified errors for more robust execution")
        performance = self.demo_results.get('performance_analysis', {})
        if performance.get('document_processing_time', 0) > 30:
            improvements.append("Optimize document processing performance")
        if performance.get('risk_assessment_time', 0) > 20:
            improvements.append("Enhance risk assessment execution speed")
            
        return {
            "success_rate": self._calculate_success_rate(),
            "achievements": achievements,
            "improvements": improvements,
            "overall_status": "successful" if self._calculate_success_rate() > 75 else "needs_improvement"
        }
    
    def close(self):
        """Clean up resources."""
        try:
            if self.workflow:
                self.workflow.close()
            if self.risk_agent:
                self.risk_agent.close()
            if self.neo4j_connection:
                self.neo4j_connection.close()
            logger.info("🧹 Demo resources cleaned up successfully")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {str(e)}")


def load_configuration() -> Dict[str, Any]:
    """Load configuration from environment variables and .env file."""
    from dotenv import load_dotenv
    
    # Load .env file
    env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    load_dotenv(env_path)
    
    config = {
        'llama_parse_api_key': os.getenv('LLAMA_PARSE_API_KEY'),
        'neo4j_uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        'neo4j_username': os.getenv('NEO4J_USERNAME', 'neo4j'),
        'neo4j_password': os.getenv('NEO4J_PASSWORD'),
        'neo4j_database': os.getenv('NEO4J_DATABASE', 'neo4j'),
        'llm_model': 'gpt-4o'  # Default model
    }
    
    # Validate required configuration
    required_keys = ['llama_parse_api_key', 'neo4j_password']
    for key in required_keys:
        if not config[key]:
            raise ValueError(f"Required configuration missing: {key}")
    
    return config


def main():
    """Main demonstration function with command-line argument support."""
    parser = argparse.ArgumentParser(
        description='EHS AI Platform - Risk Assessment Workflow Demonstration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 demo_risk_assessment_workflow.py
  python3 demo_risk_assessment_workflow.py --sample-document --enable-traces
  python3 demo_risk_assessment_workflow.py --document-path /path/to/document.pdf --facility-id FAC_001
  python3 demo_risk_assessment_workflow.py --no-neo4j-queries --no-traces
        """
    )
    
    parser.add_argument('--document-path', type=str, help='Path to specific document to process')
    parser.add_argument('--facility-id', type=str, default='DEMO_FACILITY_001', help='Facility ID for demonstration')
    parser.add_argument('--sample-document', action='store_true', help='Use a sample document for demonstration')
    parser.add_argument('--enable-traces', action='store_true', default=True, help='Enable LangSmith trace analysis')
    parser.add_argument('--no-traces', dest='enable_traces', action='store_false', help='Disable LangSmith traces')
    parser.add_argument('--query-neo4j', action='store_true', default=True, help='Query Neo4j database')
    parser.add_argument('--no-neo4j-queries', dest='query_neo4j', action='store_false', help='Skip Neo4j queries')
    parser.add_argument('--llm-model', type=str, default='gpt-4o', help='LLM model to use')
    parser.add_argument('--output-file', type=str, help='Save results to JSON file')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_configuration()
        
        # Apply command line overrides
        config.update({
            'document_path': args.document_path,
            'facility_id': args.facility_id,
            'show_traces': args.enable_traces,
            'query_neo4j': args.query_neo4j,
            'llm_model': args.llm_model
        })
        
        # Create and run demonstration
        demo = RiskAssessmentDemo(config)
        results = demo.run_complete_demonstration()
        
        # Save results if requested
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"💾 Results saved to: {args.output_file}")
        
        # Clean up
        demo.close()
        
        # Exit with appropriate code
        success_rate = results.get('performance_analysis', {}).get('success_rate', 0)
        if success_rate >= 75:
            logger.info("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
            sys.exit(0)
        else:
            logger.warning("⚠️  DEMONSTRATION COMPLETED WITH ISSUES")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Demonstration interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()