# EHS AI Platform - Risk Assessment Workflow Demonstration

## Overview

This comprehensive demonstration showcases the complete risk assessment workflow integration in the EHS AI Platform. The demo includes:

- **Document Processing**: Complete document ingestion through enhanced workflow
- **Risk Assessment**: AI-powered risk analysis with detailed factors and recommendations  
- **LangSmith Integration**: Comprehensive tracing and monitoring of all operations
- **Neo4j Storage**: Data persistence and relationship mapping
- **Performance Analysis**: Detailed metrics and optimization suggestions
- **Error Handling**: Robust error management with clear reporting

## Quick Start

### Simple Demo Run
```bash
# Run the complete demonstration with all features
python3 run_risk_demo.py

# Quick demo without detailed queries (faster)
python3 run_risk_demo.py --quick

# Save results to file for analysis
python3 run_risk_demo.py --save-results
```

### Advanced Usage
```bash
# Use specific document
python3 src/workflows/demo_risk_assessment_workflow.py --document-path /path/to/document.pdf

# Specify facility ID
python3 src/workflows/demo_risk_assessment_workflow.py --facility-id FACILITY_ABC_001

# Disable specific components
python3 src/workflows/demo_risk_assessment_workflow.py --no-neo4j-queries --no-traces

# Use different LLM model
python3 src/workflows/demo_risk_assessment_workflow.py --llm-model gpt-4o-mini
```

## What the Demo Shows

### 1. Document Processing Workflow 📄
- **File Validation**: Checks document format and accessibility
- **Document Parsing**: Extracts text and structure using LlamaParse
- **Data Extraction**: Identifies EHS-relevant information
- **Quality Validation**: Ensures data completeness and accuracy
- **Duplicate Detection**: Prevents duplicate document processing
- **Phase 1 Enhancements**: Audit trail, rejection handling, pro-rating

### 2. Risk Assessment Engine 🎯
- **Data Collection**: Gathers environmental, health, safety, and compliance data
- **Risk Analysis**: AI-powered identification of risk factors
- **Risk Scoring**: Quantitative assessment with severity and probability
- **Risk Categorization**: Classification by type (environmental, health, safety, compliance)
- **Recommendation Generation**: Actionable mitigation strategies
- **Confidence Assessment**: Quality metrics for assessment reliability

### 3. LangSmith Tracing 🔍
- **Complete Execution Traces**: Every step of the workflow is traced
- **Performance Monitoring**: Detailed timing and resource usage
- **Error Tracking**: Comprehensive error capture and analysis
- **Custom Tagging**: Organized traces by document and assessment ID
- **Interactive Dashboard**: Real-time monitoring through LangSmith UI

### 4. Neo4j Data Integration 💾
- **Document Storage**: Complete document metadata and relationships
- **Risk Data Persistence**: Risk factors, assessments, and recommendations
- **Facility Relationships**: Links between facilities, documents, and risks
- **Queryable Structure**: Rich graph database for complex queries
- **Data Validation**: Ensures consistency and referential integrity

### 5. Performance Analysis 📈
- **Execution Timing**: Detailed breakdown of processing times
- **Resource Usage**: Memory and CPU utilization tracking
- **Success Rates**: Component-level success/failure analysis
- **Optimization Recommendations**: Specific suggestions for improvement
- **Comparative Analysis**: Workflow vs standalone assessment comparison

## Sample Output

When you run the demonstration, you'll see output like this:

```
================================================================================
EHS AI PLATFORM - RISK ASSESSMENT WORKFLOW DEMONSTRATION
================================================================================
Demonstration started at: 2025-08-28 16:30:15.123456

🔧 INITIALIZING COMPONENTS
--------------------------------------------------
1️⃣  Initializing Risk Assessment Integrated Workflow...
✅ Risk Assessment Integrated Workflow initialized successfully
2️⃣  Initializing standalone Risk Assessment Agent...
✅ Standalone Risk Assessment Agent initialized successfully
3️⃣  Establishing Neo4j database connection...
✅ Neo4j connection established successfully
4️⃣  LangSmith tracing is enabled and available
   🔍 Project: ehs-ai-demo-ingestion
✅ LangSmith integration ready

🎉 All components initialized successfully!

📄 STEP 1: DOCUMENT PROCESSING DEMONSTRATION
------------------------------------------------------------
📋 Processing Document:
   📂 Path: /path/to/electric_bill.pdf
   🆔 Document ID: demo_doc_1724864215
   📊 Document Type: utility_bill
   📋 Metadata: {
     "facility_id": "DEMO_FACILITY_001",
     "facility_name": "Demo Manufacturing Facility",
     "document_category": "environmental",
     ...
   }

🔄 Starting Document Processing Workflow...
   This will include: Validation → Parsing → Extraction → Risk Assessment

✅ Document processing completed!
   ⏱️  Total processing time: 45.23 seconds
   📊 Final status: completed
   🎯 Risk assessment status: completed

🎯 STEP 2: RISK ASSESSMENT ANALYSIS
------------------------------------------------------------
📊 Risk Assessment Results:
   🚨 Risk Level: medium
   📈 Risk Score: 65.7
   ⚠️  Risk Factors Identified: 8
   💡 Recommendations Generated: 5

🔍 DETAILED RISK FACTORS:
   1. High Energy Consumption Pattern
      Category: environmental
      Severity: 7.2/10
      Probability: 0.85
      Description: Facility shows consistently high electricity usage compared to industry benchmarks...

   2. Potential Equipment Inefficiency
      Category: operational
      Severity: 6.1/10
      Probability: 0.72
      Description: Energy usage patterns suggest possible equipment degradation or inefficient operation...

💡 TOP RECOMMENDATIONS:
   1. Implement Energy Efficiency Audit
      Priority: high
      Timeline: 30-60 days
      Impact: 8.5/10
      Description: Conduct comprehensive energy audit to identify specific efficiency improvements...

   2. Equipment Maintenance Review
      Priority: medium
      Timeline: 60-90 days
      Impact: 7.2/10
      Description: Review and optimize maintenance schedules for energy-consuming equipment...

🔬 PERFORMING STANDALONE RISK ASSESSMENT FOR COMPARISON...
✅ Risk assessment analysis completed!

🔍 STEP 3: LANGSMITH TRACE ANALYSIS
------------------------------------------------------------
📊 LangSmith Configuration:
   🔗 Endpoint: https://api.smith.langchain.com
   📁 Project: ehs-ai-demo-ingestion
   🔄 Tracing Enabled: true

📈 Trace Information:
   📄 Document ID: demo_doc_1724864215
   🎯 Risk Assessment ID: risk_demo_doc_1724864215_20250828_163015
   ⏱️  Document Processing Time: 45.23s
   ⏱️  Risk Processing Time: 23.67s

🌐 LANGSMITH TRACE VIEWING INSTRUCTIONS:
   1. Open your browser and navigate to: https://smith.langchain.com
   2. Select project: ehs-ai-demo-ingestion
   3. Look for traces with document ID: demo_doc_1724864215
   4. Look for traces with risk assessment ID: risk_demo_doc_1724864215_20250828_163015
   5. Examine the complete execution flow and performance metrics

✅ LangSmith trace analysis completed!

💾 STEP 4: NEO4J DATA QUERYING AND ANALYSIS
------------------------------------------------------------
🔍 Querying Neo4j Database:
   📄 Document ID: demo_doc_1724864215
   🏢 Facility ID: DEMO_FACILITY_001
   🎯 Risk Assessment ID: risk_demo_doc_1724864215_20250828_163015

1️⃣  Querying Document Information...
   ✅ Found document with 12 relationships

2️⃣  Querying Risk Assessment Data...
   ✅ Found risk assessment with 8 risk factors
      and 5 recommendations

3️⃣  Querying Facility Context...
   ✅ Found facility with 15 documents
      and 3 risk assessments

4️⃣  Gathering Database Statistics...
   📊 Database Node Statistics:
      Document: 127 nodes
      RiskAssessment: 23 nodes
      RiskFactor: 89 nodes
      RiskRecommendation: 67 nodes
      Facility: 12 nodes
      ...

✅ Neo4j querying completed!

📈 STEP 5: PERFORMANCE ANALYSIS
------------------------------------------------------------
⚡ Performance Summary:
   ⏱️  Total Execution Time: 89.45 seconds
   📄 Document Processing Time: 45.23 seconds
   🎯 Risk Assessment Time: 23.67 seconds
   🔧 Components Analyzed: 6
   ❌ Errors Encountered: 0
   ✅ Success Rate: 100.0%

✅ Performance analysis completed!

📋 STEP 6: GENERATING SUMMARY REPORT
------------------------------------------------------------
📊 DEMONSTRATION SUMMARY REPORT
============================================================
🕒 Duration: 89.45 seconds
🔧 Components: 6

📄 Document Processing:
   Status: success
   Processing Time: 45.23s

🎯 Risk Assessment:
   Risk Level: medium
   Risk Factors: 8
   Recommendations: 5

✅ Overall Assessment:
   Success Rate: 100.0%
   Key Achievements: 4
   Improvement Areas: 0

✅ Summary report generated!

🎊 DEMONSTRATION COMPLETED SUCCESSFULLY!
Total execution time: 0:01:29.450000

🧹 Demo resources cleaned up successfully

🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!
```

## System Requirements

### Prerequisites
- **Python 3.9+** with virtual environment
- **Neo4j Database** running on localhost:7687 (or configured endpoint)
- **API Keys**: LlamaParse API key, OpenAI API key, LangSmith API key
- **Dependencies**: All required packages installed via pip

### Environment Setup
1. **Activate virtual environment**:
   ```bash
   cd /Users/eugene/dev/ai/agentos/ehs-ai-demo/data-foundation/backend
   source venv/bin/activate  # or python3 -m venv venv && source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   - Ensure `.env` file has all required API keys
   - Verify Neo4j connection parameters
   - Check LangSmith project configuration

4. **Verify Neo4j is running**:
   ```bash
   # Check if Neo4j is accessible
   curl -u neo4j:password http://localhost:7474/
   ```

## Configuration Options

### Environment Variables (.env file)
```bash
# Required API Keys
OPENAI_API_KEY=your_openai_api_key
LLAMA_PARSE_API_KEY=your_llama_parse_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
NEO4J_DATABASE=neo4j

# LangSmith Configuration
LANGCHAIN_PROJECT=ehs-ai-demo-ingestion
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com

# Model Configuration
LLM_MODEL_CONFIG_openai_gpt_4o="gpt-4o-2024-11-20,openai_api_key"
```

### Command Line Options

#### Main Demo Script Options
```bash
--document-path PATH         # Specific document to process
--facility-id ID            # Facility identifier for demo
--sample-document           # Use built-in sample document
--enable-traces             # Enable LangSmith tracing (default)
--no-traces                 # Disable LangSmith tracing
--query-neo4j               # Query Neo4j database (default)
--no-neo4j-queries          # Skip Neo4j queries
--llm-model MODEL           # LLM model to use (default: gpt-4o)
--output-file FILE          # Save results to JSON file
```

#### Quick Runner Options
```bash
--quick                     # Skip detailed queries for faster demo
--sample-doc               # Use specific sample document
--save-results             # Save timestamped results
--facility-id ID           # Custom facility ID
--verbose                  # Enable verbose command output
```

## Troubleshooting

### Common Issues

1. **Missing API Keys**
   ```
   Error: Required configuration missing: llama_parse_api_key
   ```
   **Solution**: Add missing API keys to `.env` file

2. **Neo4j Connection Failed**
   ```
   Error: Failed to connect to Neo4j: ServiceUnavailable
   ```
   **Solutions**:
   - Ensure Neo4j is running: `sudo systemctl start neo4j`
   - Check connection parameters in `.env`
   - Verify firewall settings

3. **Sample Document Not Found**
   ```
   Error: No sample documents found for demonstration
   ```
   **Solutions**:
   - Use `--document-path` to specify a document
   - Ensure test documents exist in `test/test_documents/`
   - Check file permissions

4. **LangSmith Tracing Disabled**
   ```
   Warning: LangSmith tracing is not available
   ```
   **Solutions**:
   - Add `LANGCHAIN_API_KEY` to `.env`
   - Verify `LANGCHAIN_TRACING_V2=true`
   - Check internet connectivity

5. **Out of Memory Errors**
   ```
   Error: CUDA out of memory / System memory exceeded
   ```
   **Solutions**:
   - Use smaller model: `--llm-model gpt-4o-mini`
   - Process smaller documents
   - Increase system memory/swap

### Performance Issues

1. **Slow Document Processing**
   - Use smaller documents for testing
   - Check network connectivity to API services
   - Consider using local models if available

2. **Slow Risk Assessment**
   - Reduce assessment scope in metadata
   - Use faster LLM models for testing
   - Check Neo4j query performance

3. **High Memory Usage**
   - Close other applications
   - Use streaming processing where available
   - Monitor system resources

### Debugging Tips

1. **Enable Verbose Logging**
   ```python
   import logging
   logging.getLogger().setLevel(logging.DEBUG)
   ```

2. **Check Log Files**
   ```bash
   tail -f /tmp/demo_risk_assessment.log
   ```

3. **Verify Component Status**
   - Test Neo4j connection separately
   - Verify API keys with simple calls
   - Check document accessibility

## Integration Examples

### Using in Production

```python
from src.workflows.ingestion_workflow_with_risk_assessment import (
    create_risk_integrated_workflow
)

# Initialize workflow
workflow = create_risk_integrated_workflow(
    llama_parse_api_key="your_key",
    neo4j_uri="bolt://your_neo4j_host:7687",
    neo4j_username="neo4j", 
    neo4j_password="your_password",
    enable_risk_assessment=True
)

# Process document
result = workflow.process_document(
    file_path="/path/to/document.pdf",
    document_id="unique_id",
    metadata={"facility_id": "FAC_001"}
)

# Check risk assessment results
if result.get('risk_assessment_status') == 'completed':
    risk_level = result.get('risk_level')
    recommendations = result.get('risk_recommendations', [])
    print(f"Risk Level: {risk_level}")
    print(f"Recommendations: {len(recommendations)}")
```

### Custom Risk Assessment

```python
from src.agents.risk_assessment.agent import create_risk_assessment_agent

# Create standalone risk agent
agent = create_risk_assessment_agent()

# Assess specific facility
assessment = agent.assess_facility_risk(
    facility_id="FACILITY_001",
    assessment_scope={
        "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
        "categories": ["environmental", "safety"],
        "depth": "comprehensive"
    }
)

# Process results
if assessment.get('status') == 'completed':
    risk_assessment = assessment.get('risk_assessment')
    print(f"Overall Risk: {risk_assessment.overall_risk_level}")
    print(f"Risk Score: {risk_assessment.risk_score}")
```

## Support and Documentation

### Additional Resources
- **LangSmith Dashboard**: https://smith.langchain.com
- **Neo4j Browser**: http://localhost:7474/browser/
- **API Documentation**: Check individual service documentation
- **Troubleshooting Guide**: See project wiki

### Getting Help
1. **Check log files** for detailed error messages
2. **Review configuration** for missing or incorrect values
3. **Test components individually** to isolate issues
4. **Consult documentation** for specific components
5. **Contact development team** for persistent issues

---

*This demonstration showcases the complete EHS AI Platform risk assessment capabilities. For production deployment, additional security, scalability, and monitoring considerations should be implemented.*