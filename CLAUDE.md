# EHS AI Demo - Project Context

# RULES WHICH MUST BE FOLLOWED AT ALL TIMES AND WITHOUT EXCEPTION

- **Code Development Rules**: ou must follow these rules at all times and without exception
- **Whenever you are working on a feature you must**:
    - Split the feature into small self-contained increments
    - Fully implement one increment at at ime before proceeding onto the next one
    - Always create comprehansive tests for every increment and every feaure
    - Never use mocks for testing
    - Never create hardcoded values for testing
    - Always fully test every increment and every feature before concluding that "the feature is implemented and ready"
    - Never stop working on an increment of a feature untill its is fully tested and confirmed as 100% functional through the testing
    - Never report to the user that "the testing is finished" when in fact all you did is to create a testing framework, but have not conducted all of the actual tests with 100% success rate
- For every project you must maintain a single file in which you keep a complete inventory of all the tests related to that project
- At any point whenever you need to investigate something or to test anything - you must first look through this test inventory file to fully undertsand what existing tests are available and where these tests are located
- When you are working on fixing or improving a test you must keep the intermediate test files separately from the main project structure to avoid confision. You must create and keep these temporary files under the /tmp directory
- You must alsways keep every test file under the ../test subdirectory within the given project structure
- When working with software libraries and frameworks such as LangCHain, LangGraph, LlamaIndex, LlamaParse, neo4j-grahrag-python .. - you must first use the context7 MCP tool to get the information on the latests available version, as well as the documented examples on how to properly use these libraries
- Once you have implemented a new feature or an increment of a feature - you must run of of the tests for that feature as well as all of the tests available for all other features (i.e. conduct a regression test) to make sure that the new feature did not brake any existing features
- You must not use zen MCP tool for any tasks unless explicitly instructed by the user
- You must not perform any git commands unless explicitly instructed by the user
- you must alsways use the file-creator sub-agent for any file (Python or otherwise) creation or modification.
- you must alsways use the file-creator for any HTML, Javascript, CSS creation or modification. 
- You must always use Python3 in combination with venv to run any Python scripts
- You must use the markdown-doc-expert to read, write and edit any markdown (.md) files
- When asked to read any project documentation you must always use the markdown-doc-expert sub-agent
- You must use the git-workflow to run any git commands, but do so only when explicitly instructed bye the user
- You must only run git commands when explicitly instructed
- You must use the test-runner sub-agent to run any  tests
- Prior to running any tests you must first verify whether a given test will require an API key and set one appropriately
- Using curl for API testing counts as testing, so you must use the test-runner sub-agent to carrie it out
- You must always read .env file for a given project to obtain the API keys
- when you do any testing you must perform the testing and the background by using the '&' at the end of the test command. you must make sure that every test writes a detailed log file in which every action is captured. as you running a test you must follow that log file (i.e. tail -f) to keep following the progress.
- You must avoid creating simplifications
- You must avoid using mocks
- You must avoid any hardcoding unless explicitly instructed to do so
- If a given task requires and API key you must not use any generic placeholder value, instead you must inform the user of the need to provide you with the real API key
- You must always keep the implementation plan and implementation status documents up to date, and update the implementation status after each feature or function has been implemented
- You must not conclude that any feature or function is implemented and ready untill you abtain 100% test-based evidence
- You must create comprehensive tests for every feature and function and must run these tests any time you have made a code change
- You must keep updating the current status memory after implementing every feature or function
- You must not make any speculative conclusions regarding the SDKs or APIs. Instead you must use the Context7 MCP tool to obtain the latest information on a particular SDK, API or library
- Before creating a new functionality which involves making use of a particular SDK, API, or library you should use the exa search MCP tool to obtain the information on the real-world best practices and specific well-regarded reference implementations
- You must use zen MCP ONLY when asked to commenicate with Google Gemini and NOT for any other purpose
- You must not attemp to create any new Python venv untill you are 100% certain that a given project does not have one already



## Project Overview
This project implements an AI-powered Environmental Health and Safety (EHS) solution based on Verdantix's evaluation criteria for industry-leading ESG platforms.

## Key Verdantix-Recognized AI Capabilities

### 1. AI-Driven Sustainability Data Management and Automation
- **Goal**: Transform messy, disparate data into reliable, auditable assets
- **Key Features**:
  - Automated data ingestion from PDFs (utility bills, invoices, reports)
  - RAG for extracting structured data from unstructured documents
  - AI-powered data quality validation and anomaly detection
  - Automated Scope 3 emissions calculation and aggregation

### 2. AI-Enabled Compliance and Reporting
- **Goal**: Reduce compliance risk and administrative burden
- **Key Features**:
  - Regulatory horizon scanning and impact analysis
  - Automated compliance gap analysis against frameworks (GRI, SASB, CSRD)
  - Generative AI for disclosure drafting

### 3. Predictive Analytics for ESG Risk and Performance
- **Goal**: Shift from reactive reporting to proactive risk management
- **Key Features**:
  - Environmental incident prediction using ML models
  - Climate-related physical risk forecasting
  - Prescriptive recommendations for performance improvement

## Architecture Design

### Tech Stack
- **LangGraph**: Central orchestration for stateful, multi-agent workflows
- **LangChain**: Building specialized agents
- **LlamaIndex**: Contextual knowledge base and RAG
- **LlamaParse**: Document parsing (PDFs to structured data)
- **Neo4j-graphrag-python**: Knowledge graph construction and querying

### Three-Phase Implementation

#### Phase 1: Data Foundation (Ingestion & Structuring)
1. Parse documents with LlamaParse
2. Extract entities and relationships with LangChain
3. Populate Neo4j knowledge graph
   - Nodes: Facility, UtilityBill, SupplierReport, Equipment, Permit
   - Relationships: Document-to-Facility, Facility-to-Permit connections

#### Phase 2: Intelligence Core (Analysis & Prescription)
1. Query knowledge graph for trends and risks
2. Analyst Agent: Monitor metrics and predict negative outcomes
3. Recommendation Agent: Generate specific mitigation actions
4. Example: "Facility X on track to exceed water permit by 15%" → "Preventative maintenance on Pump #3 could reduce usage by 20%"

#### Phase 3: Communication Layer (Reporting)
1. Create contextual knowledge base with LlamaIndex (previous reports, policies, frameworks)
2. Writer Agent: Synthesize performance data and planned actions
3. Generate draft disclosures using RAG

## Implementation Status
- Disclosure generation capability: **COMPLETED**
- Data ingestion & knowledge graph: **TO BE IMPLEMENTED**
- Prescriptive intelligence: **TO BE IMPLEMENTED**

## Key Repository References

### For Data Ingestion & Knowledge Graph
- **Primary**: `data-foundation` - Complete full-stack application for transforming unstructured data into Neo4j graphs (renamed from neo4j-labs/llm-graph-builder)
- **Alternative**: `rathcoding/knowledge-graph-rag` - Minimal implementation focusing on PDF→LLM→Neo4j pipeline
- **Tutorial**: `Joshua-Yu/graph-rag` - Colab notebook demonstrating LlamaParse to Neo4j workflow

### For Prescriptive Intelligence
- **Primary**: `chrisshayan/agentic-los` - Best example of stateful multi-agent system using LangGraph
- **Alternative**: `NirDiamant/GenAI_Agents` - Simpler example of conditional routing in LangGraph

## Project-Specific Notes
- Focus on three high-value use cases identified by Verdantix for ESG platform leadership
- Knowledge graph approach chosen for handling complex ESG data relationships
- Stateful agent orchestration critical for prescriptive recommendations
- System designed to transform data into "actionable intelligence" for proactive risk management