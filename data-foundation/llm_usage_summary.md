# LLM Usage Summary in data-foundation

## Files with LLM invocations found:

1. **backend/src/QA_integration.py**
   - Line 236: `rag_chain.invoke()` - RAG chain for Q&A
   - Line 277: `doc_retriever.invoke()` - Document retrieval
   - Line 513: `summarization_chain.invoke()` - Chat summarization
   - Line 554: `graph_chain.invoke()` - Graph-based queries

2. **backend/src/post_processing.py**
   - Line 212: `chain.invoke()` - Node/relationship label mapping

3. **backend/src/communities.py**
   - Line 299: `chain.invoke()` - Community summarization

4. **backend/src/shared/schema_extraction.py**
   - Line 57: `runnable.invoke()` - Schema extraction from text
   - Line 81: `runnable.invoke()` - Schema extraction with descriptions

5. **backend/src/extractors/ehs_extractors.py**
   - Already has transcript logging implemented

## Files that need investigation:
- backend/src/workflows/*.py files
- backend/src/parsers/llama_parser.py
- backend/src/phase1_enhancements/*.py files

## Action items:
1. Add transcript logging to each LLM invocation
2. Import transcript forwarder in each file
3. Log both prompts and responses