"""
Response Generator for EHS RAG Agent

This module provides LLM-based response generation, template-based formatting,
source citation formatting, confidence calculation, and answer validation.
"""

import json
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Import LangChain components
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.chat_models.base import BaseChatModel
from langchain_openai import ChatOpenAI

# Import local components
from .query_router import QueryClassification, IntentType
from .context_builder import ContextWindow, ContextSource

# Import logging and monitoring
from ..utils.logging import get_ehs_logger, performance_logger, log_context
from ..utils.tracing import trace_function, SpanKind

logger = get_ehs_logger(__name__)


class ResponseFormat(str, Enum):
    """Different response format types."""
    
    NARRATIVE = "narrative"        # Natural language response
    STRUCTURED = "structured"     # Structured data response
    SUMMARY = "summary"           # Brief summary response
    DETAILED = "detailed"         # Detailed analytical response
    RECOMMENDATION = "recommendation"  # Action-oriented response


@dataclass
class ResponseSource:
    """Source citation in response."""
    
    id: str
    title: str
    relevance_score: float
    content_snippet: str
    metadata: Dict[str, Any]


@dataclass
class GeneratedResponse:
    """Generated response with metadata and sources."""
    
    content: str
    sources: List[ResponseSource]
    confidence_score: float
    response_format: ResponseFormat
    metadata: Dict[str, Any]
    generation_time_ms: float = 0.0
    validation_passed: bool = True
    validation_issues: List[str] = None
    
    def __post_init__(self):
        """Initialize validation issues list if None."""
        if self.validation_issues is None:
            self.validation_issues = []


class ResponseGenerator:
    """
    Response Generator for RAG processing.
    
    Provides LLM-based response generation with template-based formatting,
    source citation management, confidence calculation, and answer validation.
    """
    
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        max_length: int = 1000,
        include_sources: bool = True,
        validate_responses: bool = True,
        temperature: float = 0.1
    ):
        """
        Initialize the response generator.
        
        Args:
            llm: Language model for generation (defaults to GPT-3.5-turbo)
            max_length: Maximum response length
            include_sources: Whether to include source citations
            validate_responses: Whether to validate generated responses
            temperature: Temperature for LLM generation
        """
        self.llm = llm or ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=temperature,
            max_tokens=max_length
        )
        
        self.max_length = max_length
        self.include_sources = include_sources
        self.validate_responses = validate_responses
        self.temperature = temperature
        
        # Response templates for different intents
        self.response_templates = self._initialize_templates()
        
        logger.info(
            "Response Generator initialized",
            model=getattr(self.llm, 'model_name', 'unknown'),
            max_length=max_length,
            include_sources=include_sources,
            validate_responses=validate_responses
        )
    
    def _initialize_templates(self) -> Dict[IntentType, Dict[str, str]]:
        """Initialize response templates for different intent types."""
        return {
            IntentType.CONSUMPTION_ANALYSIS: {
                "system_prompt": """You are an expert EHS data analyst specializing in utility consumption analysis. 
                Generate a comprehensive response about consumption patterns, trends, and insights based on the provided data.
                Focus on actionable insights and highlight any concerning trends or opportunities for improvement.""",
                
                "format_instructions": """Structure your response as follows:
                1. Current Consumption Summary
                2. Trends and Patterns
                3. Key Insights
                4. Recommendations (if applicable)
                
                Use specific data points and metrics when available. Cite sources using [Source X] format."""
            },
            
            IntentType.COMPLIANCE_CHECK: {
                "system_prompt": """You are an expert EHS compliance specialist. 
                Analyze the provided compliance data and generate a clear response about regulatory status, 
                violations, and compliance requirements. Prioritize accuracy and regulatory precision.""",
                
                "format_instructions": """Structure your response as follows:
                1. Compliance Status Overview
                2. Current Violations or Issues (if any)
                3. Regulatory Requirements
                4. Next Actions Required
                
                Be specific about regulations and deadlines. Cite all sources for compliance information."""
            },
            
            IntentType.RISK_ASSESSMENT: {
                "system_prompt": """You are an expert EHS risk assessment specialist. 
                Evaluate the provided data to identify risks, assess their severity, and provide risk mitigation insights.
                Focus on environmental and safety risks with clear risk levels and mitigation strategies.""",
                
                "format_instructions": """Structure your response as follows:
                1. Risk Assessment Summary
                2. Identified Risks and Severity Levels
                3. Risk Factors and Contributing Elements
                4. Mitigation Recommendations
                
                Use clear risk categorization (High/Medium/Low) and cite data sources."""
            },
            
            IntentType.EMISSION_TRACKING: {
                "system_prompt": """You are an expert EHS emissions tracking specialist. 
                Analyze emission data, carbon footprint information, and environmental impact metrics.
                Provide insights on emission trends, targets, and reduction opportunities.""",
                
                "format_instructions": """Structure your response as follows:
                1. Emission Levels Summary
                2. Trends and Comparisons
                3. Target Performance
                4. Reduction Opportunities
                
                Include specific emission values and units. Cite all data sources."""
            },
            
            IntentType.EQUIPMENT_EFFICIENCY: {
                "system_prompt": """You are an expert EHS equipment efficiency analyst. 
                Evaluate equipment performance, efficiency metrics, and maintenance data.
                Focus on optimization opportunities and performance improvements.""",
                
                "format_instructions": """Structure your response as follows:
                1. Equipment Performance Summary
                2. Efficiency Metrics and Trends
                3. Maintenance and Operational Insights
                4. Optimization Recommendations
                
                Include specific efficiency ratings and performance data. Cite sources."""
            },
            
            IntentType.PERMIT_STATUS: {
                "system_prompt": """You are an expert EHS permit compliance specialist. 
                Analyze permit status, expiration dates, renewal requirements, and compliance obligations.
                Provide clear information about permit validity and required actions.""",
                
                "format_instructions": """Structure your response as follows:
                1. Permit Status Overview
                2. Expiration Dates and Renewals
                3. Compliance Requirements
                4. Action Items and Deadlines
                
                Be specific about dates and requirements. Cite permit sources."""
            },
            
            IntentType.GENERAL_INQUIRY: {
                "system_prompt": """You are an expert EHS data analyst. 
                Provide comprehensive information based on the available EHS data.
                Ensure your response is informative, accurate, and helpful for EHS decision-making.""",
                
                "format_instructions": """Structure your response clearly with:
                1. Direct answer to the query
                2. Supporting information and context
                3. Additional relevant insights
                4. Data sources and references
                
                Tailor the structure to best address the specific query."""
            }
        }
    
    @performance_logger(include_args=True, include_result=False)
    @trace_function("generate_response", SpanKind.CLIENT, {"component": "response_generator", "service": "openai"})
    async def generate_response(
        self,
        query: str,
        classification: QueryClassification,
        context_window: ContextWindow,
        response_format: ResponseFormat = ResponseFormat.NARRATIVE
    ) -> GeneratedResponse:
        """
        Generate response using LLM with context.
        
        Args:
            query: Original user query
            classification: Query classification result
            context_window: Built context window
            response_format: Desired response format
            
        Returns:
            GeneratedResponse with content and metadata
        """
        with log_context(
            component="response_generator",
            operation="generate_response",
            intent_type=classification.intent_type.value,
            context_length=context_window.total_length,
            source_count=context_window.source_count
        ):
            start_time = datetime.utcnow()
            
            logger.debug(
                "Generating response",
                query_preview=query[:100],
                intent_type=classification.intent_type.value,
                context_length=context_window.total_length
            )
            
            try:
                # Step 1: Prepare prompt with context
                system_prompt, user_prompt = await self._prepare_prompts(
                    query, classification, context_window, response_format
                )
                
                # Step 2: Generate response using LLM
                raw_response = await self._call_llm(system_prompt, user_prompt)
                
                # Step 3: Process and format response
                formatted_response = await self._format_response(
                    raw_response, response_format
                )
                
                # Step 4: Extract and format source citations
                response_sources = await self._extract_response_sources(
                    formatted_response, context_window
                )
                
                # Step 5: Calculate confidence score
                confidence_score = await self._calculate_response_confidence(
                    formatted_response, context_window, classification
                )
                
                # Step 6: Validate response if enabled
                validation_passed = True
                validation_issues = []
                
                if self.validate_responses:
                    validation_passed, validation_issues = await self._validate_response(
                        formatted_response, query, context_window
                    )
                
                # Create response object
                generation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                response = GeneratedResponse(
                    content=formatted_response,
                    sources=response_sources,
                    confidence_score=confidence_score,
                    response_format=response_format,
                    metadata={
                        "query": query,
                        "intent_type": classification.intent_type.value,
                        "context_sources": context_window.source_count,
                        "llm_model": getattr(self.llm, 'model_name', 'unknown'),
                        "generation_timestamp": datetime.utcnow().isoformat()
                    },
                    generation_time_ms=generation_time,
                    validation_passed=validation_passed,
                    validation_issues=validation_issues
                )
                
                logger.info(
                    "Response generated successfully",
                    response_length=len(formatted_response),
                    source_count=len(response_sources),
                    confidence_score=confidence_score,
                    validation_passed=validation_passed,
                    generation_time_ms=generation_time
                )
                
                return response
                
            except Exception as e:
                generation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                logger.error(
                    "Response generation failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    generation_time_ms=generation_time,
                    exc_info=True
                )
                
                # Return error response
                return GeneratedResponse(
                    content="I apologize, but I encountered an error while generating the response. Please try again.",
                    sources=[],
                    confidence_score=0.0,
                    response_format=response_format,
                    metadata={"error": str(e)},
                    generation_time_ms=generation_time,
                    validation_passed=False,
                    validation_issues=[f"Generation failed: {str(e)}"]
                )
    
    @trace_function("prepare_prompts", SpanKind.INTERNAL, {"response_step": "prompt_preparation"})
    async def _prepare_prompts(
        self,
        query: str,
        classification: QueryClassification,
        context_window: ContextWindow,
        response_format: ResponseFormat
    ) -> Tuple[str, str]:
        """Prepare system and user prompts for LLM."""
        
        # Get templates for this intent type
        templates = self.response_templates.get(
            classification.intent_type,
            self.response_templates[IntentType.GENERAL_INQUIRY]
        )
        
        # Build system prompt
        system_prompt = templates["system_prompt"]
        
        # Add format-specific instructions
        if response_format == ResponseFormat.SUMMARY:
            system_prompt += "\n\nProvide a concise summary response (2-3 paragraphs maximum)."
        elif response_format == ResponseFormat.DETAILED:
            system_prompt += "\n\nProvide a comprehensive, detailed analysis with specific data points."
        elif response_format == ResponseFormat.RECOMMENDATION:
            system_prompt += "\n\nFocus on actionable recommendations and next steps."
        
        system_prompt += f"\n\n{templates['format_instructions']}"
        
        # Add source citation instructions if enabled
        if self.include_sources:
            system_prompt += "\n\nIMPORTANT: Cite sources using [Source X] format where X is the source number from the context."
        
        # Build user prompt with context
        user_prompt = f"""Based on the following context information, please answer this EHS query:

Query: {query}

Context Information:
{context_window.content}

Please provide a comprehensive response that directly addresses the query using the available information."""
        
        return system_prompt, user_prompt
    
    @trace_function("call_llm", SpanKind.CLIENT, {"service": "openai"})
    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM to generate response."""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            raise
    
    @trace_function("format_response", SpanKind.INTERNAL, {"response_step": "formatting"})
    async def _format_response(self, raw_response: str, response_format: ResponseFormat) -> str:
        """Format the raw LLM response according to the specified format."""
        
        # Basic cleanup
        formatted = raw_response.strip()
        
        # Format-specific processing
        if response_format == ResponseFormat.SUMMARY:
            # Ensure summary is concise
            paragraphs = formatted.split('\n\n')
            if len(paragraphs) > 3:
                formatted = '\n\n'.join(paragraphs[:3])
                
        elif response_format == ResponseFormat.STRUCTURED:
            # Ensure structured format with clear sections
            if not re.search(r'^\d+\.|\n\d+\.', formatted):
                # Add basic structure if missing
                lines = formatted.split('\n')
                structured_lines = []
                section_num = 1
                
                for line in lines:
                    if line.strip() and not line.startswith(' '):
                        structured_lines.append(f"{section_num}. {line.strip()}")
                        section_num += 1
                    else:
                        structured_lines.append(line)
                
                formatted = '\n'.join(structured_lines)
        
        # Ensure length constraints
        if len(formatted) > self.max_length:
            # Truncate while preserving structure
            formatted = self._truncate_response(formatted)
        
        return formatted
    
    def _truncate_response(self, response: str) -> str:
        """Truncate response while preserving structure."""
        if len(response) <= self.max_length:
            return response
        
        # Try to break at paragraph boundaries
        paragraphs = response.split('\n\n')
        truncated_parts = []
        current_length = 0
        
        for paragraph in paragraphs:
            if current_length + len(paragraph) + 20 > self.max_length:  # Leave room for ellipsis
                break
            truncated_parts.append(paragraph)
            current_length += len(paragraph) + 2
        
        truncated = '\n\n'.join(truncated_parts)
        if len(truncated) < len(response):
            truncated += "\n\n[Response truncated for length]"
        
        return truncated
    
    @trace_function("extract_response_sources", SpanKind.INTERNAL, {"response_step": "source_extraction"})
    async def _extract_response_sources(
        self,
        response: str,
        context_window: ContextWindow
    ) -> List[ResponseSource]:
        """Extract and format source citations from response."""
        if not self.include_sources:
            return []
        
        # Find source citations in response
        citation_pattern = r'\[Source (\d+)\]'
        citations = re.findall(citation_pattern, response)
        
        response_sources = []
        
        for citation in citations:
            try:
                source_index = int(citation) - 1  # Convert to 0-based index
                
                if 0 <= source_index < len(context_window.sources):
                    context_source = context_window.sources[source_index]
                    
                    response_source = ResponseSource(
                        id=context_source.id,
                        title=context_source.get_citation(),
                        relevance_score=context_source.relevance_score,
                        content_snippet=self._create_content_snippet(context_source.content),
                        metadata=context_source.metadata
                    )
                    
                    response_sources.append(response_source)
                    
            except (ValueError, IndexError):
                logger.warning(f"Invalid source citation: {citation}")
                continue
        
        # Remove duplicates
        unique_sources = []
        seen_ids = set()
        
        for source in response_sources:
            if source.id not in seen_ids:
                unique_sources.append(source)
                seen_ids.add(source.id)
        
        return unique_sources
    
    def _create_content_snippet(self, content: str, max_length: int = 150) -> str:
        """Create a content snippet for source citation."""
        if len(content) <= max_length:
            return content
        
        # Try to break at sentence boundaries
        sentences = content.split('.')
        snippet_parts = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) + 1 > max_length - 3:  # Leave room for ellipsis
                break
            snippet_parts.append(sentence)
            current_length += len(sentence) + 1
        
        snippet = '.'.join(snippet_parts)
        if len(snippet) < len(content):
            snippet += "..."
        
        return snippet
    
    @trace_function("calculate_response_confidence", SpanKind.INTERNAL, {"response_step": "confidence_calculation"})
    async def _calculate_response_confidence(
        self,
        response: str,
        context_window: ContextWindow,
        classification: QueryClassification
    ) -> float:
        """Calculate confidence score for the generated response."""
        
        # Base confidence from classification
        base_confidence = classification.confidence_score
        
        # Context quality factor
        context_quality = min(context_window.average_relevance, 1.0)
        
        # Source coverage factor (based on number of sources used)
        citation_pattern = r'\[Source \d+\]'
        citations_used = len(set(re.findall(citation_pattern, response)))
        source_coverage = min(citations_used / max(context_window.source_count, 1), 1.0)
        
        # Response completeness factor (based on length and structure)
        completeness = min(len(response) / max(self.max_length * 0.5, 1), 1.0)
        
        # Combine factors
        confidence = (
            0.4 * base_confidence +
            0.3 * context_quality +
            0.2 * source_coverage +
            0.1 * completeness
        )
        
        return min(max(confidence, 0.0), 1.0)
    
    @trace_function("validate_response", SpanKind.INTERNAL, {"response_step": "validation"})
    async def _validate_response(
        self,
        response: str,
        query: str,
        context_window: ContextWindow
    ) -> Tuple[bool, List[str]]:
        """Validate the generated response for quality and accuracy."""
        issues = []
        
        # Check minimum length
        if len(response.strip()) < 50:
            issues.append("Response is too short")
        
        # Check for hallucinations (response contains information not in context)
        if self._check_for_hallucinations(response, context_window):
            issues.append("Response may contain information not supported by sources")
        
        # Check for proper source citations (if enabled)
        if self.include_sources:
            citation_pattern = r'\[Source \d+\]'
            citations = re.findall(citation_pattern, response)
            
            if not citations:
                issues.append("Response lacks proper source citations")
            
            # Validate citation numbers
            max_source_index = context_window.source_count
            for citation in citations:
                source_num = int(re.search(r'\d+', citation).group())
                if source_num > max_source_index:
                    issues.append(f"Invalid source citation: Source {source_num}")
        
        # Check for query relevance
        if not self._check_query_relevance(response, query):
            issues.append("Response may not adequately address the query")
        
        # Check for appropriate tone and format
        if not self._check_response_format(response):
            issues.append("Response format or tone may be inappropriate")
        
        validation_passed = len(issues) == 0
        
        return validation_passed, issues
    
    def _check_for_hallucinations(self, response: str, context_window: ContextWindow) -> bool:
        """Check if response contains information not in context."""
        # Simple heuristic: check for specific numbers/dates not in context
        response_numbers = re.findall(r'\b\d{1,4}(?:\.\d+)?\b', response)
        context_content = context_window.content.lower()
        
        for number in response_numbers:
            if number not in context_content and len(number) > 2:  # Ignore small numbers
                return True
        
        return False
    
    def _check_query_relevance(self, response: str, query: str) -> bool:
        """Check if response is relevant to the query."""
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        response_terms = set(re.findall(r'\b\w+\b', response.lower()))
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        query_terms -= stop_words
        
        if not query_terms:
            return True  # Can't determine relevance
        
        # Check overlap
        overlap = len(query_terms.intersection(response_terms))
        relevance_ratio = overlap / len(query_terms)
        
        return relevance_ratio >= 0.3  # At least 30% term overlap
    
    def _check_response_format(self, response: str) -> bool:
        """Check if response has appropriate format and tone."""
        # Check for minimum structure (paragraphs or sections)
        has_structure = (
            '\n\n' in response or  # Multiple paragraphs
            re.search(r'^\d+\.', response, re.MULTILINE) or  # Numbered sections
            ':' in response  # Section headers
        )
        
        # Check for appropriate length distribution
        sentences = response.split('.')
        avg_sentence_length = sum(len(s.strip()) for s in sentences if s.strip()) / max(len(sentences), 1)
        
        # Reasonable sentence length (not too short or too long)
        appropriate_length = 20 <= avg_sentence_length <= 200
        
        return has_structure and appropriate_length
    
    def get_stats(self) -> Dict[str, Any]:
        """Get response generator statistics."""
        return {
            "llm_model": getattr(self.llm, 'model_name', 'unknown'),
            "max_length": self.max_length,
            "include_sources": self.include_sources,
            "validate_responses": self.validate_responses,
            "temperature": self.temperature,
            "total_responses_generated": 0,  # Would be tracked in production
            "average_generation_time_ms": 0.0,  # Would be calculated from history
            "average_confidence_score": 0.0,     # Would be calculated from history
            "validation_pass_rate": 0.0          # Would be calculated from history
        }