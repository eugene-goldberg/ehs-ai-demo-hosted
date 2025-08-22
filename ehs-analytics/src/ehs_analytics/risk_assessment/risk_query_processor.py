"""
Risk-Aware Query Processor for EHS Analytics

This module provides risk-aware query processing that integrates risk assessment
capabilities with the existing RAG system, enhancing query results with risk context,
predictive insights, and actionable recommendations.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import json

# Import risk assessment components
from . import RiskSeverity, RiskFactor, RiskAssessment, BaseRiskAnalyzer
from .electricity_risk import ElectricityRiskAnalyzer
from .water_risk import WaterRiskAnalyzer
from .waste_risk import WasteRiskAnalyzer
from .time_series import TimeSeriesPredictor

# Import existing system components
from ..agents.query_router import IntentType, QueryClassification, EntityExtraction
from ..retrieval.base import RetrievalResult, RetrievalMetadata, QueryType
from ..agents.context_builder import ContextWindow
from ..agents.response_generator import GeneratedResponse

# Import utilities
from ..utils.logging import get_ehs_logger, performance_logger, log_context
from ..utils.monitoring import get_ehs_monitor
from ..utils.tracing import trace_function, SpanKind

logger = get_ehs_logger(__name__)


class RiskContextType(str, Enum):
    """Types of risk context that can be added to queries."""
    
    CURRENT_RISK = "current_risk"
    PREDICTIVE_RISK = "predictive_risk"
    TREND_ANALYSIS = "trend_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    RECOMMENDATIONS = "recommendations"
    COMPLIANCE_RISK = "compliance_risk"


class RiskFilterLevel(str, Enum):
    """Risk filtering levels."""
    
    ALL = "all"
    HIGH_ONLY = "high_only"
    MEDIUM_AND_HIGH = "medium_and_high"
    ANOMALIES_ONLY = "anomalies_only"


@dataclass
class RiskEnhancedData:
    """Container for data enhanced with risk information."""
    
    original_data: Dict[str, Any]
    risk_score: float
    risk_severity: RiskSeverity
    risk_factors: List[RiskFactor]
    risk_context: str
    forecast_data: Optional[Dict[str, Any]] = None
    anomaly_detected: bool = False
    recommendations: List[str] = field(default_factory=list)


@dataclass
class RiskQueryContext:
    """Risk context for query processing."""
    
    risk_types: List[RiskContextType]
    facility_filters: List[str]
    time_horizon_days: int = 30
    include_forecasts: bool = True
    anomaly_threshold: float = 0.8
    min_risk_score: float = 0.0
    filter_level: RiskFilterLevel = RiskFilterLevel.ALL


@dataclass
class RiskAwareResponse:
    """Enhanced response with risk awareness."""
    
    original_response: GeneratedResponse
    risk_summary: str
    risk_alerts: List[str]
    predictive_insights: List[str]
    recommendations: List[str]
    risk_score: float
    anomalies_detected: int
    forecast_data: Optional[Dict[str, Any]] = None


class RiskAwareQueryProcessor:
    """
    Risk-Aware Query Processor that enhances EHS queries with risk assessment,
    predictive insights, and actionable recommendations.
    
    This processor integrates seamlessly with the existing RAG architecture,
    adding risk context to facility data, equipment monitoring, and compliance queries.
    """
    
    def __init__(
        self,
        neo4j_driver=None,
        llm=None,  # Add support for LLM parameter
        risk_analyzers: Optional[Dict[str, BaseRiskAnalyzer]] = None,
        predictor: Optional[TimeSeriesPredictor] = None,
        cache_ttl_minutes: int = 15,
        max_concurrent_assessments: int = 5,
        risk_threshold: float = 0.7,
        enable_risk_filtering: bool = True
    ):
        """
        Initialize the Risk-Aware Query Processor.
        
        Args:
            neo4j_driver: Neo4j driver instance (optional)
            llm: Language model interface (optional)
            risk_analyzers: Dictionary of risk analyzers by domain
            predictor: Time series predictor for forecasting
            cache_ttl_minutes: Cache time-to-live in minutes
            max_concurrent_assessments: Max concurrent risk assessments
            risk_threshold: Risk score threshold for filtering
            enable_risk_filtering: Whether to enable risk-based filtering
        """
        # Store driver and LLM interface
        self.neo4j_driver = neo4j_driver
        self.llm_interface = llm
        self.risk_threshold = risk_threshold
        self.enable_risk_filtering = enable_risk_filtering
        
        # Initialize risk analyzers
        self.risk_analyzers = risk_analyzers or {
            'electricity': ElectricityRiskAnalyzer(),
            'water': WaterRiskAnalyzer(),
            'waste': WasteRiskAnalyzer()
        }
        
        self.predictor = predictor or TimeSeriesPredictor()
        self.cache_ttl_minutes = cache_ttl_minutes
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_assessments)
        
        # Risk assessment cache
        self._risk_cache: Dict[str, Tuple[datetime, RiskAssessment]] = {}
        self._forecast_cache: Dict[str, Tuple[datetime, Dict[str, Any]]] = {}
        
        # Performance monitoring
        self.monitor = get_ehs_monitor()
        
        logger.info(
            "Risk-Aware Query Processor initialized",
            available_analyzers=list(self.risk_analyzers.keys()),
            cache_ttl_minutes=cache_ttl_minutes,
            max_concurrent_assessments=max_concurrent_assessments,
            has_neo4j=bool(self.neo4j_driver),
            has_llm=bool(self.llm_interface),
            risk_threshold=risk_threshold,
            enable_risk_filtering=enable_risk_filtering
        )
    
    @performance_logger(include_args=True)
    @trace_function("enhance_query_results", SpanKind.INTERNAL, {"component": "risk_processor"})
    async def enhance_query_results(
        self,
        query: str,
        classification: QueryClassification,
        retrieval_results: List[RetrievalResult],
        risk_context: Optional[RiskQueryContext] = None
    ) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        """
        Enhance query results with risk assessment data.
        
        Args:
            query: Original query string
            classification: Query classification result
            retrieval_results: Results from retrievers
            risk_context: Risk context configuration
            
        Returns:
            Tuple of (enhanced_results, risk_metadata)
        """
        with log_context(
            component="risk_processor",
            operation="enhance_query_results",
            intent_type=classification.intent_type.value
        ):
            start_time = datetime.utcnow()
            
            logger.info(
                "Starting risk enhancement of query results",
                query_preview=query[:100],
                intent_type=classification.intent_type.value,
                result_count=len(retrieval_results)
            )
            
            try:
                # Set default risk context based on query type
                if not risk_context:
                    risk_context = self._create_default_risk_context(classification)
                
                # Process results in parallel for performance
                enhanced_results = []
                risk_metadata = {
                    "total_assessments": 0,
                    "high_risk_items": 0,
                    "anomalies_detected": 0,
                    "forecasts_generated": 0,
                    "processing_time_ms": 0.0
                }
                
                # Enhance each retrieval result
                enhancement_tasks = []
                for result in retrieval_results:
                    if result.success and result.data:
                        task = asyncio.create_task(
                            self._enhance_retrieval_result(result, risk_context, classification)
                        )
                        enhancement_tasks.append((result, task))
                
                # Wait for all enhancements to complete
                for original_result, task in enhancement_tasks:
                    try:
                        enhanced_data, item_metadata = await task
                        
                        # Create enhanced result
                        enhanced_result = RetrievalResult(
                            data=enhanced_data,
                            metadata=RetrievalMetadata(
                                strategy=original_result.metadata.strategy,
                                query_type=original_result.metadata.query_type,
                                confidence_score=original_result.metadata.confidence_score,
                                execution_time_ms=original_result.metadata.execution_time_ms,
                                source_nodes=original_result.metadata.source_nodes,
                                cypher_query=original_result.metadata.cypher_query,
                                additional_info={
                                    **original_result.metadata.additional_info,
                                    "risk_enhanced": True,
                                    "risk_metadata": item_metadata
                                }
                            ),
                            success=True,
                            message=original_result.message
                        )
                        
                        enhanced_results.append(enhanced_result)
                        
                        # Update metadata
                        risk_metadata["total_assessments"] += item_metadata.get("assessments_performed", 0)
                        risk_metadata["high_risk_items"] += item_metadata.get("high_risk_count", 0)
                        risk_metadata["anomalies_detected"] += item_metadata.get("anomalies", 0)
                        risk_metadata["forecasts_generated"] += item_metadata.get("forecasts", 0)
                        
                    except Exception as e:
                        logger.error(
                            "Failed to enhance retrieval result",
                            result_id=id(original_result),
                            error=str(e)
                        )
                        # Keep original result on enhancement failure
                        enhanced_results.append(original_result)
                
                # Sort by risk if applicable
                if risk_context.filter_level != RiskFilterLevel.ALL:
                    enhanced_results = self._filter_and_sort_by_risk(enhanced_results, risk_context)
                
                processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                risk_metadata["processing_time_ms"] = processing_time
                
                logger.info(
                    "Risk enhancement completed",
                    enhanced_results=len(enhanced_results),
                    total_assessments=risk_metadata["total_assessments"],
                    high_risk_items=risk_metadata["high_risk_items"],
                    processing_time_ms=processing_time
                )
                
                return enhanced_results, risk_metadata
                
            except Exception as e:
                logger.error(
                    "Risk enhancement failed",
                    query=query,
                    error=str(e),
                    exc_info=True
                )
                # Return original results on failure
                return retrieval_results, {"error": str(e)}
    
    @trace_function("generate_risk_summary", SpanKind.INTERNAL, {"workflow_step": "risk_summary"})
    async def generate_risk_summary(
        self,
        enhanced_results: List[RetrievalResult],
        classification: QueryClassification,
        risk_metadata: Dict[str, Any]
    ) -> str:
        """
        Generate a natural language risk summary from enhanced results.
        
        Args:
            enhanced_results: Risk-enhanced retrieval results
            classification: Query classification
            risk_metadata: Risk processing metadata
            
        Returns:
            Natural language risk summary
        """
        with log_context(workflow_step="risk_summary"):
            logger.debug("Generating risk summary from enhanced results")
            
            try:
                # Extract risk information
                high_risk_count = risk_metadata.get("high_risk_items", 0)
                anomalies_count = risk_metadata.get("anomalies_detected", 0)
                total_assessments = risk_metadata.get("total_assessments", 0)
                
                # Build summary components
                summary_parts = []
                
                # Overall risk status
                if high_risk_count == 0:
                    summary_parts.append("No high-risk conditions detected in the analyzed data.")
                elif high_risk_count == 1:
                    summary_parts.append("1 high-risk condition identified requiring attention.")
                else:
                    summary_parts.append(f"{high_risk_count} high-risk conditions identified requiring attention.")
                
                # Anomaly detection
                if anomalies_count > 0:
                    summary_parts.append(f"{anomalies_count} anomalies detected in operational patterns.")
                
                # Intent-specific insights
                if classification.intent_type == IntentType.RISK_ASSESSMENT:
                    summary_parts.append("Detailed risk analysis provided with predictive insights.")
                elif classification.intent_type == IntentType.COMPLIANCE_CHECK:
                    summary_parts.append("Compliance status evaluated with risk-based prioritization.")
                elif classification.intent_type == IntentType.CONSUMPTION_ANALYSIS:
                    summary_parts.append("Consumption patterns analyzed for efficiency and risk factors.")
                elif classification.intent_type == IntentType.EQUIPMENT_EFFICIENCY:
                    summary_parts.append("Equipment performance assessed with failure risk analysis.")
                
                # Forecasting information
                if risk_metadata.get("forecasts_generated", 0) > 0:
                    summary_parts.append("Predictive forecasts included for proactive risk management.")
                
                summary = " ".join(summary_parts)
                
                logger.debug(
                    "Risk summary generated",
                    summary_length=len(summary),
                    components_included=len(summary_parts)
                )
                
                return summary
                
            except Exception as e:
                logger.error("Failed to generate risk summary", error=str(e))
                return "Risk analysis completed. Review detailed results for specific insights."
    
    @trace_function("get_risk_recommendations", SpanKind.INTERNAL, {"workflow_step": "recommendations"})
    async def get_risk_recommendations(
        self,
        enhanced_results: List[RetrievalResult],
        classification: QueryClassification,
        entities: EntityExtraction
    ) -> List[str]:
        """
        Generate context-aware risk recommendations.
        
        Args:
            enhanced_results: Risk-enhanced results
            classification: Query classification
            entities: Extracted entities from query
            
        Returns:
            List of actionable recommendations
        """
        with log_context(workflow_step="recommendations"):
            logger.debug("Generating risk-based recommendations")
            
            try:
                recommendations = []
                
                # Extract high-risk items
                high_risk_items = []
                for result in enhanced_results:
                    if result.success and result.data:
                        for item in result.data:
                            if isinstance(item, dict) and item.get("risk_severity") == RiskSeverity.HIGH.value:
                                high_risk_items.append(item)
                
                # Generate recommendations based on intent type
                if classification.intent_type == IntentType.RISK_ASSESSMENT:
                    recommendations.extend(self._generate_risk_assessment_recommendations(high_risk_items))
                
                elif classification.intent_type == IntentType.COMPLIANCE_CHECK:
                    recommendations.extend(self._generate_compliance_recommendations(high_risk_items, entities))
                
                elif classification.intent_type == IntentType.CONSUMPTION_ANALYSIS:
                    recommendations.extend(self._generate_consumption_recommendations(high_risk_items, entities))
                
                elif classification.intent_type == IntentType.EQUIPMENT_EFFICIENCY:
                    recommendations.extend(self._generate_equipment_recommendations(high_risk_items, entities))
                
                # Add generic recommendations if no specific ones generated
                if not recommendations and high_risk_items:
                    recommendations.append("Review high-risk items identified in the analysis for immediate attention.")
                    recommendations.append("Consider implementing additional monitoring for elevated risk conditions.")
                
                logger.debug(
                    "Risk recommendations generated",
                    recommendation_count=len(recommendations),
                    high_risk_items=len(high_risk_items)
                )
                
                return recommendations[:10]  # Limit to top 10 recommendations
                
            except Exception as e:
                logger.error("Failed to generate recommendations", error=str(e))
                return ["Review analysis results and consult with EHS specialists for guidance."]
    
    @trace_function("enhance_response", SpanKind.INTERNAL, {"workflow_step": "response_enhancement"})
    async def enhance_response(
        self,
        response: GeneratedResponse,
        enhanced_results: List[RetrievalResult],
        risk_metadata: Dict[str, Any],
        classification: QueryClassification
    ) -> RiskAwareResponse:
        """
        Enhance a generated response with risk context.
        
        Args:
            response: Original generated response
            enhanced_results: Risk-enhanced results
            risk_metadata: Risk metadata
            classification: Query classification
            
        Returns:
            Risk-aware enhanced response
        """
        with log_context(workflow_step="response_enhancement"):
            logger.debug("Enhancing response with risk context")
            
            try:
                # Generate risk summary
                risk_summary = await self.generate_risk_summary(
                    enhanced_results, classification, risk_metadata
                )
                
                # Generate recommendations
                recommendations = await self.get_risk_recommendations(
                    enhanced_results, classification, classification.entities_identified
                )
                
                # Extract risk alerts
                risk_alerts = self._extract_risk_alerts(enhanced_results)
                
                # Generate predictive insights
                predictive_insights = self._extract_predictive_insights(enhanced_results)
                
                # Calculate overall risk score
                overall_risk_score = self._calculate_overall_risk_score(enhanced_results)
                
                # Extract forecast data
                forecast_data = self._extract_forecast_data(enhanced_results)
                
                enhanced_response = RiskAwareResponse(
                    original_response=response,
                    risk_summary=risk_summary,
                    risk_alerts=risk_alerts,
                    predictive_insights=predictive_insights,
                    recommendations=recommendations,
                    risk_score=overall_risk_score,
                    anomalies_detected=risk_metadata.get("anomalies_detected", 0),
                    forecast_data=forecast_data
                )
                
                logger.debug(
                    "Response enhancement completed",
                    risk_score=overall_risk_score,
                    alerts_count=len(risk_alerts),
                    recommendations_count=len(recommendations)
                )
                
                return enhanced_response
                
            except Exception as e:
                logger.error("Failed to enhance response", error=str(e))
                
                # Return minimal enhancement on failure
                return RiskAwareResponse(
                    original_response=response,
                    risk_summary="Risk analysis completed.",
                    risk_alerts=[],
                    predictive_insights=[],
                    recommendations=[],
                    risk_score=0.0,
                    anomalies_detected=0
                )
    
    async def _enhance_retrieval_result(
        self,
        result: RetrievalResult,
        risk_context: RiskQueryContext,
        classification: QueryClassification
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Enhance a single retrieval result with risk data."""
        enhanced_data = []
        metadata = {
            "assessments_performed": 0,
            "high_risk_count": 0,
            "anomalies": 0,
            "forecasts": 0
        }
        
        for item in result.data:
            if isinstance(item, dict):
                try:
                    enhanced_item = await self._enhance_single_item(item, risk_context)
                    enhanced_data.append(enhanced_item)
                    
                    # Update metadata
                    metadata["assessments_performed"] += 1
                    if enhanced_item.get("risk_severity") == RiskSeverity.HIGH.value:
                        metadata["high_risk_count"] += 1
                    if enhanced_item.get("anomaly_detected"):
                        metadata["anomalies"] += 1
                    if enhanced_item.get("forecast_data"):
                        metadata["forecasts"] += 1
                        
                except Exception as e:
                    logger.error(f"Failed to enhance item: {e}")
                    enhanced_data.append(item)  # Keep original on failure
            else:
                enhanced_data.append(item)
        
        return enhanced_data, metadata
    
    async def _enhance_single_item(
        self,
        item: Dict[str, Any],
        risk_context: RiskQueryContext
    ) -> Dict[str, Any]:
        """Enhance a single data item with risk assessment."""
        enhanced_item = item.copy()
        
        # Determine which risk analyzer to use
        analyzer = self._select_risk_analyzer(item)
        if not analyzer:
            return enhanced_item
        
        # Get or compute risk assessment
        cache_key = self._generate_cache_key(item, analyzer.__class__.__name__)
        risk_assessment = await self._get_cached_risk_assessment(cache_key, item, analyzer)
        
        if risk_assessment:
            # Create risk context description from assessment
            risk_context_desc = self._generate_risk_context_description(risk_assessment)
            
            # Add risk information to item
            enhanced_item.update({
                "risk_score": risk_assessment.overall_score,
                "risk_severity": risk_assessment.severity.value,
                "risk_factors": [factor.to_dict() for factor in risk_assessment.factors],
                "risk_context": risk_context_desc,
                "risk_metadata": {
                    "analyzer_used": analyzer.__class__.__name__,
                    "assessment_timestamp": risk_assessment.timestamp.isoformat(),
                    "confidence": risk_assessment.confidence_score
                }
            })
            
            # Add forecast if requested
            if risk_context.include_forecasts:
                forecast = await self._get_forecast_data(item, risk_context.time_horizon_days)
                if forecast:
                    enhanced_item["forecast_data"] = forecast
            
            # Check for anomalies
            anomaly_detected = risk_assessment.overall_score > risk_context.anomaly_threshold
            enhanced_item["anomaly_detected"] = anomaly_detected
        
        return enhanced_item
    
    def _generate_risk_context_description(self, risk_assessment: RiskAssessment) -> str:
        """Generate a descriptive risk context from the assessment."""
        try:
            # Use assessment type and severity to create description
            severity_desc = {
                RiskSeverity.LOW: "Low risk conditions detected",
                RiskSeverity.MEDIUM: "Medium risk conditions require monitoring",  
                RiskSeverity.HIGH: "High risk conditions require immediate attention",
                RiskSeverity.CRITICAL: "Critical risk conditions demand urgent action"
            }
            
            base_desc = severity_desc.get(risk_assessment.severity, "Risk assessment completed")
            
            # Add factor count if available
            if risk_assessment.factors:
                high_risk_factors = [f for f in risk_assessment.factors if f.severity in [RiskSeverity.HIGH, RiskSeverity.CRITICAL]]
                if high_risk_factors:
                    base_desc += f" with {len(high_risk_factors)} high-priority factors"
            
            return base_desc
            
        except Exception as e:
            logger.error(f"Failed to generate risk context description: {e}")
            return "Risk assessment completed"
    
    def _create_default_risk_context(self, classification: QueryClassification) -> RiskQueryContext:
        """Create default risk context based on query classification."""
        intent_context_map = {
            IntentType.RISK_ASSESSMENT: RiskQueryContext(
                risk_types=[RiskContextType.CURRENT_RISK, RiskContextType.PREDICTIVE_RISK],
                facility_filters=[],
                time_horizon_days=30,
                include_forecasts=True,
                filter_level=RiskFilterLevel.ALL
            ),
            IntentType.COMPLIANCE_CHECK: RiskQueryContext(
                risk_types=[RiskContextType.COMPLIANCE_RISK, RiskContextType.CURRENT_RISK],
                facility_filters=classification.entities_identified.facilities,
                time_horizon_days=90,
                include_forecasts=False,
                filter_level=RiskFilterLevel.MEDIUM_AND_HIGH
            ),
            IntentType.CONSUMPTION_ANALYSIS: RiskQueryContext(
                risk_types=[RiskContextType.TREND_ANALYSIS, RiskContextType.ANOMALY_DETECTION],
                facility_filters=classification.entities_identified.facilities,
                time_horizon_days=60,
                include_forecasts=True,
                filter_level=RiskFilterLevel.ALL
            ),
            IntentType.EQUIPMENT_EFFICIENCY: RiskQueryContext(
                risk_types=[RiskContextType.PREDICTIVE_RISK, RiskContextType.TREND_ANALYSIS],
                facility_filters=classification.entities_identified.facilities,
                time_horizon_days=30,
                include_forecasts=True,
                filter_level=RiskFilterLevel.MEDIUM_AND_HIGH
            )
        }
        
        return intent_context_map.get(
            classification.intent_type,
            RiskQueryContext(
                risk_types=[RiskContextType.CURRENT_RISK],
                facility_filters=[],
                time_horizon_days=30,
                include_forecasts=False,
                filter_level=RiskFilterLevel.ALL
            )
        )
    
    def _select_risk_analyzer(self, item: Dict[str, Any]) -> Optional[BaseRiskAnalyzer]:
        """Select appropriate risk analyzer for data item."""
        # Simple selection based on data content
        if any(key in item for key in ['electricity', 'power', 'energy', 'kwh']):
            return self.risk_analyzers.get('electricity')
        elif any(key in item for key in ['water', 'gallons', 'usage']):
            return self.risk_analyzers.get('water')
        elif any(key in item for key in ['waste', 'disposal', 'tons']):
            return self.risk_analyzers.get('waste')
        
        # Default to first available analyzer
        return next(iter(self.risk_analyzers.values())) if self.risk_analyzers else None
    
    def _generate_cache_key(self, item: Dict[str, Any], analyzer_name: str) -> str:
        """Generate cache key for risk assessment."""
        # Create a simple hash of relevant item data
        key_data = {
            'facility': item.get('facility'),
            'timestamp': item.get('timestamp'),
            'analyzer': analyzer_name
        }
        return str(hash(json.dumps(key_data, sort_keys=True)))
    
    async def _get_cached_risk_assessment(
        self,
        cache_key: str,
        item: Dict[str, Any],
        analyzer: BaseRiskAnalyzer
    ) -> Optional[RiskAssessment]:
        """Get cached risk assessment or compute new one."""
        # Check cache
        if cache_key in self._risk_cache:
            cached_time, assessment = self._risk_cache[cache_key]
            if datetime.utcnow() - cached_time < timedelta(minutes=self.cache_ttl_minutes):
                return assessment
        
        # Compute new assessment
        try:
            loop = asyncio.get_event_loop()
            assessment = await loop.run_in_executor(
                self.executor,
                analyzer.analyze,
                item
            )
            
            # Cache result
            self._risk_cache[cache_key] = (datetime.utcnow(), assessment)
            return assessment
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return None
    
    async def _get_forecast_data(
        self,
        item: Dict[str, Any],
        horizon_days: int
    ) -> Optional[Dict[str, Any]]:
        """Get forecast data for item."""
        try:
            # Generate simple cache key for forecasts
            cache_key = f"forecast_{hash(str(item))}"
            
            # Check cache
            if cache_key in self._forecast_cache:
                cached_time, forecast = self._forecast_cache[cache_key]
                if datetime.utcnow() - cached_time < timedelta(minutes=self.cache_ttl_minutes):
                    return forecast
            
            # Generate forecast
            loop = asyncio.get_event_loop()
            forecast = await loop.run_in_executor(
                self.executor,
                self.predictor.predict,
                item,
                horizon_days
            )
            
            # Cache result
            self._forecast_cache[cache_key] = (datetime.utcnow(), forecast)
            return forecast
            
        except Exception as e:
            logger.error(f"Forecast generation failed: {e}")
            return None
    
    def _filter_and_sort_by_risk(
        self,
        results: List[RetrievalResult],
        risk_context: RiskQueryContext
    ) -> List[RetrievalResult]:
        """Filter and sort results by risk level."""
        # This would implement the actual filtering logic
        return results  # Placeholder implementation
    
    def _extract_risk_alerts(self, results: List[RetrievalResult]) -> List[str]:
        """Extract risk alerts from enhanced results."""
        alerts = []
        for result in results:
            if result.success and result.data:
                for item in result.data:
                    if isinstance(item, dict):
                        if item.get("risk_severity") == RiskSeverity.HIGH.value:
                            facility = item.get("facility", "Unknown facility")
                            alerts.append(f"HIGH RISK: {facility} requires immediate attention")
                        if item.get("anomaly_detected"):
                            alerts.append(f"ANOMALY: Unusual patterns detected in {facility}")
        return alerts[:5]  # Limit to top 5 alerts
    
    def _extract_predictive_insights(self, results: List[RetrievalResult]) -> List[str]:
        """Extract predictive insights from forecast data."""
        insights = []
        for result in results:
            if result.success and result.data:
                for item in result.data:
                    if isinstance(item, dict) and item.get("forecast_data"):
                        # Extract insights from forecast data
                        forecast = item["forecast_data"]
                        if forecast.get("trend") == "increasing":
                            insights.append(f"Upward trend predicted for {item.get('facility', 'facility')}")
        return insights[:3]  # Limit to top 3 insights
    
    def _calculate_overall_risk_score(self, results: List[RetrievalResult]) -> float:
        """Calculate overall risk score from results."""
        risk_scores = []
        for result in results:
            if result.success and result.data:
                for item in result.data:
                    if isinstance(item, dict) and "risk_score" in item:
                        risk_scores.append(item["risk_score"])
        
        return sum(risk_scores) / len(risk_scores) if risk_scores else 0.0
    
    def _extract_forecast_data(self, results: List[RetrievalResult]) -> Optional[Dict[str, Any]]:
        """Extract and aggregate forecast data."""
        forecasts = {}
        for result in results:
            if result.success and result.data:
                for item in result.data:
                    if isinstance(item, dict) and item.get("forecast_data"):
                        facility = item.get("facility", "unknown")
                        forecasts[facility] = item["forecast_data"]
        
        return forecasts if forecasts else None
    
    # Recommendation generators
    def _generate_risk_assessment_recommendations(self, high_risk_items: List[Dict]) -> List[str]:
        """Generate recommendations for risk assessment queries."""
        recommendations = []
        if high_risk_items:
            recommendations.append("Implement enhanced monitoring for high-risk facilities")
            recommendations.append("Schedule immediate EHS inspections for flagged locations")
            recommendations.append("Review and update risk mitigation procedures")
        return recommendations
    
    def _generate_compliance_recommendations(self, high_risk_items: List[Dict], entities: EntityExtraction) -> List[str]:
        """Generate recommendations for compliance queries."""
        recommendations = []
        if high_risk_items:
            recommendations.append("Prioritize compliance reviews for high-risk areas")
            recommendations.append("Update regulatory compliance tracking procedures")
        if entities.regulations:
            recommendations.append(f"Review requirements for {', '.join(entities.regulations)}")
        return recommendations
    
    def _generate_consumption_recommendations(self, high_risk_items: List[Dict], entities: EntityExtraction) -> List[str]:
        """Generate recommendations for consumption analysis queries."""
        recommendations = []
        if high_risk_items:
            recommendations.append("Investigate consumption anomalies for efficiency opportunities")
            recommendations.append("Implement energy management controls in high-usage areas")
        return recommendations
    
    def _generate_equipment_recommendations(self, high_risk_items: List[Dict], entities: EntityExtraction) -> List[str]:
        """Generate recommendations for equipment efficiency queries."""
        recommendations = []
        if high_risk_items:
            recommendations.append("Schedule preventive maintenance for at-risk equipment")
            recommendations.append("Consider equipment upgrades for consistently poor performers")
        if entities.equipment:
            recommendations.append(f"Prioritize maintenance for {', '.join(entities.equipment)}")
        return recommendations
    
    async def enhance_query_with_risk_context(
        self,
        query: str,
        classification: QueryClassification,
        entities: EntityExtraction,
        risk_context: Optional[RiskQueryContext] = None
    ) -> str:
        """
        Enhance a query with risk context information.
        
        Args:
            query: Original query string
            classification: Query classification results
            entities: Extracted entities from query
            risk_context: Optional risk context configuration
            
        Returns:
            Enhanced query string with risk context
        """
        enhanced_query_parts = [query]
        
        # Set default risk context if not provided
        if not risk_context:
            risk_context = self._create_default_risk_context(classification)
        
        # Add risk type context
        if risk_context.risk_types:
            risk_types_str = ", ".join([rt.value for rt in risk_context.risk_types])
            enhanced_query_parts.append(f"Consider risk factors: {risk_types_str}")
        
        # Add facility filter context
        if entities.facilities and risk_context.facility_filters:
            facility_filter = f"Focus on facilities: {', '.join(entities.facilities)}"
            enhanced_query_parts.append(facility_filter)
        
        # Add time horizon context
        if risk_context.time_horizon_days > 0:
            enhanced_query_parts.append(f"Include {risk_context.time_horizon_days}-day forecast")
        
        # Add risk level filter
        if risk_context.min_risk_score > 0:
            enhanced_query_parts.append(f"Filter for risk scores above {risk_context.min_risk_score}")
        
        # Add anomaly detection context
        if RiskContextType.ANOMALY_DETECTION in risk_context.risk_types:
            enhanced_query_parts.append("Include anomaly detection analysis")
        
        # Add predictive context
        if risk_context.include_forecasts:
            enhanced_query_parts.append("Include predictive risk insights")
        
        # Combine all parts
        enhanced_query = " | ".join(enhanced_query_parts)
        
        logger.debug(
            "Enhanced query with risk context",
            original_query=query[:100],
            enhanced_query=enhanced_query[:200],
            risk_types=len(risk_context.risk_types),
            facilities=len(entities.facilities) if entities.facilities else 0
        )
        
        return enhanced_query
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the risk processor."""
        analyzer_health = {}
        for name, analyzer in self.risk_analyzers.items():
            try:
                # Simple health check - could be enhanced
                analyzer_health[name] = {"status": "healthy"}
            except Exception as e:
                analyzer_health[name] = {"status": "unhealthy", "error": str(e)}
        
        return {
            "status": "healthy",
            "analyzers": analyzer_health,
            "cache_size": len(self._risk_cache),
            "forecast_cache_size": len(self._forecast_cache),
            "predictor_status": "available" if self.predictor else "unavailable"
        }
    
    def clear_cache(self):
        """Clear all cached risk assessments and forecasts."""
        self._risk_cache.clear()
        self._forecast_cache.clear()
        logger.info("Risk processor cache cleared")


async def create_risk_aware_processor(
    risk_analyzers: Optional[Dict[str, BaseRiskAnalyzer]] = None,
    predictor: Optional[TimeSeriesPredictor] = None,
    **kwargs
) -> RiskAwareQueryProcessor:
    """
    Factory function to create a risk-aware query processor.
    
    Args:
        risk_analyzers: Dictionary of risk analyzers
        predictor: Time series predictor
        **kwargs: Additional configuration options
        
    Returns:
        Initialized RiskAwareQueryProcessor
    """
    logger.info("Creating risk-aware query processor")
    
    processor = RiskAwareQueryProcessor(
        risk_analyzers=risk_analyzers,
        predictor=predictor,
        **kwargs
    )
    
    logger.info("Risk-aware query processor created successfully")
    return processor

# Create aliases for backward compatibility with tests
RiskQueryEnhancer = RiskAwareQueryProcessor
RiskFilteringRetriever = RiskAwareQueryProcessor