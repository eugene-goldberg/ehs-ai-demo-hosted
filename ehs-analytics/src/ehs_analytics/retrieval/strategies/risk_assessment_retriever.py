"""
Risk Assessment Retriever for comprehensive EHS risk analysis.

This module implements a specialized retriever that integrates all Phase 3 risk assessment
components to provide comprehensive risk analysis based on query parameters.
It supports risk domain filtering, time range analysis, forecasting, and anomaly detection.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass

from neo4j import GraphDatabase

from ..base import BaseRetriever, RetrievalResult, RetrievalMetadata, RetrievalStrategy, QueryType, EHSSchemaAware
from ...models import RiskAssessment, RiskSeverity, RiskFactor, RiskThresholds
from ...risk_assessment.water_risk import WaterConsumptionRiskAnalyzer
from ...risk_assessment.electricity_risk import ElectricityRiskAnalyzer
from ...risk_assessment.waste_risk import WasteGenerationRiskAnalyzer
from ...risk_assessment.time_series import TimeSeriesAnalyzer
from ...risk_assessment.forecasting import ForecastingEngine
from ...risk_assessment.anomaly_detection import AnomalyDetectionSystem

logger = logging.getLogger(__name__)


@dataclass
class RiskAnalysisConfig:
    """Configuration for risk assessment retrieval."""
    risk_domains: Optional[List[str]] = None  # ['water', 'electricity', 'waste']
    time_range_days: int = 30
    include_forecasting: bool = False
    include_anomaly_detection: bool = False
    forecast_horizon_days: int = 7
    confidence_threshold: float = 0.7
    max_facilities: int = 10
    aggregation_level: str = "facility"  # "facility", "equipment", "system"


class RiskAssessmentRetriever(BaseRetriever, EHSSchemaAware):
    """
    Comprehensive risk assessment retriever that integrates all Phase 3 components.
    
    This retriever performs comprehensive risk analysis by:
    - Integrating water, electricity, and waste risk analyzers
    - Performing time series analysis on historical data
    - Providing forecasting capabilities when requested
    - Detecting anomalies in consumption patterns
    - Aggregating results into comprehensive risk assessments
    """

    def __init__(
        self,
        neo4j_driver,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Risk Assessment Retriever.
        
        Args:
            neo4j_driver: Neo4j database driver
            config: Configuration dictionary
        """
        super().__init__(config or {})
        self.neo4j_driver = neo4j_driver
        
        # Initialize configuration
        self.risk_config = RiskAnalysisConfig(**config) if config else RiskAnalysisConfig()
        
        # Initialize risk analyzers
        self._init_analyzers()
        
        logger.info("Initialized Risk Assessment Retriever")

    def _init_analyzers(self):
        """Initialize all risk assessment components."""
        try:
            # Initialize domain-specific risk analyzers
            self.water_analyzer = WaterConsumptionRiskAnalyzer()
            self.electricity_analyzer = ElectricityRiskAnalyzer()
            self.waste_analyzer = WasteGenerationRiskAnalyzer()
            
            # Initialize analysis engines
            self.time_series_analyzer = TimeSeriesAnalyzer()
            self.forecasting_engine = ForecastingEngine()
            self.anomaly_detector = AnomalyDetectionSystem()
            
            logger.info("Risk analysis components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize risk analyzers: {e}")
            raise

    async def initialize(self) -> None:
        """Initialize the risk assessment retriever."""
        if self._initialized:
            logger.debug("Risk assessment retriever already initialized")
            return
            
        try:
            logger.info("Initializing Risk Assessment Retriever")
            
            # Verify Neo4j connection using proper async session handling
            session = self.neo4j_driver.session()
            try:
                result = await asyncio.to_thread(session.run, "RETURN 1 AS test")
                test_record = await asyncio.to_thread(result.single)
                if test_record["test"] != 1:
                    raise RuntimeError("Neo4j connection test failed")
            finally:
                await asyncio.to_thread(session.close)
            
            # Initialize analyzers
            await self._init_analyzer_connections()
            
            self._initialized = True
            logger.info("Risk Assessment Retriever initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize risk assessment retriever: {e}")
            raise

    async def _init_analyzer_connections(self):
        """Initialize database connections for analyzers."""
        tasks = []
        
        # Initialize each analyzer if they have async init methods
        for analyzer in [self.water_analyzer, self.electricity_analyzer, 
                        self.waste_analyzer, self.time_series_analyzer,
                        self.forecasting_engine, self.anomaly_detector]:
            if hasattr(analyzer, 'initialize'):
                tasks.append(analyzer.initialize())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def get_strategy(self) -> RetrievalStrategy:
        """Get the retrieval strategy identifier."""
        return RetrievalStrategy.HYBRID  # Risk assessment uses hybrid approach

    async def validate_query(self, query: str) -> bool:
        """Validate if the query can be processed by this retriever."""
        # Check for risk-related keywords
        risk_keywords = [
            'risk', 'assessment', 'analysis', 'compliance', 'threshold',
            'anomaly', 'forecast', 'trend', 'consumption', 'efficiency',
            'waste', 'water', 'electricity', 'safety', 'environmental'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in risk_keywords)

    async def retrieve(
        self,
        query: str,
        query_type: Optional[str] = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> RetrievalResult:
        """
        Perform comprehensive risk assessment based on query parameters.
        
        Args:
            query: Natural language query
            query_type: Type of query (optional)
            limit: Maximum number of results
            **kwargs: Additional parameters including:
                - risk_domains: List of risk domains to analyze
                - time_range: Time range for analysis
                - include_forecasting: Whether to include forecasts
                - include_anomaly_detection: Whether to detect anomalies
                - facility_filter: Specific facility to analyze
                
        Returns:
            RetrievalResult with comprehensive risk assessment data
        """
        try:
            start_time = datetime.now()
            
            # Extract parameters from kwargs
            config_updates = self._extract_analysis_parameters(kwargs)
            
            # Parse query to determine analysis scope
            analysis_scope = self._parse_risk_query(query, config_updates)
            
            # Retrieve relevant data from Neo4j
            raw_data = await self._retrieve_risk_data(analysis_scope)
            
            # Perform comprehensive risk analysis
            risk_assessment = await self._perform_risk_analysis(raw_data, analysis_scope)
            
            # Format response
            response = self._format_risk_response(risk_assessment, query, start_time)
            
            logger.info(f"Risk assessment completed in {response.metadata.execution_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return RetrievalResult(
                success=False,
                data=[],
                metadata=RetrievalMetadata(
                    query=query,
                    query_type=query_type or "risk",
                    strategy=self.get_strategy(),
                    execution_time=processing_time,
                    total_results=0,
                    error_message=str(e)
                )
            )

    def _extract_analysis_parameters(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract risk analysis parameters from kwargs."""
        params = {}
        
        # Risk domains
        if "risk_domains" in kwargs:
            params["risk_domains"] = kwargs["risk_domains"]
        elif "domains" in kwargs:
            params["risk_domains"] = kwargs["domains"]
        
        # Time range
        if "time_range" in kwargs:
            params["time_range"] = kwargs["time_range"]
        elif "days" in kwargs:
            params["time_range_days"] = kwargs["days"]
        
        # Analysis options
        params["include_forecasting"] = kwargs.get("include_forecasting", False)
        params["include_anomaly_detection"] = kwargs.get("include_anomaly_detection", False)
        params["forecast_horizon_days"] = kwargs.get("forecast_horizon", 7)
        
        # Filtering
        if "facility" in kwargs:
            params["facility_filter"] = kwargs["facility"]
        if "equipment" in kwargs:
            params["equipment_filter"] = kwargs["equipment"]
        
        return params

    def _parse_risk_query(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the query to determine analysis scope and parameters."""
        query_lower = query.lower()
        scope = {
            "query": query,
            "risk_domains": params.get("risk_domains", []),
            "time_range_days": params.get("time_range_days", 30),
            "include_forecasting": params.get("include_forecasting", False),
            "include_anomaly_detection": params.get("include_anomaly_detection", False),
            "facility_filter": params.get("facility_filter"),
            "equipment_filter": params.get("equipment_filter")
        }
        
        # Infer risk domains from query if not specified
        if not scope["risk_domains"]:
            if any(word in query_lower for word in ["water", "hydro", "aqua"]):
                scope["risk_domains"].append("water")
            if any(word in query_lower for word in ["electricity", "electric", "power", "energy"]):
                scope["risk_domains"].append("electricity")
            if any(word in query_lower for word in ["waste", "disposal", "garbage", "trash"]):
                scope["risk_domains"].append("waste")
            
            # Default to all domains if none specified
            if not scope["risk_domains"]:
                scope["risk_domains"] = ["water", "electricity", "waste"]
        
        # Infer analysis options from query
        if any(word in query_lower for word in ["forecast", "predict", "future", "projection"]):
            scope["include_forecasting"] = True
        
        if any(word in query_lower for word in ["anomaly", "unusual", "outlier", "abnormal"]):
            scope["include_anomaly_detection"] = True
        
        # Infer time range from query
        if "month" in query_lower:
            scope["time_range_days"] = 30
        elif "quarter" in query_lower:
            scope["time_range_days"] = 90
        elif "year" in query_lower:
            scope["time_range_days"] = 365
        elif "week" in query_lower:
            scope["time_range_days"] = 7
        
        return scope

    async def _retrieve_risk_data(self, scope: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant data from Neo4j for risk analysis."""
        data = {
            "water_data": [],
            "electricity_data": [],
            "waste_data": [],
            "facilities": [],
            "equipment": [],
            "permits": []
        }
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=scope["time_range_days"])
        
        try:
            session = self.neo4j_driver.session()
            try:
                # Retrieve facility information
                facilities_query = """
                MATCH (f:Facility)
                WHERE ($facility_filter IS NULL OR f.name = $facility_filter)
                RETURN f.id AS facility_id, f.name AS facility_name, 
                       f.location AS location, f.type AS facility_type
                LIMIT $max_facilities
                """
                
                facilities_result = await asyncio.to_thread(
                    session.run,
                    facilities_query,
                    facility_filter=scope.get("facility_filter"),
                    max_facilities=self.risk_config.max_facilities
                )
                
                facilities_records = await asyncio.to_thread(facilities_result.data)
                for record in facilities_records:
                    data["facilities"].append({
                        "facility_id": record["facility_id"],
                        "facility_name": record["facility_name"],
                        "location": record["location"],
                        "facility_type": record["facility_type"]
                    })
                
                # Retrieve consumption data based on risk domains
                if "water" in scope["risk_domains"]:
                    data["water_data"] = await self._retrieve_water_data(session, start_date, end_date, scope)
                
                if "electricity" in scope["risk_domains"]:
                    data["electricity_data"] = await self._retrieve_electricity_data(session, start_date, end_date, scope)
                
                if "waste" in scope["risk_domains"]:
                    data["waste_data"] = await self._retrieve_waste_data(session, start_date, end_date, scope)
                
                # Retrieve permits for compliance analysis
                permits_query = """
                MATCH (p:Permit)-[:ISSUED_TO]->(f:Facility)
                WHERE ($facility_filter IS NULL OR f.name = $facility_filter)
                  AND p.expiry_date > $current_date
                RETURN p.permit_number AS permit_number, p.type AS permit_type,
                       p.daily_limit AS daily_limit, p.monthly_limit AS monthly_limit,
                       p.annual_limit AS annual_limit, p.expiry_date AS expiry_date,
                       f.id AS facility_id
                """
                
                permits_result = await asyncio.to_thread(
                    session.run,
                    permits_query,
                    facility_filter=scope.get("facility_filter"),
                    current_date=datetime.now()
                )
                
                permits_records = await asyncio.to_thread(permits_result.data)
                for record in permits_records:
                    data["permits"].append({
                        "permit_number": record["permit_number"],
                        "permit_type": record["permit_type"],
                        "daily_limit": record["daily_limit"],
                        "monthly_limit": record["monthly_limit"],
                        "annual_limit": record["annual_limit"],
                        "expiry_date": record["expiry_date"],
                        "facility_id": record["facility_id"]
                    })
            finally:
                await asyncio.to_thread(session.close)
        
        except Exception as e:
            logger.error(f"Failed to retrieve risk data: {e}")
            raise
        
        return data

    async def _retrieve_water_data(self, session, start_date: datetime, end_date: datetime, scope: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve water consumption data."""
        query = """
        MATCH (w:WaterConsumption)-[:RECORDED_AT]->(f:Facility)
        WHERE w.timestamp >= $start_date AND w.timestamp <= $end_date
          AND ($facility_filter IS NULL OR f.name = $facility_filter)
        RETURN w.timestamp AS timestamp, w.consumption_gallons AS consumption,
               f.id AS facility_id, f.name AS facility_name,
               w.meter_id AS meter_id, w.equipment_id AS equipment_id
        ORDER BY w.timestamp DESC
        """
        
        result = await asyncio.to_thread(
            session.run,
            query,
            start_date=start_date,
            end_date=end_date,
            facility_filter=scope.get("facility_filter")
        )
        
        records = await asyncio.to_thread(result.data)
        return [dict(record) for record in records]

    async def _retrieve_electricity_data(self, session, start_date: datetime, end_date: datetime, scope: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve electricity consumption data."""
        query = """
        MATCH (e:ElectricityConsumption)-[:RECORDED_AT]->(f:Facility)
        WHERE e.timestamp >= $start_date AND e.timestamp <= $end_date
          AND ($facility_filter IS NULL OR f.name = $facility_filter)
        RETURN e.timestamp AS timestamp, e.consumption_kwh AS consumption,
               e.peak_demand AS peak_demand, f.id AS facility_id, f.name AS facility_name,
               e.meter_id AS meter_id, e.equipment_id AS equipment_id
        ORDER BY e.timestamp DESC
        """
        
        result = await asyncio.to_thread(
            session.run,
            query,
            start_date=start_date,
            end_date=end_date,
            facility_filter=scope.get("facility_filter")
        )
        
        records = await asyncio.to_thread(result.data)
        return [dict(record) for record in records]

    async def _retrieve_waste_data(self, session, start_date: datetime, end_date: datetime, scope: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve waste generation data."""
        query = """
        MATCH (w:WasteRecord)-[:GENERATED_AT]->(f:Facility)
        WHERE w.generation_date >= $start_date AND w.generation_date <= $end_date
          AND ($facility_filter IS NULL OR f.name = $facility_filter)
        RETURN w.generation_date AS timestamp, w.amount AS amount,
               w.waste_type AS waste_type, w.disposal_method AS disposal_method,
               f.id AS facility_id, f.name AS facility_name
        ORDER BY w.generation_date DESC
        """
        
        result = await asyncio.to_thread(
            session.run,
            query,
            start_date=start_date,
            end_date=end_date,
            facility_filter=scope.get("facility_filter")
        )
        
        records = await asyncio.to_thread(result.data)
        return [dict(record) for record in records]

    async def _perform_risk_analysis(self, raw_data: Dict[str, Any], scope: Dict[str, Any]) -> RiskAssessment:
        """Perform comprehensive risk analysis using all available analyzers."""
        risk_factors = []
        recommendations = []
        metadata = {
            "analysis_scope": scope,
            "data_points": {
                "water": len(raw_data["water_data"]),
                "electricity": len(raw_data["electricity_data"]),
                "waste": len(raw_data["waste_data"]),
                "facilities": len(raw_data["facilities"])
            }
        }
        
        try:
            # Perform domain-specific risk analysis
            analysis_tasks = []
            
            if "water" in scope["risk_domains"] and raw_data["water_data"]:
                analysis_tasks.append(self._analyze_water_risk(raw_data["water_data"], raw_data["permits"]))
            
            if "electricity" in scope["risk_domains"] and raw_data["electricity_data"]:
                analysis_tasks.append(self._analyze_electricity_risk(raw_data["electricity_data"]))
            
            if "waste" in scope["risk_domains"] and raw_data["waste_data"]:
                analysis_tasks.append(self._analyze_waste_risk(raw_data["waste_data"]))
            
            # Execute domain analyses in parallel
            if analysis_tasks:
                domain_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
                
                for result in domain_results:
                    if isinstance(result, Exception):
                        logger.error(f"Domain analysis failed: {result}")
                        continue
                    
                    if isinstance(result, RiskAssessment):
                        risk_factors.extend(result.factors)
                        recommendations.extend(result.recommendations)
            
            # Perform time series analysis if requested
            if scope.get("include_forecasting") or scope.get("include_anomaly_detection"):
                time_series_data = self._prepare_time_series_data(raw_data)
                
                if time_series_data:
                    ts_analysis = await self._perform_time_series_analysis(
                        time_series_data, scope
                    )
                    risk_factors.extend(ts_analysis.factors)
                    recommendations.extend(ts_analysis.recommendations)
                    metadata["time_series_analysis"] = ts_analysis.metadata
            
            # Create comprehensive risk assessment
            if not risk_factors:
                # Create a default assessment if no data available
                risk_factors.append(RiskFactor(
                    name="Data Availability",
                    value=0.1,
                    weight=1.0,
                    severity=RiskSeverity.LOW,
                    description="Limited data available for comprehensive risk assessment"
                ))
                recommendations.append("Increase data collection frequency for better risk assessment")
            
            assessment = RiskAssessment.from_factors(
                factors=risk_factors,
                recommendations=recommendations,
                assessment_type="comprehensive_risk",
                assessment_id=str(uuid.uuid4()),
                metadata=metadata
            )
            
            return assessment
            
        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")
            # Return minimal assessment on error
            return RiskAssessment(
                overall_score=0.5,
                severity=RiskSeverity.MEDIUM,
                factors=[RiskFactor(
                    name="Analysis Error",
                    value=0.5,
                    weight=1.0,
                    severity=RiskSeverity.MEDIUM,
                    description=f"Risk analysis failed: {str(e)}"
                )],
                recommendations=["Review data quality and system configuration"],
                assessment_type="error",
                confidence_score=0.1,
                metadata={"error": str(e)}
            )

    async def _analyze_water_risk(self, water_data: List[Dict[str, Any]], permits: List[Dict[str, Any]]) -> RiskAssessment:
        """Analyze water consumption risks."""
        try:
            analysis_data = {
                "consumption_data": water_data,
                "permits": permits,
                "analysis_type": "water_consumption"
            }
            
            return await asyncio.to_thread(
                self.water_analyzer.analyze,
                analysis_data
            )
            
        except Exception as e:
            logger.error(f"Water risk analysis failed: {e}")
            return RiskAssessment(
                overall_score=0.5,
                severity=RiskSeverity.MEDIUM,
                factors=[RiskFactor(
                    name="Water Analysis Error",
                    value=0.5,
                    weight=1.0,
                    severity=RiskSeverity.MEDIUM,
                    description=f"Water risk analysis failed: {str(e)}"
                )],
                recommendations=["Review water consumption data quality"]
            )

    async def _analyze_electricity_risk(self, electricity_data: List[Dict[str, Any]]) -> RiskAssessment:
        """Analyze electricity consumption risks."""
        try:
            analysis_data = {
                "consumption_data": electricity_data,
                "analysis_type": "electricity_consumption"
            }
            
            return await asyncio.to_thread(
                self.electricity_analyzer.analyze,
                analysis_data
            )
            
        except Exception as e:
            logger.error(f"Electricity risk analysis failed: {e}")
            return RiskAssessment(
                overall_score=0.5,
                severity=RiskSeverity.MEDIUM,
                factors=[RiskFactor(
                    name="Electricity Analysis Error",
                    value=0.5,
                    weight=1.0,
                    severity=RiskSeverity.MEDIUM,
                    description=f"Electricity risk analysis failed: {str(e)}"
                )],
                recommendations=["Review electricity consumption data quality"]
            )

    async def _analyze_waste_risk(self, waste_data: List[Dict[str, Any]]) -> RiskAssessment:
        """Analyze waste generation risks."""
        try:
            analysis_data = {
                "waste_data": waste_data,
                "analysis_type": "waste_generation"
            }
            
            return await asyncio.to_thread(
                self.waste_analyzer.analyze,
                analysis_data
            )
            
        except Exception as e:
            logger.error(f"Waste risk analysis failed: {e}")
            return RiskAssessment(
                overall_score=0.5,
                severity=RiskSeverity.MEDIUM,
                factors=[RiskFactor(
                    name="Waste Analysis Error",
                    value=0.5,
                    weight=1.0,
                    severity=RiskSeverity.MEDIUM,
                    description=f"Waste risk analysis failed: {str(e)}"
                )],
                recommendations=["Review waste generation data quality"]
            )

    def _prepare_time_series_data(self, raw_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Prepare time series data for analysis."""
        time_series_data = {}
        
        # Process water consumption data
        if raw_data["water_data"]:
            time_series_data["water"] = [
                {
                    "timestamp": record["timestamp"],
                    "value": record["consumption"],
                    "facility_id": record["facility_id"]
                }
                for record in raw_data["water_data"]
                if record.get("consumption") is not None
            ]
        
        # Process electricity consumption data
        if raw_data["electricity_data"]:
            time_series_data["electricity"] = [
                {
                    "timestamp": record["timestamp"],
                    "value": record["consumption"],
                    "facility_id": record["facility_id"]
                }
                for record in raw_data["electricity_data"]
                if record.get("consumption") is not None
            ]
        
        # Process waste generation data
        if raw_data["waste_data"]:
            time_series_data["waste"] = [
                {
                    "timestamp": record["timestamp"],
                    "value": record["amount"],
                    "facility_id": record["facility_id"]
                }
                for record in raw_data["waste_data"]
                if record.get("amount") is not None
            ]
        
        return time_series_data

    async def _perform_time_series_analysis(
        self, 
        time_series_data: Dict[str, List[Dict[str, Any]]], 
        scope: Dict[str, Any]
    ) -> RiskAssessment:
        """Perform time series analysis including forecasting and anomaly detection."""
        risk_factors = []
        recommendations = []
        analysis_metadata = {}
        
        try:
            for domain, data in time_series_data.items():
                if not data:
                    continue
                
                # Perform trend analysis
                trend_analysis = await asyncio.to_thread(
                    self.time_series_analyzer.analyze_trends,
                    data
                )
                
                # Add trend risk factor
                trend_score = abs(trend_analysis.get("trend_slope", 0)) / 100  # Normalize
                trend_score = min(trend_score, 1.0)
                
                risk_factors.append(RiskFactor(
                    name=f"{domain.title()} Consumption Trend",
                    value=trend_score,
                    weight=0.3,
                    severity=RiskSeverity.LOW if trend_score < 0.5 else RiskSeverity.MEDIUM,
                    description=f"Trend analysis for {domain} consumption",
                    metadata=trend_analysis
                ))
                
                # Perform forecasting if requested
                if scope.get("include_forecasting"):
                    forecast_result = await asyncio.to_thread(
                        self.forecasting_engine.forecast,
                        data,
                        horizon_days=scope.get("forecast_horizon_days", 7)
                    )
                    
                    forecast_risk = self._evaluate_forecast_risk(forecast_result)
                    risk_factors.append(forecast_risk)
                    
                    if forecast_result.get("trend", "stable") == "increasing":
                        recommendations.append(f"Monitor {domain} consumption - forecast shows increasing trend")
                
                # Perform anomaly detection if requested
                if scope.get("include_anomaly_detection"):
                    anomalies = await asyncio.to_thread(
                        self.anomaly_detector.detect_anomalies,
                        data
                    )
                    
                    if anomalies:
                        anomaly_risk = RiskFactor(
                            name=f"{domain.title()} Anomalies",
                            value=min(len(anomalies) / 10, 1.0),  # Normalize by expected anomaly count
                            weight=0.4,
                            severity=RiskSeverity.HIGH if len(anomalies) > 5 else RiskSeverity.MEDIUM,
                            description=f"Detected {len(anomalies)} anomalies in {domain} data",
                            metadata={"anomaly_count": len(anomalies), "anomalies": anomalies[:5]}
                        )
                        risk_factors.append(anomaly_risk)
                        recommendations.append(f"Investigate {len(anomalies)} anomalies detected in {domain} consumption")
                
                analysis_metadata[domain] = {
                    "data_points": len(data),
                    "trend_analysis": trend_analysis,
                    "anomaly_count": len(anomalies) if scope.get("include_anomaly_detection") else 0
                }
        
        except Exception as e:
            logger.error(f"Time series analysis failed: {e}")
            risk_factors.append(RiskFactor(
                name="Time Series Analysis Error",
                value=0.3,
                weight=0.1,
                severity=RiskSeverity.LOW,
                description=f"Time series analysis failed: {str(e)}"
            ))
        
        return RiskAssessment.from_factors(
            factors=risk_factors,
            recommendations=recommendations,
            assessment_type="time_series",
            metadata=analysis_metadata
        )

    def _evaluate_forecast_risk(self, forecast_result: Dict[str, Any]) -> RiskFactor:
        """Evaluate risk based on forecast results."""
        trend = forecast_result.get("trend", "stable")
        confidence = forecast_result.get("confidence", 0.5)
        
        if trend == "increasing":
            risk_value = 0.7 * confidence
            severity = RiskSeverity.MEDIUM if risk_value < 0.6 else RiskSeverity.HIGH
        elif trend == "decreasing":
            risk_value = 0.3 * confidence
            severity = RiskSeverity.LOW
        else:
            risk_value = 0.2
            severity = RiskSeverity.LOW
        
        return RiskFactor(
            name="Forecast Risk",
            value=risk_value,
            weight=0.2,
            severity=severity,
            description=f"Risk based on forecast trend: {trend}",
            metadata=forecast_result
        )

    def _format_risk_response(
        self, 
        risk_assessment: RiskAssessment, 
        query: str, 
        start_time: datetime
    ) -> RetrievalResult:
        """Format the risk assessment into a retrieval result."""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Convert risk assessment to response format
        response_data = [risk_assessment.to_dict()]
        
        # Add summary statistics
        summary = {
            "total_risk_factors": len(risk_assessment.factors),
            "critical_factors": len(risk_assessment.get_critical_factors()),
            "high_risk_factors": len(risk_assessment.get_high_risk_factors()),
            "overall_risk_level": risk_assessment.severity.value,
            "confidence_score": risk_assessment.confidence_score,
            "assessment_timestamp": risk_assessment.timestamp.isoformat()
        }
        
        response_data.append({"summary": summary})
        
        return RetrievalResult(
            success=True,
            data=response_data,
            metadata=RetrievalMetadata(
                query=query,
                strategy=self.get_strategy(),
                execution_time=processing_time,
                total_results=len(response_data),
                config_used={
                    "risk_domains": self.risk_config.risk_domains,
                    "time_range_days": self.risk_config.time_range_days,
                    "include_forecasting": self.risk_config.include_forecasting,
                    "include_anomaly_detection": self.risk_config.include_anomaly_detection
                }
            )
        )

    async def cleanup(self) -> None:
        """Clean up resources used by the retriever."""
        try:
            # Cleanup analyzer resources if they have cleanup methods
            cleanup_tasks = []
            
            for analyzer in [self.water_analyzer, self.electricity_analyzer, 
                           self.waste_analyzer, self.time_series_analyzer,
                           self.forecasting_engine, self.anomaly_detector]:
                if hasattr(analyzer, 'cleanup'):
                    cleanup_tasks.append(analyzer.cleanup())
            
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
            await super().cleanup()
            logger.info("Risk Assessment Retriever cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")