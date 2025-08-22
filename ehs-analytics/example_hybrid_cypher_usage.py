"""
Example usage of the HybridCypher Retriever for temporal EHS analytics.

This script demonstrates how to use the new HybridCypher retriever for
complex temporal queries involving trends, patterns, and correlations
in EHS data.
"""

# Example queries that the HybridCypher Retriever can handle:

EXAMPLE_QUERIES = [
    {
        "query": "Show emission trends for facilities that had safety incidents in the past 6 months",
        "expected_type": "SEQUENCE_ANALYSIS",
        "description": "Correlates incidents with subsequent emission patterns"
    },
    {
        "query": "Find equipment with increasing failure rates near permit renewal dates",
        "expected_type": "COMPLIANCE_TIMELINE", 
        "description": "Identifies compliance risks through temporal correlation"
    },
    {
        "query": "Analyze seasonal patterns in water consumption across all facilities",
        "expected_type": "PATTERN_DETECTION",
        "description": "Detects cyclical consumption patterns"
    },
    {
        "query": "Detect anomalous energy usage spikes in the last quarter",
        "expected_type": "ANOMALY_DETECTION",
        "description": "Identifies unusual consumption patterns"
    },
    {
        "query": "Show correlation between production increases and waste generation over time",
        "expected_type": "CORRELATION_ANALYSIS",
        "description": "Analyzes temporal relationships between metrics"
    },
    {
        "query": "Track risk escalation patterns for facilities with multiple violations",
        "expected_type": "RISK_PROGRESSION",
        "description": "Monitors safety and compliance risk evolution"
    }
]

def demonstrate_configuration_usage():
    """
    Demonstrate different configuration options for various use cases.
    
    Note: This is example code showing how the system would be configured.
    Actual usage requires proper Neo4j driver and OpenAI API setup.
    """
    
    print("=== HybridCypher Retriever Configuration Examples ===\n")
    
    # Example 1: Analytics workload configuration
    print("1. Analytics Configuration:")
    print("   - Optimized for deep temporal analysis")
    print("   - Enhanced pattern detection enabled") 
    print("   - Comprehensive performance profile")
    print("   - Extended result limits for thorough analysis")
    
    # Example 2: Compliance monitoring configuration  
    print("\n2. Compliance Configuration:")
    print("   - Emphasizes graph traversal for compliance relationships")
    print("   - Step decay function for deadline awareness")
    print("   - Extended 3-year compliance history window")
    print("   - Deadline proximity boosting enabled")
    
    # Example 3: Risk assessment configuration
    print("\n3. Risk Assessment Configuration:")
    print("   - Balanced retrieval methods for comprehensive analysis")
    print("   - Aggressive anomaly detection (2.0 std dev)")
    print("   - Short 2-month time window for recent risk factors")
    print("   - Exponential decay emphasizing recent events")

def demonstrate_temporal_query_types():
    """Demonstrate the different types of temporal queries supported."""
    
    print("\n=== Supported Temporal Query Types ===\n")
    
    query_types = {
        "TREND_ANALYSIS": {
            "description": "Analyzes temporal trends and changes over time",
            "keywords": ["trend", "increase", "decrease", "growth", "decline"],
            "use_cases": ["Consumption trends", "Emission changes", "Efficiency evolution"]
        },
        "PATTERN_DETECTION": {
            "description": "Detects recurring patterns and cycles",
            "keywords": ["pattern", "cycle", "seasonal", "recurring"],
            "use_cases": ["Seasonal consumption", "Business cycles", "Maintenance patterns"]
        },
        "CORRELATION_ANALYSIS": {
            "description": "Analyzes relationships between different metrics over time",
            "keywords": ["correlation", "relationship", "impact", "effect"],
            "use_cases": ["Production vs emissions", "Weather vs consumption"]
        },
        "ANOMALY_DETECTION": {
            "description": "Identifies unusual patterns and outliers",
            "keywords": ["anomaly", "unusual", "spike", "outlier"],
            "use_cases": ["Consumption spikes", "Equipment failures", "Compliance violations"]
        },
        "SEQUENCE_ANALYSIS": {
            "description": "Analyzes temporal sequences and event ordering",
            "keywords": ["before", "after", "sequence", "timeline"],
            "use_cases": ["Incident-emission correlation", "Compliance sequences"]
        },
        "COMPLIANCE_TIMELINE": {
            "description": "Tracks compliance deadlines and renewal cycles",
            "keywords": ["permit", "deadline", "renewal", "compliance"],
            "use_cases": ["Permit renewals", "Inspection schedules", "Compliance tracking"]
        },
        "RISK_PROGRESSION": {
            "description": "Monitors risk escalation and safety evolution",
            "keywords": ["risk", "incident", "escalation", "safety"],
            "use_cases": ["Incident progression", "Risk escalation", "Safety trends"]
        }
    }
    
    for query_type, info in query_types.items():
        print(f"{query_type}:")
        print(f"  Description: {info['description']}")
        print(f"  Keywords: {', '.join(info['keywords'])}")
        print(f"  Use Cases: {', '.join(info['use_cases'])}")
        print()

def demonstrate_temporal_patterns():
    """Demonstrate EHS-specific temporal patterns."""
    
    print("\n=== EHS Temporal Patterns ===\n")
    
    patterns = {
        "Seasonal Energy Consumption": {
            "pattern": "Higher consumption in winter (Dec-Feb) and summer (Jun-Aug)",
            "peak_factor": 1.3,
            "business_impact": "Budget planning, capacity management"
        },
        "Compliance Cycles": {
            "pattern": "Air permits: 5yr renewal, 12mo reporting, 24mo inspections",
            "lead_time": "6 months advance planning required",
            "business_impact": "Resource allocation, deadline management"
        },
        "Equipment Degradation": {
            "pattern": "Efficiency drop → Maintenance alert → Increased emissions → Failure",
            "time_window": "6 months typical progression",
            "business_impact": "Predictive maintenance, cost optimization"
        },
        "Incident Escalation": {
            "pattern": "Minor incident → Investigation → Corrective action → Major incident",
            "time_window": "90 days risk window",
            "business_impact": "Safety management, risk prevention"
        }
    }
    
    for pattern_name, details in patterns.items():
        print(f"{pattern_name}:")
        print(f"  Pattern: {details['pattern']}")
        if 'peak_factor' in details:
            print(f"  Peak Factor: {details['peak_factor']}x baseline")
        if 'time_window' in details:
            print(f"  Time Window: {details['time_window']}")
        if 'lead_time' in details:
            print(f"  Lead Time: {details['lead_time']}")
        print(f"  Business Impact: {details['business_impact']}")
        print()

def main():
    """Main demonstration function."""
    
    print("HybridCypher Retriever for Temporal EHS Analytics")
    print("=" * 50)
    
    print("\n=== Example Queries ===\n")
    
    for i, example in enumerate(EXAMPLE_QUERIES, 1):
        print(f"{i}. Query: \"{example['query']}\"")
        print(f"   Type: {example['expected_type']}")
        print(f"   Description: {example['description']}")
        print()
    
    demonstrate_configuration_usage()
    demonstrate_temporal_query_types()
    demonstrate_temporal_patterns()
    
    print("\n=== Implementation Features ===\n")
    
    features = [
        "Combined vector similarity, fulltext, and graph traversal with temporal awareness",
        "Support for 7 different temporal query types with automatic classification", 
        "EHS-specific seasonal patterns (energy, water, emissions, incidents)",
        "Compliance cycle tracking with deadline proximity boosting",
        "Advanced anomaly detection with configurable sensitivity",
        "Event sequence pattern matching for risk progression",
        "Configurable temporal weight decay functions (linear, exponential, logarithmic)",
        "Performance optimization profiles (speed, accuracy, balanced, comprehensive)",
        "Time window auto-adjustment based on query scope and data availability",
        "Pattern-aware result boosting and relevance scoring"
    ]
    
    for feature in features:
        print(f"• {feature}")
    
    print("\n=== Next Steps ===\n")
    print("1. Set up Neo4j driver with EHS data schema")
    print("2. Configure OpenAI embeddings for vector search") 
    print("3. Initialize HybridCypher retriever with appropriate configuration")
    print("4. Test with sample temporal queries")
    print("5. Tune configuration based on performance metrics")

if __name__ == "__main__":
    main()